import asyncio
import json
import re
import time
from typing import Any, List

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.config_loader import Config
from models.schemas import PhaseTelemetry
from observabilty.metrics_engine import ObservabilityManager, CostEngine

litellm.enable_cache = False


class LLMGateway:
    def __init__(self, obs_manager: ObservabilityManager, max_concurrent_requests: int = 15):
        self.obs = obs_manager
        self.session_telemetry: List[PhaseTelemetry] = []
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    def get_session_telemetry(self) -> List[PhaseTelemetry]:
        return self.session_telemetry

    def reset_session_telemetry(self):
        self.session_telemetry = []

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
                litellm.exceptions.RateLimitError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.Timeout
        )),
        reraise=True
    )
    async def _safe_acompletion(self, **kwargs) -> Any:
        async with self._semaphore:
            return await litellm.acompletion(**kwargs)

    async def _execute_with_telemetry(self, prompt: str, model: str, agent_role: str, call_kwargs: dict) -> Any:
        start_time = time.time()

        response = await self._safe_acompletion(**call_kwargs)

        total_time_s = time.time() - start_time

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        response_text = response.choices[0].message.content if response.choices else ""

        if model in Config.LOCAL_MODELS:
            cost_usd = CostEngine.calculate_local_cost(total_time_s, hourly_tco_usd=1.5)
        else:
            cost_usd = litellm.cost_calculator.cost_per_token(model, tokens_in, tokens_out) or 0.0

        self.obs.log_task(
            model_name=model, phase=agent_role, prompt=prompt, response=response_text,
            tokens_in=tokens_in, tokens_out=tokens_out, total_time_s=total_time_s,
            ttft_ms=0.0, cost_usd=cost_usd
        )

        self.session_telemetry.append(PhaseTelemetry(
            agent_role=agent_role, model_name=model, tokens_in=tokens_in,
            tokens_out=tokens_out, cost_usd=cost_usd, time_s=total_time_s, ttft_ms=0.0,
            prompt_chars=len(prompt), response_chars=len(response_text)
        ))

        return response

    @staticmethod
    def _format_schema_contract(schema_class: type[BaseModel]) -> str:
        lines = []
        for name, info in schema_class.model_fields.items():
            is_list = "List" in str(info.annotation)
            shape = "lista stringów" if is_list else "tekst"
            desc = info.description or ""
            lines.append(f"- \"{name}\" ({shape}): {desc}".rstrip())
        fields = "\n".join(lines)
        return (
            "\n\n<WYMAGANY FORMAT WYJŚCIA>\n"
            "Zwróć WYŁĄCZNIE poprawny obiekt JSON z DOKŁADNIE tymi polami "
            "(bez żadnego dodatkowego tekstu przed ani po):\n"
            f"{fields}\n"
            "Jeśli nie potrafisz wygenerować poprawnego JSON, użyj tagów XML o tych samych nazwach, "
            "np. <thematic_summary>treść</thematic_summary>.\n"
        )

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        cleaned = raw_text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence:
            cleaned = fence.group(1).strip()
        brace = re.search(r"\{.*}", cleaned, re.DOTALL)
        if brace:
            return brace.group(0).strip()
        return cleaned

    @staticmethod
    def _parse_xml_fallback(raw_text: str, schema_class: type[BaseModel]) -> dict:
        parsed_data = {}
        for field_name, field_info in schema_class.model_fields.items():
            annotation = str(field_info.annotation)
            is_list = "List" in annotation or "list" in annotation
            is_str = "str" in annotation and not is_list

            # Only attempt to fill plain string or list-of-string fields from tags.
            # Numeric / nested-model / optional fields are left to their schema defaults.
            if not (is_list or is_str):
                continue

            match = re.search(f"<{field_name}>(.*?)</{field_name}>", raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if is_list:
                    parsed_data[field_name] = [
                        re.sub(r"^\s*[-*•]\s+", "", line).strip()
                        for line in content.split('\n')
                        if re.sub(r"^\s*[-*•]\s+", "", line).strip()
                    ]
                else:
                    parsed_data[field_name] = content
            else:
                parsed_data[field_name] = [] if is_list else ""
        return parsed_data

    async def execute_structured(
            self, prompt: str, schema_class: type[BaseModel], model: str,
            agent_role: str = "Unassigned Agent", **kwargs
    ) -> BaseModel:

        full_prompt = f"{prompt}{self._format_schema_contract(schema_class)}"

        if model in Config.COMMERCIAL_MODELS:
            call_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "response_format": schema_class,
            }
            call_kwargs.update(kwargs)
            response = await self._execute_with_telemetry(full_prompt, model, agent_role, call_kwargs)
            if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
                return response.choices[0].message.parsed

        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
        }
        call_kwargs.update(kwargs)
        response = await self._execute_with_telemetry(full_prompt, model, agent_role, call_kwargs)
        raw_text = response.choices[0].message.content or ""

        try:
            return schema_class.model_validate(json.loads(self._extract_json_object(raw_text)))
        except (json.JSONDecodeError, ValidationError):
            return schema_class.model_validate(self._parse_xml_fallback(raw_text, schema_class))

    async def execute_raw(
            self, prompt: str, model: str, agent_role: str = "Unassigned Agent", **kwargs
    ) -> str:
        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        call_kwargs.update(kwargs)

        response = await self._execute_with_telemetry(prompt, model, agent_role, call_kwargs)
        return response.choices[0].message.content
