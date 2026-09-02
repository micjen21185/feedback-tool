import asyncio
import json
import litellm
import re
import time
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Any, List, Optional

from core.config_loader import Config
from models.schemas import PhaseTelemetry
from observabilty.metrics_engine import ObservabilityManager, CostEngine

litellm.enable_cache = False

# Transient errors that are always worth retrying (network/rate blips).
_TRANSIENT_ERRORS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
)

# Hard per-model OUTPUT-token caps that would otherwise error if exceeded.
# claude-3-5-sonnet-20240620 allows only 4096 output tokens without the max-tokens beta header.
_MODEL_MAX_OUTPUT_TOKENS = {
    "claude-3-5-sonnet-20240620": 4096,
}


class LLMGateway:
    def __init__(self, obs_manager: ObservabilityManager, max_concurrent_requests: int = 15,
                 default_timeout: Optional[int] = None):
        self.obs = obs_manager
        self.session_telemetry: List[PhaseTelemetry] = []
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.default_timeout = default_timeout or Config.LLM_REQUEST_TIMEOUT

    def get_session_telemetry(self) -> List[PhaseTelemetry]:
        return self.session_telemetry

    def reset_session_telemetry(self):
        self.session_telemetry = []

    async def _safe_acompletion(self, retry_on_timeout: bool = True, **kwargs) -> Any:
        # Timeouts are retried for the small map-phase calls (a blip may clear), but NOT for
        # the heavy reduce/monolith calls: retrying a call that is slow *because the work is
        # genuinely large* just burns 4x the wall-clock and times out again anyway.
        retry_types = _TRANSIENT_ERRORS + ((litellm.exceptions.Timeout,) if retry_on_timeout else ())

        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(retry_types),
            reraise=True
        )
        async def _call() -> Any:
            async with self._semaphore:
                return await litellm.acompletion(**kwargs)

        return await _call()

    async def _execute_with_telemetry(self, prompt: str, model: str, agent_role: str, call_kwargs: dict,
                                      retry_on_timeout: bool = True) -> Any:
        start_time = time.time()

        # For local Ollama models, pass the base URL explicitly (with a safe default) instead
        # of relying on litellm's implicit OLLAMA_API_BASE env lookup. Lets the same image talk
        # to a containerized Ollama (GCloud) or a host-installed one (local) via one env var.
        if model.startswith("ollama/") and "api_base" not in call_kwargs:
            call_kwargs["api_base"] = Config.OLLAMA_API_BASE

        # Clamp requested output tokens to the model's hard cap (e.g. Claude 3.5 Sonnet = 4096
        # without the beta header) so a large HEGEMON_MAX_TOKENS doesn't error on that model.
        model_cap = _MODEL_MAX_OUTPUT_TOKENS.get(model)
        if model_cap is not None and call_kwargs.get("max_tokens", 0) > model_cap:
            call_kwargs["max_tokens"] = model_cap

        response = await self._safe_acompletion(retry_on_timeout=retry_on_timeout, **call_kwargs)

        total_time_s = time.time() - start_time

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        response_text = response.choices[0].message.content if response.choices else ""

        if model in Config.LOCAL_MODELS:
            cost_usd = CostEngine.calculate_local_cost(total_time_s, hourly_tco_usd=1.5)
        else:
            # completion_cost returns a single combined float. (cost_per_token returns a
            # (prompt_cost, completion_cost) TUPLE, which SQLite then rejects as a parameter.)
            try:
                cost_usd = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                cost_usd = 0.0

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
            agent_role: str = "Unassigned Agent", retry_on_timeout: bool = True, **kwargs
    ) -> BaseModel:

        full_prompt = f"{prompt}{self._format_schema_contract(schema_class)}"
        kwargs.setdefault("timeout", self.default_timeout)

        if model in Config.COMMERCIAL_MODELS:
            call_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "response_format": schema_class,
            }
            call_kwargs.update(kwargs)
            response = await self._execute_with_telemetry(full_prompt, model, agent_role, call_kwargs,
                                                          retry_on_timeout=retry_on_timeout)
            if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
                return response.choices[0].message.parsed

        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
        }
        call_kwargs.update(kwargs)
        response = await self._execute_with_telemetry(full_prompt, model, agent_role, call_kwargs,
                                                      retry_on_timeout=retry_on_timeout)
        raw_text = response.choices[0].message.content or ""

        try:
            return schema_class.model_validate(json.loads(self._extract_json_object(raw_text)))
        except (json.JSONDecodeError, ValidationError):
            return schema_class.model_validate(self._parse_xml_fallback(raw_text, schema_class))

    async def execute_raw(
            self, prompt: str, model: str, agent_role: str = "Unassigned Agent",
            retry_on_timeout: bool = True, **kwargs
    ) -> str:
        kwargs.setdefault("timeout", self.default_timeout)
        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        call_kwargs.update(kwargs)

        response = await self._execute_with_telemetry(prompt, model, agent_role, call_kwargs,
                                                      retry_on_timeout=retry_on_timeout)
        return response.choices[0].message.content
