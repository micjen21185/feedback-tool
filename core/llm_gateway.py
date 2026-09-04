import asyncio
import json
import litellm
import logging
import re
import time
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from typing import Any, List, Optional

from core.config_loader import Config
from models.schemas import PhaseTelemetry
from observabilty.metrics_engine import ObservabilityManager, CostEngine

litellm.enable_cache = False
logger = logging.getLogger(__name__)

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

# Substrings that indicate a NON-retryable failure. Retrying these is pointless — the situation
# is identical on the next attempt — so we fail fast instead of churning through 4 backoff cycles.
# Covers: OOM / killed local runner, AND "model not found" (wrong/absent Ollama model name).
_FATAL_LOCAL_MARKERS = (
    "process has terminated",
    'signal "killed"',
    "signal: killed",
    "out of memory",
    "cudamalloc",
    "failed to allocate",
    "model not found",
    "model '",  # ollama: "model 'X' not found, try pulling it first"
    "not found, try pulling",
    "no such model",
    "pull the model",
)


def _is_fatal_local_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_LOCAL_MARKERS)


def _is_capacity_error(exc: BaseException) -> bool:
    """True for provider capacity/availability failures where trying a DIFFERENT model helps:
    rate limits, overload, quota. NOT for model-not-found, context-length, auth, or local OOM —
    a fallback model wouldn't fix those (and could mask a real config bug)."""
    if _is_fatal_local_error(exc):
        return False
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True
    msg = str(exc).lower()
    markers = ("rate limit", "429", "overloaded", "capacity", "quota", "resource_exhausted",
               "too many requests", "503", "service unavailable")
    # Exclude context-length errors that sometimes co-occur with generic messages.
    if "context length" in msg or "context window" in msg or "maximum context" in msg:
        return False
    return any(m in msg for m in markers)


def check_ollama_models(models: list, api_base: Optional[str] = None) -> list:
    """Preflight: return the subset of the given ollama/* models that are NOT present on the
    Ollama server. Empty list = all good. Used to fail fast with a clear message instead of
    hanging/erroring mid-run on a model-name mismatch. Non-ollama models are ignored.
    Network/parse errors return [] (don't block the run on a flaky preflight)."""
    import urllib.request
    import json as _json

    base = (api_base or Config.OLLAMA_API_BASE).rstrip("/")
    wanted = [m[len("ollama/"):] for m in models if m.startswith("ollama/")]
    if not wanted:
        return []
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        available = {m.get("name", "") for m in data.get("models", [])}
        # Ollama tags include an implicit ':latest'; match both bare and :latest forms.
        available_bare = {n.split(":")[0] for n in available}
        missing = []
        for w in wanted:
            if w in available:
                continue
            if ":" not in w and (w in available_bare or f"{w}:latest" in available):
                continue
            missing.append(w)
        return missing
    except Exception:
        return []


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

        def _should_retry(exc: BaseException) -> bool:
            # Never retry an OOM / killed-runner error, even though it arrives as an
            # APIConnectionError — the memory situation won't change on the next try.
            if _is_fatal_local_error(exc):
                return False
            return isinstance(exc, retry_types)

        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(_should_retry),
            reraise=True
        )
        async def _call() -> Any:
            async with self._semaphore:
                return await litellm.acompletion(**kwargs)

        return await _call()

    @staticmethod
    def _message_text(response: Any) -> str:
        # Prefer visible content, but fall back to the reasoning/thinking channel. Some Ollama
        # models (and reasoning-tuned models) put ALL output into `reasoning_content` and leave
        # `content` empty — reading only `content` yields tokens_out>0 but response_chars=0.
        if not response.choices:
            return ""
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if content:
            return content
        for attr in ("reasoning_content", "reasoning"):
            alt = getattr(msg, attr, None)
            if alt:
                return alt
        return ""

    async def _execute_with_telemetry(self, prompt: str, model: str, agent_role: str, call_kwargs: dict,
                                      retry_on_timeout: bool = True) -> Any:
        start_time = time.time()

        # For local Ollama models, pass the base URL explicitly (with a safe default) instead
        # of relying on litellm's implicit OLLAMA_API_BASE env lookup. Lets the same image talk
        # to a containerized Ollama (GCloud) or a host-installed one (local) via one env var.
        if model.startswith("ollama/") and "api_base" not in call_kwargs:
            call_kwargs["api_base"] = Config.OLLAMA_API_BASE

        # Suppress (or tune) Ollama's hidden reasoning trace via the standard reasoning_effort
        # param — the OpenAI-compat endpoint has no think= flag. "none" keeps the whole output
        # budget for the actual report instead of a reasoning trace that lands in reasoning_content.
        if (model.startswith("ollama/") and Config.OLLAMA_REASONING_EFFORT
                and "reasoning_effort" not in call_kwargs):
            call_kwargs["reasoning_effort"] = Config.OLLAMA_REASONING_EFFORT

        # Clamp requested output tokens to the model's hard cap (e.g. Claude 3.5 Sonnet = 4096
        # without the beta header) so a large HEGEMON_MAX_TOKENS doesn't error on that model.
        model_cap = _MODEL_MAX_OUTPUT_TOKENS.get(model)
        if model_cap is not None and call_kwargs.get("max_tokens", 0) > model_cap:
            call_kwargs["max_tokens"] = model_cap

        try:
            response = await self._safe_acompletion(retry_on_timeout=retry_on_timeout, **call_kwargs)
        except Exception as exc:
            # Fallback ONLY on capacity errors (rate limit / overload / quota), and only if a
            # fallback model is configured and differs from the one that just failed. We swap the
            # model here — not via litellm's own fallbacks — so telemetry records the model that
            # ACTUALLY served the request (provenance matters for the research comparison).
            fallback = Config.FALLBACK_MODEL
            if fallback and fallback != model and _is_capacity_error(exc):
                logger.warning("[fallback] '%s' capacity error (%s) → retrying on '%s'.",
                               model, type(exc).__name__, fallback)
                model = fallback
                agent_role = f"{agent_role} [fallback→{fallback}]"
                call_kwargs["model"] = fallback
                # Re-apply the fallback model's own output cap; drop ollama-only api_base if the
                # fallback is a commercial model.
                cap = _MODEL_MAX_OUTPUT_TOKENS.get(fallback)
                if cap is not None and call_kwargs.get("max_tokens", 0) > cap:
                    call_kwargs["max_tokens"] = cap
                if not fallback.startswith("ollama/"):
                    call_kwargs.pop("api_base", None)
                    call_kwargs.pop("reasoning_effort", None)
                response = await self._safe_acompletion(retry_on_timeout=retry_on_timeout, **call_kwargs)
            else:
                raise

        total_time_s = time.time() - start_time

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        response_text = self._message_text(response)

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

    @staticmethod
    def _repair_json(text: str):
        """Best-effort recovery of JSON-ish output from small models: strip trailing prose after
        the last closing brace, remove trailing commas, and normalize smart quotes. Returns a
        parsed dict/list on success, else None. Does NOT fabricate content — only fixes syntax."""
        import json as _json
        s = (text or "").strip()
        if not s:
            return None
        # Isolate the outermost {...} if there's leading/trailing prose.
        first, last = s.find("{"), s.rfind("}")
        if first != -1 and last != -1 and last > first:
            s = s[first:last + 1]
        # Normalize common small-model quirks.
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas before } or ]
        for candidate in (s, s.replace("'", '"')):
            try:
                return _json.loads(candidate)
            except Exception:
                continue
        return None

    @staticmethod
    def _looks_empty(obj: BaseModel) -> bool:
        """True if a parsed structured object has no meaningful content — a sign the parse failed
        rather than the model genuinely finding nothing. Checks common content fields."""
        d = obj.model_dump()
        content_keys = ("thematic_summary", "factual_errors", "scored_errors", "anomalies",
                        "scored_anomalies", "dominant_tendencies", "justification")
        present = [k for k in content_keys if k in d]
        if not present:
            return False  # schema has no content fields we track → don't flag
        return all(not d.get(k) for k in present)

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
        raw_text = self._message_text(response)

        # Parse chain: strict JSON → tolerant JSON repair → XML-tag fallback. Small local models
        # (e.g. llama3.1:8b) often emit JSON-ish output with fences/trailing commas/extra prose, or
        # ignore the tag format entirely — which previously produced an ALL-EMPTY object that then
        # looked like "no findings / perfect speech". We now try harder and log a real parse miss.
        parsed = None
        try:
            parsed = schema_class.model_validate(json.loads(self._extract_json_object(raw_text)))
        except (json.JSONDecodeError, ValidationError):
            repaired = self._repair_json(self._extract_json_object(raw_text))
            if repaired is not None:
                try:
                    parsed = schema_class.model_validate(repaired)
                except ValidationError:
                    parsed = None
        if parsed is None:
            parsed = schema_class.model_validate(self._parse_xml_fallback(raw_text, schema_class))

        # Diagnostic: if the model clearly produced text (tokens spent) but every meaningful field
        # came back empty, the parse FAILED — surface it instead of silently scoring it as clean.
        if raw_text.strip() and self._looks_empty(parsed):
            logger.warning("[%s] structured parse yielded an EMPTY object from %d chars of model output "
                           "(model=%s). Raw head: %s",
                           agent_role, len(raw_text), model, raw_text[:300].replace("\n", " "))
        return parsed

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
        return self._message_text(response)
