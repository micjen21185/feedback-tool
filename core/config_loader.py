import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    OLLAMA_MODELS_PATH = os.getenv("OLLAMA_MODELS_PATH", r"E:\OllamaModels")

    # Base URL for the Ollama server. Read explicitly (not just relied upon via litellm's
    # env auto-detection) so it is guaranteed to be passed and has a safe default.
    #   - GCloud / containerized Ollama: default "http://ollama:11434" (compose service name)
    #   - Local dev with host-installed Ollama: set OLLAMA_API_BASE=http://host.docker.internal:11434
    #     (or http://localhost:11434 when running the app outside Docker).
    OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    # Controls thinking/reasoning on Ollama's OpenAI-compatible endpoint (there is no think=
    # flag there — Ollama maps the standard `reasoning_effort` param). "none" disables thinking
    # so the whole output budget goes to the actual report instead of being burned on a hidden
    # reasoning trace (which was arriving in `reasoning_content` with empty `content`).
    # Set to "low"/"medium"/"high" to re-enable reasoning, or "" to omit the param entirely.
    OLLAMA_REASONING_EFFORT = os.getenv("OLLAMA_REASONING_EFFORT", "none")

    # LLMLingua-2 prompt-compression model. Must be a real HF identifier — the small multilingual
    # BERT is the right default (handles Polish, task-agnostic, small/fast). The previous value
    # "llmlingua-small" was NOT a valid model id and caused a load error.
    LLMLINGUA_MODEL = os.getenv("LLMLINGUA_MODEL", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")

    # Fallback model used when the primary is rate-limited / overloaded / over-quota (capacity
    # errors ONLY — not model-not-found, context-length, auth, or local OOM). Empty = disabled.
    # gpt-4o-mini is a cheap, high-availability default; set "" to turn fallback off.
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

    # Lightweight model used for utility tasks (gatekeeper TAK/NIE, query generation, context compression).
    UTILITY_MODEL = os.getenv("UTILITY_MODEL", "ollama/llama3.2:1b")

    # Per-request timeout (seconds) for the small MAP-phase calls. Long enough for a slow local
    # model (14B doing CoT/GoT can take minutes per chunk) but still bounded so a genuinely
    # unreachable/hung Ollama fails in reasonable time rather than the full LLM_REQUEST_TIMEOUT.
    MAP_REQUEST_TIMEOUT = int(os.getenv("MAP_REQUEST_TIMEOUT", "600"))

    # Per-request LLM timeout (seconds). litellm defaults to 600; long talks + slow local
    # models on the monolith / Hegemon reduce phase routinely exceed that.
    LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "1800"))

    # Longer ceiling for the heavy reduce/monolith calls that produce the full essay.
    HEGEMON_REQUEST_TIMEOUT = int(os.getenv("HEGEMON_REQUEST_TIMEOUT", "3600"))

    # Max OUTPUT tokens for the final report. 8192 gives headroom for a deep multi-paragraph
    # essay on a 40+ min talk. NOTE: claude-3-5-sonnet-20240620 caps at 4096 without a beta
    # header, so the gateway clamps that model back to 4096 automatically.
    HEGEMON_MAX_TOKENS = int(os.getenv("HEGEMON_MAX_TOKENS", "8192"))

    # Output cap for per-chunk MAP agents (factual/linguistic). Their output is a small structured
    # object (a few anomalies + short summary), so bound it — across ~60 map calls an unbounded
    # response is wasted tokens and latency. ~1024 is plenty for the schema they return.
    MAP_MAX_TOKENS = int(os.getenv("MAP_MAX_TOKENS", "1024"))

    # Directory for map-phase checkpoints so a crashed long run can resume mid-way.
    CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", os.getenv("LLM_BENCHMARK_DB", "llm_benchmark.db"))

    # Where batch run results are written incrementally (one JSON per run). On Cloud Run the FS is
    # read-only except /tmp, so set RUNS_DIR=/tmp/runs there. Default "runs" for local/GCloud-VM.
    RUNS_DIR = os.getenv("RUNS_DIR", "runs")

    # Where saved MAP-phase results are written (reusable expensive map+combine output). Same
    # /tmp caveat on Cloud Run. Default "maps".
    MAPS_DIR = os.getenv("MAPS_DIR", "maps")

    # LLM-as-judge grounding: total characters of transcript the judge may see, split evenly
    # across start/middle/end regions (deterministic — reproducible across runs). Larger =
    # better-grounded correctness scoring, but more judge tokens and risk of the judge itself
    # hitting lost-in-the-middle. 0 disables multi-region (falls back to a single head slice).
    JUDGE_EXCERPT_CHARS = int(os.getenv("JUDGE_EXCERPT_CHARS", "6000"))
    JUDGE_EXCERPT_REGIONS = int(os.getenv("JUDGE_EXCERPT_REGIONS", "3"))

    # Idea 3 — timestamp-targeted probing: for up to N timestamps the report cites, extract a
    # transcript window (± this many chars) around the mapped position and give it to the judge,
    # so it can verify the report's claim AT that point is actually supported. 0 disables probing.
    JUDGE_PROBE_TIMESTAMPS = int(os.getenv("JUDGE_PROBE_TIMESTAMPS", "5"))
    JUDGE_PROBE_WINDOW_CHARS = int(os.getenv("JUDGE_PROBE_WINDOW_CHARS", "600"))

    LOCAL_MODELS = [
        "ollama/llama3.1:70b",
        "ollama/gemma3:27b",
        "ollama/qwen3.5:35b",
        "ollama/gemma3:4b",
        "ollama/gemma4:e4b",
        "ollama/mistral:7b",
        "ollama/llama3.2:1b",
        "ollama/llama3.2:3b",
        "ollama/llama3.1:8b",
        "ollama/Speakleash/bielik-minitron-7B-v3.0-instruct:Q5_K_M",
        "ollama/Speakleash/bielik-11b-v3.0-instruct:Q5_K_M",
        "ollama/dimpigulsky/pllum-12b",
        "ollama/bbaranow/pllum-12B-q4_k_m",
        "ollama/antoniprzybylik/llama-pllum",
        "ollama/antoniprzybylik/llama-pllum:70b"
    ]

    COMMERCIAL_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20240620",
        "gemini/gemini-3.1-pro",
        "gemini/gemini-3.6-flash"
    ]

    @classmethod
    def get_all_models(cls) -> list[str]:
        return cls.COMMERCIAL_MODELS + cls.LOCAL_MODELS
