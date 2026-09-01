import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    OLLAMA_MODELS_PATH = os.getenv("OLLAMA_MODELS_PATH", r"E:\OllamaModels")

    # Lightweight model used for utility tasks (gatekeeper TAK/NIE, query generation, context compression).
    UTILITY_MODEL = os.getenv("UTILITY_MODEL", "ollama/llama3.2:1b")

    # Per-request LLM timeout (seconds). litellm defaults to 600; long talks + slow local
    # models on the monolith / Hegemon reduce phase routinely exceed that.
    LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "1800"))

    # Longer ceiling for the heavy reduce/monolith calls that produce the full essay.
    HEGEMON_REQUEST_TIMEOUT = int(os.getenv("HEGEMON_REQUEST_TIMEOUT", "3600"))

    # Max OUTPUT tokens for the final report. 4096 is the safe floor (Claude 3.5 Sonnet caps
    # at 4096 without the max-tokens beta header). Capable models can go to 8192+.
    HEGEMON_MAX_TOKENS = int(os.getenv("HEGEMON_MAX_TOKENS", "4096"))

    # Directory for map-phase checkpoints so a crashed long run can resume mid-way.
    CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", os.getenv("LLM_BENCHMARK_DB", "llm_benchmark.db"))

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
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash"
    ]

    @classmethod
    def get_all_models(cls) -> list[str]:
        return cls.COMMERCIAL_MODELS + cls.LOCAL_MODELS
