import os
import sqlite3
import time
import uuid
from typing import Dict, Any, Optional


class CostEngine:
    @staticmethod
    def calculate_cloud_cost(tokens_in: int, tokens_out: int, price_in_1m: float, price_out_1m: float) -> float:
        """Strategia Chmurowa: Płatność za tokeny."""
        return (tokens_in / 1_000_000) * price_in_1m + (tokens_out / 1_000_000) * price_out_1m

    @staticmethod
    def calculate_local_cost(total_time_s: float, hourly_tco_usd: float) -> float:
        """Strategia Lokalna: Płatność za czas pracy serwera (TCO)."""
        return (total_time_s / 3600.0) * hourly_tco_usd


class ObservabilityManager:
    def __init__(self, db_path: Optional[str] = None):
        # Configurable via env so cloud (read-only FS) can point at a writable path (e.g. /tmp).
        self.db_path = db_path or os.getenv("LLM_BENCHMARK_DB", "llm_benchmark.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS benchmark_logs
                         (
                             request_id
                             TEXT
                             PRIMARY
                             KEY,
                             timestamp
                             REAL,
                             model_name
                             TEXT,
                             phase
                             TEXT,
                             prompt_chars
                             INTEGER,
                             response_chars
                             INTEGER,
                             tokens_in
                             INTEGER,
                             tokens_out
                             INTEGER,
                             ttft_ms
                             REAL,
                             total_time_s
                             REAL,
                             cps
                             REAL,
                             estimated_cost_usd
                             REAL,
                             input_cpt
                             REAL,
                             output_cpt
                             REAL
                         )
                         ''')

    def log_task(self, model_name: str, phase: str, prompt: str, response: str,
                 tokens_in: int, tokens_out: int, total_time_s: float,
                 ttft_ms: float, cost_usd: float) -> Dict[str, Any]:
        """
        Zapisuje miary do bazy, wliczając PODATEK JĘZYKOWY (Characters Per Token - CPT).
        Wysokie CPT (~4.0) = Tanie polskie słowa. Niskie CPT (~2.0) = Drogi "podatek językowy".
        """
        prompt_chars = len(prompt)
        response_chars = len(response)

        cps = response_chars / total_time_s if total_time_s > 0 else 0.0

        input_cpt = prompt_chars / tokens_in if tokens_in > 0 else 0.0
        output_cpt = response_chars / tokens_out if tokens_out > 0 else 0.0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                         INSERT INTO benchmark_logs
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                         ''', (
                             str(uuid.uuid4()), time.time(), model_name, phase, prompt_chars,
                             response_chars, tokens_in, tokens_out, ttft_ms, total_time_s,
                             cps, cost_usd, input_cpt, output_cpt
                         ))

        return {
            "model": model_name,
            "tokens_total": tokens_in + tokens_out,
            "time_s": round(total_time_s, 2),
            "ttft_ms": round(ttft_ms, 2),
            "cps": round(cps, 1),
            "cost_usd": round(cost_usd, 6),
            "input_token_density": round(input_cpt, 2),
            "output_token_density": round(output_cpt, 2)
        }
