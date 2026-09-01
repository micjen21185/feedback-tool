from typing import Dict, Any

from core.llm_gateway import LLMGateway
from core.reasoning.strategies.cot_strategy import CoTStrategy
from core.reasoning.strategies.got_strategy import GoTStrategy
from core.reasoning.strategies.zero_shot_strategy import ZeroShotStrategy
from models.schemas import ChunkPayload, FactualOutput


class ReasoningEngine:
    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway

        self.strategies = {
            "Zero-Shot": ZeroShotStrategy(gateway),
            "CoT": CoTStrategy(gateway),
            "GoT": GoTStrategy(gateway)
        }

    async def process(self, chunk: ChunkPayload, context_data: Dict[str, Any], mode: str, model: str) -> FactualOutput:
        strategy = self.strategies.get(mode, self.strategies["Zero-Shot"])

        try:
            return await strategy.execute(chunk, context_data, model)
        except Exception as e:
            print(f"[Fallback] Strategia {mode} dla chunka {chunk.chunk_meta.index} napotkała błąd: {e}")
            print("[Fallback] Downgrade do trybu Zero-Shot...")

            return await self.strategies["Zero-Shot"].execute(chunk, context_data, model)
