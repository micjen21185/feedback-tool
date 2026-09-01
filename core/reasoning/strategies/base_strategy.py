from abc import ABC, abstractmethod
from typing import Dict, Any

from core.llm_gateway import LLMGateway
from models.schemas import ChunkPayload, FactualOutput


class BaseReasoningStrategy(ABC):
    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway

    @abstractmethod
    async def execute(self, chunk: ChunkPayload, context: Dict[str, Any], model: str) -> FactualOutput:
        pass
