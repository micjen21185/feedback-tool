from typing import Dict, Any

from core.config_loader import Config
from core.reasoning.strategies.base_strategy import BaseReasoningStrategy
from models.schemas import ChunkPayload, FactualOutput


class ZeroShotStrategy(BaseReasoningStrategy):
    async def execute(self, chunk: ChunkPayload, context: Dict[str, Any], model: str) -> FactualOutput:
        prompt = f"""
{context.get('base_instruction', '')}

<DANE ZEWNĘTRZNE (RAG / WYSZUKIWARKA)>
{context.get('rag', 'Brak kontekstu zewnętrznego.')}

<KONTEKST HISTORYCZNY (Ostatnie 30s)>
{context.get('historical_summary', '')}

<WYPOWIEDŹ DO OCENY>
{chunk.text_data.clean_text}

ZADANIE (wykonaj OBA kroki):
1. ZAWSZE wypełnij pole `thematic_summary`: 1–2 zdania streszczające, O CZYM MÓWI ten fragment (temat, tezy).
   To pole jest OBOWIĄZKOWE i nigdy nie może być puste — nawet gdy nie ma żadnych błędów.
2. Jeśli w wypowiedzi są twarde błędy merytoryczne (nie licząc dozwolonych uproszczeń dla widowni),
   wypisz je w `scored_errors` (z wagą severity i, jeśli podano źródło, statusem weryfikacji).
   Jeśli błędów nie ma — zostaw `scored_errors` pustą listą. Brak błędów jest normalny.
"""
        output: FactualOutput = await self.gateway.execute_structured(
            prompt=prompt,
            schema_class=FactualOutput,
            model=model,
            agent_role="Factual Agent (Zero-Shot)",
            max_tokens=Config.MAP_MAX_TOKENS,
            timeout=Config.MAP_REQUEST_TIMEOUT
        )

        output.chunk_id = f"chunk_{chunk.chunk_meta.index}"
        output.start_time = chunk.chunk_meta.start_time
        return output
