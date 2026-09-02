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

Przeanalizuj powyższą wypowiedź. Jeśli znajdziesz błędy merytoryczne (niebędące dozwolonymi uproszczeniami), wpisz je na listę. Stwórz krótkie podsumowanie tematyczne.
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
