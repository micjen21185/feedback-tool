from typing import Dict, Any

from core.config_loader import Config
from core.reasoning.strategies.base_strategy import BaseReasoningStrategy
from core.reasoning.transient_schemas import CoTTransientOutput
from models.schemas import ChunkPayload, FactualOutput


class CoTStrategy(BaseReasoningStrategy):
    async def execute(self, chunk: ChunkPayload, context: Dict[str, Any], model: str) -> FactualOutput:
        clean_text = chunk.text_data.clean_text
        rag_data = context.get('rag', 'Brak kontekstu zewnętrznego.')

        reasoning_prompt = f"""
        {context.get('base_instruction', '')}

        <WYPOWIEDŹ DO OCENY>
        {clean_text}

        <DANE ZEWNĘTRZNE (RAG)>
        {rag_data}

        Wykonaj dogłębną analizę krok po kroku (Chain of Thought). 
        1. Jakie są główne tezy prelegenta?
        2. Jak mają się one do danych RAG?
        3. Czy występują błędy merytoryczne (pamiętając o poziomie widowni)?
        Myśl głośno, zapisz cały swój proces dedukcji.
        """

        raw_reasoning = await self.gateway.execute_raw(
            prompt=reasoning_prompt,
            model=model,
            agent_role="Factual Agent (CoT - Reasoning Phase)",
            timeout=Config.MAP_REQUEST_TIMEOUT
        )

        extraction_prompt = f"""
        Na podstawie poniższej analizy, wyekstrahuj znalezione błędy merytoryczne oraz stwórz krótkie podsumowanie.

        <ANALIZA KROK PO KROKU>
        {raw_reasoning}
        """

        transient_output: CoTTransientOutput = await self.gateway.execute_structured(
            prompt=extraction_prompt,
            schema_class=CoTTransientOutput,
            model=model,
            agent_role="Factual Agent (CoT - Extraction Phase)",
            max_tokens=Config.MAP_MAX_TOKENS,
            timeout=Config.MAP_REQUEST_TIMEOUT
        )

        return FactualOutput(
            chunk_id=f"chunk_{chunk.chunk_meta.index}",
            start_time=chunk.chunk_meta.start_time,
            factual_errors=transient_output.factual_errors,
            scored_errors=transient_output.scored_errors,
            thematic_summary=transient_output.thematic_summary
        )
