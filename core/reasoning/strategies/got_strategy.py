import asyncio
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple

from core.config_loader import Config
from core.reasoning.strategies.base_strategy import BaseReasoningStrategy
from models.schemas import ChunkPayload, FactualOutput, SeverityItem


class ConvergenceOutput(BaseModel):
    factual_errors: List[str]
    scored_errors: List[SeverityItem] = Field(
        default_factory=list,
        description="Te same błędy z przypisaną wagą (severity): LOW / MEDIUM / HIGH / CRITICAL."
    )
    thematic_summary: str


class ThoughtScore(BaseModel):
    is_plausible: bool = Field(description="Czy hipoteza ma sens w kontekście RAG i tematu?")
    score: int = Field(ge=0, le=10, description="Ocena prawdopodobieństwa tej hipotezy od 0 do 10")


class GoTStrategy(BaseReasoningStrategy):
    async def execute(self, chunk: ChunkPayload, context: Dict[str, Any], model: str) -> FactualOutput:
        speaker_text = chunk.text_data.clean_text
        base_instruction = context.get('base_instruction', '')
        rag_data = context.get('rag', '')

        async def generate_thought(branch_id: int) -> str:
            prompt = f"""
            Fragment wypowiedzi prelegenta (transkrypcja może zawierać zniekształcenia): "{speaker_text}".
            Wygeneruj JEDNĄ logiczną hipotezę (Wersja {branch_id}) dotyczącą tego, co prelegent MIAŁ NA MYŚLI.
            Nie zakładaj z góry, że wystąpił błąd. Skup się na unikalnej interpretacji, innej niż oczywiste skojarzenia.
            """
            return await self.gateway.execute_raw(
                prompt=prompt,
                model=model,
                agent_role=f"GoT - Generator {branch_id}"
            )

        thoughts = await asyncio.gather(*(generate_thought(i) for i in range(1, 4)))

        async def score_thought(thought: str) -> Tuple[str, ThoughtScore]:
            prompt = f"""
            Oceń sensowność poniższej hipotezy w kontekście twardych danych.
            Hipoteza: {thought}
            Kontekst (RAG): {rag_data}
            """
            score = await self.gateway.execute_structured(
                prompt=prompt,
                schema_class=ThoughtScore,
                model=model,
                agent_role="GoT - Scorer"
            )
            return thought, score

        scored_thoughts = await asyncio.gather(*(score_thought(t) for t in thoughts))

        valid_thoughts = [
            thought for thought, score in scored_thoughts
            if score.is_plausible and score.score >= 6
        ]

        if not valid_thoughts:
            valid_thoughts = [speaker_text]
        else:
            valid_thoughts.sort(key=lambda t: next(s.score for th, s in scored_thoughts if th == t), reverse=True)

        formatted_thoughts = "\n".join([f"- {t}" for t in valid_thoughts])

        convergence_prompt = f"""
        {base_instruction}

        Wyselekcjonowane i zweryfikowane hipotezy wypowiedzi prelegenta (od najbardziej prawdopodobnej):
        {formatted_thoughts}

        <DANE ZEWNĘTRZNE (RAG)>
        {rag_data}

        Zidentyfikuj ostateczny sens, zderz go z RAG i wygeneruj końcową ocenę merytoryczną.
        """

        converged_data: ConvergenceOutput = await self.gateway.execute_structured(
            prompt=convergence_prompt,
            schema_class=ConvergenceOutput,
            model=model,
            agent_role="Factual Agent (GoT - Convergence)",
            max_tokens=Config.MAP_MAX_TOKENS
        )

        return FactualOutput(
            chunk_id=f"chunk_{chunk.chunk_meta.index}",
            start_time=chunk.chunk_meta.start_time,
            factual_errors=converged_data.factual_errors,
            scored_errors=converged_data.scored_errors,
            thematic_summary=converged_data.thematic_summary
        )
