from typing import List

from pydantic import BaseModel, Field

from models.schemas import SeverityItem


class CoTTransientOutput(BaseModel):
    thought_process: str = Field(
        description="Zapisz krok po kroku swoje rozumowanie: 1) Analiza intencji prelegenta, 2) Zderzenie z faktami (kontekstem), 3) Konkluzja i uzasadnienie błędu."
    )
    factual_errors: List[str] = Field(
        description="Lista wyłapanych błędów merytorycznych. Pusta lista jeśli brak."
    )
    scored_errors: List[SeverityItem] = Field(
        default_factory=list,
        description="Te same błędy z przypisaną wagą (severity): LOW / MEDIUM / HIGH / CRITICAL."
    )
    thematic_summary: str = Field(
        description="Podsumowanie tematyczne fragmentu (max 2 zdania)."
    )
