from typing import Any, Dict

from core.llm_gateway import LLMGateway
from core.reasoning.reasoning_engine import ReasoningEngine
from models.schemas import ChunkPayload, FactualOutput, LectureMetadata
from tools.gatekeeper import KnowledgeGatekeeper
from tools.knowledge_engine import KnowledgeEngine


class FactualAgent:
    def __init__(
            self,
            reasoning_engine: ReasoningEngine,
            gatekeeper: KnowledgeGatekeeper,
            knowledge_engine: KnowledgeEngine,
            gateway: LLMGateway,
            config: dict
    ):
        self.reasoning = reasoning_engine
        self.gatekeeper = gatekeeper
        self.knowledge = knowledge_engine
        self.gateway = gateway
        self.config = config
        # Per-talk budget: cap how many chunks may trigger RAG + CoT/GoT escalation,
        # to bound cost/latency on long talks. Default: max(8, unlimited if not set).
        self._rag_budget = config.get("rag_budget", 12)
        self._rag_used = 0

    def _build_factual_context_rules(self, knowledge_level: str) -> str:
        return f"""Jesteś weryfikatorem merytoryki. Twoim zadaniem jest ocena faktów, ale z pełną świadomością, że prelegent mówi do widowni na poziomie: {knowledge_level}.

<KONTRAKT OSADZENIA (ANTY-HALUCYNACJA) — NADRZĘDNY>
1. Oceniaj WYŁĄCZNIE to, co prelegent NAPRAWDĘ powiedział w dostarczonym fragmencie. NIGDY nie wymyślaj tez, liczb, twierdzeń ani tematów, których w tekście nie ma.
2. Każdy zgłoszony błąd MUSI zawierać dosłowny cytat z wypowiedzi (fragment w cudzysłowie), do którego się odnosisz. Jeśli nie potrafisz zacytować konkretnego fragmentu, NIE zgłaszaj błędu.
3. Brak błędów to POPRAWNA i częsta odpowiedź. Jeśli fragment jest trywialny, krótki, poprawny lub to zwykła narracja — zwróć PUSTE listy błędów. Nie szukaj problemów na siłę.
4. Nie oceniaj stylu, tempa ani dykcji — to zadanie innego agenta. Zajmujesz się tylko faktami.
5. Jeśli transkrypcja jest niewyraźna/zniekształcona, załóż życzliwą interpretację i NIE traktuj zniekształcenia transkrypcji jako błędu merytorycznego prelegenta.

ŻELAZNA ZASADA UPROSZCZEŃ (DOPASOWANIE DO WIDOWNI):
Zanim uznasz fragment za "błąd merytoryczny" (Factual Error), sprawdź poziom wiedzy słuchaczy.
- Jeśli poziom to "Podstawowy" (Beginner/Junior), prelegent ma pełne prawo do skrótów myślowych, analogii i pomijania skrajnych wyjątków akademickich. Upraszczanie trudnych definicji (np. "Zmienna to takie pudełko na dane") NIE JEST BŁĘDEM, lecz znakomitą praktyką pedagogiczną. Nie umieszczaj takich uproszczeń w tablicy `factual_errors`.
- Jeśli poziom to "Ekspert" (Senior/Architekt), wymagaj absolutnej precyzji akademickiej i technicznej. Uproszczenia traktuj jako brak kompetencji.

PRZYKŁAD EWALUACJI:
Sytuacja: Mówca twierdzi, że "Ziemia to idealna kula".
Ocena dla widowni 'Podstawowy': Ignoruj. To dopuszczalne uproszczenie.
Ocena dla widowni 'Ekspert': Zaraportuj błąd (Ziemia to elipsoida obrotowa/geoida).
"""

    async def analyze(self, chunk: ChunkPayload, metadata: LectureMetadata, use_tools: bool) -> FactualOutput:
        clean_text = chunk.text_data.clean_text
        slide_ocr = chunk.context_data.pdf_text if chunk.context_data.pdf_text else "Brak danych OCR dla tego slajdu."

        pedagogical_rules = self._build_factual_context_rules(metadata.knowledge_level)

        base_context_prompt = f"""
ROLA PRELEGENTA: {metadata.speaker_role}
TEMAT GŁÓWNY: {metadata.main_topic}
GRUPA DOCELOWA: {metadata.target_audience}

{pedagogical_rules}

<KONTEKST WIZUALNY (Slajdy)>
Aktualny Slajd ID: {chunk.chunk_meta.slide_id} (Czy powrót: {chunk.chunk_meta.is_return_to_slide})
Treść slajdu (OCR): {slide_ocr}
"""

        context_data: Dict[str, Any] = {
            "rag": "Brak zewnętrznego kontekstu.",
            "historical_summary": getattr(chunk.trailing_fact_summary, 'prev_summary', 'Brak'),
            "base_instruction": base_context_prompt
        }

        selected_mode = "Zero-Shot"

        # Reserve a budget slot up-front (before any await) so concurrent chunks can't
        # all pass the check and overshoot the cap. Release it if the gate says "no".
        if use_tools and self._rag_used < self._rag_budget:
            self._rag_used += 1
            if await self.gatekeeper.needs_external_knowledge(chunk):
                context_data["rag"] = await self.knowledge.retrieve(clean_text)
                if chunk.linguistic_data.avg_transcription_confidence < 0.75 or chunk.linguistic_data.unclear_words_count > 2:
                    selected_mode = "GoT"
                else:
                    selected_mode = "CoT"
            else:
                self._rag_used -= 1

        return await self.reasoning.process(
            chunk=chunk,
            context_data=context_data,
            mode=selected_mode,
            model=self.config.get("factual_model")
        )
