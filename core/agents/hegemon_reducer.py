import re
from typing import List

from core.llm_gateway import LLMGateway
from models.schemas import HegemonOutput, LectureMetadata, DeepAnalysis, ConstructiveFeedback


class HegemonReducer:
    def __init__(self, gateway: LLMGateway, hegemon_model: str):
        self.gateway = gateway
        self.model = hegemon_model

    def _build_hegemon_prompt(self, role: str, audience: str, topic: str, level: str) -> str:
        return f"""Jesteś Głównym Ewaluatorem (Hegemonem) i wybitnym mentorem wystąpień publicznych. 
Twój cel to wygenerowanie najwyższej klasy, bezlitosnego, ale wysoce konstruktywnego raportu mentorskiego.

PROFIL WYSTĄPIENIA:
- Rola prelegenta: {role}
- Grupa docelowa: {audience} (Poziom: {level})
- Temat wystąpienia: {topic}

<ŻELAZNE ZASADY EWALUACJI>
1. KORELACJA DWUTOROWA: Analizuj chronologię. Jeśli w osi behawioralnej widzisz skok stresu (np. pauzy, wypełniacze), sprawdź co działo się w osi merytorycznej i wyciągnij wnioski (Sytuacja -> Stres -> Wpływ).
2. ZASADY KOMUNIKACYJNE: Punktuj nadużywanie strony biernej, brak pauz przed puentą oraz rozwlekłość.
3. MODEL SBI: Stosuj twardo model Sytuacja-Zachowanie-Wpływ. Zero pustych pochwał typu "dobra robota".
4. HIERARCHIA WAGI (SEVERITY): Obserwacje są otagowane wagą (CRITICAL/HIGH/MEDIUM/LOW). Najpierw omawiaj CRITICAL i HIGH (np. wulgaryzmy, rażące błędy, utrata wątku). Drobiazgi (LOW) najwyżej zbiorczo. Nie stawiaj drobiazgu ponad problemem krytycznym.
5. WERDYKT ILOŚCIOWY: Blok <WERDYKT ILOŚCIOWY> zawiera gotowe, deterministyczne oceny (np. tempo mowy). Opieraj się na nich, nie zaprzeczaj im.
6. PREZENTACJA (jeśli dostępna): Jeśli otrzymasz sekcję pokrycia slajdów i przepływu, skomentuj pominięte punkty, powroty do slajdów oraz zbyt krótkie/długie slajdy (np. zasugeruj mniej, ale bogatszych slajdów).

<WYMOGI FORMATOWANIA>
ZABRANIA CI SIĘ używania formatu JSON. Wygeneruj odpowiedź używając wyłącznie poniższych tagów. 
Wewnątrz tagu <executive_summary> napisz wieloakapitowy, głęboki esej mentorski używając formatowania Markdown (pogrubienia, cytaty).
ZASADA ZNACZNIKÓW CZASU: przy każdej konkretnej obserwacji podawaj znacznik czasu w formacie [MM:SS] (np. "[12:30] utrata wątku"). Rozłóż obserwacje po całym wystąpieniu — od początku, przez środek, aż po koniec — nie skupiaj się tylko na początku i końcu.

Wymagane tagi:
<factual_summary> (Ocena merytoryki) </factual_summary>
<linguistic_summary> (Ocena dynamiki i języka) </linguistic_summary>
<missed_context> (Wypunktowane pominięte wątki - opcjonalnie) </missed_context>
<executive_summary> (Twój główny esej mentorski) </executive_summary>
<strengths>
- (mocna strona 1)
- (mocna strona 2)
</strengths>
<areas_for_improvement>
- (obszar 1)
</areas_for_improvement>
<actionable_tips>
- (porada 1)
</actionable_tips>
<overall_message> (Jedno, mocne zdanie podsumowujące dla prelegenta) </overall_message>
"""

    def _extract_tag(self, text: str, tag: str) -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_list(self, text: str, tag: str) -> List[str]:
        content = self._extract_tag(text, tag)
        items = []
        for line in content.split('\n'):
            # Strip only a single leading bullet marker so inline Markdown (**bold**) is preserved.
            cleaned = re.sub(r"^\s*[-*•]\s+", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    async def generate_report(self, metadata: LectureMetadata, thematic_blocks: List[str],
                              behavioral_profiles: List[str], presentation_context: str = "",
                              scorecard=None) -> HegemonOutput:
        system_prompt = self._build_hegemon_prompt(
            role=metadata.speaker_role, audience=metadata.target_audience,
            topic=metadata.main_topic, level=metadata.knowledge_level
        )

        presentation_block = ""
        if presentation_context:
            presentation_block = f"\n<POKRYCIE SLAJDÓW I PRZEPŁYW PREZENTACJI>\n{presentation_context}\n"

        score_block = ""
        if scorecard is not None:
            slide_part = ""
            if scorecard.slide_coverage_score is not None:
                slide_part = f" | Pokrycie slajdów: {scorecard.slide_coverage_score}/100"
            score_block = (
                f"\n<OCENA ILOŚCIOWA (deterministyczna, użyj jako kotwicy TONU)>\n"
                f"Ocena łączna: {scorecard.overall_score}/100 → WERDYKT: {scorecard.readiness_verdict}\n"
                f"Merytoryka: {scorecard.factual_score}/100 | Język: {scorecard.linguistic_score}/100{slide_part}\n"
                f"Dopasuj ton eseju do werdyktu: przy wysokim wyniku szczerze chwal i doceniaj, "
                f"przy niskim jasno zaznacz, że wystąpienie wymaga poważnej pracy. Nie zaprzeczaj werdyktowi.\n"
            )

        thematic_text = "\n".join(thematic_blocks) if isinstance(thematic_blocks, list) else str(thematic_blocks)
        behavioral_text = "\n".join(behavioral_profiles) if isinstance(behavioral_profiles, list) else str(
            behavioral_profiles)

        user_prompt = f"""
<CHRONOLOGICZNE BLOKI MERYTORYCZNE (Thematic Blocks)>
{thematic_text}

<CHRONOLOGICZNE PROFILE BEHAWIORALNE (Behavioral Profiles)>
{behavioral_text}
{presentation_block}{score_block}
Wygeneruj raport używając tagów.
"""
        raw_response = await self.gateway.execute_raw(
            prompt=f"{system_prompt}\n{user_prompt}",
            model=self.model,
            agent_role="Hegemon (Reduce Phase)",
            max_tokens=4096,
            temperature=0.3
        )

        return HegemonOutput(
            analysis=DeepAnalysis(
                factual_summary=self._extract_tag(raw_response, "factual_summary"),
                linguistic_summary=self._extract_tag(raw_response, "linguistic_summary"),
                missed_context=self._extract_list(raw_response, "missed_context")
            ),
            feedback=ConstructiveFeedback(
                executive_summary_markdown=self._extract_tag(raw_response, "executive_summary"),
                strengths=self._extract_list(raw_response, "strengths"),
                areas_for_improvement=self._extract_list(raw_response, "areas_for_improvement"),
                actionable_tips=self._extract_list(raw_response, "actionable_tips"),
                overall_message=self._extract_tag(raw_response, "overall_message")
            )
        )
