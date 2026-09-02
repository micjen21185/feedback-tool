import re
from typing import Any, List

from core.config_loader import Config
from models.schemas import (
    LectureMetadata, DeepAnalysis, ConstructiveFeedback,
    HegemonOutput, SystemConfiguration, ScoreCard, readiness_verdict
)


class MonolithNakedPipeline:
    def __init__(self, gateway: Any, config: SystemConfiguration):
        self.gateway = gateway
        self.config = config

    def _extract_tag(self, text: str, tag: str) -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback for unclosed/truncated tags: open tag to next opening tag or end of text.
        open_match = re.search(f"<{tag}>(.*?)(?=<[a-z_]+>|$)", text, re.DOTALL | re.IGNORECASE)
        return open_match.group(1).strip() if open_match else ""

    def _extract_list(self, text: str, tag: str) -> List[str]:
        content = self._extract_tag(text, tag)
        items = []
        for line in content.split('\n'):
            # Strip only a single leading bullet marker so inline Markdown (**bold**) is preserved.
            cleaned = re.sub(r"^\s*[-*•]\s+", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    async def execute(self, metadata: LectureMetadata, raw_text: str) -> HegemonOutput:
        role = metadata.speaker_role.strip() if metadata.speaker_role else 'Nie określono'
        audience = metadata.target_audience.strip() if metadata.target_audience else 'Nie określono'
        topic = metadata.main_topic.strip() if metadata.main_topic else 'Nie określono'
        level = metadata.knowledge_level.strip() if metadata.knowledge_level else 'Nie określono'

        metrics_block = f"""
- Czas trwania: {metadata.total_duration_sec}s
- Liczba słów: {metadata.total_words}
- Tempo mowy (min/max): {metadata.slowest_chunk_wpm} / {metadata.fastest_chunk_wpm} WPM
- Słowa-wypełniacze: {metadata.total_filler_words}
- Znaczące pauzy: {metadata.total_significant_pauses} (łącznie {metadata.total_significant_pauses_duration_sec}s)
- Niewyraźne słowa: {metadata.total_unclear_words}
"""
        prompt = f"""Jesteś wybitnym mentorem ds. wystąpień publicznych. 
Oceń poniższą transkrypcję. Bądź obiektywny – weryfikuj tekst pod kątem ewentualnych zająknięć i pauz wykorzystując podane metryki.

<CONTEXT>
Prelegent: {role} | Widownia: {audience} (Poziom: {level}) | Temat: {topic}
Metryki: {metrics_block}
</CONTEXT>

<SAFEGUARD>
Jeśli w profilu widzisz "Nie określono", oceniaj to jako uniwersalną przemowę publiczną. ZABRANIA CI SIĘ zmyślać grupy docelowej.
</SAFEGUARD>

<WYMOGI FORMATOWANIA>
Nie używaj JSON! Przy każdej konkretnej obserwacji podawaj znacznik czasu [MM:SS] i rozłóż uwagi po całym wystąpieniu (początek, środek, koniec). Zwróć raport w tagach:
<factual_summary> (Analiza merytoryki) </factual_summary>
<linguistic_summary> (Analiza języka) </linguistic_summary>
<executive_summary> (Wieloakapitowy esej z feedbackiem mentorskim, formatowanie Markdown) </executive_summary>
<strengths>
- (zalety od myślników)
</strengths>
<areas_for_improvement>
- (wady od myślników)
</areas_for_improvement>
<actionable_tips>
- (konkretne rady)
</actionable_tips>
<overall_message> (Złota myśl na koniec) </overall_message>
<score_overall> (Liczba całkowita 0-100 oceniająca całość wystąpienia; 100 = perfekcyjne) </score_overall>

<TEXT>
{raw_text}
</TEXT>
"""
        raw_response = await self.gateway.execute_raw(
            prompt=prompt, model=self.config.hegemon_model, agent_role="Hegemon (Scenario 1)",
            max_tokens=Config.HEGEMON_MAX_TOKENS, timeout=Config.HEGEMON_REQUEST_TIMEOUT,
            retry_on_timeout=False
        )

        analysis = DeepAnalysis(
            factual_summary=self._extract_tag(raw_response, "factual_summary"),
            linguistic_summary=self._extract_tag(raw_response, "linguistic_summary")
        )
        feedback = ConstructiveFeedback(
            executive_summary_markdown=self._extract_tag(raw_response, "executive_summary"),
            strengths=self._extract_list(raw_response, "strengths"),
            areas_for_improvement=self._extract_list(raw_response, "areas_for_improvement"),
            actionable_tips=self._extract_list(raw_response, "actionable_tips"),
            overall_message=self._extract_tag(raw_response, "overall_message")
        )
        nothing_parsed = not any([
            analysis.factual_summary, analysis.linguistic_summary,
            feedback.executive_summary_markdown, feedback.strengths,
            feedback.areas_for_improvement, feedback.actionable_tips
        ])
        if nothing_parsed and raw_response.strip():
            feedback.executive_summary_markdown = (
                    "> ⚠️ Model nie użył wymaganych znaczników — poniżej surowa odpowiedź.\n\n"
                    + raw_response.strip()
            )

        return HegemonOutput(
            analysis=analysis,
            feedback=feedback,
            scorecard=self._parse_scorecard(raw_response),
            raw_reducer_response=raw_response,
            reducer_input=prompt
        )

    def _parse_scorecard(self, text: str):
        raw = self._extract_tag(text, "score_overall")
        match = re.search(r"\d{1,3}", raw)
        if not match:
            return None
        score = float(min(100, int(match.group(0))))
        return ScoreCard(overall_score=score, readiness_verdict=readiness_verdict(score))


class MonolithTwoPhasePipeline:
    def __init__(self, gateway: Any, config: SystemConfiguration):
        self.gateway = gateway
        self.config = config

    def _extract_tag(self, text: str, tag: str) -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback for unclosed/truncated tags: open tag to next opening tag or end of text.
        open_match = re.search(f"<{tag}>(.*?)(?=<[a-z_]+>|$)", text, re.DOTALL | re.IGNORECASE)
        return open_match.group(1).strip() if open_match else ""

    def _extract_list(self, text: str, tag: str) -> List[str]:
        content = self._extract_tag(text, tag)
        items = []
        for line in content.split('\n'):
            # Strip only a single leading bullet marker so inline Markdown (**bold**) is preserved.
            cleaned = re.sub(r"^\s*[-*•]\s+", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    async def execute(self, metadata: LectureMetadata, formatted_text: str) -> HegemonOutput:
        context_block = f"""
Prelegent: {metadata.speaker_role} | Widownia: {metadata.target_audience} (Poziom: {metadata.knowledge_level})
WPM: {metadata.slowest_chunk_wpm}/{metadata.fastest_chunk_wpm} | Wypełniacze: {metadata.total_filler_words}
"""

        phase_1_prompt = f"""Jesteś analitykiem mowy. Twoim zadaniem jest chłodna, obiektywna ocena transkrypcji.
<CONTEXT>{context_block}</CONTEXT>

<INSTRUCTIONS>
Przeanalizuj logikę, merytorykę i użycie tagów akustycznych (np. [pauza: Xs], [niewyraźne]).
Zwróć odpowiedź w poniższych tagach (NIE UŻYWAJ JSON):
<factual_summary> (Ocena logiki wywodu) </factual_summary>
<linguistic_summary> (Ocena wpływu tagów i akustyki) </linguistic_summary>
<missed_context> (Wypunktuj potencjalne braki lub niedopowiedzenia) </missed_context>

<TEXT>
{formatted_text}
</TEXT>
"""
        raw_analysis = await self.gateway.execute_raw(
            prompt=phase_1_prompt, model=self.config.hegemon_model, agent_role="Hegemon (Scenariusz 2 - Analiza)",
            max_tokens=Config.HEGEMON_MAX_TOKENS, timeout=Config.HEGEMON_REQUEST_TIMEOUT,
            retry_on_timeout=False
        )

        analysis_obj = DeepAnalysis(
            factual_summary=self._extract_tag(raw_analysis, "factual_summary"),
            linguistic_summary=self._extract_tag(raw_analysis, "linguistic_summary"),
            missed_context=self._extract_list(raw_analysis, "missed_context")
        )

        phase_2_prompt = f"""Jesteś doświadczonym, empatycznym trenerem wystąpień.
<CONTEXT>{context_block}</CONTEXT>

<INPUT_ANALYSIS>
Twoja głęboka analiza z poprzedniej fazy:
Merytoryka: {analysis_obj.factual_summary}
Język i dykcja: {analysis_obj.linguistic_summary}
Braki: {', '.join(analysis_obj.missed_context) if analysis_obj.missed_context else 'Brak istotnych luk.'}
</INPUT_ANALYSIS>

<INSTRUCTIONS>
Sformułuj ostateczny feedback dla prelegenta w oparciu o powyższą analizę. Używaj modelu SBI (Sytuacja-Zachowanie-Wpływ).
Przy każdej konkretnej obserwacji podawaj znacznik czasu [MM:SS] i rozłóż uwagi po całym wystąpieniu (początek, środek, koniec).
NIE UŻYWAJ JSON. Użyj tagów:
<executive_summary> (Wieloakapitowy esej mentorski Markdown) </executive_summary>
<strengths> (lista od myślników) </strengths>
<areas_for_improvement> (lista od myślników) </areas_for_improvement>
<actionable_tips> (lista porad od myślników) </actionable_tips>
<overall_message> (Zdanie motywujące) </overall_message>

<SCORING_RUBRIC>
Oceń wystąpienie w 5 wymiarach, każdy 0-100, wraz z jednym zdaniem uzasadnienia. Użyj dokładnie tych tagów:
<score_factual> (0-100) </score_factual> <justify_factual> (uzasadnienie) </justify_factual>
<score_linguistic> (0-100) </score_linguistic> <justify_linguistic> (uzasadnienie) </justify_linguistic>
<score_structure> (0-100) </score_structure> <justify_structure> (uzasadnienie) </justify_structure>
<score_tempo> (0-100) </score_tempo> <justify_tempo> (uzasadnienie) </justify_tempo>
<score_confidence> (0-100) </score_confidence> <justify_confidence> (uzasadnienie) </justify_confidence>
</SCORING_RUBRIC>
</INSTRUCTIONS>
"""
        raw_feedback = await self.gateway.execute_raw(
            prompt=phase_2_prompt, model=self.config.hegemon_model, agent_role="Hegemon (Scenariusz 2 - Feedback)",
            max_tokens=Config.HEGEMON_MAX_TOKENS, timeout=Config.HEGEMON_REQUEST_TIMEOUT,
            retry_on_timeout=False
        )

        feedback_obj = ConstructiveFeedback(
            executive_summary_markdown=self._extract_tag(raw_feedback, "executive_summary"),
            strengths=self._extract_list(raw_feedback, "strengths"),
            areas_for_improvement=self._extract_list(raw_feedback, "areas_for_improvement"),
            actionable_tips=self._extract_list(raw_feedback, "actionable_tips"),
            overall_message=self._extract_tag(raw_feedback, "overall_message")
        )

        nothing_parsed = not any([
            analysis_obj.factual_summary, analysis_obj.linguistic_summary,
            feedback_obj.executive_summary_markdown, feedback_obj.strengths,
            feedback_obj.areas_for_improvement, feedback_obj.actionable_tips
        ])
        if nothing_parsed and raw_feedback.strip():
            feedback_obj.executive_summary_markdown = (
                    "> ⚠️ Model nie użył wymaganych znaczników — poniżej surowa odpowiedź.\n\n"
                    + raw_feedback.strip()
            )

        return HegemonOutput(
            analysis=analysis_obj,
            feedback=feedback_obj,
            scorecard=self._parse_scorecard(raw_feedback),
            raw_reducer_response=f"--- FAZA 1 (Analiza) ---\n{raw_analysis}\n\n--- FAZA 2 (Feedback) ---\n{raw_feedback}",
            reducer_input=f"--- PROMPT FAZY 1 ---\n{phase_1_prompt}\n\n--- PROMPT FAZY 2 ---\n{phase_2_prompt}"
        )

    def _parse_score(self, text: str, tag: str) -> float:
        raw = self._extract_tag(text, tag)
        match = re.search(r"\d{1,3}", raw)
        return float(min(100, int(match.group(0)))) if match else 100.0

    def _parse_scorecard(self, text: str):
        factual = self._parse_score(text, "score_factual")
        linguistic = self._parse_score(text, "score_linguistic")
        structure = self._parse_score(text, "score_structure")
        tempo = self._parse_score(text, "score_tempo")
        confidence = self._parse_score(text, "score_confidence")
        # Weighted overall: factual dominant, matching the swarm's factual-heavy blend.
        overall = round(
            0.35 * factual + 0.20 * linguistic + 0.20 * structure + 0.15 * tempo + 0.10 * confidence, 1
        )
        return ScoreCard(
            factual_score=factual,
            linguistic_score=round((linguistic + structure + tempo + confidence) / 4, 1),
            overall_score=overall,
            readiness_verdict=readiness_verdict(overall)
        )
