import itertools
import re
from typing import Dict, Tuple, List

from models.schemas import (
    FinalReport, JudgeRubric, ScenarioEvaluation, PairwisePreference, EvaluationReport
)

_REGIONS = 3  # start / middle / end
_TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})]")


class EvaluationEngine:
    """
    Tier 1 (absolute rubric) + Tier 2 (pairwise preference) LLM-as-judge over multiple
    scenario reports produced for the SAME input. Judge calls go through the gateway so
    their tokens/cost are measured too. Also aggregates telemetry and token density
    (chars/token) per scenario — useful for comparing Polish vs. English small models.
    """

    def __init__(self, gateway, judge_model: str):
        self.gateway = gateway
        self.judge_model = judge_model

    @staticmethod
    def _report_text(report: FinalReport) -> str:
        fb = report.feedback
        parts = [
            f"OCENA MERYTORYCZNA: {report.analysis.factual_summary}",
            f"OCENA JĘZYKOWA: {report.analysis.linguistic_summary}",
            f"ESEJ: {fb.executive_summary_markdown}",
            f"MOCNE STRONY: {'; '.join(fb.strengths)}",
            f"DO POPRAWY: {'; '.join(fb.areas_for_improvement)}",
            f"WSKAZÓWKI: {'; '.join(fb.actionable_tips)}",
            f"PRZESŁANIE: {fb.overall_message}",
        ]
        return "\n".join(parts)

    @staticmethod
    def _density(report: FinalReport) -> Tuple[float, float]:
        # True language-tax density = characters per token, aggregated over all phases.
        # Higher chars/token = cheaper (more chars packed per token, typical for well-tokenized text).
        pc = sum(p.prompt_chars for p in report.telemetry.phase_details)
        rc = sum(p.response_chars for p in report.telemetry.phase_details)
        tin = report.telemetry.total_tokens_in
        tout = report.telemetry.total_tokens_out
        in_density = round(pc / tin, 2) if tin else 0.0
        out_density = round(rc / tout, 2) if tout else 0.0
        return in_density, out_density

    @staticmethod
    def _parsed_timestamps(report: FinalReport) -> List[float]:
        text = "\n".join([
            report.analysis.factual_summary,
            report.analysis.linguistic_summary,
            "\n".join(report.analysis.missed_context),
            report.feedback.executive_summary_markdown,
            "\n".join(report.feedback.strengths),
            "\n".join(report.feedback.areas_for_improvement),
            "\n".join(report.feedback.actionable_tips),
        ])
        return [float(m.group(1)) * 60 + float(m.group(2)) for m in _TS_RE.finditer(text)]

    @staticmethod
    def positional_recall(report: FinalReport, duration_sec: float, regions: int = _REGIONS) -> List[float]:
        # Fraction of cited timestamps falling in each equal time-region (start..end).
        # A dip in the middle region is the lost-in-the-middle symptom.
        # Returns [] (not zeros) when the report has NO parseable [MM:SS] markers, so
        # "model didn't emit timestamps" is distinguishable from "covered evenly / poorly".
        ts = EvaluationEngine._parsed_timestamps(report)
        if not ts or duration_sec <= 0:
            return []
        counts = [0] * regions
        for t in ts:
            idx = min(regions - 1, int((t / duration_sec) * regions))
            counts[idx] += 1
        total = sum(counts)
        return [round(c / total, 2) for c in counts] if total else []

    @staticmethod
    def reduce_fidelity(report: FinalReport, duration_sec: float, regions: int = _REGIONS) -> List[float]:
        # Swarm only: per region, fraction of map-finding regions that survived into the report.
        # Compares where the map found things vs. where the report cites things.
        map_ts = report.map_timestamps or []
        if not map_ts or duration_sec <= 0:
            return []
        report_ts = EvaluationEngine._parsed_timestamps(report)

        def bucket(values):
            b = [0] * regions
            for t in values:
                b[min(regions - 1, int((t / duration_sec) * regions))] += 1
            return b

        map_b = bucket(map_ts)
        rep_b = bucket(report_ts)
        out = []
        for i in range(regions):
            if map_b[i] == 0:
                out.append(1.0)  # nothing to preserve in this region
            else:
                out.append(round(min(1.0, rep_b[i] / map_b[i]), 2))
        return out

    @staticmethod
    def _lost_in_middle(curve: List[float]) -> bool:
        # Middle region materially lower than the mean of the edges.
        if len(curve) < 3:
            return False
        edges = (curve[0] + curve[-1]) / 2
        return curve[len(curve) // 2] < 0.5 * edges and edges > 0

    @staticmethod
    def _normalize_winner(raw: str, name_a: str, name_b: str) -> str:
        # Map the judge's free-text winner back to a known scenario name, TIE, or UNCLEAR.
        r = (raw or "").strip().upper()
        if "TIE" in r or "REMIS" in r:
            return "TIE"
        if name_a.upper() in r or r in ("A", "RAPORT A", "PIERWSZY"):
            return name_a
        if name_b.upper() in r or r in ("B", "RAPORT B", "DRUGI"):
            return name_b
        return "UNCLEAR"

    async def _judge_absolute(self, transcript_excerpt: str, scenario_name: str,
                              report: FinalReport) -> JudgeRubric:
        prompt = f"""Jesteś surowym sędzią jakości feedbacku mentorskiego dla wystąpień publicznych.
Oceniasz JAKOŚĆ poniższego raportu (nie samo wystąpienie).

<FRAGMENT TRANSKRYPCJI (kontekst)>
{transcript_excerpt}

<RAPORT DO OCENY (scenariusz: {scenario_name})>
{self._report_text(report)}

Oceń raport w 5 wymiarach 0-10 (actionability, specificity, correctness, tone, groundedness)
i podaj krótkie uzasadnienie. Zwróć wynik zgodnie ze schematem."""
        return await self.gateway.execute_structured(
            prompt=prompt,
            schema_class=JudgeRubric,
            model=self.judge_model,
            agent_role="Evaluator (Judge - Absolute)"
        )

    async def _judge_pairwise(self, transcript_excerpt: str,
                              name_a: str, report_a: FinalReport,
                              name_b: str, report_b: FinalReport) -> PairwisePreference:
        prompt = f"""Jesteś sędzią porównującym dwa raporty mentorskie dla TEGO SAMEGO wystąpienia.
Wybierz, który jest BARDZIEJ UŻYTECZNY dla prelegenta (konkretność, trafność, ton, brak halucynacji).

<FRAGMENT TRANSKRYPCJI>
{transcript_excerpt}

<RAPORT A ({name_a})>
{self._report_text(report_a)}

<RAPORT B ({name_b})>
{self._report_text(report_b)}

W polu 'winner' wpisz DOKŁADNIE "{name_a}" lub "{name_b}", albo "TIE". Podaj krótki 'reason'.
NIE nagradzaj rozwlekłości ani długości — oceniaj wyłącznie użyteczność, konkretność i trafność dla prelegenta."""
        return await self.gateway.execute_structured(
            prompt=prompt,
            schema_class=PairwisePreference,
            model=self.judge_model,
            agent_role="Evaluator (Judge - Pairwise)"
        )

    async def evaluate(self, transcript_excerpt: str,
                       reports: Dict[str, FinalReport],
                       duration_sec: float = 0.0) -> EvaluationReport:
        result = EvaluationReport()

        # --- Tier 1: absolute rubric per scenario ---
        for name, report in reports.items():
            try:
                rubric = await self._judge_absolute(transcript_excerpt, name, report)
            except Exception as e:
                rubric = JudgeRubric(justification=f"[Sędzia zawiódł: {e}]")
            total = rubric.actionability + rubric.specificity + rubric.correctness + rubric.tone + rubric.groundedness
            in_density, out_density = self._density(report)
            pos_recall = self.positional_recall(report, duration_sec)
            red_fidelity = self.reduce_fidelity(report, duration_sec)
            result.per_scenario.append(ScenarioEvaluation(
                scenario_name=name,
                rubric=rubric,
                rubric_total=total,
                total_tokens_in=report.telemetry.total_tokens_in,
                total_tokens_out=report.telemetry.total_tokens_out,
                total_cost_usd=report.telemetry.total_cost_usd,
                total_time_s=report.telemetry.total_time_s,
                input_token_density=in_density,
                output_token_density=out_density,
                positional_recall=pos_recall,
                reduce_fidelity=red_fidelity,
                lost_in_middle_flag=self._lost_in_middle(pos_recall),
            ))

        # --- Tier 2: pairwise preferences over all scenario pairs ---
        names = list(reports.keys())
        for a, b in itertools.combinations(names, 2):
            try:
                pref = await self._judge_pairwise(transcript_excerpt, a, reports[a], b, reports[b])
                pref.winner = self._normalize_winner(pref.winner, a, b)
            except Exception as e:
                pref = PairwisePreference(winner="UNCLEAR", reason=f"[Sędzia zawiódł: {e}]")
            result.pairwise.append(pref)

        # Aggregate judge cost from the session telemetry (judge calls were logged too).
        judge_calls = [t for t in self.gateway.get_session_telemetry() if "Evaluator" in t.agent_role]
        result.judge_tokens_in = sum(t.tokens_in for t in judge_calls)
        result.judge_tokens_out = sum(t.tokens_out for t in judge_calls)

        if result.per_scenario:
            best = max(result.per_scenario, key=lambda s: s.rubric_total)
            result.summary = (
                f"Najwyższa ocena jakości: {best.scenario_name} ({best.rubric_total}/50). "
                f"Porównaj z kosztem tokenowym każdego scenariusza w tabeli."
            )
        return result
