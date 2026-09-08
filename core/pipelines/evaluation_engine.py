import itertools
import math
import re
from typing import Dict, Tuple, List, Any

from core.config_loader import Config
from models.schemas import (
    FinalReport, JudgeRubric, ScenarioEvaluation, PairwisePreference, EvaluationReport
)

_REGIONS = 3  # start / middle / end

# Regex obsługujący zarówno format minutowy [12:34] jak i sekundowy z telemetrii np. [1550.704s] lub [112.8s]
_TS_RE_MIN = re.compile(r"\[(\d{1,2}):(\d{2})\]")
_TS_RE_SEC = re.compile(r"\[(\d+(?:\.\d+)?)[sS]\]")


class EvaluationEngine:
    """
    Tier 1 (absolute rubric) + Tier 2 (pairwise preference) LLM-as-judge over multiple
    scenario reports produced for the SAME input. Judge calls go through the gateway so
    their tokens/cost are measured too. Also aggregates telemetry and token density
    (chars/token) per scenario — useful for comparing Polish vs. English small models.
    """

    def __init__(self, gateway, judge_model: str,
                 excerpt_chars: int = None, excerpt_regions: int = None,
                 probe_timestamps: int = None, probe_window_chars: int = None,
                 focus_instruction: str = ""):
        self.gateway = gateway
        self.judge_model = judge_model
        self.excerpt_chars = Config.JUDGE_EXCERPT_CHARS if excerpt_chars is None else excerpt_chars
        self.excerpt_regions = Config.JUDGE_EXCERPT_REGIONS if excerpt_regions is None else excerpt_regions
        self.probe_timestamps = Config.JUDGE_PROBE_TIMESTAMPS if probe_timestamps is None else probe_timestamps
        self.probe_window_chars = Config.JUDGE_PROBE_WINDOW_CHARS if probe_window_chars is None else probe_window_chars
        self.focus_instruction = (focus_instruction or "").strip()

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
    def _calculate_language_tax(report: FinalReport, total_words: int) -> float:
        """
        Oblicza wskaźnik Tokens-Per-Word (TPW) dla fazy Map (wejście).
        Udowadnia 'Podatek Językowy' dla polskiego tekstu.
        """
        if not total_words or total_words == 0:
            return 0.0

        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []
        map_tokens_in = sum(
            p.tokens_in for p in details
            if "reduce" not in p.agent_role.lower() and "hegemon" not in p.agent_role.lower() and "evaluator" not in p.agent_role.lower()
        )
        return round(map_tokens_in / total_words, 2)

    @staticmethod
    def _split_phase_telemetry(report: FinalReport) -> dict:
        """
        Rozbija koszty i tokeny na poszczególne ramy architektoniczne (Map vs Reduce).
        """
        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []

        map_cost = 0.0
        reduce_cost = 0.0

        for p in details:
            role = p.agent_role.lower()
            if "reduce" in role or "hegemon" in role or "got " in role:
                reduce_cost += p.cost_usd
            elif "evaluator" in role or "judge" in role:
                pass
            else:
                map_cost += p.cost_usd

        return {
            "map_total_usd": round(map_cost, 5),
            "reduce_usd": round(reduce_cost, 5)
        }

    @staticmethod
    def _calculate_alignment_error(report: FinalReport, exp_factual: float, exp_linguistic: float) -> float:
        """
        Zwraca RMSE (Root Mean Square Error) - średnie odchylenie w punktach (0-100),
        co jest czytelne dla człowieka (np. 'pomylił się średnio o 12 punktów').
        """
        if not report.scorecard or report.scorecard.factual_score is None or report.scorecard.linguistic_score is None:
            return 0.0

        factual_diff = report.scorecard.factual_score - exp_factual
        ling_diff = report.scorecard.linguistic_score - exp_linguistic

        mse = (math.pow(factual_diff, 2) + math.pow(ling_diff, 2)) / 2
        return round(math.sqrt(mse), 1)

    @staticmethod
    def build_grounding_excerpt(transcript: str, total_chars: int = None, regions: int = None) -> str:
        transcript = transcript or ""
        total_chars = Config.JUDGE_EXCERPT_CHARS if total_chars is None else total_chars
        regions = Config.JUDGE_EXCERPT_REGIONS if regions is None else regions

        if total_chars <= 0 or regions <= 1 or len(transcript) <= total_chars:
            return transcript[:total_chars] if total_chars > 0 else transcript

        per_region = total_chars // regions
        n = len(transcript)
        region_span = n // regions
        labels = ["POCZĄTEK", "ŚRODEK", "KONIEC"] if regions == 3 else [f"REGION {i + 1}" for i in range(regions)]
        parts = []
        for i in range(regions):
            start = i * region_span
            slice_txt = transcript[start:start + per_region].strip()
            label = labels[i] if i < len(labels) else f"REGION {i + 1}"
            parts.append(f"[{label} ~{int(100 * start / n)}%]\n{slice_txt}")
        return "\n\n(…)\n\n".join(parts)

    @staticmethod
    def _ground_truth_findings(report: FinalReport, max_items: int = 12) -> str:
        lines = []
        sc = report.scorecard
        if sc is not None:
            lines.append(
                f"OCENA DETERMINISTYCZNA: łącznie {sc.overall_score}/100 "
                f"(merytoryka {sc.factual_score}, język {sc.linguistic_score})."
            )
        if report.analysis.missed_context:
            lines.append("POMINIĘTE WĄTKI (wg pipeline): " + "; ".join(report.analysis.missed_context[:max_items]))
        if report.analysis.unverified_claims:
            lines.append(
                "NIEPOTWIERDZONE (do weryfikacji, NIE błędy): "
                + "; ".join(report.analysis.unverified_claims[:max_items])
            )
        return "\n".join(lines) if lines else ""

    @staticmethod
    def build_timestamp_probes(report: FinalReport, transcript: str, duration_sec: float,
                               max_probes: int = None, window: int = None) -> str:
        max_probes = Config.JUDGE_PROBE_TIMESTAMPS if max_probes is None else max_probes
        window = Config.JUDGE_PROBE_WINDOW_CHARS if window is None else window
        transcript = transcript or ""
        if max_probes <= 0 or duration_sec <= 0 or not transcript:
            return ""

        timestamps = sorted(set(EvaluationEngine._parsed_timestamps(report)))
        if not timestamps:
            return ""

        if len(timestamps) > max_probes:
            step = len(timestamps) / max_probes
            timestamps = [timestamps[int(i * step)] for i in range(max_probes)]

        n = len(transcript)
        probes = []
        for ts in timestamps:
            frac = min(1.0, max(0.0, ts / duration_sec))
            center = int(frac * n)
            start = max(0, center - window // 2)
            snippet = transcript[start:start + window].strip()
            if snippet:
                mm, ss = int(ts // 60), int(ts % 60)
                probes.append(f"[{mm:02d}:{ss:02d}] (transkrypcja w tym miejscu):\n{snippet}")
        return "\n\n".join(probes)

    @staticmethod
    def _density(report: FinalReport) -> Tuple[float, float]:
        pc = sum(p.prompt_chars for p in report.telemetry.phase_details) if report.telemetry.phase_details else 0
        rc = sum(p.response_chars for p in report.telemetry.phase_details) if report.telemetry.phase_details else 0
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
            "\n".join(report.analysis.unverified_claims)
        ])

        ts = []
        for m in _TS_RE_MIN.finditer(text):
            ts.append(float(m.group(1)) * 60 + float(m.group(2)))

        for m in _TS_RE_SEC.finditer(text):
            ts.append(float(m.group(1)))

        return ts

    @staticmethod
    def positional_recall(report: FinalReport, duration_sec: float, regions: int = _REGIONS) -> List[float]:
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
                out.append(1.0)
            else:
                out.append(round(min(1.0, rep_b[i] / map_b[i]), 2))
        return out

    @staticmethod
    def _lost_in_middle(curve: List[float]) -> str:
        """
        Zwraca string dla UI: 'Brak danych' (jeśli model nie podał znaczników),
        'TAK' (jeśli zgubił środek), 'NIE' (jeśli pokrył równomiernie).
        """
        if not curve or sum(curve) == 0:
            return "Brak danych"

        if len(curve) < 3:
            return "NIE"

        edges = (curve[0] + curve[-1]) / 2
        if curve[len(curve) // 2] < 0.5 * edges and edges > 0:
            return "TAK"
        return "NIE"

    @staticmethod
    def _normalize_winner(raw: str, name_a: str, name_b: str) -> str:
        r = (raw or "").strip().upper()
        if "TIE" in r or "REMIS" in r:
            return "TIE"
        if name_a.upper() in r or r in ("A", "RAPORT A", "PIERWSZY"):
            return name_a
        if name_b.upper() in r or r in ("B", "RAPORT B", "DRUGI"):
            return name_b
        return "UNCLEAR"

    async def _judge_absolute(self, transcript_excerpt: str, scenario_name: str,
                              report: FinalReport, probes: str = "") -> JudgeRubric:
        ground_truth = self._ground_truth_findings(report)
        ground_block = f"\n<USTALENIA PIPELINE (kotwica do weryfikacji groundedness)>\n{ground_truth}\n" if ground_truth else ""
        probe_block = (
            f"\n<SONDY CZASOWE — transkrypcja przy znacznikach [MM:SS] lub [SS.s] cytowanych w raporcie>\n{probes}\n"
            if probes else ""
        )
        focus_block = (
            f"\n<SZCZEGÓLNY NACISK OD UŻYTKOWNIKA>\n{self.focus_instruction}\n"
            if self.focus_instruction else ""
        )
        prompt = f"""Jesteś surowym sędzią jakości feedbacku mentorskiego dla wystąpień publicznych.
Oceniasz JAKOŚĆ poniższego raportu (nie samo wystąpienie).

<FRAGMENT TRANSKRYPCJI — wiele regionów: początek/środek/koniec>
{transcript_excerpt}
{ground_block}{probe_block}{focus_block}
<RAPORT DO OCENY (scenariusz: {scenario_name})>
{self._report_text(report)}

ZASADY OCENY:
1. Groundedness (Ugruntowanie): Sprawdź, czy twierdzenia raportu mają DOKŁADNE POKRYCIE w powyższych fragmentach transkrypcji. Surowo karz za halucynacje.
2. Positional Recall (Sondy Czasowe): Masz fragmenty z miejsc zacytowanych w raporcie. Jeśli raport nie cytuje błędów z całego nagrania (ze ŚRODKA i KOŃCA) lub wnioski w sondach są zmyślone - drastycznie tnij ocenę.
3. Tool Adherence (Lenistwo): Jeśli raport dotyczy scenariusza z RAG/WEB, sprawdź czy kategorycznie odrzuca kłamstwa. Jeśli model "zgaduje" lub asekuruje się statusem UNVERIFIED dla oczywistych bzdur - karz za lenistwo narzędziowe.
4. Vague Praise vs Actionability: Policz konkretne rady. Jeśli raport "leje wodę" ("musisz pracować nad dynamiką"), obniż Actionability do minimum. Wymagaj bezwzględnego wskazywania błędów z markerami czasu.

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
                       duration_sec: float = 0.0,
                       total_words: int = 0,
                       expected_factual: float = 70.0,
                       expected_linguistic: float = 30.0) -> Tuple[EvaluationReport, Dict[str, dict]]:

        result = EvaluationReport()
        extra_metrics = {}

        grounded_excerpt = self.build_grounding_excerpt(transcript_excerpt, self.excerpt_chars, self.excerpt_regions)

        for name, report in reports.items():
            probes = ""
            try:
                probes = self.build_timestamp_probes(report, transcript_excerpt, duration_sec,
                                                     self.probe_timestamps, self.probe_window_chars)
                rubric = await self._judge_absolute(grounded_excerpt, name, report, probes=probes)
            except Exception as e:
                rubric = JudgeRubric(justification=f"[Sędzia zawiódł: {e}]")

            total = rubric.actionability + rubric.specificity + rubric.correctness + rubric.tone + rubric.groundedness
            in_density, out_density = self._density(report)
            pos_recall = self.positional_recall(report, duration_sec)
            red_fidelity = self.reduce_fidelity(report, duration_sec)

            phase_costs = self._split_phase_telemetry(report)
            tpw = self._calculate_language_tax(report, total_words)
            alignment_error = self._calculate_alignment_error(report, expected_factual, expected_linguistic)
            lost_in_middle_str = self._lost_in_middle(pos_recall)

            extra_metrics[name] = {
                "rmse": alignment_error,
                "tpw": tpw,
                "costs": phase_costs,
                "lost_in_middle": lost_in_middle_str,
                "pos_recall_raw": pos_recall,
                "density_in": in_density,
                "density_out": out_density
            }

            ground_truth = self._ground_truth_findings(report)
            evidence = (
                f"=== FRAGMENTY TRANSKRYPCJI (start/środek/koniec) ===\n{grounded_excerpt}\n\n"
                f"=== USTALENIA PIPELINE ===\n{ground_truth or '(brak)'}\n\n"
                f"=== SONDY CZASOWE ===\n{probes or '(brak — raport nie cytował znaczników [MM:SS])'}"
            )

            # Bezpieczne dla Pydantic: wysyłamy tylko bool, a interfejs w app.py odczyta stringa z extra_metrics.
            # Zabezpiecza to przed ValidationError.
            safe_bool_flag = (lost_in_middle_str == "TAK")

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
                lost_in_middle_flag=safe_bool_flag,
                judge_evidence=evidence,
            ))

        names = list(reports.keys())
        for a, b in itertools.combinations(names, 2):
            try:
                pref = await self._judge_pairwise(grounded_excerpt, a, reports[a], b, reports[b])
                pref.winner = self._normalize_winner(pref.winner, a, b)
            except Exception as e:
                pref = PairwisePreference(winner="UNCLEAR", reason=f"[Sędzia zawiódł: {e}]")
            result.pairwise.append(pref)

        judge_calls = [t for t in self.gateway.get_session_telemetry() if "Evaluator" in t.agent_role]
        result.judge_tokens_in = sum(t.tokens_in for t in judge_calls)
        result.judge_tokens_out = sum(t.tokens_out for t in judge_calls)

        if result.per_scenario:
            best = max(result.per_scenario, key=lambda s: s.rubric_total)
            result.summary = (
                f"Najwyższa ocena jakości: {best.scenario_name} ({best.rubric_total}/50). "
                f"Porównaj z kosztem tokenowym każdego scenariusza w tabeli."
            )

        return result, extra_metrics