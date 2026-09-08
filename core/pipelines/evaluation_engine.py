import itertools
import math
import re
import json
from typing import Dict, Tuple, List, Any

from core.config_loader import Config
from models.schemas import (
    FinalReport, JudgeRubric, ScenarioEvaluation, PairwisePreference, EvaluationReport
)

_REGIONS = 3  # start / middle / end

# Regex obsługujący format minutowy [12:34] jak i sekundowy [1550.704s]
_TS_RE_MIN = re.compile(r"\[(\d{1,2}):(\d{2})\]")
_TS_RE_SEC = re.compile(r"\[(\d+(?:\.\d+)?)[sS]\]")


class EvaluationEngine:
    """
    Tier 1 LLM-as-judge. Ocenianie względem Złotego Wzorca (plik JSON z błędami).
    Wylicza błędy MSE, gęstość tokenów na agenta, oraz precyzyjny procent wykrytych
    błędów krytycznych (Recall) na podstawie Golden Setu.
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
        if not total_words or total_words == 0:
            return 0.0

        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []
        map_tokens_in = sum(
            p.tokens_in for p in details
            if
            "reduce" not in p.agent_role.lower() and "hegemon" not in p.agent_role.lower() and "evaluator" not in p.agent_role.lower()
        )
        return round(map_tokens_in / total_words, 2)

    @staticmethod
    def _phase_densities(report: FinalReport) -> dict:
        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []

        def calc(role_kw: str):
            pc = sum(p.prompt_chars for p in details if role_kw in p.agent_role.lower())
            tin = sum(p.tokens_in for p in details if role_kw in p.agent_role.lower())
            return round(pc / tin, 2) if tin > 0 else 0.0

        return {
            "factual_density": calc("factual"),
            "linguistic_density": calc("linguistic"),
            "reduce_density": calc("reduce") or calc("hegemon")
        }

    @staticmethod
    def _split_phase_telemetry(report: FinalReport) -> dict:
        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []

        map_cost = 0.0
        reduce_cost = 0.0
        prior_tokens_total = 0
        hegemon_tokens_in = 0
        hegemon_tokens_out = 0

        for p in details:
            role = p.agent_role.lower()
            if "reduce" in role or "hegemon" in role or "got " in role:
                reduce_cost += p.cost_usd
                hegemon_tokens_in += p.tokens_in
                hegemon_tokens_out += p.tokens_out
            elif "evaluator" in role or "judge" in role:
                pass
            else:
                map_cost += p.cost_usd
                prior_tokens_total += (p.tokens_in + p.tokens_out)

        return {
            "map_total_usd": round(map_cost, 5),
            "reduce_usd": round(reduce_cost, 5),
            "prior_tokens_total": prior_tokens_total,
            "hegemon_tokens_in": hegemon_tokens_in,
            "hegemon_tokens_out": hegemon_tokens_out
        }

    @staticmethod
    def _calculate_alignment_error(report: FinalReport, exp_factual: float, exp_linguistic: float) -> float:
        if not report.scorecard or report.scorecard.factual_score is None or report.scorecard.linguistic_score is None:
            return 0.0

        factual_diff = report.scorecard.factual_score - exp_factual
        ling_diff = report.scorecard.linguistic_score - exp_linguistic

        mse = (math.pow(factual_diff, 2) + math.pow(ling_diff, 2)) / 2
        return round(math.sqrt(mse), 1)

    @staticmethod
    def _calculate_error_recall(golden_json_str: str, report_timestamps: List[float],
                                tolerance_sec: float = 90.0) -> float:
        """
        Parsuje Golden Set JSON, wyciąga błędy o skali HIGH/CRITICAL, i sprawdza,
        czy oceniany raport wyłapał znaczniki czasu w pobliżu tych błędów.
        Zwraca procent wykrycia (0-100) lub -1.0 jeśli brak danych/błędny JSON.
        """
        if not golden_json_str.strip():
            return -1.0

        try:
            data = json.loads(golden_json_str)
            target_ts = []

            def extract_ts(obj):
                if isinstance(obj, dict):
                    scale = str(obj.get("scale", obj.get("severity", ""))).upper()
                    if scale in ["HIGH", "CRITICAL", "WYSOKA", "KRYTYCZNA"]:
                        t = obj.get("time", obj.get("czas"))
                        if t is not None:
                            try:
                                target_ts.append(float(t))
                            except ValueError:
                                pass
                    for k, v in obj.items():
                        extract_ts(v)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_ts(item)

            extract_ts(data)

            if not target_ts:
                return -1.0

            caught = 0
            for t_target in target_ts:
                if any(abs(t_target - t_rep) <= tolerance_sec for t_rep in report_timestamps):
                    caught += 1

            return round((caught / len(target_ts)) * 100, 2)
        except Exception:
            return -1.0

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
                              report: FinalReport, probes: str = "",
                              golden_factual: str = "", golden_linguistic: str = "") -> JudgeRubric:
        ground_truth = self._ground_truth_findings(report)
        ground_block = f"\n<USTALENIA PIPELINE (kotwica do weryfikacji groundedness)>\n{ground_truth}\n" if ground_truth else ""
        probe_block = (
            f"\n<SONDY CZASOWE — transkrypcja przy znacznikach cytowanych w raporcie>\n{probes}\n"
            if probes else ""
        )
        focus_block = (
            f"\n<SZCZEGÓLNY NACISK OD UŻYTKOWNIKA>\n{self.focus_instruction}\n"
            if self.focus_instruction else ""
        )

        golden_block = ""
        if golden_factual or golden_linguistic:
            golden_block = "\n<ZŁOTY WZORZEC (GOLDEN SET) - OCZEKIWANE BŁĘDY DO WYKRYCIA>\n"
            if golden_factual:
                golden_block += f"--- BŁĘDY MERYTORYCZNE ---\n{golden_factual}\n\n"
            if golden_linguistic:
                golden_block += f"--- BŁĘDY LINGWISTYCZNE ---\n{golden_linguistic}\n\n"

        prompt = f"""Jesteś surowym ekspertem MLOps i sędzią (LLM-as-a-Judge) jakości systemów AI.
Oceniasz JAKOŚĆ poniższego raportu z analizy przemówienia.

{golden_block}
<FRAGMENT TRANSKRYPCJI — opcjonalny kontekst>
{transcript_excerpt}
{ground_block}{probe_block}{focus_block}

<RAPORT DO OCENY (scenariusz: {scenario_name})>
{self._report_text(report)}

ZASADY OCENY:
1. Tabela Wykrywalności (KRYTYCZNE): W swoim uzasadnieniu wygeneruj krótką tabelę Markdown. Zestaw w niej duże i krytyczne błędy z pliku ZŁOTY WZORZEC (jeśli podano) z tym, co raport FAKTYCZNIE wykrył. Tabela ma mieć kolumny: | Błąd ze Złotego Wzorca | Oczekiwany Czas | Czy raport go wyłapał? |. Jeśli raport pominął usterki z dalszej części wykładu, tnij ocenę Groundedness i Actionability.
2. Groundedness (Ugruntowanie): Karz za halucynacje. Jeśli raport zmyśla wnioski, na które nie ma dowodów, obniż ocenę.
3. Tool Adherence: W scenariuszach z RAG/WEB model musi kategorycznie odrzucać kłamstwa. Karz za lenistwo (asekurowanie się statusem UNVERIFIED dla jawnych kłamstw historycznych).
4. Actionability vs Vague Praise: Policz konkretne rady. "Lanie wody" bez podawania konkretnych [MM:SS] to ocena minimalna.

Oceń raport w 5 wymiarach 0-10 (actionability, specificity, correctness, tone, groundedness)
i podaj uzasadnienie (wraz z Tabelą Błędów). Zwróć wynik zgodnie ze schematem."""
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
W polu 'winner' wpisz DOKŁADNIE "{name_a}" lub "{name_b}", albo "TIE"."""
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
                       expected_linguistic: float = 30.0,
                       golden_factual: str = "",
                       golden_linguistic: str = "") -> Tuple[EvaluationReport, Dict[str, dict]]:

        result = EvaluationReport()
        extra_metrics = {}

        grounded_excerpt = self.build_grounding_excerpt(transcript_excerpt, self.excerpt_chars, self.excerpt_regions)

        for name, report in reports.items():
            probes = ""
            try:
                probes = self.build_timestamp_probes(report, transcript_excerpt, duration_sec,
                                                     self.probe_timestamps, self.probe_window_chars)
                rubric = await self._judge_absolute(grounded_excerpt, name, report, probes=probes,
                                                    golden_factual=golden_factual, golden_linguistic=golden_linguistic)
            except Exception as e:
                rubric = JudgeRubric(justification=f"[Sędzia zawiódł: {e}]")

            total = rubric.actionability + rubric.specificity + rubric.correctness + rubric.tone + rubric.groundedness
            pos_recall = self.positional_recall(report, duration_sec)
            red_fidelity = self.reduce_fidelity(report, duration_sec)

            report_ts = self._parsed_timestamps(report)
            error_recall_pct = -1.0

            # Ewaluacja wykrywalności jeśli wgrano JSON z merytoryką
            if golden_factual.strip() and golden_factual.strip().startswith("{"):
                error_recall_pct = self._calculate_error_recall(golden_factual, report_ts)

            phase_costs = self._split_phase_telemetry(report)
            tpw = self._calculate_language_tax(report, total_words)
            alignment_error = self._calculate_alignment_error(report, expected_factual, expected_linguistic)
            densities = self._phase_densities(report)

            extra_metrics[name] = {
                "rmse": alignment_error,
                "tpw": tpw,
                "costs": phase_costs,
                "error_recall_pct": error_recall_pct,
                "factual_density": densities.get("factual_density", 0.0),
                "linguistic_density": densities.get("linguistic_density", 0.0),
                "reduce_density": densities.get("reduce_density", 0.0)
            }

            ground_truth = self._ground_truth_findings(report)
            evidence = (
                f"=== ZŁOTY WZORZEC MERYTORYCZNY ===\n{golden_factual or '(brak)'}\n\n"
                f"=== ZŁOTY WZORZEC LINGWISTYCZNY ===\n{golden_linguistic or '(brak)'}\n\n"
                f"=== USTALENIA PIPELINE ===\n{ground_truth or '(brak)'}\n\n"
                f"=== SONDY CZASOWE ===\n{probes or '(brak)'}"
            )

            result.per_scenario.append(ScenarioEvaluation(
                scenario_name=name,
                rubric=rubric,
                rubric_total=total,
                total_tokens_in=report.telemetry.total_tokens_in,
                total_tokens_out=report.telemetry.total_tokens_out,
                total_cost_usd=report.telemetry.total_cost_usd,
                total_time_s=report.telemetry.total_time_s,
                input_token_density=0.0,
                output_token_density=0.0,
                positional_recall=pos_recall,
                reduce_fidelity=red_fidelity,
                lost_in_middle_flag=False,
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
            result.summary = f"Sędzia (Hegemon): Najwyższa ocena {best.scenario_name} ({best.rubric_total}/50)."

        return result, extra_metrics