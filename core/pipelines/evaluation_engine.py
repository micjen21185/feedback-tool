import itertools
import json
import math
import re
from typing import Dict, Tuple, List, Any

from core.config_loader import Config
from models.schemas import (
    FinalReport, JudgeRubric, ScenarioEvaluation, PairwisePreference, EvaluationReport
)

_REGIONS = 3  # start / middle / end

# Regex obsługujący format minutowy [12:34] jak i sekundowy [1550.704s]
_TS_RE_MIN = re.compile(r"\[(\d{1,2}):(\d{2})\]")
_TS_RE_SEC = re.compile(r"\[(\d+(?:\.\d+)?)[sS]\]")

# Marker, którym gateway oznacza rolę po przełączeniu na model zapasowy, np.
# "Hegemon (...) [fallback→gpt-4o-mini]". Pozwala wykryć, że zadanie NIE zostało
# wykonane skonfigurowanym modelem, tylko modelem awaryjnym.
_FALLBACK_RE = re.compile(r"\[fallback→(?P<model>[^\]]+)\]")


def _classify_phase(agent_role: str) -> str:
    """Jednoznacznie klasyfikuje fazę telemetryczną na podstawie roli agenta.

    Zwraca jedną z: 'judge', 'reduce', 'map'.
    - 'judge'  – wywołania sędziego (nie liczone do kosztów pipeline'u).
    - 'reduce' – Hegemon / reduktor / monolit / Graph-of-Thoughts (faza scalająca).
    - 'map'    – agenci per-chunk (merytoryczny, językowy, prezentacyjny, utility).

    Dopasowanie jest odporne na sufiks fallbacku (np. '[fallback→gpt-4o-mini]')
    i na wielkość liter.
    """
    role = (agent_role or "").lower()
    if "evaluator" in role or "judge" in role or "sędzia" in role:
        return "judge"
    # 'got' = Graph of Thoughts (faza reduce). Dopasowujemy jako całe słowo, żeby
    # nie wpaść na przypadkowe wystąpienia liter w innych rolach.
    if ("reduce" in role or "hegemon" in role or "monolit" in role
            or "monolith" in role or re.search(r"\bgot\b", role)):
        return "reduce"
    return "map"


def _detect_scenario(report: FinalReport, name: str = "") -> dict:
    """Rozpoznaje typ scenariusza z etykiety i telemetrii oraz jakie sekcje raport
    MÓGŁ wygenerować (żeby sprawiedliwie karać za BRAK informacji, którą scenariusz
    był w stanie dostarczyć — ale nie za sekcje, których dany scenariusz mieć nie może).

    Zwraca:
      - kind: 'monolith' | 'swarm' | 'unknown'
      - is_presentation: bool (czy scenariusz prezentacyjny — dopiero wtedy oczekujemy pokrycia slajdów)
      - expects: zbiór sekcji, które ten scenariusz POWINIEN zawierać.
    """
    label = (name or "").upper()
    details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []
    has_map = any(_classify_phase(p.agent_role) == "map" for p in details)

    if "MONOLITH" in label:
        kind = "monolith"
    elif "SWARM" in label:
        kind = "swarm"
    else:
        kind = "swarm" if has_map else ("monolith" if details else "unknown")

    is_presentation = ("PRESENTATION" in label) or bool(
        report.analysis.slide_coverage or report.analysis.presentation_flow
    )

    # Sekcje oczekiwane od KAŻDEGO raportu (niezależnie od architektury):
    expects = {"factual_summary", "linguistic_summary", "executive_summary",
               "strengths", "areas_for_improvement", "actionable_tips", "scorecard"}
    # Pokrycia slajdów oczekujemy TYLKO w scenariuszu prezentacyjnym.
    if is_presentation:
        expects.add("slide_coverage")
    return {"kind": kind, "is_presentation": is_presentation, "expects": expects}


def _missing_sections(report: FinalReport, expects: set) -> List[str]:
    """Zwraca listę oczekiwanych sekcji, które są PUSTE w raporcie. To są braki, za
    które sędzia MA karać (informacja, którą scenariusz mógł dostarczyć, a nie dostarczył)."""
    a, fb, sc = report.analysis, report.feedback, report.scorecard
    present = {
        "factual_summary": bool((a.factual_summary or "").strip()),
        "linguistic_summary": bool((a.linguistic_summary or "").strip()),
        "executive_summary": bool((fb.executive_summary_markdown or "").strip()),
        "strengths": bool(fb.strengths),
        "areas_for_improvement": bool(fb.areas_for_improvement),
        "actionable_tips": bool(fb.actionable_tips),
        "scorecard": sc is not None and sc.overall_score is not None,
        "slide_coverage": bool(a.slide_coverage),
    }
    labels = {
        "factual_summary": "podsumowanie merytoryczne",
        "linguistic_summary": "analiza językowa",
        "executive_summary": "esej mentorski (executive summary)",
        "strengths": "mocne strony",
        "areas_for_improvement": "obszary do poprawy",
        "actionable_tips": "konkretne wskazówki",
        "scorecard": "ocena punktowa (scorecard)",
        "slide_coverage": "pokrycie slajdów",
    }
    return [labels[k] for k in expects if not present.get(k, False)]


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
            p.tokens_in for p in details if _classify_phase(p.agent_role) == "map"
        )
        return round(map_tokens_in / total_words, 2)

    @staticmethod
    def _phase_densities(report: FinalReport) -> dict:
        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []

        def calc(predicate):
            pc = sum(p.prompt_chars for p in details if predicate(p))
            tin = sum(p.tokens_in for p in details if predicate(p))
            return round(pc / tin, 2) if tin > 0 else 0.0

        return {
            "factual_density": calc(lambda p: "factual" in p.agent_role.lower()),
            "linguistic_density": calc(lambda p: "linguistic" in p.agent_role.lower()),
            "reduce_density": calc(lambda p: _classify_phase(p.agent_role) == "reduce"),
        }

    @staticmethod
    def _split_phase_telemetry(report: FinalReport) -> dict:
        details = report.telemetry.phase_details if report.telemetry and report.telemetry.phase_details else []

        map_cost = 0.0
        reduce_cost = 0.0
        prior_tokens_total = 0

        # Faza reduce (Hegemon/monolit) potrafi mieć KILKA wywołań (Analiza, Feedback,
        # Scoring). Sumowanie tokens_in po wszystkich fazach podwójnie liczy kontekst
        # przenoszony między fazami (analiza trafia ponownie do promptu feedbacku), przez
        # co "Hegemon IN" był sztucznie zawyżony. Zamiast tego:
        #  - hegemon_tokens_in  = tokens_in NAJCIĘŻSZEJ fazy reduce (realny rozmiar wejścia,
        #                         zwykle faza z pełną transkrypcją) — nie suma z double-count,
        #  - hegemon_tokens_out = SUMA wygenerowanych tokenów (to jest addytywne i sensowne),
        #  - liczby per-fazowe zachowujemy w reduce_phases do wglądu/audytu.
        reduce_phases = []
        hegemon_tokens_out = 0

        # Wykrycie modeli, które FAKTYCZNIE wykonały każdą fazę (nie skonfigurowanych).
        reduce_models = []  # (agent_role, model_name)
        map_models = set()
        fallback_models = set()

        for p in details:
            kind = _classify_phase(p.agent_role)
            fb = _FALLBACK_RE.search(p.agent_role or "")
            if fb:
                fallback_models.add(fb.group("model"))

            if kind == "reduce":
                reduce_cost += p.cost_usd
                hegemon_tokens_out += p.tokens_out
                reduce_phases.append({
                    "role": p.agent_role,
                    "model": p.model_name,
                    "tokens_in": p.tokens_in,
                    "tokens_out": p.tokens_out,
                    "cost_usd": round(p.cost_usd, 5),
                })
                reduce_models.append((p.agent_role, p.model_name))
            elif kind == "judge":
                continue
            else:  # map
                map_cost += p.cost_usd
                prior_tokens_total += (p.tokens_in + p.tokens_out)
                if p.model_name:
                    map_models.add(p.model_name)

        # Realny rozmiar wejścia Hegemona = największe pojedyncze wejście fazy reduce.
        hegemon_tokens_in = max((rp["tokens_in"] for rp in reduce_phases), default=0)
        # Sumaryczne wejście reduce (dla porównań kosztowych) — jawnie oddzielone, żeby nie
        # mylić go z rozmiarem pojedynczego promptu.
        hegemon_tokens_in_sum = sum(rp["tokens_in"] for rp in reduce_phases)

        # Model, który faktycznie pełnił rolę Hegemona: bierzemy z najcięższej fazy reduce.
        heaviest = max(reduce_phases, key=lambda rp: rp["tokens_in"], default=None)
        actual_hegemon_model = heaviest["model"] if heaviest else ""
        # Jeśli różne fazy reduce zrobiły różne modele (np. część padła na fallback) — zbierz je.
        distinct_reduce_models = sorted({rp["model"] for rp in reduce_phases if rp["model"]})

        return {
            "map_total_usd": round(map_cost, 5),
            "reduce_usd": round(reduce_cost, 5),
            "prior_tokens_total": prior_tokens_total,
            "hegemon_tokens_in": hegemon_tokens_in,
            "hegemon_tokens_in_sum": hegemon_tokens_in_sum,
            "hegemon_tokens_out": hegemon_tokens_out,
            "reduce_phases": reduce_phases,
            "actual_hegemon_model": actual_hegemon_model,
            "distinct_reduce_models": distinct_reduce_models,
            "map_models": sorted(map_models),
            "fallback_used": bool(fallback_models),
            "fallback_models": sorted(fallback_models),
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
    def _extract_golden_errors(golden_json_str: str) -> List[dict]:
        """Parsuje Golden Set JSON i zwraca PŁASKĄ listę WSZYSTKICH błędów (każda skala:
        LOW/MEDIUM/HIGH/CRITICAL), niezależnie od zagnieżdżenia (np. pod
        'evaluation_ground_truth.factual_errors'). Każdy błąd: {id, time, scale, description}.

        Wcześniej recall liczył tylko HIGH/CRITICAL, przez co połowa błędów ze Złotego
        Wzorca (MEDIUM/LOW) była po cichu pomijana i model nigdy nie był rozliczany ze
        WSZYSTKICH usterek, które powinien wyłapać.
        """
        if not golden_json_str or not golden_json_str.strip():
            return []
        try:
            data = json.loads(golden_json_str)
        except Exception:
            return []

        errors: List[dict] = []

        def looks_like_error(obj: dict) -> bool:
            keys = {k.lower() for k in obj.keys()}
            has_scale = bool(keys & {"scale", "severity", "waga"})
            has_time = bool(keys & {"time", "czas", "timestamp"})
            has_desc = bool(keys & {"description", "opis", "text", "id"})
            return (has_scale or has_time) and has_desc

        def walk(obj):
            if isinstance(obj, dict):
                if looks_like_error(obj):
                    scale = str(obj.get("scale", obj.get("severity", obj.get("waga", "")))).upper()
                    t = obj.get("time", obj.get("czas", obj.get("timestamp")))
                    try:
                        t = float(t) if t is not None else None
                    except (TypeError, ValueError):
                        t = None
                    errors.append({
                        "id": obj.get("id", ""),
                        "time": t,
                        "scale": scale or "MEDIUM",
                        "description": obj.get("description", obj.get("opis", obj.get("text", ""))),
                    })
                    # nie schodź głębiej w rozpoznany błąd (unikamy podwójnego liczenia)
                    return
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return errors

    @staticmethod
    def _format_golden_errors(errors: List[dict]) -> str:
        """Formatuje pełną listę błędów Złotego Wzorca do promptu sędziego, tak by sędzia
        widział KAŻDĄ usterkę (wszystkie skale), a nie tylko duże/krytyczne."""
        if not errors:
            return ""
        order = {"CRITICAL": 0, "KRYTYCZNA": 0, "HIGH": 1, "WYSOKA": 1,
                 "MEDIUM": 2, "ŚREDNIA": 2, "LOW": 3, "NISKA": 3}
        lines = []
        for e in sorted(errors, key=lambda x: (order.get(x["scale"], 2), x["time"] if x["time"] is not None else 0)):
            t = e["time"]
            ts = "—" if t is None else f"{int(t // 60):02d}:{int(t % 60):02d}"
            eid = f"{e['id']} " if e.get("id") else ""
            lines.append(f"- [{e['scale']}] {eid}(oczekiwany czas {ts}): {e['description']}")
        return "\n".join(lines)

    @staticmethod
    def _keywords(text: str, top: int = 8) -> set:
        """Wyciąga charakterystyczne słowa-klucze z opisu błędu (do dopasowania tematycznego).
        Pomija krótkie i pospolite wyrazy; zachowuje liczby (np. '2026', '83', 'WPM')."""
        stop = {
            "oraz", "przez", "jest", "jako", "brak", "sie", "się", "tego", "tym", "nie",
            "dla", "the", "and", "with", "błąd", "blad", "błędu", "chunk", "chunku", "czas",
            "czasu", "opis", "high", "critical", "medium", "low", "wysoka", "krytyczna",
            "niska", "srednia", "średnia", "scale", "skala", "raport", "raportu",
        }
        words = re.findall(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", (text or "").lower())
        kws = [w for w in words if (len(w) >= 5 or w.isdigit()) and w not in stop]
        # zachowaj kolejność, unikaj duplikatów, ogranicz liczbę
        seen, out = set(), []
        for w in kws:
            if w not in seen:
                out.append(w);
                seen.add(w)
            if len(out) >= top:
                break
        return set(out)

    @staticmethod
    def _thematic_hit(error_desc: str, report_text_low: str, min_overlap: int = 2) -> bool:
        """True, jeśli tekst raportu zawiera wystarczająco dużo słów-kluczy z opisu błędu —
        czyli raport OPISUJE ten błąd merytorycznie, nawet jeśli nie podał znacznika czasu."""
        kws = EvaluationEngine._keywords(error_desc)
        if not kws:
            return False
        hits = sum(1 for k in kws if k in report_text_low)
        need = min(min_overlap, len(kws))
        return hits >= need

    @staticmethod
    def _report_fulltext_low(report: FinalReport) -> str:
        a, fb = report.analysis, report.feedback
        return "\n".join([
            a.factual_summary or "", a.linguistic_summary or "",
            "\n".join(a.missed_context or []), "\n".join(a.unverified_claims or []),
            fb.executive_summary_markdown or "",
            "\n".join(fb.strengths or []), "\n".join(fb.areas_for_improvement or []),
            "\n".join(fb.actionable_tips or []), fb.overall_message or "",
        ]).lower()

    @staticmethod
    def _calculate_error_recall(golden_errors: List[dict], report_timestamps: List[float],
                                report_text_low: str = "", tolerance_sec: float = 90.0) -> dict:
        """Liczy wykrywalność (recall) błędów ze Złotego Wzorca. Błąd uznajemy za wykryty, gdy:
          (a) raport cytuje znacznik czasu w pobliżu błędu (± tolerance_sec), LUB
          (b) raport OPISUJE błąd tematycznie (pokrycie słów-kluczy z opisu) — bo część błędów
              (zwł. globalne, time=0, jak tempo/pauzy) nie wymaga dokładnego znacznika czasu.
        Bierze pod uwagę WSZYSTKIE skale.

        Zwraca słownik:
          - overall_pct, critical_pct, by_scale{scale:{total,caught}}, total, caught,
          - caught_by_time, caught_by_theme (rozbicie sposobu dopasowania).
        """
        empty = {"overall_pct": -1.0, "critical_pct": -1.0, "by_scale": {},
                 "total": 0, "caught": 0, "caught_by_time": 0, "caught_by_theme": 0}
        if not golden_errors:
            return empty

        # Błąd bez czasu (time=None) lub globalny (time=0.0) liczymy TYLKO tematycznie.
        scorable = [e for e in golden_errors if (e["time"] is not None) or report_text_low]
        if not scorable:
            return empty

        by_scale: Dict[str, dict] = {}
        caught_total = caught_time = caught_theme = 0
        crit_total = crit_caught = 0

        for e in scorable:
            t = e["time"]
            time_hit = (t is not None and t > 0 and
                        any(abs(t - tr) <= tolerance_sec for tr in report_timestamps))
            theme_hit = bool(report_text_low) and EvaluationEngine._thematic_hit(e["description"], report_text_low)
            hit = time_hit or theme_hit

            bucket = by_scale.setdefault(e["scale"], {"total": 0, "caught": 0})
            bucket["total"] += 1
            if hit:
                bucket["caught"] += 1
                caught_total += 1
                if time_hit:
                    caught_time += 1
                elif theme_hit:
                    caught_theme += 1
            if e["scale"] in ("CRITICAL", "HIGH", "KRYTYCZNA", "WYSOKA"):
                crit_total += 1
                if hit:
                    crit_caught += 1

        return {
            "overall_pct": round((caught_total / len(scorable)) * 100, 2),
            "critical_pct": round((crit_caught / crit_total) * 100, 2) if crit_total else -1.0,
            "by_scale": by_scale,
            "total": len(scorable),
            "caught": caught_total,
            "caught_by_time": caught_time,
            "caught_by_theme": caught_theme,
        }

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
        # UWAGA: celowo NIE podajemy sędziemu deterministycznej oceny (overall_score) raportu —
        # to samo-przyznana ocena, która zakotwiczała sędziego w górę (zwł. przy monolitach
        # z wysokim self-scorem) i zaburzała sprawiedliwość. Podajemy tylko treść do weryfikacji.
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
    def _lost_in_middle(pos_recall: List[float]) -> bool:
        """True, gdy środkowy region ma wyraźnie niższe pokrycie znaczników niż skrajne
        — sygnał 'lost in the middle'. Wymaga 3 regionów i realnego pokrycia na krańcach."""
        if not pos_recall or len(pos_recall) < 3:
            return False
        start, middle, end = pos_recall[0], pos_recall[len(pos_recall) // 2], pos_recall[-1]
        edges = (start + end) / 2.0
        # środek < 60% średniej krańców ORAZ krańce faktycznie coś pokrywają
        return edges > 0.15 and middle < 0.6 * edges

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
                              golden_factual: str = "", golden_linguistic: str = "",
                              scenario_info: dict = None, missing_sections: List[str] = None) -> JudgeRubric:
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

        # Blok sprawiedliwości: mówimy sędziemu, jakiego TYPU jest raport i czego można od niego
        # oczekiwać, oraz WYPUNKTOWUJEMY braki, za które MA karać (informacja, którą scenariusz
        # mógł dostarczyć, a nie dostarczył) — ale nie karze za sekcje, których scenariusz mieć nie może.
        scenario_info = scenario_info or {}
        missing_sections = missing_sections or []
        kind = scenario_info.get("kind", "unknown")
        is_pres = scenario_info.get("is_presentation", False)
        fairness_block = (
                "\n<KONTEKST SCENARIUSZA I SPRAWIEDLIWOŚĆ OCENY>\n"
                f"Typ architektury raportu: {kind.upper()}"
                + (" (scenariusz PREZENTACYJNY — oczekuj analizy pokrycia slajdów)\n" if is_pres else "\n")
                + "ZASADA SPRAWIEDLIWOŚCI: Oceniaj po JAKOŚCI i KOMPLETNOŚCI informacji, a nie po długości.\n"
                  "- Raport monolityczny NIE jest z definicji gorszy ani lepszy od roju — liczy się treść.\n"
                  "- KARZ za BRAK informacji, którą ten scenariusz mógł dostarczyć (płytka analiza "
                  "merytoryczna/językowa, brak konkretów, ogólniki, brak oceny mocnych stron/obszarów do "
                  "poprawy/wskazówek). Płytki, ubogi raport MA dostać niską ocenę specificity i actionability.\n"
                  "- NIE karz za brak sekcji, których dany scenariusz mieć NIE MOŻE (np. brak pokrycia "
                  "slajdów w scenariuszu nieprezentacyjnym).\n"
        )
        if missing_sections:
            fairness_block += (
                    "WYKRYTE BRAKI W TYM RAPORCIE (obniż ocenę odpowiednio — to informacja, której "
                    "zabrakło, mimo że scenariusz mógł ją dostarczyć):\n- "
                    + "\n- ".join(missing_sections) + "\n"
            )
        else:
            fairness_block += "Wszystkie oczekiwane sekcje są obecne (nie oznacza to jeszcze wysokiej jakości).\n"

        golden_block = ""
        if golden_factual or golden_linguistic:
            golden_block = (
                "\n<ZŁOTY WZORZEC (GOLDEN SET) — PEŁNA LISTA BŁĘDÓW, KTÓRE RAPORT POWINIEN WYKRYĆ>\n"
                "UWAGA: poniżej wymieniono WSZYSTKIE błędy (każda skala: LOW/MEDIUM/HIGH/CRITICAL), "
                "nie tylko te największe. Rozlicz raport ze WSZYSTKICH z nich.\n"
            )
            if golden_factual:
                golden_block += f"--- BŁĘDY MERYTORYCZNE ---\n{golden_factual}\n\n"
            if golden_linguistic:
                golden_block += f"--- BŁĘDY LINGWISTYCZNE ---\n{golden_linguistic}\n\n"

        prompt = f"""Jesteś surowym ekspertem MLOps i sędzią (LLM-as-a-Judge) jakości systemów AI.
Oceniasz JAKOŚĆ poniższego raportu z analizy przemówienia.
{fairness_block}
{golden_block}
<FRAGMENT TRANSKRYPCJI — opcjonalny kontekst>
{transcript_excerpt}
{ground_block}{probe_block}{focus_block}

<RAPORT DO OCENY (scenariusz: {scenario_name})>
{self._report_text(report)}

ZASADY OCENY:
1. Tabela Wykrywalności: W swoim uzasadnieniu wygeneruj tabelę Markdown zestawiającą KAŻDY błąd z pełnej listy ZŁOTY WZORZEC (wszystkie skale, nie tylko duże/krytyczne) z tym, co raport FAKTYCZNIE wykrył. Kolumny: | Błąd ze Złotego Wzorca | Skala | Oczekiwany Czas | Czy raport go wyłapał? |. Błąd uznaj za wyłapany tylko, gdy raport odnosi się do niego merytorycznie (a nie przypadkowo trafia w pobliski znacznik czasu). Jeśli raport pominął usterki — zwłaszcza z dalszej części wykładu — tnij ocenę Groundedness i Actionability proporcjonalnie do liczby i wagi pominięć.
2. Groundedness (Ugruntowanie): Karz za halucynacje. Jeśli raport zmyśla wnioski, na które nie ma dowodów, obniż ocenę.
3. Tool Adherence: W scenariuszach z RAG/WEB model musi kategorycznie odrzucać kłamstwa. Karz za lenistwo (asekurowanie się statusem UNVERIFIED dla jawnych kłamstw historycznych).
4. Actionability vs Vague Praise: Policz konkretne rady. "Lanie wody" bez podawania konkretnych [MM:SS] to ocena minimalna.

Oceń raport w 5 wymiarach 0-10 (actionability, specificity, correctness, tone, groundedness)
i podaj uzasadnienie (wraz z Tabelą Błędów obejmującą wszystkie skale). Zwróć wynik zgodnie ze schematem."""
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

        # Parsujemy Złoty Wzorzec RAZ: pełne (wszystkie skale) listy błędów dla obu wymiarów.
        # Używamy ich zarówno do promptu sędziego (żeby widział KAŻDY błąd), jak i do recall.
        golden_factual_errors = self._extract_golden_errors(golden_factual)
        golden_linguistic_errors = self._extract_golden_errors(golden_linguistic)
        all_golden_errors = golden_factual_errors + golden_linguistic_errors

        golden_factual_fmt = self._format_golden_errors(golden_factual_errors) or golden_factual
        golden_linguistic_fmt = self._format_golden_errors(golden_linguistic_errors) or golden_linguistic

        for name, report in reports.items():
            scenario_info = _detect_scenario(report, name)
            missing = _missing_sections(report, scenario_info["expects"])
            report_text_low = self._report_fulltext_low(report)

            probes = ""
            try:
                probes = self.build_timestamp_probes(report, transcript_excerpt, duration_sec,
                                                     self.probe_timestamps, self.probe_window_chars)
                rubric = await self._judge_absolute(grounded_excerpt, name, report, probes=probes,
                                                    golden_factual=golden_factual_fmt,
                                                    golden_linguistic=golden_linguistic_fmt,
                                                    scenario_info=scenario_info,
                                                    missing_sections=missing)
            except Exception as e:
                rubric = JudgeRubric(justification=f"[Sędzia zawiódł: {e}]")

            total = rubric.actionability + rubric.specificity + rubric.correctness + rubric.tone + rubric.groundedness
            pos_recall = self.positional_recall(report, duration_sec)
            red_fidelity = self.reduce_fidelity(report, duration_sec)
            lost_in_middle = self._lost_in_middle(pos_recall)

            report_ts = self._parsed_timestamps(report)

            # Wykrywalność liczona na PEŁNYM zbiorze błędów (merytoryczne + lingwistyczne,
            # wszystkie skale), z dopasowaniem czasowym LUB tematycznym (dla błędów globalnych).
            recall = self._calculate_error_recall(all_golden_errors, report_ts, report_text_low)
            error_recall_pct = recall["overall_pct"]

            phase_costs = self._split_phase_telemetry(report)
            tpw = self._calculate_language_tax(report, total_words)
            alignment_error = self._calculate_alignment_error(report, expected_factual, expected_linguistic)
            densities = self._phase_densities(report)

            extra_metrics[name] = {
                "rmse": alignment_error,
                "tpw": tpw,
                "costs": phase_costs,
                "error_recall_pct": error_recall_pct,
                "error_recall_critical_pct": recall["critical_pct"],
                "error_recall_detail": recall,
                "caught_by_time": recall.get("caught_by_time", 0),
                "caught_by_theme": recall.get("caught_by_theme", 0),
                "scenario_kind": scenario_info["kind"],
                "is_presentation": scenario_info["is_presentation"],
                "missing_sections": missing,
                "lost_in_middle": lost_in_middle,
                "actual_hegemon_model": phase_costs.get("actual_hegemon_model", ""),
                "fallback_used": phase_costs.get("fallback_used", False),
                "fallback_models": phase_costs.get("fallback_models", []),
                "factual_density": densities.get("factual_density", 0.0),
                "linguistic_density": densities.get("linguistic_density", 0.0),
                "reduce_density": densities.get("reduce_density", 0.0)
            }

            ground_truth = self._ground_truth_findings(report)
            recall_line = (
                f"Wykryte błędy: {recall['caught']}/{recall['total']} "
                f"({error_recall_pct}%)  |  CRIT/HIGH: {recall['critical_pct']}%  "
                f"[czasowo: {recall.get('caught_by_time', 0)}, tematycznie: {recall.get('caught_by_theme', 0)}]"
                if recall["total"] else "Wykryte błędy: brak danych (Złoty Wzorzec pusty)"
            )
            missing_line = ("BRAKI (informacja, której zabrakło): " + "; ".join(missing)) if missing \
                else "Braki: brak (wszystkie oczekiwane sekcje obecne)"
            lim_line = "TAK — środek słabiej pokryty niż krańce" if lost_in_middle else "nie wykryto"
            evidence = (
                f"=== TYP SCENARIUSZA === {scenario_info['kind'].upper()}"
                f"{' / PREZENTACJA' if scenario_info['is_presentation'] else ''}\n"
                f"=== KOMPLETNOŚĆ === {missing_line}\n"
                f"=== LOST-IN-THE-MIDDLE === {lim_line}  (positional_recall={pos_recall})\n\n"
                f"=== WYKRYWALNOŚĆ BŁĘDÓW ===\n{recall_line}\n\n"
                f"=== ZŁOTY WZORZEC MERYTORYCZNY (pełna lista) ===\n{golden_factual_fmt or '(brak)'}\n\n"
                f"=== ZŁOTY WZORZEC LINGWISTYCZNY (pełna lista) ===\n{golden_linguistic_fmt or '(brak)'}\n\n"
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
                lost_in_middle_flag=lost_in_middle,
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
