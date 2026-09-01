import difflib
import logging
from collections import Counter
from typing import List, Tuple, Any, Optional

from models.schemas import LinguisticOutput, FactualOutput, LectureMetadata, readiness_verdict

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
_SEVERITY_WEIGHT = {"CRITICAL": 15, "HIGH": 7, "MEDIUM": 3, "LOW": 1}

# Track blend and readiness thresholds.
_FACTUAL_WEIGHT = 0.6
_LINGUISTIC_WEIGHT = 0.4

# Hierarchical reduce: above this many windows, collapse LOW/MEDIUM into segment summaries
# while keeping CRITICAL/HIGH verbatim, to bound the Hegemon prompt size.
_REDUCE_WINDOW_THRESHOLD = 15
_REDUCE_SEGMENTS = 5
_KEEP_VERBATIM = ("CRITICAL", "HIGH")


def _wpm_band(audience: str) -> Tuple[int, int]:
    a = (audience or "").lower()
    if any(k in a for k in ("ekspert", "senior", "architekt", "nauk")):
        return 110, 150
    if any(k in a for k in ("student", "podstaw", "junior", "begin")):
        return 100, 140
    return 110, 160


class CombineEngine:
    def __init__(self, gateway: Any, config: Any):
        self.gateway = gateway
        self.config = config
        self.similarity_threshold = 0.85

        self.compressor = None
        if getattr(self.config, 'use_llmlingua', False):
            try:
                from llmlingua import PromptCompressor
                self.compressor = PromptCompressor("llmlingua-small")
                logger.info("LLMLingua loaded into RAM.")
            except ImportError:
                logger.error("llmlingua is not installed. Install it (uncomment in requirements.txt).")
                self.config.use_llmlingua = False

    def _is_duplicate_error(self, new_error: str, seen_errors: List[str]) -> bool:
        new_norm = new_error.lower().strip()
        for seen in seen_errors:
            if difflib.SequenceMatcher(None, new_norm, seen).ratio() > self.similarity_threshold:
                return True
        return False

    def build_metrics_verdict(self, metadata: Optional[LectureMetadata],
                              ling_results: List[LinguisticOutput]) -> str:
        if metadata is None:
            return ""

        lines = ["<WERDYKT ILOŚCIOWY (obliczony deterministycznie)>"]

        # Tempo / WPM verdict
        mean_wpm = 0
        if metadata.total_duration_sec > 0:
            mean_wpm = round(metadata.total_words / (metadata.total_duration_sec / 60.0))
        low, high = _wpm_band(metadata.target_audience)
        if mean_wpm:
            if mean_wpm < low:
                tempo = f"ZA WOLNO ({mean_wpm} WPM, zalecane {low}-{high})"
            elif mean_wpm > high:
                tempo = f"ZA SZYBKO ({mean_wpm} WPM, zalecane {low}-{high})"
            else:
                tempo = f"OK ({mean_wpm} WPM, w zakresie {low}-{high})"
            lines.append(
                f"- Tempo mowy: {tempo}. Skrajne okna: {metadata.slowest_chunk_wpm}-{metadata.fastest_chunk_wpm} WPM.")

        lines.append(
            f"- Wypełniacze łącznie: {metadata.total_filler_words} | "
            f"Powtarzalne tendencje: {metadata.total_repeated_tendencies} | "
            f"Niewyraźne słowa: {metadata.total_unclear_words}"
        )
        lines.append(
            f"- Znaczące pauzy: {metadata.total_significant_pauses} "
            f"(łącznie {metadata.total_significant_pauses_duration_sec}s) | "
            f"Śr. pewność transkrypcji: {metadata.overall_transcription_confidence}"
        )

        clean_windows = sum(1 for l in ling_results if not l.scored_anomalies and not l.anomalies)
        lines.append(f"- Okna bez anomalii lingwistycznych: {clean_windows} / {len(ling_results)}")
        return "\n".join(lines)

    @staticmethod
    def _sev_value(item) -> str:
        return item.severity.value if hasattr(item.severity, "value") else str(item.severity)

    def compute_scorecard(
            self,
            ling_results: List[LinguisticOutput],
            fact_results: List[FactualOutput],
            slide_coverage: Optional[list] = None
    ):
        # Import here to avoid a circular import at module load.
        from models.schemas import ScoreCard

        n_windows = max(1, len(ling_results) or len(fact_results))

        # Penalty per track = sum of severity weights, normalized per window, scaled to 0-100.
        # Fallback: if the model returned plain items but no scored items, count each plain
        # item at the MEDIUM weight so the score does not silently stay at 100.
        def _track_penalty(items_scored, items_plain) -> int:
            if items_scored:
                return sum(_SEVERITY_WEIGHT.get(self._sev_value(it), 3) for it in items_scored)
            return len(items_plain) * _SEVERITY_WEIGHT["MEDIUM"]

        ling_penalty = sum(_track_penalty(l.scored_anomalies, l.anomaly_texts()) for l in ling_results)
        fact_penalty = sum(_track_penalty(f.scored_errors, f.error_texts()) for f in fact_results)

        # Normalize: each window can absorb ~one HIGH (7 pts) before the score drops materially.
        ling_score = max(0.0, 100.0 - (ling_penalty / n_windows) * (100.0 / 21.0))
        fact_score = max(0.0, 100.0 - (fact_penalty / n_windows) * (100.0 / 21.0))

        slide_score = None
        weights = {"factual": _FACTUAL_WEIGHT, "linguistic": _LINGUISTIC_WEIGHT}
        if slide_coverage:
            total_pts = sum(len(c.covered_points) + len(c.missed_points) for c in slide_coverage)
            covered = sum(len(c.covered_points) for c in slide_coverage)
            slide_score = 100.0 if total_pts == 0 else round(100.0 * covered / total_pts, 1)
            # Rebalance: factual 0.45, linguistic 0.30, slides 0.25 when presentation data exists.
            weights = {"factual": 0.45, "linguistic": 0.30, "slides": 0.25}

        overall = weights["factual"] * fact_score + weights["linguistic"] * ling_score
        if slide_score is not None:
            overall += weights["slides"] * slide_score
        overall = round(overall, 1)

        return ScoreCard(
            factual_score=round(fact_score, 1),
            linguistic_score=round(ling_score, 1),
            slide_coverage_score=slide_score,
            overall_score=overall,
            readiness_verdict=readiness_verdict(overall)
        )

    def _build_behavioral_detailed(self, ling_results: List[LinguisticOutput]) -> List[str]:
        profiles = []
        clean_streak = 0
        for l_out in ling_results:
            items = l_out.scored_anomalies
            if not items:
                clean_streak += 1
                continue
            if clean_streak:
                profiles.append(f"[czyste] {clean_streak} kolejnych okien bez anomalii.")
                clean_streak = 0

            profile_str = f"[{l_out.start_time}s] Anomalie okna:\n"
            for it in sorted(items, key=lambda x: _SEVERITY_ORDER.get(self._sev_value(x), 1), reverse=True):
                profile_str += f" - ({self._sev_value(it)}) {it.text}\n"
            if l_out.dominant_tendencies:
                profile_str += f"   Dominująca tendencja: {l_out.dominant_tendencies}\n"
            profiles.append(profile_str)

        if clean_streak:
            profiles.append(f"[czyste] {clean_streak} kolejnych okien bez anomalii.")
        return profiles

    def _build_behavioral_hierarchical(self, ling_results: List[LinguisticOutput]) -> List[str]:
        profiles = []

        keep = []
        for l_out in ling_results:
            for it in l_out.scored_anomalies:
                if self._sev_value(it) in _KEEP_VERBATIM:
                    keep.append(f"[{l_out.start_time}s] ({self._sev_value(it)}) {it.text}")

        if keep:
            profiles.append("<KLUCZOWE ANOMALIE (CRITICAL/HIGH, zachowane w całości)>\n" + "\n".join(keep))

        segments = self._segment(ling_results)
        seg_lines = ["<PODSUMOWANIE SEGMENTAMI (pozostałe, drobne)>"]
        for (lo, hi), group in segments:
            minor = sum(
                1 for l_out in group for it in l_out.scored_anomalies
                if self._sev_value(it) not in _KEEP_VERBATIM
            )
            tendencies = [l_out.dominant_tendencies for l_out in group if l_out.dominant_tendencies]
            top_tend = Counter(tendencies).most_common(1)
            tend_str = f", przewaga: {top_tend[0][0]}" if top_tend else ""
            seg_lines.append(f"Segment {int(lo)}-{int(hi)}s: {len(group)} okien, {minor} drobnych anomalii{tend_str}")
        profiles.append("\n".join(seg_lines))
        return profiles

    def _build_thematic_detailed(self, fact_results: List[FactualOutput]) -> List[str]:
        blocks = []
        seen_errors: List[str] = []
        for f_out in fact_results:
            block_str = f"[{f_out.start_time}s] Temat: {f_out.thematic_summary}\n"
            unique = []
            for it in f_out.scored_errors:
                if not self._is_duplicate_error(it.text, seen_errors):
                    seen_errors.append(it.text.lower().strip())
                    unique.append((self._sev_value(it), it.text))
            unique.sort(key=lambda t: _SEVERITY_ORDER.get(t[0], 1), reverse=True)
            if unique:
                block_str += " NOWE błędy (wg wagi):\n" + "\n".join([f"  -> ({sev}) {err}" for sev, err in unique])
            blocks.append(block_str)
        return blocks

    def _build_thematic_hierarchical(self, fact_results: List[FactualOutput]) -> List[str]:
        blocks = []
        seen_errors: List[str] = []

        keep = []
        for f_out in fact_results:
            for it in f_out.scored_errors:
                if self._is_duplicate_error(it.text, seen_errors):
                    continue
                seen_errors.append(it.text.lower().strip())
                if self._sev_value(it) in _KEEP_VERBATIM:
                    keep.append(f"[{f_out.start_time}s] ({self._sev_value(it)}) {it.text}")

        if keep:
            blocks.append("<KLUCZOWE BŁĘDY MERYTORYCZNE (CRITICAL/HIGH)>\n" + "\n".join(keep))

        segments = self._segment(fact_results)
        seg_lines = ["<PODSUMOWANIE TEMATYCZNE SEGMENTAMI>"]
        for (lo, hi), group in segments:
            topics = [f.thematic_summary for f in group if f.thematic_summary]
            topic_str = "; ".join(topics[:4]) if topics else "brak wyraźnych tematów"
            seg_lines.append(f"Segment {int(lo)}-{int(hi)}s: {topic_str}")
        blocks.append("\n".join(seg_lines))
        return blocks

    @staticmethod
    def _segment(items):
        # Split time-sorted items into up to _REDUCE_SEGMENTS contiguous groups by index.
        if not items:
            return []
        n = len(items)
        seg_count = min(_REDUCE_SEGMENTS, n)
        size = -(-n // seg_count)  # ceil division
        result = []
        for i in range(0, n, size):
            group = items[i:i + size]
            lo = group[0].start_time
            hi = group[-1].start_time
            result.append(((lo, hi), group))
        return result

    async def aggregate_dual_track(
            self,
            mapped_results: List[Tuple[Any, Any]],
            metadata: Optional[LectureMetadata] = None
    ) -> Tuple[List[str], List[str]]:
        ling_results: List[LinguisticOutput] = [res[0] for res in mapped_results if res[0] is not None]
        fact_results: List[FactualOutput] = [res[1] for res in mapped_results if res[1] is not None]

        ling_results.sort(key=lambda x: x.start_time)
        fact_results.sort(key=lambda x: x.start_time)

        # Normalize: if a model returned only plain lists (no scored items), backfill scored
        # items at MEDIUM so the severity-based builders/scoring have data to work with.
        from models.schemas import SeverityItem, AnomalySeverity
        for l_out in ling_results:
            if not l_out.scored_anomalies and l_out.anomalies:
                l_out.scored_anomalies = [SeverityItem(text=a, severity=AnomalySeverity.MEDIUM) for a in
                                          l_out.anomalies]
        for f_out in fact_results:
            if not f_out.scored_errors and f_out.factual_errors:
                f_out.scored_errors = [SeverityItem(text=e, severity=AnomalySeverity.MEDIUM) for e in
                                       f_out.factual_errors]

        # --- Behavioral (linguistic) track ---
        behavioral_profiles = []

        verdict = self.build_metrics_verdict(metadata, ling_results)
        if verdict:
            behavioral_profiles.append(verdict)

        global_anomaly_severity: Counter = Counter()
        for l_out in ling_results:
            for item in l_out.scored_anomalies:
                global_anomaly_severity[self._sev_value(item)] += 1

        if global_anomaly_severity:
            tally = " | ".join(f"{k}: {global_anomaly_severity[k]}" for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if
                               global_anomaly_severity.get(k))
            behavioral_profiles.append(f"<ROZKŁAD WAGI ANOMALII>\n{tally}")

        if len(ling_results) > _REDUCE_WINDOW_THRESHOLD:
            behavioral_profiles.extend(self._build_behavioral_hierarchical(ling_results))
        else:
            behavioral_profiles.extend(self._build_behavioral_detailed(ling_results))

        # --- Factual track ---
        if len(fact_results) > _REDUCE_WINDOW_THRESHOLD:
            thematic_blocks = self._build_thematic_hierarchical(fact_results)
        else:
            thematic_blocks = self._build_thematic_detailed(fact_results)

        if getattr(self.config, 'use_llmlingua', False) and self.compressor:
            raw_thematic_text = "\n".join(thematic_blocks)

            compressed_result = self.compressor.compress_prompt(
                context=[raw_thematic_text],
                instruction="Zachowaj daty, liczby, błędy i kluczowe pojęcia techniczne.",
                rate=0.55,
                force_tokens=['\n', ':', '[', ']', '->']
            )

            thematic_blocks = [compressed_result["compressed_prompt"]]

        return thematic_blocks, behavioral_profiles
