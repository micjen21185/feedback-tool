import asyncio
from typing import Dict, List

from models.schemas import (
    ChunkPayload, SlideSummary,
    SlideCoverage, PresentationFlow, SlideFlowStat
)

VERY_SHORT_SEC = 15.0
LONG_PER_POINT_SEC = 90.0


class SlideCoverageEngine:
    """Scenario-5 reduce phase: per-slide coverage analysis + deterministic flow stats."""

    def __init__(self, gateway, model: str):
        self.gateway = gateway
        self.model = model

    def build_presentation_flow(
            self,
            slide_summaries: Dict[str, SlideSummary]
    ) -> PresentationFlow:
        stats: Dict[int, SlideFlowStat] = {}

        for key, summary in slide_summaries.items():
            sid = summary.slide_id
            total = sum(max(0.0, a.end_time - a.start_time) for a in summary.timeline_appearances)
            appearances = max(1, len(summary.timeline_appearances))
            stats[sid] = SlideFlowStat(slide_id=sid, time_on_slide_sec=round(total, 1), appearances=appearances)

        very_short = [s.slide_id for s in stats.values() if s.time_on_slide_sec < VERY_SHORT_SEC]

        very_long = []
        for key, summary in slide_summaries.items():
            sid = summary.slide_id
            points = max(1, len([ln for ln in (summary.pdf_text or "").splitlines() if ln.strip()]))
            if sid in stats and stats[sid].time_on_slide_sec > points * LONG_PER_POINT_SEC:
                very_long.append(sid)

        ordered = sorted(stats.values(), key=lambda s: s.slide_id)
        summary_bits = []
        if very_short:
            summary_bits.append(f"Slajdy pokazane bardzo krótko (<{int(VERY_SHORT_SEC)}s): {very_short}")
        if very_long:
            summary_bits.append(f"Slajdy z nadmiernym czasem względem treści: {very_long}")
        if not summary_bits:
            summary_bits.append("Rozkład czasu na slajdach wygląda zbalansowanie.")

        return PresentationFlow(
            total_slides=len(stats),
            slide_stats=ordered,
            very_short_slides=very_short,
            very_long_slides=very_long,
            flow_summary=" ".join(summary_bits)
        )

    async def analyze_slide_coverage(
            self,
            chunks: List[ChunkPayload],
            slide_summaries: Dict[str, SlideSummary]
    ) -> List[SlideCoverage]:
        # Group chunks by slide_id across the whole talk (returns share the same slide_id).
        by_slide: Dict[int, List[ChunkPayload]] = {}
        returned: Dict[int, bool] = {}
        for ch in chunks:
            sid = ch.chunk_meta.slide_id
            if sid is None:
                continue
            by_slide.setdefault(sid, []).append(ch)
            if ch.chunk_meta.is_return_to_slide:
                returned[sid] = True

        async def _one(sid: int, slide_chunks: List[ChunkPayload]) -> SlideCoverage:
            folder_key = f"Slide_{sid:02d}"
            slide_summary = slide_summaries.get(folder_key)
            slide_text = slide_summary.pdf_text if slide_summary else "Brak treści OCR."

            slide_chunks_sorted = sorted(slide_chunks, key=lambda c: c.chunk_meta.start_time)
            speech = "\n".join(c.text_data.clean_text for c in slide_chunks_sorted)

            time_on_slide = 0.0
            if slide_summary:
                time_on_slide = sum(
                    max(0.0, a.end_time - a.start_time) for a in slide_summary.timeline_appearances
                )

            prompt = f"""Jesteś analitykiem prezentacji. Porównaj TREŚĆ SLAJDU z tym, CO PRELEGENT POWIEDZIAŁ o tym slajdzie.

<TREŚĆ SLAJDU (OCR)>
{slide_text}

<WYPOWIEDŹ PRELEGENTA DOTYCZĄCA TEGO SLAJDU (chronologicznie, łącznie z ewentualnymi powrotami)>
{speech}

Zadanie:
- Wypisz punkty slajdu, które prelegent OMÓWIŁ (covered_points).
- Wypisz punkty slajdu, które POMINĄŁ (missed_points).
- Oceń, czy jeśli wrócił do slajdu później, to uzupełnił wcześniejsze braki (completed_on_return).
"""
            coverage: SlideCoverage = await self.gateway.execute_structured(
                prompt=prompt,
                schema_class=SlideCoverage,
                model=self.model,
                agent_role=f"Slide Coverage (Reduce) - Slide {sid}"
            )
            coverage.slide_id = sid
            coverage.returned_later = returned.get(sid, False)
            coverage.time_on_slide_sec = round(time_on_slide, 1)
            if not coverage.dwell_verdict:
                if time_on_slide and time_on_slide < VERY_SHORT_SEC:
                    coverage.dwell_verdict = "ZA_KRÓTKO"
                else:
                    coverage.dwell_verdict = "OK"
            return coverage

        tasks = [_one(sid, chs) for sid, chs in sorted(by_slide.items())]
        return list(await asyncio.gather(*tasks)) if tasks else []

    @staticmethod
    def format_for_reducer(slide_coverage: List[SlideCoverage], flow: PresentationFlow) -> str:
        lines = [f"Slajdów łącznie: {flow.total_slides}. {flow.flow_summary}"]
        for cov in slide_coverage:
            missed = ", ".join(cov.missed_points) if cov.missed_points else "brak"
            ret = " (powrót później)" if cov.returned_later else ""
            lines.append(
                f"- Slajd {cov.slide_id}{ret}: czas {cov.time_on_slide_sec}s [{cov.dwell_verdict}]; "
                f"pominięte punkty: {missed}"
            )
        return "\n".join(lines)
