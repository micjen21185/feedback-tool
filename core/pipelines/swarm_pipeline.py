import asyncio
import logging
from typing import List, Tuple, Any

from models.schemas import (
    LectureMetadata, ChunkPayload, HegemonOutput, DeepAnalysis, ConstructiveFeedback
)

logger = logging.getLogger(__name__)


class SwarmNaivePipeline:
    def __init__(self, linguistic_agent, factual_agent, combine_engine, hegemon, config, use_tools: bool = False,
                 progress_cb=None):
        self.linguistic_agent = linguistic_agent
        self.factual_agent = factual_agent
        self.combine_engine = combine_engine
        self.hegemon = hegemon
        self.config = config
        self.use_tools = use_tools
        self._progress_cb = progress_cb or (lambda _msg: None)

    def _progress(self, msg: str):
        # Mirror to console log so progress/hangs are visible in the terminal even when
        # Streamlit's st.status doesn't stream live during the blocking run.
        logger.info("[swarm] %s", msg)
        self._progress_cb(msg)

    async def execute(self, metadata: LectureMetadata, chunks: List[ChunkPayload],
                      presentation_context: str = "", slide_coverage: list = None) -> HegemonOutput:
        batch_size = 4
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        self._progress(f"🧠 Faza MAP: {len(chunks)} chunków w {len(batches)} batchach (po {batch_size}).")

        # Batches run sequentially so trailing state flows ACROSS batch boundaries too
        # (in-batch chunks still run concurrently inside _process_batch — map-reduce preserved).
        mapped_results = []
        carry_ling = None
        carry_fact = None
        for bi, batch in enumerate(batches, start=1):
            if batch:
                if carry_ling is not None:
                    batch[0].trailing_linguistics = carry_ling
                if carry_fact is not None:
                    batch[0].trailing_fact_summary = carry_fact

            self._progress(f"   ⏳ MAP batch {bi}/{len(batches)} (agent merytoryczny + językowy równolegle)…")
            batch_out = await self._process_batch(batch, metadata)
            mapped_results.extend(batch_out)

            if batch_out:
                last_ling, last_fact = batch_out[-1]
                carry_ling = getattr(last_ling, 'next_state', None) if last_ling else carry_ling
                carry_fact = getattr(last_fact, 'next_state', None) if last_fact else carry_fact

        ling_results = [r[0] for r in mapped_results if r[0] is not None]
        fact_results = [r[1] for r in mapped_results if r[1] is not None]

        self._progress("🔗 Faza COMBINE: agregacja i redukcja ustaleń z chunków (deterministyczna)…")
        thematic_blocks, behavioral_profiles = await self.combine_engine.aggregate_dual_track(mapped_results, metadata)
        scorecard = self.combine_engine.compute_scorecard(ling_results, fact_results, slide_coverage)
        self._progress(
            f"   ✅ Zredukowano do {len(thematic_blocks)} bloków merytorycznych "
            f"i {len(behavioral_profiles)} profili behawioralnych. Ocena: {scorecard.overall_score}/100."
        )

        # Claims not confirmed against any trusted source (RAG/slides) — computed deterministically
        # from the map's verification_status. Passed to the reducer AND kept as the authoritative list.
        unverified_claims = self.combine_engine.collect_unverified_claims(fact_results)

        self._progress("🏛️ Faza REDUCE: Hegemon generuje raport końcowy…")
        try:
            report = await self.hegemon.generate_report(
                metadata=metadata,
                thematic_blocks=thematic_blocks,
                behavioral_profiles=behavioral_profiles,
                presentation_context=presentation_context,
                scorecard=scorecard,
                unverified_claims=unverified_claims
            )
            self._progress("   ✅ Hegemon zakończył raport.")
        except Exception as e:
            # The reduce call failed (e.g. timeout on a 40+ min talk). Do NOT lose the whole
            # run: the scorecard is deterministic and the map findings are already aggregated,
            # so return a degraded-but-valid report built from what we have.
            logger.error("Hegemon reduce failed (%s). Returning degraded report from map findings.", e)
            self._progress(f"   ⚠️ Hegemon zawiódł ({e}). Zwracam raport awaryjny z ustaleń fazy map.")
            report = self._build_fallback_report(thematic_blocks, behavioral_profiles)
        report.scorecard = scorecard

        # Timestamps of chunks that produced ANY finding in the map phase — used by the
        # evaluator to measure how much of each time-region survived into the final report.
        map_ts = set()
        for r in ling_results:
            if r.anomaly_texts():
                map_ts.add(round(r.start_time, 1))
        for r in fact_results:
            if r.error_texts():
                map_ts.add(round(r.start_time, 1))
        # Authoritative (deterministic) unverified list — overrides whatever the essay re-mentioned.
        report.analysis.unverified_claims = unverified_claims

        report.map_timestamps = sorted(map_ts)

        # Non-penalizing substance density: how many map windows carried substantive factual
        # content (a non-empty thematic summary). Observability only — does NOT affect the score.
        report.total_windows = len(fact_results)
        report.substantive_windows = sum(1 for r in fact_results if (r.thematic_summary or "").strip())
        return report

    async def _process_batch(self, batch: List[ChunkPayload], metadata: LectureMetadata) -> List[Tuple[Any, Any]]:
        batch_results = []
        use_tools = self.use_tools

        for i, chunk in enumerate(batch):
            ling_task = self.linguistic_agent.analyze(chunk, metadata)
            fact_task = self.factual_agent.analyze(chunk, metadata, use_tools=use_tools)

            ling_out, fact_out = await asyncio.gather(ling_task, fact_task)
            batch_results.append((ling_out, fact_out))

            if i + 1 < len(batch):
                if ling_out and getattr(ling_out, 'next_state', None) is not None:
                    batch[i + 1].trailing_linguistics = ling_out.next_state

                if fact_out and getattr(fact_out, 'next_state', None) is not None:
                    batch[i + 1].trailing_fact_summary = fact_out.next_state

        return batch_results

    @staticmethod
    def _build_fallback_report(thematic_blocks: List[str], behavioral_profiles: List[str]) -> HegemonOutput:
        # Degraded report used when the Hegemon reduce call fails. Surfaces the raw aggregated
        # map findings so the run is not wasted; the deterministic scorecard is attached by the caller.
        thematic_text = "\n".join(thematic_blocks) if isinstance(thematic_blocks, list) else str(thematic_blocks)
        behavioral_text = "\n".join(behavioral_profiles) if isinstance(behavioral_profiles, list) else str(
            behavioral_profiles)
        notice = ("⚠️ Faza reduce (Hegemon) nie powiodła się — prawdopodobnie przekroczono limit czasu na długim "
                  "wystąpieniu. Poniżej surowe, zagregowane ustalenia z fazy map oraz deterministyczna ocena punktowa.")
        return HegemonOutput(
            analysis=DeepAnalysis(
                factual_summary=f"[Raport awaryjny] {notice}\n\n{thematic_text}",
                linguistic_summary=behavioral_text,
                missed_context=[]
            ),
            feedback=ConstructiveFeedback(
                executive_summary_markdown=(
                    f"> {notice}\n\n"
                    f"### Ustalenia merytoryczne (map)\n{thematic_text}\n\n"
                    f"### Profile behawioralne (map)\n{behavioral_text}"
                ),
                strengths=[],
                areas_for_improvement=[],
                actionable_tips=["Uruchom ponownie fazę reduce (mniejszy model Hegemona lub większy limit czasu)."],
                overall_message=notice
            )
        )
