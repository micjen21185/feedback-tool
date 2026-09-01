import asyncio
from typing import List, Tuple, Any

from models.schemas import (
    LectureMetadata, ChunkPayload, HegemonOutput
)


class SwarmNaivePipeline:
    def __init__(self, linguistic_agent, factual_agent, combine_engine, hegemon, config, use_tools: bool = False):
        self.linguistic_agent = linguistic_agent
        self.factual_agent = factual_agent
        self.combine_engine = combine_engine
        self.hegemon = hegemon
        self.config = config
        self.use_tools = use_tools

    async def execute(self, metadata: LectureMetadata, chunks: List[ChunkPayload],
                      presentation_context: str = "", slide_coverage: list = None) -> HegemonOutput:
        batch_size = 4
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

        # Batches run sequentially so trailing state flows ACROSS batch boundaries too
        # (in-batch chunks still run concurrently inside _process_batch — map-reduce preserved).
        mapped_results = []
        carry_ling = None
        carry_fact = None
        for batch in batches:
            if batch:
                if carry_ling is not None:
                    batch[0].trailing_linguistics = carry_ling
                if carry_fact is not None:
                    batch[0].trailing_fact_summary = carry_fact

            batch_out = await self._process_batch(batch, metadata)
            mapped_results.extend(batch_out)

            if batch_out:
                last_ling, last_fact = batch_out[-1]
                carry_ling = getattr(last_ling, 'next_state', None) if last_ling else carry_ling
                carry_fact = getattr(last_fact, 'next_state', None) if last_fact else carry_fact

        ling_results = [r[0] for r in mapped_results if r[0] is not None]
        fact_results = [r[1] for r in mapped_results if r[1] is not None]

        thematic_blocks, behavioral_profiles = await self.combine_engine.aggregate_dual_track(mapped_results, metadata)
        scorecard = self.combine_engine.compute_scorecard(ling_results, fact_results, slide_coverage)

        report = await self.hegemon.generate_report(
            metadata=metadata,
            thematic_blocks=thematic_blocks,
            behavioral_profiles=behavioral_profiles,
            presentation_context=presentation_context,
            scorecard=scorecard
        )
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
        report.map_timestamps = sorted(map_ts)
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
