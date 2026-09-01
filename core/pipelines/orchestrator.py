import asyncio
import time
from typing import List, Dict, Optional

from core.agents.factual_agent import FactualAgent
from core.agents.hegemon_reducer import HegemonReducer
from core.agents.linguistic_agent import LinguisticAgent
from core.config_loader import Config
from core.pipelines.combine_engine import CombineEngine
from core.pipelines.monolith_pipelines import MonolithNakedPipeline, MonolithTwoPhasePipeline
from core.pipelines.slide_coverage_engine import SlideCoverageEngine
from core.pipelines.swarm_pipeline import SwarmNaivePipeline
from core.reasoning.reasoning_engine import ReasoningEngine
from models.schemas import (
    SystemConfiguration, LectureMetadata, FinalReport, TelemetryReport,
    ExperimentScenario,
    ChunkPayload, SlideSummary, TimelinePayload, HegemonOutput
)
from tools.gatekeeper import KnowledgeGatekeeper
from tools.knowledge_engine import KnowledgeEngine


class Orchestrator:
    def __init__(self, config: SystemConfiguration, gateway):
        self.config = config
        self.gateway = gateway

    def execute_pipeline(
            self,
            metadata: LectureMetadata,
            raw_text: str = "",
            formatted_text: str = "",
            chunks: List[ChunkPayload] = None,
            timeline: Optional[TimelinePayload] = None,
            slide_summaries: Dict[str, SlideSummary] = None,
            knowledge_base_bytes: bytes = None
    ) -> FinalReport:

        chunks = chunks or []
        slide_summaries = slide_summaries or {}
        start_time = time.time()

        if self.config.scenario == ExperimentScenario.MONOLITH_NAKED:
            hegemon_out = asyncio.run(self._execute_scenario_1_monolith_naked(metadata, raw_text))

        elif self.config.scenario == ExperimentScenario.MONOLITH_TWO_PHASE_FORMATTED:
            hegemon_out = asyncio.run(self._execute_scenario_2_monolith_two_phase_formatted(metadata, formatted_text))

        elif self.config.scenario == ExperimentScenario.SWARM_NAIVE_NO_RAG:
            hegemon_out = asyncio.run(self._execute_scenario_3_swarm_naive_no_rag(metadata, chunks))

        elif self.config.scenario == ExperimentScenario.SWARM_NAIVE_RAG_WEB:
            hegemon_out = asyncio.run(
                self._execute_scenario_4_swarm_naive_rag_web(metadata, chunks, knowledge_base_bytes))

        elif self.config.scenario == ExperimentScenario.SWARM_PRESENTATION_RAG_WEB:
            hegemon_out = asyncio.run(self._execute_scenario_5_swarm_presentation_rag_web(
                metadata, chunks, timeline, slide_summaries, knowledge_base_bytes
            ))
        else:
            raise ValueError("Nieznany scenariusz eksperymentu!")

        session_telemetry = self.gateway.get_session_telemetry()

        telemetry = TelemetryReport(
            phase_details=session_telemetry,
            total_cost_usd=sum(t.cost_usd for t in session_telemetry),
            total_tokens_in=sum(t.tokens_in for t in session_telemetry),
            total_tokens_out=sum(t.tokens_out for t in session_telemetry),
            map_phases_count=sum(1 for t in session_telemetry if "Map" in t.agent_role),
            reduce_phases_count=sum(
                1 for t in session_telemetry if "Reduce" in t.agent_role or "Hegemon" in t.agent_role),
            total_time_s=time.time() - start_time
        )

        self.gateway.reset_session_telemetry()

        return FinalReport(
            analysis=hegemon_out.analysis,
            feedback=hegemon_out.feedback,
            telemetry=telemetry,
            scorecard=hegemon_out.scorecard,
            map_timestamps=hegemon_out.map_timestamps
        )

    async def _execute_scenario_1_monolith_naked(self, metadata, raw_text) -> HegemonOutput:
        pipeline = MonolithNakedPipeline(self.gateway, self.config)
        return await pipeline.execute(metadata, raw_text)

    async def _execute_scenario_2_monolith_two_phase_formatted(self, metadata, formatted_text) -> HegemonOutput:
        pipeline = MonolithTwoPhasePipeline(self.gateway, self.config)
        return await pipeline.execute(metadata, formatted_text)

    def _build_swarm_pipeline(self, use_tools: bool = False):
        reasoning_engine = ReasoningEngine(self.gateway)
        gatekeeper = KnowledgeGatekeeper(self.gateway, lightweight_model=Config.UTILITY_MODEL)
        knowledge_engine = KnowledgeEngine(self.gateway, model_name=Config.UTILITY_MODEL)

        factual_config = {
            "factual_model": self.config.agent_models.factual_model
        }

        factual_agent = FactualAgent(
            reasoning_engine=reasoning_engine,
            gatekeeper=gatekeeper,
            knowledge_engine=knowledge_engine,
            gateway=self.gateway,
            config=factual_config
        )

        linguistic_agent = LinguisticAgent(
            gateway=self.gateway,
            config_model=self.config.agent_models.linguistic_model
        )

        combine_engine = CombineEngine(self.gateway, self.config)
        hegemon_reducer = HegemonReducer(self.gateway, self.config.hegemon_model)

        return SwarmNaivePipeline(
            linguistic_agent=linguistic_agent,
            factual_agent=factual_agent,
            combine_engine=combine_engine,
            hegemon=hegemon_reducer,
            config=self.config,
            use_tools=use_tools
        ), knowledge_engine

    async def _execute_scenario_3_swarm_naive_no_rag(self, metadata: LectureMetadata,
                                                     chunks: List[ChunkPayload]) -> HegemonOutput:
        pipeline, _ = self._build_swarm_pipeline(use_tools=False)
        return await pipeline.execute(metadata, chunks)

    async def _execute_scenario_4_swarm_naive_rag_web(self, metadata: LectureMetadata, chunks: List[ChunkPayload],
                                                      knowledge_base_bytes: bytes = None) -> HegemonOutput:
        pipeline, knowledge_engine = self._build_swarm_pipeline(use_tools=True)

        if knowledge_base_bytes:
            await knowledge_engine.build_knowledge_base(knowledge_base_bytes)

        return await pipeline.execute(metadata, chunks)

    async def _execute_scenario_5_swarm_presentation_rag_web(
            self, metadata: LectureMetadata, chunks: List[ChunkPayload], timeline: Optional[TimelinePayload],
            slide_summaries: Dict[str, SlideSummary], knowledge_base_bytes: bytes = None
    ) -> HegemonOutput:
        pipeline, knowledge_engine = self._build_swarm_pipeline(use_tools=True)

        if knowledge_base_bytes:
            await knowledge_engine.build_knowledge_base(knowledge_base_bytes)

        if timeline and timeline.global_timeline:
            for event in timeline.global_timeline:
                slide_id = event.slide_id
                slide_text = "Brak danych OCR dla tego slajdu."
                if slide_id is not None:
                    folder_key = f"Slide_{slide_id:02d}"
                    if folder_key in slide_summaries:
                        slide_text = slide_summaries[folder_key].pdf_text

                for chunk_idx in event.chunk_indices:
                    for chunk in chunks:
                        if chunk.chunk_meta.index == chunk_idx:
                            chunk.chunk_meta.slide_id = slide_id
                            chunk.chunk_meta.is_return_to_slide = event.is_return
                            chunk.context_data.pdf_text = slide_text
                            break

        # Presentation-specific reduce: compute slide coverage + flow BEFORE the Hegemon,
        # so the Hegemon can comment on them; then attach the structured results to the report.
        coverage_engine = SlideCoverageEngine(self.gateway, self.config.hegemon_model)
        slide_coverage = await coverage_engine.analyze_slide_coverage(chunks, slide_summaries)
        presentation_flow = coverage_engine.build_presentation_flow(slide_summaries)
        presentation_context = coverage_engine.format_for_reducer(slide_coverage, presentation_flow)

        hegemon_out = await pipeline.execute(
            metadata, chunks,
            presentation_context=presentation_context,
            slide_coverage=slide_coverage
        )

        hegemon_out.analysis.slide_coverage = slide_coverage
        hegemon_out.analysis.presentation_flow = presentation_flow

        return hegemon_out
