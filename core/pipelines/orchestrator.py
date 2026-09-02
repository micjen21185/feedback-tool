import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
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
    ExperimentScenario, MapResult,
    ChunkPayload, SlideSummary, TimelinePayload, HegemonOutput
)
from tools.gatekeeper import KnowledgeGatekeeper
from tools.knowledge_engine import KnowledgeEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: SystemConfiguration, gateway, progress_cb=None):
        self.config = config
        self.gateway = gateway
        # Optional callback(str) for live step logging in the UI. No-op if not provided.
        self._progress = progress_cb or (lambda _msg: None)

    def _log(self, msg: str):
        # Mirror to the console/IDE log too: Streamlit's st.status often won't stream writes
        # live during a long blocking asyncio.run, so the terminal is where you actually see
        # progress in real time (and where a hang becomes visible).
        logger.info("[pipeline] %s", msg)
        self._progress(msg)

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

        self._log(f"▶️ Start scenariusza: {self.config.scenario.name}")
        if chunks:
            self._log(f"📦 Wczytano {len(chunks)} chunków do analizy (faza map).")

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
            map_timestamps=hegemon_out.map_timestamps,
            raw_reducer_response=hegemon_out.raw_reducer_response,
            reducer_input=hegemon_out.reducer_input,
            substantive_windows=hegemon_out.substantive_windows,
            total_windows=hegemon_out.total_windows
        )

    def _telemetry_snapshot(self, start_time: float) -> TelemetryReport:
        """Build a TelemetryReport from the gateway's current session telemetry, then reset it."""
        st = self.gateway.get_session_telemetry()
        report = TelemetryReport(
            phase_details=list(st),
            total_cost_usd=sum(t.cost_usd for t in st),
            total_tokens_in=sum(t.tokens_in for t in st),
            total_tokens_out=sum(t.tokens_out for t in st),
            map_phases_count=sum(1 for t in st if "Map" in t.agent_role),
            reduce_phases_count=sum(1 for t in st if "Reduce" in t.agent_role or "Hegemon" in t.agent_role),
            total_time_s=time.time() - start_time,
        )
        self.gateway.reset_session_telemetry()
        return report

    def execute_map_only(
            self,
            metadata: LectureMetadata,
            chunks: List[ChunkPayload] = None,
            timeline: Optional[TimelinePayload] = None,
            slide_summaries: Dict[str, SlideSummary] = None,
            knowledge_base_bytes: bytes = None,
            source_label: str = "",
            input_fingerprint: str = "",
    ) -> MapResult:
        """Run ONLY the swarm MAP + COMBINE phases and return a reusable MapResult. The expensive
        part (~60 agent calls) runs once; the reduce can then be run separately with any Hegemon
        model via execute_reduce_from_map(). Only valid for swarm scenarios (3/4/5)."""
        chunks = chunks or []
        slide_summaries = slide_summaries or {}
        start_time = time.time()
        self._log(f"▶️ MAP-only: {self.config.scenario.name} ({len(chunks)} chunków)")

        combine, presentation_context = asyncio.run(
            self._run_map_combine_for_scenario(metadata, chunks, timeline, slide_summaries, knowledge_base_bytes)
        )
        telemetry = self._telemetry_snapshot(start_time)
        sc = combine.get("scorecard")
        return MapResult(
            map_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(),
            scenario_name=self.config.scenario.name,
            input_fingerprint=input_fingerprint,
            source_label=source_label,
            factual_model=self.config.agent_models.factual_model,
            linguistic_model=self.config.agent_models.linguistic_model,
            utility_model=Config.UTILITY_MODEL,
            thematic_blocks=combine.get("thematic_blocks", []),
            behavioral_profiles=combine.get("behavioral_profiles", []),
            scorecard=sc,
            unverified_claims=combine.get("unverified_claims", []),
            map_timestamps=combine.get("map_timestamps", []),
            substantive_windows=combine.get("substantive_windows", 0),
            total_windows=combine.get("total_windows", 0),
            presentation_context=presentation_context,
            duration_sec=metadata.total_duration_sec,
            target_audience=metadata.target_audience,
            knowledge_level=metadata.knowledge_level,
            main_topic=metadata.main_topic,
            speaker_role=metadata.speaker_role,
            map_telemetry=telemetry,
        )

    def execute_reduce_from_map(self, map_result: MapResult) -> FinalReport:
        """Run ONLY the reduce, using a previously-saved MapResult as the evidence, with THIS
        orchestrator's configured Hegemon model. Lets you compare reducers on identical map
        evidence without re-running the map."""
        start_time = time.time()
        self._log(f"▶️ REDUCE-only na zapisanej fazie MAP ({map_result.map_id}) → Hegemon={self.config.hegemon_model}")

        # Rebuild the metadata the reducer needs from the saved MapResult.
        metadata = LectureMetadata(
            speaker_role=map_result.speaker_role, target_audience=map_result.target_audience,
            main_topic=map_result.main_topic, knowledge_level=map_result.knowledge_level or "Podstawowy",
            total_duration_sec=map_result.duration_sec,
        )
        pipeline, _ = self._build_swarm_pipeline(use_tools=False)
        combine = {
            "thematic_blocks": map_result.thematic_blocks,
            "behavioral_profiles": map_result.behavioral_profiles,
            "scorecard": map_result.scorecard,
            "unverified_claims": map_result.unverified_claims,
            "map_timestamps": map_result.map_timestamps,
            "total_windows": map_result.total_windows,
            "substantive_windows": map_result.substantive_windows,
        }
        hegemon_out = asyncio.run(
            pipeline.run_reduce(metadata, combine, presentation_context=map_result.presentation_context)
        )
        reduce_telemetry = self._telemetry_snapshot(start_time)

        # Cost the reduce ON TOP of the already-spent map cost (so the total reflects the full
        # pipeline even though the map ran earlier). Combine phase_details from both.
        map_tel = map_result.map_telemetry or TelemetryReport()
        telemetry = TelemetryReport(
            phase_details=list(map_tel.phase_details) + list(reduce_telemetry.phase_details),
            total_cost_usd=map_tel.total_cost_usd + reduce_telemetry.total_cost_usd,
            total_tokens_in=map_tel.total_tokens_in + reduce_telemetry.total_tokens_in,
            total_tokens_out=map_tel.total_tokens_out + reduce_telemetry.total_tokens_out,
            map_phases_count=map_tel.map_phases_count,
            reduce_phases_count=reduce_telemetry.reduce_phases_count,
            total_time_s=map_tel.total_time_s + reduce_telemetry.total_time_s,
        )
        return FinalReport(
            analysis=hegemon_out.analysis,
            feedback=hegemon_out.feedback,
            telemetry=telemetry,
            scorecard=hegemon_out.scorecard,
            map_timestamps=hegemon_out.map_timestamps,
            raw_reducer_response=hegemon_out.raw_reducer_response,
            reducer_input=hegemon_out.reducer_input,
            substantive_windows=hegemon_out.substantive_windows,
            total_windows=hegemon_out.total_windows,
        )

    async def _run_map_combine_for_scenario(self, metadata, chunks, timeline, slide_summaries, knowledge_base_bytes):
        """Shared: build the swarm pipeline for the current scenario, apply scenario-5 presentation
        wiring if needed, run map+combine, and return (combine_dict, presentation_context)."""
        use_tools = self.config.scenario in (
            ExperimentScenario.SWARM_NAIVE_RAG_WEB, ExperimentScenario.SWARM_PRESENTATION_RAG_WEB)
        pipeline, knowledge_engine = self._build_swarm_pipeline(use_tools=use_tools)
        if use_tools and knowledge_base_bytes:
            await knowledge_engine.build_knowledge_base(knowledge_base_bytes)

        presentation_context = ""
        slide_cov = None
        if self.config.scenario == ExperimentScenario.SWARM_PRESENTATION_RAG_WEB:
            self._apply_presentation_wiring(chunks, timeline, slide_summaries)
            coverage_engine = SlideCoverageEngine(self.gateway, self.config.hegemon_model)
            slide_cov = await coverage_engine.analyze_slide_coverage(chunks, slide_summaries)
            presentation_flow = coverage_engine.build_presentation_flow(slide_summaries)
            presentation_context = coverage_engine.format_for_reducer(slide_cov, presentation_flow)

        combine = await pipeline.run_map_combine(metadata, chunks, slide_coverage=slide_cov)
        return combine, presentation_context

    @staticmethod
    def _apply_presentation_wiring(chunks, timeline, slide_summaries):
        """Attach slide_id / OCR text to chunks from the timeline (mirrors scenario-5 setup)."""
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
            use_tools=use_tools,
            progress_cb=self._progress
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
