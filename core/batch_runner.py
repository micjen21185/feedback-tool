"""
Batch runner: execute several scenarios sequentially on the SAME input + models, producing a
RunResult per scenario. Long-blocking by design (run on a machine that tolerates long jobs).
Each run reuses the exact single-run construction so batch and single-run results are identical.
"""
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from core.config_loader import Config
from core.llm_gateway import LLMGateway
from core.pipelines.orchestrator import Orchestrator
from models.schemas import (
    AgentModelsConfig, ChunkPayload, ExperimentScenario, LectureMetadata, RunModels, RunResult,
    SlideSummary, SystemConfiguration, TimelinePayload,
)
from observabilty.metrics_engine import ObservabilityManager

# Scenarios that need RAG/web tools. Used to set use_tools per scenario.
_RAG_SCENARIOS = {ExperimentScenario.SWARM_NAIVE_RAG_WEB, ExperimentScenario.SWARM_PRESENTATION_RAG_WEB}


def scenarios_for_batch(has_presentation: bool) -> List[ExperimentScenario]:
    """1–4 by default; add scenario 5 only when presentation/slide data exists."""
    base = [
        ExperimentScenario.MONOLITH_NAKED,
        ExperimentScenario.MONOLITH_TWO_PHASE_FORMATTED,
        ExperimentScenario.SWARM_NAIVE_NO_RAG,
        ExperimentScenario.SWARM_NAIVE_RAG_WEB,
    ]
    if has_presentation:
        base.append(ExperimentScenario.SWARM_PRESENTATION_RAG_WEB)
    return base


def _build_metadata(md: dict, speaker_role: str, target_audience: str, main_topic: str,
                    knowledge_level: str) -> LectureMetadata:
    return LectureMetadata(
        speaker_role=speaker_role,
        target_audience=target_audience,
        main_topic=main_topic,
        knowledge_level=knowledge_level,
        has_knowledge_base_file=md.get("has_knowledge_base_file", False),
        total_duration_sec=md.get("total_duration_sec", 0.0),
        total_words=md.get("total_words", 0),
        fastest_chunk_wpm=md.get("fastest_chunk_wpm", 0),
        slowest_chunk_wpm=md.get("slowest_chunk_wpm", 0),
        total_filler_words=md.get("total_filler_words", 0),
        total_repeated_tendencies=md.get("total_repeated_tendencies", 0),
        total_significant_pauses=md.get("total_significant_pauses", 0),
        total_significant_pauses_duration_sec=md.get("total_significant_pauses_duration_sec", 0.0),
        total_unclear_words=md.get("total_unclear_words", 0),
        overall_transcription_confidence=md.get("overall_transcription_confidence", 0.0),
    )


def run_batch(
        scenarios: List[ExperimentScenario],
        zip_data: dict,
        speaker_role: str, target_audience: str, main_topic: str, knowledge_level: str,
        hegemon_model: str, factual_model: str, linguistic_model: str,
        use_llmlingua: bool,
        input_fingerprint: str,
        source_label: str,
        knowledge_base_bytes: Optional[bytes] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
) -> List[RunResult]:
    """Run each scenario in order; return a RunResult per scenario. A single scenario failure is
    captured in its report (via the pipeline's own fallbacks) and does not abort the batch."""
    progress = progress_cb or (lambda _m: None)
    md = zip_data["metadata"]
    metadata = _build_metadata(md, speaker_role, target_audience, main_topic, knowledge_level)

    parsed_chunks = [ChunkPayload(**c) for c in zip_data.get("chunks", [])]
    parsed_summaries = {k: SlideSummary(**v) for k, v in zip_data.get("slide_summaries", {}).items()}
    timeline = zip_data.get("timeline") or {}
    parsed_timeline = TimelinePayload(**timeline) if timeline else None

    results: List[RunResult] = []
    for i, scenario in enumerate(scenarios, start=1):
        progress(f"══════ [{i}/{len(scenarios)}] Scenariusz: {scenario.name} ══════")
        system_config = SystemConfiguration(
            scenario=scenario,
            hegemon_model=hegemon_model,
            agent_models=AgentModelsConfig(factual_model=factual_model, linguistic_model=linguistic_model),
            use_tools=scenario in _RAG_SCENARIOS,
            use_llmlingua=use_llmlingua,
        )
        gateway = LLMGateway(ObservabilityManager())
        orchestrator = Orchestrator(system_config, gateway=gateway, progress_cb=progress)
        report = orchestrator.execute_pipeline(
            metadata=metadata,
            raw_text=zip_data.get("raw_text", ""),
            formatted_text=zip_data.get("formatted_text", ""),
            chunks=parsed_chunks,
            timeline=parsed_timeline,
            slide_summaries=parsed_summaries,
            knowledge_base_bytes=knowledge_base_bytes,
        )
        results.append(RunResult(
            run_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(),
            scenario_name=scenario.name,
            models=RunModels(
                hegemon_model=hegemon_model, factual_model=factual_model,
                linguistic_model=linguistic_model, utility_model=Config.UTILITY_MODEL,
            ),
            use_tools=system_config.use_tools,
            use_llmlingua=use_llmlingua,
            input_fingerprint=input_fingerprint,
            source_label=source_label,
            duration_sec=metadata.total_duration_sec,
            total_words=metadata.total_words,
            main_topic=metadata.main_topic,
            target_audience=metadata.target_audience,
            knowledge_level=metadata.knowledge_level,
            report=report,
        ))
        progress(f"   ✅ Zapisano wynik scenariusza {scenario.name}.")
    return results
