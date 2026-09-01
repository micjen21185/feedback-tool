"""
Offline smoke test: runs every scenario over the sample ZIPs in smoke_tests/ using a
MOCK gateway (no external LLM calls), then runs one evaluation comparison. Verifies the
whole pipeline plumbing — ZIP parse, map, combine, reduce, slide coverage, scoring,
and the evaluation engine — does not crash and produces structured, non-empty output.

Run:  .venv\\Scripts\\python.exe -m smoke_tests.run_smoke
Flip to a live model by passing a real LLMGateway instead of MockGateway.
"""
import asyncio
import json
import zipfile
from pathlib import Path
from typing import Dict

from pydantic import BaseModel

from core.pipelines.evaluation_engine import EvaluationEngine
from core.pipelines.orchestrator import Orchestrator
from models.schemas import (
    ExperimentScenario, LectureMetadata, SystemConfiguration, AgentModelsConfig,
    ChunkPayload, SlideSummary, TimelinePayload, FinalReport
)

SMOKE_DIR = Path(__file__).parent


class MockGateway:
    """Deterministic stand-in for LLMGateway — returns valid structured/tag output, no I/O."""

    def __init__(self):
        self.session_telemetry = []

    def get_session_telemetry(self):
        return self.session_telemetry

    def reset_session_telemetry(self):
        self.session_telemetry = []

    async def execute_raw(self, prompt: str, model: str, agent_role: str = "", **kwargs) -> str:
        # Gatekeeper expects TAK/NIE.
        if "Knowledge Gatekeeper" in agent_role:
            return "NIE"
        if "GoT - Generator" in agent_role:
            return "Hipoteza: prelegent mówił o architekturze systemu."
        # Tag-based reports (Hegemon / monolith) — include [MM:SS] markers across the talk.
        return (
            "<factual_summary>[00:10] Wstęp ok. [01:00] Środek: drobne uproszczenie. "
            "[02:20] Zakończenie zebrane.</factual_summary>"
            "<linguistic_summary>[00:30] Tempo dobre. [01:30] pauza.</linguistic_summary>"
            "<missed_context>[01:10] pominięto koszty</missed_context>"
            "<executive_summary>**Dobre wystąpienie.** [00:10] mocny start, [01:00] środek do dopracowania, "
            "[02:20] solidne zamknięcie.</executive_summary>"
            "<strengths>- [00:10] jasny wstęp\n- [02:20] dobre podsumowanie</strengths>"
            "<areas_for_improvement>- [01:00] rozwiń środek</areas_for_improvement>"
            "<actionable_tips>- [01:30] rób krótsze pauzy</actionable_tips>"
            "<overall_message>Solidnie, z potencjałem.</overall_message>"
            "<score_factual>80</score_factual><score_linguistic>75</score_linguistic>"
            "<score_structure>70</score_structure><score_tempo>65</score_tempo>"
            "<score_confidence>85</score_confidence>"
            "<score_overall>78</score_overall>"
        )

    async def execute_structured(self, prompt: str, schema_class, model: str,
                                 agent_role: str = "", **kwargs) -> BaseModel:
        # Build a minimal valid instance of whatever schema is requested.
        from models.schemas import (
            LinguisticOutput, FactualOutput, SeverityItem, AnomalySeverity,
            SlideCoverage, JudgeRubric, PairwisePreference
        )
        from core.reasoning.strategies.got_strategy import ThoughtScore, ConvergenceOutput
        from core.reasoning.transient_schemas import CoTTransientOutput

        if schema_class is LinguisticOutput:
            return LinguisticOutput(scored_anomalies=[SeverityItem(text="pauza", severity=AnomalySeverity.MEDIUM)],
                                    dominant_tendencies="tempo")
        if schema_class is FactualOutput:
            return FactualOutput(scored_errors=[SeverityItem(text="uproszczenie", severity=AnomalySeverity.LOW)],
                                 thematic_summary="architektura")
        if schema_class is ThoughtScore:
            return ThoughtScore(is_plausible=True, score=8)
        if schema_class is ConvergenceOutput:
            return ConvergenceOutput(factual_errors=["uproszczenie"],
                                     scored_errors=[SeverityItem(text="uproszczenie", severity=AnomalySeverity.LOW)],
                                     thematic_summary="architektura")
        if schema_class is CoTTransientOutput:
            return CoTTransientOutput(thought_process="analiza", factual_errors=["uproszczenie"],
                                      scored_errors=[SeverityItem(text="uproszczenie", severity=AnomalySeverity.LOW)],
                                      thematic_summary="architektura")
        if schema_class is SlideCoverage:
            return SlideCoverage(slide_id=0, covered_points=["punkt A"], missed_points=["punkt B"])
        if schema_class is JudgeRubric:
            return JudgeRubric(actionability=7, specificity=6, correctness=8, tone=7, groundedness=7,
                               justification="ok")
        if schema_class is PairwisePreference:
            return PairwisePreference(winner="TIE", reason="porównywalne")
        # Fallback: empty instance.
        return schema_class()


def load_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()

        def find(suffix):
            return next((n for n in names if n.endswith(suffix)), None)

        md = json.loads(z.read(find("metadata.json")))
        raw = z.read(find("full_raw_text.txt")).decode("utf-8") if find("full_raw_text.txt") else ""
        fmt = z.read(find("full_formatted_text.txt")).decode("utf-8") if find("full_formatted_text.txt") else ""
        tl = json.loads(z.read(find("timeline.json"))) if find("timeline.json") else {}
        chunks, summaries = [], {}
        for n in names:
            if n.endswith(".json"):
                fname = n.split("/")[-1]
                if "chunk_" in fname:
                    chunks.append(json.loads(z.read(n)))
                elif "slide_summary" in fname:
                    folder = n.split("/")[-2] if "/" in n else "global"
                    summaries[folder] = json.loads(z.read(n))
        chunks.sort(key=lambda x: x.get("chunk_meta", {}).get("start_time", 0.0))
        return {"md": md, "raw": raw, "fmt": fmt, "timeline": tl, "chunks": chunks, "summaries": summaries}


def build_metadata(md: dict) -> LectureMetadata:
    return LectureMetadata(
        speaker_role=md.get("speaker_role", ""), target_audience=md.get("target_audience", ""),
        main_topic=md.get("main_topic", ""), knowledge_level=md.get("knowledge_level", "Podstawowy"),
        has_knowledge_base_file=md.get("has_knowledge_base_file", False),
        total_duration_sec=md.get("total_duration_sec", 0.0), total_words=md.get("total_words", 0),
        fastest_chunk_wpm=md.get("fastest_chunk_wpm", 0), slowest_chunk_wpm=md.get("slowest_chunk_wpm", 0),
        total_filler_words=md.get("total_filler_words", 0),
        total_repeated_tendencies=md.get("total_repeated_tendencies", 0),
        total_significant_pauses=md.get("total_significant_pauses", 0),
        total_significant_pauses_duration_sec=md.get("total_significant_pauses_duration_sec", 0.0),
        total_unclear_words=md.get("total_unclear_words", 0),
        overall_transcription_confidence=md.get("overall_transcription_confidence", 0.0),
    )


def run_scenario(data: dict, scenario: ExperimentScenario) -> FinalReport:
    metadata = build_metadata(data["md"])
    config = SystemConfiguration(
        scenario=scenario, hegemon_model="mock",
        agent_models=AgentModelsConfig(factual_model="mock", linguistic_model="mock"),
    )
    orch = Orchestrator(config, gateway=MockGateway())
    chunks = [ChunkPayload(**c) for c in data["chunks"]]
    summaries = {k: SlideSummary(**v) for k, v in data["summaries"].items()}
    timeline = TimelinePayload(**data["timeline"]) if data["timeline"] else None
    return orch.execute_pipeline(
        metadata=metadata, raw_text=data["raw"], formatted_text=data["fmt"],
        chunks=chunks, timeline=timeline, slide_summaries=summaries,
    )


def assert_report(report: FinalReport, label: str):
    assert report.feedback.executive_summary_markdown, f"{label}: empty essay"
    assert report.analysis.factual_summary, f"{label}: empty factual summary"
    assert report.scorecard is not None, f"{label}: no scorecard"
    print(f"  {label}: overall={report.scorecard.overall_score} "
          f"strengths={len(report.feedback.strengths)} "
          f"slides={len(report.analysis.slide_coverage)} "
          f"map_ts={len(report.map_timestamps)}  OK")


def main():
    zips = sorted(SMOKE_DIR.glob("*.zip"))
    assert zips, "No smoke_tests/*.zip found"

    # Presentation ZIP (has Slide_ folders) → scenario 5; others → scenarios 1-4.
    pres_zip = next((z for z in zips if _has_slides(z)), zips[0])
    flat_zip = next((z for z in zips if not _has_slides(z)), zips[0])

    print("=== Scenarios 1-4 on flat ZIP ===")
    data = load_zip(flat_zip)
    for sc in [ExperimentScenario.MONOLITH_NAKED, ExperimentScenario.MONOLITH_TWO_PHASE_FORMATTED,
               ExperimentScenario.SWARM_NAIVE_NO_RAG, ExperimentScenario.SWARM_NAIVE_RAG_WEB]:
        assert_report(run_scenario(data, sc), sc.name)

    print("=== Scenario 5 on presentation ZIP (18 slides, chunks for 3) ===")
    pres = load_zip(pres_zip)
    rep5 = run_scenario(pres, ExperimentScenario.SWARM_PRESENTATION_RAG_WEB)
    assert_report(rep5, "SWARM_PRESENTATION_RAG_WEB")

    print("=== Evaluation over two scenarios ===")
    reports: Dict[str, FinalReport] = {
        "MONOLITH_NAKED": run_scenario(data, ExperimentScenario.MONOLITH_NAKED),
        "SWARM_NAIVE_NO_RAG": run_scenario(data, ExperimentScenario.SWARM_NAIVE_NO_RAG),
    }
    engine = EvaluationEngine(MockGateway(), judge_model="mock")
    eval_report = asyncio.run(engine.evaluate(data["raw"][:2000], reports,
                                              duration_sec=data["md"].get("total_duration_sec", 0.0)))
    assert len(eval_report.per_scenario) == 2, "eval missing scenarios"
    for se in eval_report.per_scenario:
        print(f"  {se.scenario_name}: quality={se.rubric_total}/50 "
              f"density_in={se.input_token_density} recall={se.positional_recall} "
              f"reduce_fidelity={se.reduce_fidelity} lim={se.lost_in_middle_flag}")
    print(f"  pairwise: {[p.winner for p in eval_report.pairwise]}")

    print("\nALL_SMOKE_TESTS_PASSED")


def _has_slides(zip_path: Path) -> bool:
    with zipfile.ZipFile(zip_path) as z:
        return any("Slide_" in n for n in z.namelist())


if __name__ == "__main__":
    main()
