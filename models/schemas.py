from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VerificationStatus(str, Enum):
    """Trust provenance for a factual claim, based on the source-trust hierarchy:
    RAG (PDF) + slide summaries are treated as ground truth we believe."""
    SUPPORTED_BY_SOURCE = "SUPPORTED_BY_SOURCE"  # matches RAG/slides → trusted, NOT an error
    CONTRADICTS_SOURCE = "CONTRADICTS_SOURCE"  # RAG/slides say otherwise → high-confidence error
    UNVERIFIED = "UNVERIFIED"  # checkable but not in any source → needs review
    NOT_APPLICABLE = "NOT_APPLICABLE"  # opinion/narration/trivial → not a factual claim


class SeverityItem(BaseModel):
    text: str = Field(description="Opis obserwacji (anomalii lub błędu).")
    severity: AnomalySeverity = Field(
        default=AnomalySeverity.MEDIUM,
        description="Waga obserwacji: LOW (drobiazg), MEDIUM (warte uwagi), HIGH (poważny problem), CRITICAL (rażący błąd, np. wulgaryzm, utrata wątku)."
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.NOT_APPLICABLE,
        description="Skąd wiadomo o tym fakcie: SUPPORTED_BY_SOURCE (zgodny z RAG/slajdami), CONTRADICTS_SOURCE (sprzeczny ze źródłem), UNVERIFIED (niepotwierdzony — wymaga uwagi), NOT_APPLICABLE (nie jest twardym faktem)."
    )


class ExperimentScenario(int, Enum):
    MONOLITH_NAKED = 1
    MONOLITH_TWO_PHASE_FORMATTED = 2
    SWARM_NAIVE_NO_RAG = 3
    SWARM_NAIVE_RAG_WEB = 4
    SWARM_PRESENTATION_RAG_WEB = 5


class LectureMetadata(BaseModel):
    speaker_role: str = Field(..., description="Rola prelegenta")
    target_audience: str = Field(..., description="Grupa docelowa")
    main_topic: str = Field(..., description="Główny temat wystąpienia")
    knowledge_level: str = Field(..., description="Poziom wiedzy słuchaczy")
    has_knowledge_base_file: bool = False
    total_duration_sec: float = 0.0
    total_words: int = 0
    fastest_chunk_wpm: int = 0
    slowest_chunk_wpm: int = 0
    total_filler_words: int = 0
    total_repeated_tendencies: int = 0
    total_significant_pauses: int = 0
    total_significant_pauses_duration_sec: float = 0.0
    total_unclear_words: int = 0
    overall_transcription_confidence: float = 0.0


class AgentModelsConfig(BaseModel):
    factual_model: str
    linguistic_model: str


class SystemConfiguration(BaseModel):
    scenario: ExperimentScenario
    hegemon_model: str
    agent_models: AgentModelsConfig
    use_tools: bool = False
    use_llmlingua: bool = False


class SlideCoverage(BaseModel):
    slide_id: int
    covered_points: List[str] = Field(default_factory=list, description="Punkty slajdu, które prelegent omówił.")
    missed_points: List[str] = Field(default_factory=list, description="Punkty slajdu, których prelegent nie poruszył.")
    returned_later: bool = Field(default=False,
                                 description="Czy prelegent wrócił do tego slajdu w późniejszym momencie.")
    completed_on_return: bool = Field(default=False,
                                      description="Czy powrót do slajdu uzupełnił brakujące wcześniej punkty.")
    time_on_slide_sec: float = Field(default=0.0,
                                     description="Łączny czas spędzony na slajdzie (z uwzględnieniem powrotów).")
    dwell_verdict: str = Field(default="",
                               description="Werdykt dot. czasu: OK / ZA_KRÓTKO / ZA_DŁUGO wraz z krótkim uzasadnieniem.")


class SlideFlowStat(BaseModel):
    slide_id: int
    time_on_slide_sec: float
    appearances: int = 1


class PresentationFlow(BaseModel):
    total_slides: int = 0
    slide_stats: List[SlideFlowStat] = Field(default_factory=list)
    very_short_slides: List[int] = Field(default_factory=list,
                                         description="ID slajdów pokazanych bardzo krótko (np. < 15s).")
    very_long_slides: List[int] = Field(default_factory=list,
                                        description="ID slajdów pokazanych bardzo długo względem treści.")
    flow_summary: str = Field(default="", description="Zwięzły opis przepływu prezentacji dla reduktora.")


class DeepAnalysis(BaseModel):
    factual_summary: str
    linguistic_summary: str
    missed_context: List[str] = Field(default_factory=list)
    # Claims the pipeline could NOT confirm against any trusted source (RAG/slides) and that need
    # human/judge attention before being trusted. NOT the same as "wrong" — just "unconfirmed".
    unverified_claims: List[str] = Field(
        default_factory=list,
        description="Twierdzenia niepotwierdzone przez źródła (RAG/slajdy) — wymagają uwagi prelegenta/sędziego."
    )
    slide_coverage: List[SlideCoverage] = Field(
        default_factory=list,
        description="Analiza pokrycia slajdów (tylko scenariusz prezentacyjny)."
    )
    presentation_flow: Optional[PresentationFlow] = Field(
        default=None,
        description="Ocena przepływu prezentacji: czas na slajdach, tempo (tylko scenariusz prezentacyjny)."
    )


class ConstructiveFeedback(BaseModel):
    executive_summary_markdown: str = Field(
        description="Obszerny, wieloakapitowy feedback mentorski napisany w Markdown. Ma zawierać pochwały, twardą krytykę opartą na korelacjach z osią czasu oraz wytyczne naprawcze."
    )
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    actionable_tips: List[str] = Field(description="Lista krótkich wskazówek (równoważniki zdań).")
    overall_message: str = Field(default="")


class PhaseTelemetry(BaseModel):
    agent_role: str = Field(description="Rola agenta")
    model_name: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    time_s: float
    ttft_ms: float = 0.0
    prompt_chars: int = 0
    response_chars: int = 0


class TelemetryReport(BaseModel):
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_time_s: float = 0.0
    phase_details: List[PhaseTelemetry] = Field(default_factory=list)
    map_phases_count: int = 0
    reduce_phases_count: int = 0


class ScoreCard(BaseModel):
    factual_score: Optional[float] = Field(default=None, description="Ocena merytoryczna 0-100 (None = nie dotyczy).")
    linguistic_score: Optional[float] = Field(default=None,
                                              description="Ocena językowa/dynamiki 0-100 (None = nie dotyczy).")
    slide_coverage_score: Optional[float] = Field(default=None,
                                                  description="Ocena pokrycia slajdów 0-100 (tylko prezentacja).")
    overall_score: float = Field(default=100.0, description="Łączna ocena 0-100 (ważona).")
    readiness_verdict: str = Field(default="",
                                   description="Werdykt gotowości, np. 'Gotowe', 'Wymaga pracy', 'Niegotowe do publicznego wystąpienia'.")


def readiness_verdict(score: float) -> str:
    if score >= 85:
        return "Gotowe / doskonałe wystąpienie"
    if score >= 70:
        return "Dobre — wymaga jedynie drobnych szlifów"
    if score >= 50:
        return "Wymaga pracy przed publicznym wystąpieniem"
    return "Niegotowe do publicznego wystąpienia — konieczna gruntowna poprawa"


class MultiDimensionScore(BaseModel):
    """5-dimension scoring used by the two-phase monolith (scenario 2), extracted via
    execute_structured so it survives models that ignore XML tags."""
    score_factual: float = Field(default=100.0, ge=0, le=100, description="Ocena merytoryki 0-100.")
    score_linguistic: float = Field(default=100.0, ge=0, le=100, description="Ocena języka/dykcji 0-100.")
    score_structure: float = Field(default=100.0, ge=0, le=100, description="Ocena struktury/logiki 0-100.")
    score_tempo: float = Field(default=100.0, ge=0, le=100, description="Ocena tempa mowy 0-100.")
    score_confidence: float = Field(default=100.0, ge=0, le=100, description="Ocena pewności/płynności 0-100.")


class HegemonOutput(BaseModel):
    analysis: DeepAnalysis
    feedback: ConstructiveFeedback
    scorecard: Optional[ScoreCard] = None
    map_timestamps: List[float] = Field(
        default_factory=list,
        description="Znaczniki czasu (start_time) fragmentów, które miały ustalenia w fazie map (tylko swarm)."
    )
    # Debug/observability: the raw text the reducer/monolith produced, and the aggregated
    # input that was fed into the reducer. Surfaced in the UI so a failed tag-extraction or
    # a weak reduce can be inspected instead of showing a blank report.
    raw_reducer_response: str = Field(default="", description="Surowa odpowiedź modelu reduktora/monolitu.")
    reducer_input: str = Field(default="", description="Zagregowane dane wejściowe podane do reduktora.")
    # Non-penalizing observability (Q2): how many map windows carried substantive/checkable content
    # (a non-empty thematic summary) vs. total. Lets you SEE factual density without baking a
    # "not enough facts" penalty into the score (which would wrongly assume all talks must be fact-dense).
    substantive_windows: int = Field(default=0, description="Liczba okien map z treścią merytoryczną.")
    total_windows: int = Field(default=0, description="Łączna liczba okien fazy map.")


class FinalReport(BaseModel):
    analysis: DeepAnalysis
    feedback: ConstructiveFeedback
    telemetry: TelemetryReport
    scorecard: Optional[ScoreCard] = None
    map_timestamps: List[float] = Field(default_factory=list)
    raw_reducer_response: str = Field(default="", description="Surowa odpowiedź modelu reduktora/monolitu.")
    reducer_input: str = Field(default="", description="Zagregowane dane wejściowe podane do reduktora.")
    substantive_windows: int = Field(default=0)
    total_windows: int = Field(default=0)


class ChunkMeta(BaseModel):
    index: int
    start_time: float
    end_time: float
    slide_id: Optional[int] = None
    is_return_to_slide: bool = False
    sub_chunk_index: int = 1


class ChunkContextData(BaseModel):
    pdf_text: Optional[str] = None
    user_notes: str = ""
    auto_generated_summary: str = ""


class ChunkLinguisticData(BaseModel):
    chunk_wpm: int
    filler_words_count: int
    repeated_tendencies_count: int
    significant_pauses_count: int
    significant_pauses_duration_sec: float
    unclear_words_count: int
    avg_transcription_confidence: float
    detected_fillers: Dict[str, int] = Field(default_factory=dict)
    detected_tendencies: Dict[str, int] = Field(default_factory=dict)


class ChunkTextData(BaseModel):
    clean_text: str
    tagged_text: str


class TrailingLinguisticState(BaseModel):
    prev_filler_count: int = 0
    escalation_flag: bool = False


class TrailingFactualState(BaseModel):
    prev_summary: str = ""
    open_loops: List[str] = Field(default_factory=list)


class ChunkPayload(BaseModel):
    chunk_meta: ChunkMeta
    context_data: ChunkContextData
    linguistic_data: ChunkLinguisticData
    text_data: ChunkTextData
    trailing_linguistics: Optional[TrailingLinguisticState] = None
    trailing_fact_summary: Optional[TrailingFactualState] = None


class LinguisticOutput(BaseModel):
    chunk_id: str = ""
    start_time: float = 0.0
    scored_anomalies: List[SeverityItem] = Field(
        default_factory=list,
        description="Wykryte anomalie językowe/akustyczne, każda z przypisaną wagą (severity): LOW / MEDIUM / HIGH / CRITICAL."
    )
    anomalies: List[str] = Field(
        default_factory=list,
        description="(Opcjonalne) Płaska lista anomalii. Jeśli pominięta, zostanie wywnioskowana ze scored_anomalies."
    )
    dominant_tendencies: str = ""
    next_state: Optional[TrailingLinguisticState] = None

    def anomaly_texts(self) -> List[str]:
        if self.scored_anomalies:
            return [it.text for it in self.scored_anomalies]
        return self.anomalies


class FactualOutput(BaseModel):
    chunk_id: str = ""
    start_time: float = 0.0
    scored_errors: List[SeverityItem] = Field(
        default_factory=list,
        description="Wykryte błędy merytoryczne, każdy z przypisaną wagą (severity): LOW / MEDIUM / HIGH / CRITICAL."
    )
    factual_errors: List[str] = Field(
        default_factory=list,
        description="(Opcjonalne) Płaska lista błędów. Jeśli pominięta, zostanie wywnioskowana ze scored_errors."
    )
    thematic_summary: str = ""
    next_state: Optional[TrailingFactualState] = None

    def error_texts(self) -> List[str]:
        if self.scored_errors:
            return [it.text for it in self.scored_errors]
        return self.factual_errors


class TimelineAppearance(BaseModel):
    start_time: float
    end_time: float
    is_return: bool


class ChunkMetadataInfo(BaseModel):
    chunk_file: str
    chunk_index: int
    sub_chunk_index: int
    duration_sec: float
    unclear_words_count: int
    avg_transcription_confidence: float
    auto_summary: str


class SlideLinguisticSummary(BaseModel):
    total_filler_words: int
    total_repeated_tendencies: int
    total_unclear_words: int
    avg_confidence: float


class SlideSummary(BaseModel):
    slide_id: int
    pdf_text: str
    timeline_appearances: List[TimelineAppearance] = Field(default_factory=list)
    chunks_metadata: List[ChunkMetadataInfo] = Field(default_factory=list)
    linguistic_summary: SlideLinguisticSummary


class GlobalTimelineEvent(BaseModel):
    slide_id: Optional[int] = None
    is_return: bool = False
    chunk_indices: List[int] = Field(default_factory=list)


class TimelinePayload(BaseModel):
    global_timeline: List[GlobalTimelineEvent] = Field(default_factory=list)


class JudgeRubric(BaseModel):
    actionability: int = Field(default=0, ge=0, le=10, description="Jak konkretne i wykonalne są rady (0-10).")
    specificity: int = Field(default=0, ge=0, le=10,
                             description="Jak szczegółowy i osadzony w transkrypcji jest feedback (0-10).")
    correctness: int = Field(default=0, ge=0, le=10, description="Trafność krytyki merytorycznej (0-10).")
    tone: int = Field(default=0, ge=0, le=10, description="Adekwatność tonu do jakości wystąpienia (0-10).")
    groundedness: int = Field(default=0, ge=0, le=10,
                              description="Osadzenie ocen w faktach/danych, brak halucynacji (0-10).")
    justification: str = Field(default="", description="Krótkie uzasadnienie ocen.")


class ScenarioEvaluation(BaseModel):
    scenario_name: str
    rubric: JudgeRubric
    rubric_total: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    total_time_s: float = 0.0
    input_token_density: float = 0.0
    output_token_density: float = 0.0
    # Coverage / lost-in-the-middle metrics (by time-region: e.g. [start, middle, end]).
    positional_recall: List[float] = Field(
        default_factory=list,
        description="Udział cytowanych znaczników czasu w kolejnych regionach czasowych (0-1)."
    )
    reduce_fidelity: List[float] = Field(
        default_factory=list,
        description="Swarm: jaka część regionów z ustaleniami w fazie map przetrwała do raportu (0-1)."
    )
    lost_in_middle_flag: bool = Field(
        default=False,
        description="True, jeśli środkowy region ma wyraźnie niższe pokrycie niż skrajne."
    )
    # Transparency (Option 2): the exact grounding evidence the judge was shown — the multi-region
    # excerpt + timestamp probes + pipeline anchor. Lets the user verify WHY a groundedness score
    # was given, instead of trusting the number blindly.
    judge_evidence: str = Field(default="", description="Materiał dowodowy przekazany sędziemu (do wglądu).")


class PairwisePreference(BaseModel):
    winner: str = Field(default="TIE", description="Nazwa zwycięskiego scenariusza lub 'TIE'.")
    reason: str = Field(default="", description="Uzasadnienie preferencji.")


class EvaluationReport(BaseModel):
    per_scenario: List[ScenarioEvaluation] = Field(default_factory=list)
    pairwise: List[PairwisePreference] = Field(default_factory=list)
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0
    summary: str = ""


class RunModels(BaseModel):
    hegemon_model: str = ""
    factual_model: str = ""
    linguistic_model: str = ""
    utility_model: str = ""


class RunResult(BaseModel):
    """One complete pipeline execution — the unit of storage/comparison/export. Keyed by a
    composite identity so the SAME scenario run with DIFFERENT models is stored separately."""
    run_id: str = Field(description="Unikalny identyfikator uruchomienia.")
    created_at: str = Field(description="Znacznik czasu utworzenia (ISO).")
    scenario_name: str
    models: RunModels
    use_tools: bool = False
    use_llmlingua: bool = False
    # Input identity + human-readable file metadata for the export/comparison header.
    input_fingerprint: str = ""
    source_label: str = Field(default="", description="Nazwa/etykieta pliku wejściowego.")
    duration_sec: float = 0.0
    total_words: int = 0
    main_topic: str = ""
    target_audience: str = ""
    knowledge_level: str = ""
    report: FinalReport

    def display_label(self) -> str:
        # Compact label distinguishing same-scenario/different-model runs in selectors.
        t = self.created_at[11:16] if len(self.created_at) >= 16 else self.created_at
        return f"{self.scenario_name} · H={self.models.hegemon_model.split('/')[-1]} · {t}"


class BatchExport(BaseModel):
    """Top-level downloadable artifact: all runs of a batch + the judge comparison over them."""
    created_at: str
    source_label: str = ""
    runs: List[RunResult] = Field(default_factory=list)
    evaluation: Optional[EvaluationReport] = None
