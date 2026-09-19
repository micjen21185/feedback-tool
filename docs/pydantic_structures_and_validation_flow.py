"""
================================================================================
CRITICAL PYDANTIC STRUCTURES & VALIDATION / RETRY WORKFLOW
================================================================================

This is a *documentation + runnable-example* module. It does NOT change any
runtime behaviour of the feedback-tool. It gathers, in one place:

  1. The critical Pydantic structures that hold the system together, grouped by
     the role they play in the Map -> Combine -> Reduce pipeline.
  2. Concrete, valid example instances of each (the "best of" the schema set).
  3. A written explanation of how data validation + retrying actually works,
     mirroring `core/llm_gateway.py`.
  4. A Mermaid flowchart (paste into https://mermaid.live) of that flow.

Everything below reflects the real definitions in `models/schemas.py`,
`core/reasoning/transient_schemas.py`, and `core/llm_gateway.py`.

Run it directly to print the example instances as validated JSON:

    python docs/pydantic_structures_and_validation_flow.py
"""

from __future__ import annotations

from core.reasoning.transient_schemas import CoTTransientOutput
from models.schemas import (
    # --- Enums: the controlled vocabularies validation depends on ---
    AnomalySeverity,
    VerificationStatus,
    ExperimentScenario,
    # --- Atom: the smallest validated finding ---
    SeverityItem,
    # --- MAP-phase agent outputs (per-chunk) ---
    LinguisticOutput,
    FactualOutput,
    # --- Carry-over state threaded across chunks/batches ---
    TrailingLinguisticState,
    TrailingFactualState,
    # --- Input payload the agents consume ---
    ChunkMeta,
    ChunkContextData,
    ChunkLinguisticData,
    ChunkTextData,
    ChunkPayload,
    # --- COMBINE-phase deterministic scoring ---
    ScoreCard,
    # --- REDUCE-phase report ---
    DeepAnalysis,
    ConstructiveFeedback,
    HegemonOutput,
    # --- Persistence / reuse boundary ---
    MapResult,
)


# =============================================================================
# SECTION 1 — WHY THESE ARE THE "CRITICAL" STRUCTURES
# =============================================================================
#
# The pipeline is Map -> Combine -> Reduce. Pydantic sits at three seams:
#
#   (A) LLM boundary   : every structured LLM call is validated into a Pydantic
#                        model by LLMGateway.execute_structured(). This is where
#                        untrusted model text becomes a typed object.
#   (B) Combine seam   : per-chunk MAP outputs (LinguisticOutput / FactualOutput)
#                        are aggregated deterministically into a ScoreCard and
#                        text blocks — no LLM, so no validation surprises.
#   (C) Persistence    : MapResult is the serializable "expensive part" that gets
#                        written to disk as JSON and re-validated on load
#                        (MapResult.model_validate_json in app.py).
#
# SeverityItem is the atom every finding reduces to; the two Enums
# (AnomalySeverity, VerificationStatus) are the controlled vocabularies that
# make validation meaningful (a model can't invent a severity level).


# =============================================================================
# SECTION 2 — BEST-OF EXAMPLE INSTANCES
# Each example is a valid instance; constructing it *is* the validation.
# =============================================================================

def example_severity_item() -> SeverityItem:
    """The atom. A finding + how bad it is + whether a trusted source backs it."""
    return SeverityItem(
        text="Prelegent podał, że RSA powstał w 1965 r. (faktycznie 1977).",
        severity=AnomalySeverity.HIGH,
        verification_status=VerificationStatus.CONTRADICTS_SOURCE,
    )


def example_linguistic_output() -> LinguisticOutput:
    """One MAP window, language track. Note scored_anomalies vs anomalies:
    the flat `anomalies` list is a fallback that gets backfilled from
    scored_anomalies in CombineEngine if a model only returned plain text."""
    return LinguisticOutput(
        chunk_id="chunk_003",
        start_time=120.0,
        scored_anomalies=[
            SeverityItem(text="Nadużywanie 'znaczy' (7x)", severity=AnomalySeverity.MEDIUM),
            SeverityItem(text="Długa pauza wypełniona 'yyy'", severity=AnomalySeverity.LOW),
        ],
        dominant_tendencies="wtrącenia wypełniające",
        next_state=TrailingLinguisticState(prev_filler_count=7, escalation_flag=False),
    )


def example_factual_output() -> FactualOutput:
    """One MAP window, factual track. verification_status drives scoring:
    SUPPORTED_BY_SOURCE is ignored, UNVERIFIED is a small flat penalty,
    CONTRADICTS_SOURCE is a full-weight error."""
    return FactualOutput(
        chunk_id="chunk_003",
        start_time=120.0,
        scored_errors=[
            SeverityItem(
                text="RSA powstał w 1965 r.",
                severity=AnomalySeverity.HIGH,
                verification_status=VerificationStatus.CONTRADICTS_SOURCE,
            ),
        ],
        thematic_summary="Wprowadzenie do kryptografii asymetrycznej.",
        next_state=TrailingFactualState(
            prev_summary="Omówiono podstawy szyfrowania.",
            open_loops=["Nie dokończono wątku o kluczach publicznych."],
        ),
    )


def example_chunk_payload() -> ChunkPayload:
    """The validated INPUT an agent receives for one window."""
    return ChunkPayload(
        chunk_meta=ChunkMeta(index=3, start_time=120.0, end_time=160.0, slide_id=4),
        context_data=ChunkContextData(
            pdf_text="RSA został opracowany w 1977 roku przez Rivesta, Shamira i Adlemana.",
            user_notes="Slajd o historii kryptografii.",
            auto_generated_summary="Historia i podstawy RSA.",
        ),
        linguistic_data=ChunkLinguisticData(
            chunk_wpm=132,
            filler_words_count=7,
            repeated_tendencies_count=2,
            significant_pauses_count=1,
            significant_pauses_duration_sec=3.5,
            unclear_words_count=0,
            avg_transcription_confidence=0.91,
            detected_fillers={"znaczy": 7},
            detected_tendencies={"wtrącenia": 2},
        ),
        text_data=ChunkTextData(
            clean_text="RSA został opracowany w tysiąc dziewięćset sześćdziesiątym piątym roku...",
            tagged_text="RSA został opracowany w <FILLER>znaczy</FILLER> 1965 roku...",
        ),
    )


def example_cot_transient_output() -> CoTTransientOutput:
    """A TRANSIENT schema used only during reasoning (chain-of-thought). It is
    the direct target of execute_structured() for the factual agent's CoT step —
    a good example of a schema whose sole purpose is to shape/validate an LLM
    response, then get mapped into the durable FactualOutput."""
    return CoTTransientOutput(
        thought_process=(
            "1) Prelegent twierdzi, że RSA powstał w 1965. "
            "2) Kontekst PDF podaje rok 1977. "
            "3) Konkluzja: sprzeczność ze źródłem, błąd HIGH."
        ),
        factual_errors=["RSA powstał w 1965 r."],
        scored_errors=[
            SeverityItem(
                text="RSA powstał w 1965 r.",
                severity=AnomalySeverity.HIGH,
                verification_status=VerificationStatus.CONTRADICTS_SOURCE,
            )
        ],
        thematic_summary="Błędna data powstania RSA.",
    )


def example_scorecard() -> ScoreCard:
    """COMBINE output. Computed deterministically by CombineEngine.compute_scorecard —
    NOT produced by an LLM, so it never needs the parse/retry chain."""
    return ScoreCard(
        factual_score=78.5,
        linguistic_score=84.0,
        slide_coverage_score=None,  # None unless the presentation scenario
        overall_score=80.7,
        readiness_verdict="Dobre — wymaga jedynie drobnych szlifów",
    )


def example_hegemon_output() -> HegemonOutput:
    """REDUCE output: the final report object the Hegemon produces (also the
    target schema of a structured call, hence subject to the full parse chain)."""
    return HegemonOutput(
        analysis=DeepAnalysis(
            factual_summary="Jeden istotny błąd merytoryczny (data RSA).",
            linguistic_summary="Tempo poprawne, sporo wypełniaczy w środkowej części.",
            missed_context=["Nie omówiono zastosowań RSA w podpisie cyfrowym."],
            unverified_claims=["90% firm w PL doświadczyło ransomware w 2025 (niepotwierdzone)."],
        ),
        feedback=ConstructiveFeedback(
            executive_summary_markdown="## Podsumowanie\nDobre wystąpienie z jednym błędem faktograficznym...",
            strengths=["Jasna struktura", "Dobre tempo"],
            areas_for_improvement=["Weryfikacja dat", "Redukcja wypełniaczy"],
            actionable_tips=["Sprawdź daty przed wystąpieniem.", "Ćwicz pauzy zamiast 'znaczy'."],
            overall_message="Solidnie, wymaga drobnych korekt.",
        ),
        scorecard=example_scorecard(),
        map_timestamps=[0.0, 120.0, 240.0],
        substantive_windows=11,
        total_windows=15,
    )


def example_map_result() -> MapResult:
    """The PERSISTENCE boundary: the serialized Combine output. Written to disk
    as JSON and re-validated on load. Feed into reduce with ANY Hegemon model."""
    sc = example_scorecard()
    return MapResult(
        map_id="a1b2c3d4e5f6",
        created_at="2026-09-20T08:15:42.123456+00:00",
        scenario_name=ExperimentScenario.SWARM_NAIVE_RAG_WEB.name,
        input_fingerprint="sha256:9f2c...b71a",
        source_label="wyklad_bezpieczenstwo_it.zip",
        factual_model="openai/gpt-4o",
        linguistic_model="qwen/qwen-2.5-14b",
        utility_model="openai/gpt-4o-mini",
        thematic_blocks=[
            "[120.0s] Temat: Kryptografia asymetryczna\n  -> (HIGH) RSA powstał w 1965 (faktycznie 1977).",
        ],
        behavioral_profiles=[
            "<ROZKŁAD WAGI ANOMALII>\nHIGH: 1 | MEDIUM: 6 | LOW: 11",
        ],
        scorecard=sc,
        unverified_claims=["[240.0s] 90% firm w PL doświadczyło ransomware w 2025."],
        map_timestamps=[0.0, 120.0, 240.0],
        substantive_windows=11,
        total_windows=15,
        map_trace=["[120.0s] FAKTY: errors=['RSA 1965'] status=['CONTRADICTS_SOURCE']"],
        duration_sec=1620.0,
        target_audience="Studenci informatyki",
        knowledge_level="Średniozaawansowany",
        main_topic="Podstawy bezpieczeństwa IT",
        speaker_role="Wykładowca akademicki",
    )


# =============================================================================
# SECTION 3 — HOW VALIDATION + RETRYING WORKS (mirrors core/llm_gateway.py)
# =============================================================================
#
# There are TWO independent retry layers. Do not conflate them.
#
# LAYER 1 — TRANSPORT retry (network/provider level)
# --------------------------------------------------
#   Implemented in LLMGateway._safe_acompletion via tenacity:
#     @retry(stop=stop_after_attempt(4),
#            wait=wait_exponential(multiplier=1, min=2, max=10),
#            retry=retry_if_exception(_should_retry), reraise=True)
#
#   _should_retry() retries ONLY:
#     - litellm.RateLimitError, litellm.APIConnectionError  (always)
#     - litellm.Timeout                                     (only if retry_on_timeout=True)
#   ...and NEVER retries a fatal local error (_is_fatal_local_error):
#     OOM / "signal killed" / "model not found" / "no such model" etc.
#     Rationale: the situation is identical next attempt, so 4 backoffs are wasted.
#
#   MODEL FALLBACK (in _execute_with_telemetry): if the call still fails AND it's
#   a *capacity* error (_is_capacity_error: rate limit / 429 / overloaded / quota /
#   503), the gateway switches to Config.FALLBACK_MODEL once and retries. Capacity
#   errors are distinguished from context-length / auth / local-OOM, which a
#   fallback model would NOT fix.
#
# LAYER 2 — PARSE / VALIDATION chain (schema level, NO re-calling the model)
# -------------------------------------------------------------------------
#   Implemented in LLMGateway.execute_structured. Once text is returned, it is
#   coerced into `schema_class` through a graceful-degradation chain:
#
#     1. Commercial models: first try native response_format=schema_class, then
#        schema_class.model_validate_json(content). On ValidationError -> fall
#        through to the generic chain below (do NOT raise).
#     2. Strict JSON:   schema_class.model_validate(json.loads(_extract_json_object(text)))
#                       (_extract_json_object strips ```json fences and isolates {...})
#     3. Repaired JSON: _repair_json() fixes smart quotes + trailing commas, isolates
#                       the outer {...}, retries json.loads. It only fixes SYNTAX —
#                       it never fabricates content.
#     4. XML fallback:  _parse_xml_fallback() reads <field>…</field> tags and coerces
#                       per declared type (List[str] / str / int / float / bool),
#                       defaulting missing fields. This is why the schema contract
#                       tells the model "if you can't do JSON, use XML tags".
#
#   EMPTY-PARSE DETECTION: _looks_empty() checks known content fields
#   (thematic_summary, factual_errors, scored_anomalies, ...). If tokens were spent
#   but every meaningful field is empty, that's a *parse miss*, not a clean speech —
#   it's logged and dumped to debug_raw/ instead of being silently scored as perfect.
#
#   Key design point: Layer 2 does NOT retry the LLM. It degrades the PARSER.
#   Only Layer 1 re-issues network calls. This keeps cost bounded (a garbled
#   response is salvaged locally rather than paid for again).
#
# WORKFLOW END-TO-END
# -------------------
#   MAP:     for each ChunkPayload -> agent builds prompt -> execute_structured
#            (Layer1 transport + Layer2 parse) -> LinguisticOutput / FactualOutput
#            with next_state carried into the next chunk/batch.
#   COMBINE: CombineEngine deterministically aggregates -> ScoreCard + text blocks
#            (no LLM, no validation risk). Backfills scored_* from flat lists.
#   REDUCE:  Hegemon consumes blocks -> execute_structured(HegemonOutput). On
#            failure, swarm_pipeline._build_fallback_report() returns a degraded
#            HegemonOutput from map findings so the run is never wasted.
#   PERSIST: MapResult (combine output) <-> JSON on disk, re-validated on load.


# =============================================================================
# SECTION 4 — MERMAID FLOWCHART (paste into https://mermaid.live)
# =============================================================================

MERMAID_VALIDATION_RETRY_FLOW = r"""
flowchart TD
    A[ChunkPayload validated input] --> B[Agent builds prompt<br/>+ schema contract]
    B --> C[execute_structured schema_class]

    %% ---------- LAYER 1: TRANSPORT RETRY ----------
    subgraph L1[LAYER 1 - transport retry _safe_acompletion / tenacity]
        C --> D[litellm.acompletion<br/>under semaphore]
        D --> E{Call raised<br/>an exception?}
        E -- No --> F[Response text]
        E -- Yes --> G{_is_fatal_local_error?<br/>OOM / killed / model not found}
        G -- Yes --> H[FAIL FAST - reraise<br/>no retry]
        G -- No --> I{_should_retry?<br/>RateLimit / APIConnection<br/>/ Timeout if enabled}
        I -- Yes --> J{attempts < 4?}
        J -- Yes --> K[exponential backoff<br/>2s..10s] --> D
        J -- No --> L{_is_capacity_error?<br/>429 / overloaded / quota / 503}
        I -- No --> L
        L -- Yes --> M[switch to FALLBACK_MODEL<br/>once, re-call] --> D
        L -- No --> H
    end

    %% ---------- LAYER 2: PARSE / VALIDATION CHAIN ----------
    subgraph L2[LAYER 2 - parse chain - no LLM re-call]
        F --> N{Commercial + response_format?}
        N -- Yes --> O[model_validate_json]
        O --> P{Valid?}
        P -- Yes --> Z[Typed Pydantic object]
        P -- No --> Q
        N -- No --> Q[Strict JSON:<br/>extract_json_object + json.loads<br/>+ model_validate]
        Q --> R{Valid?}
        R -- Yes --> Z
        R -- No --> S[Repair JSON:<br/>smart quotes, trailing commas,<br/>isolate braces]
        S --> T{Valid?}
        T -- Yes --> Z
        T -- No --> U[XML fallback:<br/>field tags per declared type,<br/>defaults for missing]
        U --> Z
    end

    Z --> V{_looks_empty?<br/>tokens spent but<br/>all content fields empty}
    V -- Yes --> W[WARN + dump debug_raw/<br/>flag parse miss]
    V -- No --> X[Return object to agent]
    W --> X

    %% ---------- PIPELINE CONTEXT ----------
    X --> Y[MAP outputs:<br/>LinguisticOutput / FactualOutput<br/>+ next_state carry-over]
    Y --> AA[COMBINE deterministic:<br/>ScoreCard + thematic_blocks<br/>+ behavioral_profiles]
    AA --> AB[persist MapResult JSON]
    AA --> AC[REDUCE: Hegemon -> HegemonOutput]
    AC --> AD{Reduce failed?}
    AD -- Yes --> AE[_build_fallback_report<br/>degraded HegemonOutput]
    AD -- No --> AF[Final report]
    AE --> AF
"""


# =============================================================================
# SECTION 5 — RUNNER: constructing = validating; print as JSON
# =============================================================================

def _print_all_examples() -> None:
    examples = {
        "SeverityItem": example_severity_item(),
        "LinguisticOutput": example_linguistic_output(),
        "FactualOutput": example_factual_output(),
        "ChunkPayload": example_chunk_payload(),
        "CoTTransientOutput": example_cot_transient_output(),
        "ScoreCard": example_scorecard(),
        "HegemonOutput": example_hegemon_output(),
        "MapResult": example_map_result(),
    }
    for name, obj in examples.items():
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        # model_dump_json re-runs serialization; a round-trip proves validity.
        print(obj.model_dump_json(indent=2))

    print(f"\n{'=' * 72}\nMERMAID FLOW (paste into https://mermaid.live)\n{'=' * 72}")
    print(MERMAID_VALIDATION_RETRY_FLOW)


if __name__ == "__main__":
    _print_all_examples()
