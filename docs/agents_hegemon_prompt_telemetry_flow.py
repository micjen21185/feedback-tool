# ==============================================================================
# AGENTS & HEGEMON — PROMPT / TELEMETRY FLOW
# ==============================================================================
#
# Purpose
# -------
# A structured, human-readable map of the three LLM roles in the pipeline
# (Linguistic Agent, Factual Agent, Hegemon Reducer): what each prompt is asked
# to produce, which telemetry / quantitative data each one *reads*, and — the key
# question — which output fields the LLM authors vs. which are assigned
# deterministically by Python AFTER the call.
#
# This mirrors the real code:
#   - core/agents/linguistic_agent.py
#   - core/agents/factual_agent.py            (delegates to reasoning strategies)
#   - core/reasoning/strategies/cot_strategy.py (+ got / zero_shot)
#   - core/agents/hegemon_reducer.py
#   - core/pipelines/combine_engine.py         (deterministic aggregation)
#   - observabilty/metrics_engine.py           (telemetry sink)
#
# Legend used throughout:
#   [LLM]    = value produced by the model inside its response
#   [PY]     = value assigned/overwritten by Python (NOT decided by the model)
#   [PY->LLM] = deterministic value Python computes and INJECTS into the prompt
#              (the model reads it but does not compute it)
#
# ------------------------------------------------------------------------------
# There are TWO distinct notions of "telemetry" in this system — keep them apart:
#
#   1. LECTURE / SIGNAL METRICS  (WPM, fillers, pauses, transcription confidence)
#      Computed deterministically upstream, stored on ChunkLinguisticData /
#      LectureMetadata, and INJECTED into prompts as "<DANE ILOŚCIOWE>".
#      The agents are told: "opieraj oceny na tych danych, nie zgaduj".
#
#   2. RUNTIME / COST TELEMETRY  (tokens_in/out, time_s, cost_usd, cps, CPT)
#      Produced by the LLMGateway + ObservabilityManager AROUND every call.
#      The models NEVER see or produce these — they are pure Python instrumentation
#      (PhaseTelemetry rows + the SQLite benchmark_logs table).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ==============================================================================
# SECTION 1 — DATA MODEL FOR THIS FLOW FILE
# (small dataclasses so the summary is queryable, not just prose)
# ==============================================================================

@dataclass
class PromptRole:
    name: str  # human name of the role
    call_site: str  # where it lives in the codebase
    phase: str  # MAP / REDUCE
    gateway_method: str  # execute_structured | execute_raw
    output_format: str  # JSON schema | XML tags
    prompt_summary: str  # what the prompt asks for, condensed
    injected_metrics: List[str]  # [PY→LLM] quantitative data fed IN
    llm_authored_fields: List[str]  # [LLM] fields the model fills
    python_assigned_fields: List[str]  # [PY] fields set after the call
    runtime_telemetry: List[str]  # [PY] cost/latency instrumentation captured
    notes: str = ""


# ==============================================================================
# SECTION 2 — ROLE 1: LINGUISTIC AGENT  (MAP phase)
# core/agents/linguistic_agent.py
# ==============================================================================

LINGUISTIC_AGENT = PromptRole(
    name="Linguistic Agent",
    call_site="core/agents/linguistic_agent.py :: LinguisticAgent.analyze",
    phase="MAP (per-chunk)",
    gateway_method="execute_structured(schema_class=LinguisticOutput)",
    output_format="JSON schema (LinguisticOutput)",
    prompt_summary=(
        "System prompt casts the model as a strict speech/linguistics expert, "
        "parameterised by speaker_role + target_audience, with an escalation rule "
        "toggled by a Python-computed escalation_flag. Enforces an anti-hallucination "
        "contract (only report anomalies literally present in the tagged text or "
        "implied by the quantitative window data; empty lists are a valid answer) and "
        "a few-shot acoustic pattern library. User prompt supplies the quantitative "
        "window block + the acoustic-tagged transcript and asks for anomalies, "
        "severity-scored anomalies, and a one-sentence dominant tendency."
    ),
    injected_metrics=[
        "[PY->LLM] chunk_wpm",
        "[PY->LLM] filler_words_count + per-filler distribution (detected_fillers)",
        "[PY->LLM] repeated_tendencies_count + distribution (detected_tendencies)",
        "[PY->LLM] significant_pauses_count + significant_pauses_duration_sec",
        "[PY->LLM] unclear_words_count",
        "[PY->LLM] avg_transcription_confidence",
        "[PY->LLM] tagged_text (acoustic-tagged transcript)",
        "[PY->LLM] speaker_role, target_audience (demographic framing)",
        "[PY->LLM] escalation_flag (from previous chunk's next_state)",
    ],
    llm_authored_fields=[
        "[LLM] scored_anomalies (SeverityItem list, with severity)",
        "[LLM] anomalies (flat text list)",
        "[LLM] dominant_tendencies (one sentence)",
    ],
    python_assigned_fields=[
        "[PY] chunk_id            = f'chunk_{chunk_meta.index}'  (overwritten after call)",
        "[PY] start_time          = chunk_meta.start_time        (overwritten after call)",
        "[PY] next_state          = TrailingLinguisticState(...) built by Python",
        "[PY]   next_state.prev_filler_count = ld.filler_words_count (metric, not LLM)",
        "[PY]   next_state.escalation_flag   = _should_escalate(...) (hard metrics OR "
        "the model's HIGH/CRITICAL count OR >2 anomalies)",
    ],
    runtime_telemetry=[
        "[PY] tokens_in, tokens_out (from provider usage)",
        "[PY] time_s (wall clock around the call)",
        "[PY] cost_usd (cloud: token pricing | local: time-based TCO)",
        "[PY] prompt_chars, response_chars",
        "[PY] cps, input_cpt, output_cpt (derived in ObservabilityManager.log_task)",
        "[PY] agent_role label = 'Linguistic Agent (Map Phase)'",
    ],
    notes=(
        "Escalation is a hybrid: the DECISION to be strict next window is Python "
        "(_should_escalate), but it is fed back INTO the next prompt as text, so it "
        "shapes subsequent LLM behaviour. The model never sees runtime/cost telemetry."
    ),
)

# ==============================================================================
# SECTION 3 — ROLE 2: FACTUAL AGENT  (MAP phase, delegates to a reasoning strategy)
# core/agents/factual_agent.py  ->  core/reasoning/strategies/*.py
# ==============================================================================

FACTUAL_AGENT = PromptRole(
    name="Factual Agent",
    call_site="core/agents/factual_agent.py :: FactualAgent.analyze -> ReasoningEngine.process -> {ZeroShot|CoT|GoT}Strategy.execute",
    phase="MAP (per-chunk)",
    gateway_method=(
        "CoT/GoT: execute_raw (reasoning) THEN execute_structured("
        "schema_class=CoTTransientOutput) for extraction. Zero-Shot: structured directly."
    ),
    output_format="JSON schema (CoTTransientOutput -> mapped into FactualOutput)",
    prompt_summary=(
        "System/context prompt casts the model as a fact verifier tuned to the "
        "audience knowledge_level, injects TODAY'S DATE (so recent years aren't "
        "dismissed as 'the future'), and enforces: (a) an anti-hallucination contract "
        "requiring a literal quote per reported error; (b) a SOURCE-TRUST HIERARCHY that "
        "sets verification_status per claim (SUPPORTED_BY_SOURCE / CONTRADICTS_SOURCE / "
        "UNVERIFIED / NOT_APPLICABLE) — never call a checkable fact FALSE without a source; "
        "(c) an audience-appropriate simplification rule (beginner simplifications are not "
        "errors). CoT strategy runs a two-step 'reason out loud' then 'extract structured "
        "findings' pipeline; slide OCR + RAG context are injected when tools are used."
    ),
    injected_metrics=[
        "[PY->LLM] speaker_role, main_topic, target_audience, knowledge_level",
        "[PY->LLM] slide_id + is_return_to_slide (visual context)",
        "[PY->LLM] slide OCR text (context_data.pdf_text)",
        "[PY->LLM] historical_summary (trailing_fact_summary.prev_summary — carry-over)",
        "[PY->LLM] RAG retrieval text (only if the gatekeeper/force-year check opened it)",
        "[PY->LLM] current_date (datetime.now, computed by Python, not the model)",
        "[PY->LLM] clean_text (the utterance under evaluation)",
        "NOTE: unlike the linguistic agent, WPM/pauses are NOT fed here — this agent "
        "judges facts only, not delivery.",
    ],
    llm_authored_fields=[
        "[LLM] thought_process (CoT reasoning trace — transient, not persisted downstream)",
        "[LLM] factual_errors (flat text list)",
        "[LLM] scored_errors (SeverityItem list WITH verification_status)",
        "[LLM] thematic_summary (<=2 sentences)",
    ],
    python_assigned_fields=[
        "[PY] chunk_id      = f'chunk_{chunk_meta.index}'   (set in the strategy)",
        "[PY] start_time    = chunk_meta.start_time         (set in the strategy)",
        "[PY] next_state    = TrailingFactualState(...)     (carry-over, when built)",
        "[PY] selected reasoning MODE (Zero-Shot/CoT/GoT) — chosen by Python from "
        "transcription confidence + unclear-word count + RAG budget, NOT by the model",
        "[PY] force_external RAG trigger on regex \\b202[3-9]\\b (Python bypass)",
        "[PY] _rag_used / _rag_budget accounting (per-talk cost cap)",
    ],
    runtime_telemetry=[
        "[PY] tokens_in/out, time_s, cost_usd, prompt_chars/response_chars per call",
        "[PY] TWO telemetry rows for CoT/GoT (reasoning call + extraction call)",
        "[PY] agent_role labels: 'Factual Agent (CoT - Reasoning Phase)' and "
        "'Factual Agent (CoT - Extraction Phase)'",
    ],
    notes=(
        "The MODE decision and RAG gating are the clearest 'Python decides, LLM executes' "
        "boundary: cost/latency-shaping logic is deterministic; the model only fills the "
        "schema. thought_process is authored by the LLM but is transient — only "
        "factual_errors / scored_errors / thematic_summary survive into FactualOutput."
    ),
)

# ==============================================================================
# SECTION 4 — ROLE 3: HEGEMON REDUCER  (REDUCE phase)
# core/agents/hegemon_reducer.py
# ==============================================================================

HEGEMON_REDUCER = PromptRole(
    name="Hegemon Reducer",
    call_site="core/agents/hegemon_reducer.py :: HegemonReducer.generate_report",
    phase="REDUCE (once per run)",
    gateway_method="execute_raw (XML-tagged prose, parsed by Python regex)",
    output_format="XML tags (parsed into DeepAnalysis + ConstructiveFeedback)",
    prompt_summary=(
        "System prompt casts the model as the chief evaluator/mentor, parameterised by "
        "role/audience/level/topic. Iron rules: dual-track chronological correlation "
        "(behavioural stress vs. factual events), SBI model, SEVERITY hierarchy "
        "(discuss CRITICAL/HIGH first, batch LOW), and — critically — 'the QUANTITATIVE "
        "VERDICT block is deterministic; anchor your tone to it, do not contradict it'. "
        "Output MUST be XML tags (JSON forbidden here). The user prompt is the aggregated "
        "COMBINE output plus deterministic blocks."
    ),
    injected_metrics=[
        "[PY->LLM] thematic_blocks (deterministically aggregated in CombineEngine)",
        "[PY->LLM] behavioral_profiles (deterministic; includes the WERDYKT ILOŚCIOWY "
        "block: mean WPM band verdict, filler/pause tallies, transcription confidence)",
        "[PY->LLM] scorecard block: overall/factual/linguistic/slide scores + "
        "readiness_verdict — 'użyj jako kotwicy TONU'",
        "[PY->LLM] unverified_claims block (flag as 'wymaga weryfikacji', NOT errors)",
        "[PY->LLM] presentation_context / slide-coverage block (scenario 5 only)",
        "[PY->LLM] speaker_role, target_audience, knowledge_level, main_topic",
    ],
    llm_authored_fields=[
        "[LLM] <factual_summary>      -> DeepAnalysis.factual_summary",
        "[LLM] <linguistic_summary>   -> DeepAnalysis.linguistic_summary",
        "[LLM] <missed_context>       -> DeepAnalysis.missed_context (list)",
        "[LLM] <executive_summary>    -> ConstructiveFeedback.executive_summary_markdown",
        "[LLM] <strengths>            -> ConstructiveFeedback.strengths (list)",
        "[LLM] <areas_for_improvement>-> ConstructiveFeedback.areas_for_improvement (list)",
        "[LLM] <actionable_tips>      -> ConstructiveFeedback.actionable_tips (list)",
        "[LLM] <overall_message>      -> ConstructiveFeedback.overall_message",
    ],
    python_assigned_fields=[
        "[PY] scorecard              (attached by the pipeline AFTER reduce — the "
        "deterministic ScoreCard OVERRIDES anything the essay implies about score)",
        "[PY] analysis.unverified_claims (authoritative list carried from COMBINE, "
        "overwrites whatever the model wrote)",
        "[PY] map_timestamps, total_windows, substantive_windows (from COMBINE)",
        "[PY] raw_reducer_response, reducer_input (observability capture)",
        "[PY] regex tag extraction + open-tag fallback (partial/truncated output rescue)",
    ],
    runtime_telemetry=[
        "[PY] tokens_in/out, time_s, cost_usd for the (heavy) reduce call",
        "[PY] a SECOND telemetry row if the corrective-format retry fires",
        "[PY] agent_role: 'Hegemon (Reduce Phase)' and '... (Reduce Phase - Retry)'",
        "[PY] reduce_phases_count in the aggregated TelemetryReport",
    ],
    notes=(
        "Hegemon is prose-first: it writes the narrative, but every NUMBER shown to the "
        "user (scores, verdict, unverified list, window counts) is Python-authoritative "
        "and injected/overwritten — the model is explicitly told not to contradict them. "
        "Format retry is role-specific here (XML-tag corrective), separate from the "
        "gateway's transport retry."
    ),
)

ALL_ROLES: List[PromptRole] = [LINGUISTIC_AGENT, FACTUAL_AGENT, HEGEMON_REDUCER]

# ==============================================================================
# SECTION 5 — THE LLM vs PYTHON SPLIT, AT A GLANCE
# ==============================================================================
#
#  DATA / FIELD                          | AUTHORED BY | HOW
#  --------------------------------------+-------------+----------------------------
#  WPM, fillers, pauses, unclear words   | PYTHON      | signal metrics upstream,
#  transcription confidence              |             | injected into prompts [PY→LLM]
#  today's date                          | PYTHON      | datetime.now injected [PY→LLM]
#  escalation_flag                       | PYTHON      | _should_escalate() heuristic
#  reasoning MODE (ZeroShot/CoT/GoT)     | PYTHON      | confidence + RAG budget
#  RAG trigger (incl. \b202[3-9]\b)      | PYTHON      | gatekeeper + regex bypass
#  chunk_id / start_time                 | PYTHON      | overwritten from chunk_meta
#  next_state (trailing carry-over)      | PYTHON      | rebuilt each chunk
#  anomalies / scored_anomalies          | LLM         | linguistic schema fields
#  factual_errors / scored_errors        | LLM         | factual schema fields
#  verification_status per claim         | LLM         | but RULES are prompt-enforced
#  thematic_summary / dominant_tendencies| LLM         | schema fields
#  thematic_blocks / behavioral_profiles | PYTHON      | CombineEngine aggregation
#  ScoreCard (all scores + verdict)      | PYTHON      | compute_scorecard(), OVERRIDES
#  unverified_claims (final)             | PYTHON      | collected in COMBINE, OVERRIDES
#  map_timestamps/total/substantive      | PYTHON      | COMBINE counters
#  executive_summary + tag fields        | LLM         | Hegemon prose
#  tokens/time/cost/cps/CPT telemetry    | PYTHON      | LLMGateway + ObservabilityManager
#
# One-line takeaway: models author QUALITATIVE findings and narrative; Python owns
# every NUMBER (both the lecture signal metrics fed IN and the scores/telemetry
# produced OUT), plus all control-flow decisions (mode, escalation, RAG, retries).


# ==============================================================================
# SECTION 6 — MERMAID FLOWCHART (paste into https://mermaid.live)
# ==============================================================================

MERMAID_PROMPT_TELEMETRY_FLOW = r"""
flowchart TD
    subgraph UP[Upstream deterministic - Python]
      M1[Signal metrics: WPM, fillers,<br/>pauses, unclear, confidence]
      M2[Transcript tagging + clean_text]
      M3[Slide OCR + timeline]
    end

    M1 --> CP[ChunkPayload / LectureMetadata]
    M2 --> CP
    M3 --> CP

    %% ---------------- MAP: LINGUISTIC ----------------
    subgraph LA[Linguistic Agent - MAP]
      CP --> LP[Prompt: role+audience+escalation<br/>PY to LLM: WPM, fillers, pauses,<br/>confidence, tagged_text]
      LP --> LG[execute_structured LinguisticOutput]
      LG --> LO[LLM: anomalies, scored_anomalies,<br/>dominant_tendencies]
      LO --> LPY[PY assigns: chunk_id, start_time,<br/>next_state, escalation_flag]
    end

    %% ---------------- MAP: FACTUAL ----------------
    subgraph FA[Factual Agent - MAP]
      CP --> FMODE[PY decides MODE:<br/>ZeroShot / CoT / GoT + RAG gate]
      FMODE --> FP[Prompt: source-trust rules,<br/>today date, audience level<br/>PY to LLM: slide OCR, RAG, clean_text]
      FP --> FG[execute_raw reason + execute_structured extract]
      FG --> FO[LLM: factual_errors, scored_errors,<br/>verification_status, thematic_summary]
      FO --> FPY[PY assigns: chunk_id, start_time,<br/>next_state]
    end

    %% ---------------- COMBINE ----------------
    LPY --> CE[COMBINE - deterministic Python]
    FPY --> CE
    CE --> CB[thematic_blocks + behavioral_profiles<br/>incl. WERDYKT ILOSCIOWY]
    CE --> SCORE[ScoreCard scores + verdict]
    CE --> UNV[unverified_claims]
    CE --> CNT[map_timestamps / total / substantive]

    %% ---------------- REDUCE: HEGEMON ----------------
    subgraph HG[Hegemon Reducer - REDUCE]
      CB --> HP[Prompt: mentor rules, SBI, severity<br/>PY to LLM: score block as TONE anchor,<br/>unverified block, presentation block]
      SCORE --> HP
      UNV --> HP
      HP --> HGc[execute_raw XML tags]
      HGc --> HO[LLM: factual_summary, linguistic_summary,<br/>executive_summary, strengths, tips...]
    end

    HO --> FINAL[HegemonOutput]
    SCORE -->|PY OVERRIDES scorecard| FINAL
    UNV -->|PY OVERRIDES unverified_claims| FINAL
    CNT -->|PY sets counts| FINAL

    %% ---------------- RUNTIME TELEMETRY ----------------
    LG -.captured by.-> TEL[LLMGateway + ObservabilityManager]
    FG -.captured by.-> TEL
    HGc -.captured by.-> TEL
    TEL --> TELROWS[PY only: tokens_in/out, time_s,<br/>cost_usd, cps, CPT -> PhaseTelemetry<br/>+ SQLite benchmark_logs]
    TELNOTE[Models never see or produce<br/>runtime/cost telemetry]
    TEL --- TELNOTE
"""


# ==============================================================================
# SECTION 7 — RUNNER: print the structured summary + the Mermaid flow
# ==============================================================================

def _print_role(r: PromptRole) -> None:
    print(f"\n{'=' * 78}\n{r.name}  [{r.phase}]\n{'=' * 78}")
    print(f"Call site       : {r.call_site}")
    print(f"Gateway method  : {r.gateway_method}")
    print(f"Output format   : {r.output_format}")
    print(f"\nPrompt summary  :\n  {r.prompt_summary}")
    print("\nInjected metrics [PY -> LLM]:")
    for m in r.injected_metrics:
        print(f"  - {m}")
    print("\nLLM-authored fields [LLM]:")
    for f in r.llm_authored_fields:
        print(f"  - {f}")
    print("\nPython-assigned fields [PY]:")
    for f in r.python_assigned_fields:
        print(f"  - {f}")
    print("\nRuntime telemetry captured [PY]:")
    for t in r.runtime_telemetry:
        print(f"  - {t}")
    if r.notes:
        print(f"\nNotes           : {r.notes}")


def main() -> None:
    # Ensure non-ASCII (Polish diacritics etc.) print on a legacy cp1252 console.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("AGENTS & HEGEMON — PROMPT / TELEMETRY FLOW SUMMARY")
    for role in ALL_ROLES:
        _print_role(role)
    print(f"\n{'=' * 78}\nMERMAID FLOW (paste into https://mermaid.live)\n{'=' * 78}")
    print(MERMAID_PROMPT_TELEMETRY_FLOW)


if __name__ == "__main__":
    main()
