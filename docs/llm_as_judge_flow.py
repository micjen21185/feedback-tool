# ==============================================================================
# LLM-AS-JUDGE — PYDANTIC STRUCTURES, FLOW & METRICS
# ==============================================================================
#
# Purpose
# -------
# Explain how the evaluation layer ("Tier-1 LLM-as-a-Judge") works: which Pydantic
# structures hold the judge's output, how the judge FLOW is orchestrated, and —
# crucially — which metrics are AUTHORED BY THE JUDGE LLM vs. which are computed
# DETERMINISTICALLY by Python and either fed to the judge or reported alongside it.
#
# Mirrors the real code:
#   - core/pipelines/evaluation_engine.py   (EvaluationEngine — the whole flow)
#   - models/schemas.py                     (JudgeRubric, ScenarioEvaluation,
#                                            PairwisePreference, EvaluationReport)
#
# Legend:
#   [JUDGE]  = value produced by the judge LLM inside its structured response
#   [PY]     = value computed deterministically by Python (never by the model)
#   [PY->J]  = deterministic value Python computes and INJECTS into the judge prompt
#              (the judge reads it as grounding/anchor, but does not compute it)
#
# ------------------------------------------------------------------------------
# BIG PICTURE
# ------------------------------------------------------------------------------
# The judge does TWO kinds of LLM calls per evaluation run:
#   1. ABSOLUTE  (per report): scores one report on a 5-dimension rubric (0-10 each)
#      -> execute_structured(JudgeRubric)
#   2. PAIRWISE  (per report pair): picks the more useful of two reports
#      -> execute_structured(PairwisePreference)
#
# Around those calls, Python computes a large set of DETERMINISTIC metrics that do
# NOT depend on the judge model at all (recall vs a Golden Set, positional recall,
# lost-in-the-middle, reduce fidelity, cost/telemetry splits, RMSE alignment). Some
# of these are ALSO injected into the judge prompt as grounding so the qualitative
# score is anchored to evidence rather than vibes.

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ==============================================================================
# SECTION 1 — CRITICAL PYDANTIC STRUCTURES FOR THE JUDGE
# ==============================================================================
#
# JudgeRubric              [JUDGE authors all fields]
#   The 5-dimension score card the judge fills per report. Every field is
#   constrained 0..10 (ge=0, le=10) by Pydantic — so an out-of-range score from
#   the model is rejected at validation time. `justification` is free text
#   (and is where the judge is asked to embed the Markdown "error table").
#     actionability : int 0-10   — how concrete/executable the advice is
#     specificity   : int 0-10   — how grounded in the transcript the feedback is
#     correctness   : int 0-10   — accuracy of the substantive critique
#     tone          : int 0-10   — appropriateness of tone to quality
#     groundedness  : int 0-10   — no hallucinations; claims backed by evidence
#     justification : str        — rationale + the per-error detection table
#
# PairwisePreference       [JUDGE authors, PY normalizes `winner`]
#     winner : str  — the judge writes a scenario name or "TIE"; Python then
#                     runs _normalize_winner() to coerce it to an exact name /
#                     "TIE" / "UNCLEAR" (defends against loose model phrasing).
#     reason : str  — free-text justification.
#
# ScenarioEvaluation       [MIXED: rubric is JUDGE; the rest is PY]
#   The durable per-report evaluation record. Wraps the JudgeRubric plus all the
#   deterministic metrics Python computed:
#     rubric              : JudgeRubric   [JUDGE]
#     rubric_total        : float         [PY]  sum of the 5 dims (0..50)
#     total_tokens_in/out : int           [PY]  from report telemetry
#     total_cost_usd      : float         [PY]
#     total_time_s        : float         [PY]
#     input/output_token_density : float  [PY]  (set 0.0 here; densities live in extra_metrics)
#     positional_recall   : List[float]   [PY]  timestamp coverage per time-region
#     reduce_fidelity     : List[float]   [PY]  map->report survival per region (swarm)
#     lost_in_middle_flag : bool          [PY]  middle-region coverage collapse
#     judge_evidence      : str           [PY]  the exact grounding shown to the judge
#
# EvaluationReport         [aggregate; PY assembles]
#     per_scenario : List[ScenarioEvaluation]  [PY-assembled, contains JUDGE rubrics]
#     pairwise     : List[PairwisePreference]  [JUDGE + PY-normalized]
#     judge_tokens_in/out : int                [PY]  judge's own cost (kept SEPARATE
#                                                     from pipeline cost via phase class)
#     summary      : str                       [PY]  "best scenario" line
#
# KEY DESIGN POINT: only JudgeRubric + PairwisePreference are the judge's SUBJECTIVE
# output. Everything numeric and comparative around them is deterministic Python —
# the judge's score is one input among many, not the whole verdict.


# ==============================================================================
# SECTION 2 — QUERYABLE METRIC CATALOG
# ==============================================================================

@dataclass
class JudgeMetric:
    name: str
    origin: str  # [JUDGE] | [PY] | [PY->J]
    where: str  # function / field in evaluation_engine.py
    what: str  # what it measures


JUDGE_LLM_METRICS: List[JudgeMetric] = [
    JudgeMetric("actionability", "[JUDGE]", "JudgeRubric.actionability",
                "How concrete/executable the advice is (0-10)."),
    JudgeMetric("specificity", "[JUDGE]", "JudgeRubric.specificity",
                "How grounded in the transcript the feedback is (0-10)."),
    JudgeMetric("correctness", "[JUDGE]", "JudgeRubric.correctness",
                "Accuracy of substantive critique (0-10)."),
    JudgeMetric("tone", "[JUDGE]", "JudgeRubric.tone",
                "Appropriateness of tone to speech quality (0-10)."),
    JudgeMetric("groundedness", "[JUDGE]", "JudgeRubric.groundedness",
                "Freedom from hallucination; evidence-backed (0-10)."),
    JudgeMetric("pairwise winner", "[JUDGE]", "PairwisePreference.winner",
                "Which of two reports is more useful (then PY-normalized)."),
]

PY_DETERMINISTIC_METRICS: List[JudgeMetric] = [
    JudgeMetric("rubric_total", "[PY]", "evaluate() sum of 5 dims",
                "Sum of the judge's five dimensions (0..50)."),
    JudgeMetric("error_recall_pct", "[PY]", "_calculate_error_recall",
                "Share of Golden-Set errors the report caught (time OR thematic match)."),
    JudgeMetric("error_recall_critical_pct", "[PY]", "_calculate_error_recall",
                "Recall restricted to CRITICAL/HIGH golden errors."),
    JudgeMetric("caught_by_time / caught_by_theme", "[PY]", "_calculate_error_recall",
                "How each golden error was matched: timestamp proximity vs keyword overlap."),
    JudgeMetric("positional_recall", "[PY]", "positional_recall",
                "Distribution of cited timestamps across start/middle/end regions."),
    JudgeMetric("lost_in_middle_flag", "[PY]", "_lost_in_middle",
                "True when the middle region is markedly under-covered vs edges."),
    JudgeMetric("reduce_fidelity", "[PY]", "reduce_fidelity",
                "Swarm: fraction of map-phase findings per region that survived into the report."),
    JudgeMetric("rmse (alignment_error)", "[PY]", "_calculate_alignment_error",
                "RMSE of report's factual/linguistic scores vs expected values."),
    JudgeMetric("tpw (language tax)", "[PY]", "_calculate_language_tax",
                "Map-phase input tokens per transcript word."),
    JudgeMetric("factual/linguistic/reduce density", "[PY]", "_phase_densities",
                "prompt_chars / tokens_in per phase (characters-per-token efficiency)."),
    JudgeMetric("phase cost split", "[PY]", "_split_phase_telemetry",
                "map vs reduce cost, real Hegemon input size, actual model used, fallback detection."),
    JudgeMetric("missing_sections", "[PY]", "_missing_sections / _detect_scenario",
                "Expected sections the report left empty (fairness — only for sections the scenario COULD produce)."),
    JudgeMetric("judge_tokens_in/out", "[PY]", "evaluate() phase filter",
                "The judge's OWN token cost, kept separate from pipeline cost."),
]

INJECTED_GROUNDING: List[JudgeMetric] = [
    JudgeMetric("grounding excerpt", "[PY->J]", "build_grounding_excerpt",
                "Multi-region transcript slices (start/middle/end) to fight lost-in-the-middle."),
    JudgeMetric("timestamp probes", "[PY->J]", "build_timestamp_probes",
                "Transcript snippets at the exact timestamps the report cited (verify grounding)."),
    JudgeMetric("golden set (full list)", "[PY->J]", "_format_golden_errors",
                "EVERY golden error (all scales) so the judge builds the detection table honestly."),
    JudgeMetric("pipeline ground-truth findings", "[PY->J]", "_ground_truth_findings",
                "missed_context + unverified_claims — anchor for groundedness. NOTE: the report's own "
                "overall_score is deliberately NOT shown (it biased the judge upward)."),
    JudgeMetric("fairness / scenario block", "[PY->J]", "_detect_scenario + fairness_block",
                "Tells the judge the architecture type and what to (not) penalize; lists detected gaps."),
    JudgeMetric("focus_instruction", "[PY->J]", "self.focus_instruction",
                "Optional user emphasis passed straight into the prompt."),
]

# ==============================================================================
# SECTION 3 — THE JUDGE FLOW, STEP BY STEP (evaluate())
# ==============================================================================
#
#  0. INPUTS: transcript_excerpt, reports {name -> FinalReport}, duration_sec,
#     total_words, expected_factual/linguistic, golden_factual/linguistic (JSON).
#
#  1. PREP (once):
#     - build_grounding_excerpt(): slice transcript into regions [PY->J].
#     - _extract_golden_errors() x2: flatten Golden Set JSON into a scale-tagged
#       list of every error (LOW/MEDIUM/HIGH/CRITICAL), recursively [PY].
#     - _format_golden_errors(): render that list for the prompt [PY->J].
#
#  2. PER REPORT (loop):
#     a. _detect_scenario() + _missing_sections(): classify monolith/swarm/
#        presentation and find empty-but-expected sections (fairness) [PY].
#     b. build_timestamp_probes(): snippets at cited timestamps [PY->J].
#     c. _judge_absolute(): ONE structured judge call -> JudgeRubric [JUDGE].
#        (on exception: JudgeRubric(justification="[Sędzia zawiódł: ...]") fallback.)
#     d. Deterministic metrics [PY]:
#        rubric_total, positional_recall, reduce_fidelity, lost_in_middle,
#        error recall vs Golden Set (time OR thematic match), phase cost split,
#        language tax, RMSE alignment, phase densities.
#     e. Assemble ScoreCard-adjacent record: ScenarioEvaluation (+ judge_evidence
#        string that captures EXACTLY what the judge was shown) [PY].
#
#  3. PAIRWISE (all report pairs):
#     _judge_pairwise() -> PairwisePreference [JUDGE], winner normalized [PY].
#
#  4. AGGREGATE [PY]:
#     - judge_tokens_in/out summed from telemetry rows whose agent_role contains
#       "Evaluator" (judge cost isolated from pipeline cost via _classify_phase).
#     - summary = highest rubric_total scenario.
#     Returns (EvaluationReport, extra_metrics dict).
#
# WHY THE SPLIT MATTERS:
#   - The judge LLM provides SUBJECTIVE quality (rubric + preference).
#   - Python provides OBJECTIVE, reproducible ground truth (recall, coverage,
#     cost) and FEEDS some of it back as grounding so the subjective score is
#     anchored. The two are stored side by side but never conflated.


# ==============================================================================
# SECTION 4 — MERMAID FLOWCHART (paste into https://mermaid.live)
# ==============================================================================

MERMAID_JUDGE_FLOW = r"""
flowchart TD
    subgraph IN[Inputs]
      I1[reports: name to FinalReport]
      I2[transcript_excerpt + duration + total_words]
      I3[Golden Set JSON: factual + linguistic]
      I4[expected_factual / expected_linguistic]
    end

    %% ---------------- ONE-TIME PREP ----------------
    subgraph PREP[Prep - deterministic PY]
      I2 --> P1[build_grounding_excerpt<br/>start / middle / end slices]
      I3 --> P2[extract_golden_errors<br/>flatten all scales]
      P2 --> P3[format_golden_errors<br/>full error list for prompt]
    end

    %% ---------------- PER-REPORT LOOP ----------------
    subgraph LOOP[Per report]
      I1 --> D1[detect_scenario + missing_sections<br/>monolith/swarm/presentation + gaps]
      D1 --> D2[build_timestamp_probes<br/>snippets at cited timestamps]
      P1 --> J1
      P3 --> J1
      D1 --> J1
      D2 --> J1
      J1[_judge_absolute<br/>execute_structured JudgeRubric]
      J1 -->|LLM scores 5 dims 0-10| R1[JudgeRubric]
      J1 -.on error.-> RF[JudgeRubric fallback<br/>justification=Sedzia zawiodl]

      %% deterministic metrics computed alongside the judge
      I1 --> M1[error recall vs Golden Set<br/>time OR thematic match]
      I1 --> M2[positional_recall + lost_in_middle]
      I1 --> M3[reduce_fidelity - swarm]
      I1 --> M4[split_phase_telemetry<br/>map vs reduce cost, real model, fallback]
      I4 --> M5[alignment RMSE]
      I1 --> M6[language tax + phase densities]

      R1 --> SE[ScenarioEvaluation<br/>rubric + rubric_total + all PY metrics<br/>+ judge_evidence]
      M1 --> SE
      M2 --> SE
      M3 --> SE
      M4 --> SE
      M5 --> SE
      M6 --> SE
    end

    %% ---------------- PAIRWISE ----------------
    P1 --> PW[_judge_pairwise per pair<br/>execute_structured PairwisePreference]
    I1 --> PW
    PW -->|LLM picks winner| PWN[normalize_winner<br/>exact name / TIE / UNCLEAR]

    %% ---------------- AGGREGATE ----------------
    SE --> AGG[EvaluationReport]
    PWN --> AGG
    AGG --> T[judge_tokens_in/out<br/>isolated via classify_phase = judge]
    AGG --> S[summary: best rubric_total]

    %% telemetry note
    J1 -.telemetry.-> TEL[Evaluator judge calls<br/>counted SEPARATELY from pipeline cost]
    PW -.telemetry.-> TEL
"""


# ==============================================================================
# SECTION 5 — RUNNER
# ==============================================================================

def _print_catalog(title: str, items: List[JudgeMetric]) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for m in items:
        print(f"  {m.origin:8} {m.name}")
        print(f"           where: {m.where}")
        print(f"           what : {m.what}")


def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("LLM-AS-JUDGE — STRUCTURES, FLOW & METRICS")
    _print_catalog("JUDGE-AUTHORED METRICS [JUDGE]", JUDGE_LLM_METRICS)
    _print_catalog("DETERMINISTIC METRICS [PY]", PY_DETERMINISTIC_METRICS)
    _print_catalog("GROUNDING INJECTED INTO THE JUDGE PROMPT [PY->J]", INJECTED_GROUNDING)
    print(f"\n{'=' * 78}\nMERMAID FLOW (paste into https://mermaid.live)\n{'=' * 78}")
    print(MERMAID_JUDGE_FLOW)


if __name__ == "__main__":
    main()
