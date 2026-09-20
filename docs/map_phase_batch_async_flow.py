# ==============================================================================
# MAP PHASE — BATCHING, ASYNC & SCENARIO-5 CHUNK GROUPING
# ==============================================================================
#
# Two questions this file answers, grounded in the real code:
#
#   Q1. In scenario 5 (SWARM_PRESENTATION_RAG_WEB), are chunks analyzed ACROSS
#       SLIDES or ACROSS THE TIMELINE?
#       -> BOTH, in two SEPARATE phases:
#          * MAP (swarm) analysis   = ACROSS THE TIMELINE (chronological, batched).
#          * Slide-coverage analysis = ACROSS SLIDES (grouped by slide_id).
#
#   Q2. How do the async + batch mechanics of the MAP phase actually work?
#       -> Sequential batches of 4 chunks; INSIDE a batch the two agents per chunk
#          run concurrently (asyncio.gather); trailing state flows chunk->chunk
#          AND across batch boundaries. See SECTION 3.
#
# Source of truth:
#   - core/pipelines/orchestrator.py            (_execute_scenario_5..., _apply_presentation_wiring)
#   - core/pipelines/slide_coverage_engine.py   (analyze_slide_coverage — per-slide grouping)
#   - core/pipelines/swarm_pipeline.py          (run_map_combine / _process_batch — batching+async)
#   - core/agents/{linguistic,factual}_agent.py (the two concurrent map agents)
#
# Legend for the notes:
#   [SEQ]  = runs sequentially (ordered)
#   [CONC] = runs concurrently (asyncio.gather)
#   [TL]   = organized along the TIMELINE (chronological)
#   [SL]   = organized by SLIDE (grouped by slide_id)


# ==============================================================================
# SECTION 1 — SCENARIO 5: TWO GROUPINGS OF THE SAME CHUNKS
# ==============================================================================
#
# The SAME List[ChunkPayload] is consumed twice, differently:
#
#  (A) TIMELINE grouping  [TL]  — the swarm MAP phase (run_map_combine)
#      - Chunks are ordered by their position in the talk and processed in
#        batches of 4 (batch_size=4), batches SEQUENTIAL [SEQ].
#      - Each chunk keeps slide_id + OCR text only as CONTEXT; the linguistic
#        and factual agents judge it as a chronological window, NOT per slide.
#      - This is identical to scenarios 3 and 4 — presentation adds context,
#        it does not change the timeline-based map traversal.
#
#  (B) SLIDE grouping     [SL]  — SlideCoverageEngine.analyze_slide_coverage
#      - Chunks are bucketed by chunk_meta.slide_id into `by_slide`.
#      - Returns to a slide share the same slide_id, so a slide's speech is the
#        concatenation of ALL its chunks (sorted by start_time), including returns.
#      - ONE structured LLM call PER SLIDE compares slide OCR vs. what was said
#        -> SlideCoverage(covered_points, missed_points, returned_later, ...).
#      - These per-slide calls run CONCURRENTLY [CONC] (asyncio.gather over slides).
#
#  HOW A CHUNK GETS ITS slide_id (the wiring that enables (B)):
#      _apply_presentation_wiring / the scenario-5 loop walks timeline.global_timeline;
#      for each event it stamps every listed chunk with:
#          chunk_meta.slide_id, chunk_meta.is_return_to_slide, context_data.pdf_text (OCR).
#      Timeline events therefore MAP chunks -> slides; the timeline is the source,
#      the per-slide grouping is derived from it.
#
#  ORDER IN SCENARIO 5:
#      1) presentation wiring (stamp slide_id/OCR on chunks)              [TL->SL link]
#      2) slide coverage (per-slide) + deterministic presentation flow    [SL]
#      3) swarm MAP+COMBINE over the timeline                              [TL]
#      4) Hegemon REDUCE (gets presentation_context + slide_coverage too)
#
#  ONE-LINE ANSWER: substantive/behavioral analysis is TIMELINE-based (per chunk,
#  batched); slide-coverage analysis is SLIDE-based (grouped by slide_id). Both
#  happen in scenario 5, in different engines.


# ==============================================================================
# SECTION 2 — SLIDE GROUPING, IN PSEUDOCODE (mirrors slide_coverage_engine.py)
# ==============================================================================
#
#   by_slide = {}                       # slide_id -> [chunks]
#   returned = {}                       # slide_id -> bool (any chunk was a return)
#   for ch in chunks:
#       sid = ch.chunk_meta.slide_id
#       if sid is None: continue        # unassigned chunks are skipped here
#       by_slide[sid].append(ch)
#       if ch.chunk_meta.is_return_to_slide: returned[sid] = True
#
#   async def _one(sid, slide_chunks):
#       slide_chunks_sorted = sort by start_time          # chronological within a slide
#       speech = "\n".join(clean_text of each)            # incl. returns, in order
#       coverage = await gateway.execute_structured(SlideCoverage, ...)  # 1 call / slide
#       coverage.slide_id = sid                           # [PY] assigned
#       coverage.returned_later = returned.get(sid, False)# [PY]
#       coverage.time_on_slide_sec = sum(appearance spans)# [PY] from timeline
#       coverage.dwell_verdict = ... ("ZA_KRÓTKO" if < 15s)# [PY]
#       return coverage
#
#   tasks = [_one(sid, chs) for sid, chs in sorted(by_slide.items())]
#   return await asyncio.gather(*tasks)                   # [CONC] all slides at once
#
# Note: within a slide, speech is assembled chronologically, but the UNIT of
# analysis is the SLIDE. Contrast with the MAP phase where the unit is the CHUNK.


# ==============================================================================
# SECTION 3 — MAP PHASE BATCH + ASYNC MECHANICS (mirrors swarm_pipeline.py)
# ==============================================================================
#
#  BATCHING (run_map_combine):
#    batch_size = 4
#    batches = [chunks[i:i+4] for i in range(0, len(chunks), 4)]
#    -> batches are processed ONE AFTER ANOTHER [SEQ] (a for-loop, awaited each time),
#       NOT concurrently. This is deliberate: it lets trailing state flow ACROSS
#       batch boundaries (see carry-over below).
#
#  INSIDE A BATCH (_process_batch):
#    for each chunk i in the batch (SEQUENTIAL loop over the 4 chunks):
#        ling_task = linguistic_agent.analyze(chunk, metadata)      # coroutine
#        fact_task = factual_agent.analyze(chunk, metadata, use_tools)
#        ling_out, fact_out = await asyncio.gather(ling_task, fact_task)  # [CONC] pair
#    -> so within one chunk, the TWO agents run concurrently. The 4 chunks of a
#       batch are stepped through in order (each awaited) so intra-batch trailing
#       state can be handed to the next chunk BEFORE it runs.
#
#    IMPORTANT nuance: the code iterates chunks with `for i, chunk in enumerate(batch)`
#    and awaits gather PER CHUNK. So concurrency in the map phase is at the
#    AGENT-PAIR level (linguistic || factual), threaded through a sequential walk of
#    chunks. (The two agents are the parallel unit; chunks are sequenced so state
#    can propagate.)
#
#  TRAILING STATE (the reason batches are sequential):
#    - Linguistic: ling_out.next_state (TrailingLinguisticState: prev_filler_count,
#      escalation_flag) -> set on the NEXT chunk's trailing_linguistics.
#    - Factual: fact_out.next_state (TrailingFactualState: prev_summary, open_loops)
#      -> set on the NEXT chunk's trailing_fact_summary.
#    - Between batches: run_map_combine remembers carry_ling / carry_fact from the
#      LAST chunk of a batch and injects it into the FIRST chunk of the next batch,
#      so escalation / open-loops survive the batch boundary (map-reduce preserved).
#
#  FAILURE ISOLATION:
#    A crash in one chunk's gather is caught; that chunk yields (None, None) and the
#    batch continues. None results are filtered out before COMBINE.
#
#  AFTER MAP -> COMBINE (deterministic, no LLM):
#    aggregate_dual_track() + compute_scorecard() reduce all per-chunk outputs into
#    thematic_blocks / behavioral_profiles / ScoreCard; then REDUCE (Hegemon) runs.


# ==============================================================================
# SECTION 4 — MERMAID: MAP PHASE ASYNC + BATCH FLOW (paste into https://mermaid.live)
# ==============================================================================

MERMAID_MAP_PHASE_FLOW = r"""
flowchart TD
    START[chunks in timeline order] --> SPLIT[split into batches of 4]

    SPLIT --> B1[Batch 1 - chunks 0..3]
    B1 --> B2[Batch 2 - chunks 4..7]
    B2 --> B3[Batch N ...]
    B1 -. carry_ling / carry_fact .-> B2
    B2 -. carry across boundary .-> B3

    subgraph BATCH[Inside ONE batch - _process_batch - SEQUENTIAL walk of 4 chunks]
      direction TB
      C0[chunk i] --> GcaseA
      subgraph GcaseA[Per chunk: two agents CONCURRENT]
        direction LR
        LA[linguistic_agent.analyze]
        FA[factual_agent.analyze<br/>Zero-Shot / CoT / GoT + RAG gate]
      end
      GcaseA --> GATHER[await asyncio.gather ling, fact]
      GATHER --> OUT[LinguisticOutput + FactualOutput]
      OUT -. next_state: escalation_flag,<br/>prev_summary, open_loops .-> C1[chunk i+1]
      C1 --> GcaseA
    end

    B1 --> BATCH
    BATCH --> COLLECT[collect mapped_results<br/>None,None on chunk crash -> filtered]

    COLLECT --> COMBINE[COMBINE - deterministic, no LLM<br/>aggregate_dual_track + compute_scorecard]
    COMBINE --> REDUCE[REDUCE - Hegemon report]

    %% ---- Scenario-5 side note: the OTHER grouping ----
    subgraph S5[Scenario 5 only - runs BEFORE reduce]
      direction TB
      WIRE[presentation wiring:<br/>stamp slide_id + OCR on chunks<br/>from timeline.global_timeline]
      WIRE --> BYSLIDE[group chunks BY slide_id<br/>returns share slide_id]
      BYSLIDE --> SLIDECALLS[per-slide execute_structured SlideCoverage<br/>CONCURRENT via asyncio.gather]
      SLIDECALLS --> PCTX[presentation_context + slide_coverage<br/>fed into Hegemon]
    end
    START -. same chunks, slide grouping .-> WIRE
    PCTX -. presentation_context .-> REDUCE

    classDef tl fill:#e3f2fd,stroke:#1565c0;
    classDef sl fill:#fff3e0,stroke:#e65100;
    class START,SPLIT,B1,B2,B3,BATCH,COLLECT tl;
    class WIRE,BYSLIDE,SLIDECALLS,PCTX sl;
"""


# ==============================================================================
# SECTION 5 — QUICK REFERENCE TABLE
# ==============================================================================
#
#  ASPECT                         | MAP phase (swarm)        | Slide-coverage (scn 5)
#  -------------------------------+--------------------------+------------------------
#  Unit of analysis               | one CHUNK (window) [TL]  | one SLIDE [SL]
#  Grouping key                   | timeline order           | chunk_meta.slide_id
#  Batching                       | batches of 4, SEQUENTIAL | none (all slides)
#  Concurrency                    | ling || fact per chunk   | all slides via gather
#  Cross-item state               | trailing_* carry-over    | none (independent slides)
#  LLM calls                      | 2 per chunk (+RAG/CoT)   | 1 per slide
#  Output                         | Linguistic/FactualOutput | SlideCoverage
#  Present in scenarios           | 3, 4, 5                  | 5 only


def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(__doc__ if __doc__ else "MAP PHASE — BATCHING, ASYNC & SCENARIO-5 GROUPING")
    print("\nSee SECTION 1-3 comments for the written flow.\n")
    print("=" * 78)
    print("MERMAID FLOW (paste into https://mermaid.live)")
    print("=" * 78)
    print(MERMAID_MAP_PHASE_FLOW)


if __name__ == "__main__":
    main()
