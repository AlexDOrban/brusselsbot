# Stage 2 Decision — Reranker + Hybrid 3-way Evaluation

**Decision: KEEP hybrid, keep reranker**

_Only B passes. Reranker needed BM25's candidate pool to recover. Note: hybrid is earning its place by diversifying candidates, not by raw rank fusion._

## Configs

| Config | hybrid | rerank | File |
|--------|--------|--------|------|
| C (baseline) | False | False | `benchmark_2C_baseline.json` |
| A (rerank only) | False | True | `benchmark_2A_rerank_only.json` |
| B (full stack) | True  | True  | `benchmark_2B_hybrid_rerank.json` |

## Per-question ranks (gold source rank in top-10)

| # | Question ID | C (baseline) | A (rerank) | B (hybrid+rerank) | Winner |
|---|-------------|--------------|------------|-------------------|--------|
| 1 | Q1_noise_en | 1 | 1 | 1 | C=A=B |
| 2 | Q2_muziek_nl | 2 | 2 | 1 | B |
| 3 | Q3_sluikstort_nl | 1 | 1 | 1 | C=A=B |
| 4 | Q4_lez_social | 1 | 1 | 1 | C=A=B |
| 5 | Q5_bruxellair | 1 | 1 | 1 | C=A=B |
| 6 | Q6_grue_fr | 1 | 1 | 1 | C=A=B |
| 7 | Q7_trash_sidewalk | 1 | 3 | 2 | C |
| 8 | Q8_ev_fleet | 1 | 1 | 1 | C=A=B |
| 9 | Q9_e40_oos | OOS | OOS | OOS | — |
| 10 | Q10_alarm | 1 | 1 | 1 | C=A=B |

> Rank values: integer rank of first gold chunk, `>10` if missed, `OOS` = out-of-scope (no gold).

## Aggregate gate checks

| Metric | Threshold | C | A | B |
|--------|-----------|---|---|---|
| Hit@1 ≥ 0.889 | 0.889 | 0.889 ✅ | 0.778 ❌ | 0.889 ✅ |
| Hit@3 ≥ 1.000 | 1.000 | 1.000 ✅ | 1.000 ✅ | 1.000 ✅ |
| MRR ≥ 0.944 | 0.944 | 0.944 ✅ | 0.870 ❌ | 0.944 ✅ |
| sluikstort in top-5 | — | ✅ | ✅ | ✅ |

### All-gates verdict

- Config A (rerank only): **FAIL**
- Config B (hybrid + rerank): **PASS**
- MRR(B) − MRR(A) = +0.0741

## Raw aggregates

| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|--------|-------|-------|-------|--------|-----|
| C | 0.889 | 1.000 | 1.000 | 1.000 | 0.944 |
| A | 0.778 | 1.000 | 1.000 | 1.000 | 0.870 |
| B | 0.889 | 1.000 | 1.000 | 1.000 | 0.944 |

## Q_sluikstort deep-dive (risk #3 check)

Question: *Welke boete riskeer ik voor sluikstort?*

| Config | first_gold_rank | top-3 chunk IDs | top-3 (source, article, lang) |
|--------|-----------------|-----------------|-------------------------------|
| C | 1 | chunk_7, chunk_243, chunk_111 | police_regs·art4·nl / police_regs·art122·nl / police_regs·art57·fr |
| A | 1 | chunk_49, chunk_79, chunk_243 | police_regs·art25·nl / police_regs·art40·nl / police_regs·art122·nl |
| B | 1 | chunk_49, chunk_79, chunk_243 | police_regs·art25·nl / police_regs·art40·nl / police_regs·art122·nl |

**Reference — Stage 1 hybrid-only (no rerank):** first_gold_rank = 2.

**Risk #3 read:** the concern was that the reranker, being a semantic cross-encoder, would demote the BM25-surfaced sluikstort chunks and undo the Stage 1 win.
- Config B (hybrid+rerank) lands the gold at rank 1, vs. A (rerank only) at rank 1. **Risk #3 did NOT materialize** — the reranker kept (or improved on) BM25's candidates.

## Decision applied

- A passes: **False**
- B passes: **True**
- MRR(B) − MRR(A) = +0.0741 (threshold 0.02)

→ **KEEP hybrid, keep reranker**

_Only B passes. Reranker needed BM25's candidate pool to recover. Note: hybrid is earning its place by diversifying candidates, not by raw rank fusion._
