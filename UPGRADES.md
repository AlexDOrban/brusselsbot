# BrusselsBot — Advanced RAG Upgrades

Tracking sheet for the post-baseline RAG improvements. Every stage must be benchmarked on the same 10-question set (`src/benchmark.py`) so the deltas are comparable.

## Baseline (main branch)

10-question benchmark, top-10 retrieval, plain embedding search (`text-embedding-3-small`, cosine distance in ChromaDB).

- **Hit@1:** 0.889
- **Hit@3:** 1.000
- **Hit@5:** 1.000
- **Hit@10:** 1.000
- **MRR:** 0.944
- **Out-of-scope distance:** 0.625 (top-1 for E40 speeding question, correctly above the 0.60 caveat threshold)

Known failure modes carried into the upgrade work:

- **Keyword-rare terms buried by semantics.** `"sluikstort"` appears once in the corpus (Article 25) and doesn't surface in top-10 for the NL dumping query.
- **Cross-lingual retrieval is single-query.** An NL question doesn't pull the FR equivalents it should (the dumping topic is spread across Articles 14, 25, 29, 30, 63 in the FR half).
- **Ranking ambiguity on natural NL phrasings.** `"Mag ik 's nachts muziek maken op straat?"` ranks Article 91 above Article 88 (the gold).

## Planned stages

1. **Scaffolding** — completed on `main`: article-aware FR/NL chunking, OpenAI embeddings, ChromaDB, query rewriting with Haiku, streaming Gradio UI, benchmark harness.
2. **Hybrid BM25 + embedding retrieval** — add BM25 index over chunks (likely `rank_bm25` or `bm25s`), fuse with embedding scores via reciprocal rank fusion. Goal: surface keyword-exact terms like `sluikstort` without hurting semantic hits.
3. **bge-reranker-v2-m3** — second-stage cross-encoder rerank of the top-K from hybrid retrieval. Multilingual (EN/FR/NL). Goal: raise Hit@1 and MRR by fixing ranking ambiguity inside a correct candidate set.
4. **Cross-lingual query expansion** — translate the user query into the three corpus languages before retrieval (or use a multilingual rewrite), merge result sets. Goal: cover the case where the relevant article only exists in a language the user didn't use.
5. **Agentic retrieval loop** — let Claude decide whether to re-query with refined terms, broaden the search, or abstain. Bounded turns. Goal: handle multi-hop questions and the out-of-scope case more gracefully than a fixed threshold.
6. **RAGAS benchmark expansion** — grow the eval set and add RAGAS metrics (faithfulness, answer relevance, context precision/recall). Goal: measure answer quality, not just retrieval rank.

## Results

| Stage | Status | Hit@1 | MRR | Notes |
|-------|--------|-------|-----|-------|
| 1. Scaffolding | done | 0.889 | 0.944 | Baseline on `main`. Plain cosine, top-10. |
| 2. Hybrid BM25 + Cohere rerank-v3.5 | done (keep) | 0.605 | 0.741 | N=50 benchmark, Config B. C baseline at N=50: Hit@1=0.535, MRR=0.671. See `data/stage2_decision_n50.md`. |
| 3. Cross-lingual query expansion | planned | — | — | Needs benchmark expansion to detect gains (current set already saturates Hit@3). |
| 4. Agentic retrieval loop | planned | — | — | Latency trade-off; measure turns + cost. |
| 5. RAGAS expansion | planned | — | — | Adds faithfulness / context-precision / answer-relevance. |

## Stage 2 outcome — KEEP hybrid + Cohere rerank

Decision based on N=50 retrieval-only evaluation (`data/stage2_decision_n50.md`).

Aggregate retrieval lifts vs baseline (Config C):
- Hit@1: 0.535 → 0.605 under B (+0.070)
- Hit@3: 0.767 → 0.884 under B (+0.117)
- MRR:   0.671 → 0.741 under B (+0.070)

Per-category breakdown shows rerank earning its place specifically on multi_hop (+20pp Hit@1). BM25 contributes measurably above rerank-alone (MRR(B) − MRR(A) = +0.038), recovering 2 questions that dense-only retrieval missed entirely (Q15_demenage_fr, Q21_verhuizing_nl: rank >10 → rank 1).

OOS detection improved from 1/7 to 4/7 correctly-caveated under both rerank configs — meaningful behavioral improvement, but 3/7 OOS questions still receive confident hallucinated answers. Threshold recalibration deferred to Stage 6+.

### Decision-process notes

- Original N=10 thresholds (Hit@1 ≥ 0.889, MRR ≥ 0.944) were unreachable on the harder N=50 benchmark. Threshold re-anchoring rule extended in `src/generate_decision.py` to handle baseline drops as well as baseline rises (regression test in `tests/test_decision_gates.py`).
- RAGAS answer-quality evaluation scoped out due to local hardware constraints. Documented in `stage2_decision_n50.md` §8.

### Known limitations / Stage 6+ work

1. **Cross-lingual regression under rerank**: B and A both lose 20pp Hit@1 vs C on cross_lingual category. Cohere rerank-v3.5 appears to penalize query-language ≠ source-language matches. Stage 3 (planned: cross-lingual query expansion via Haiku translation) addresses this directly — that stage's motivating evidence is now measured.

2. **OOS retrieval-threshold calibration**: 3/7 OOS questions receive confident answers despite the corpus not containing relevant content. Caveat thresholds (dense_distance > 0.60, rerank_score < 0.30) were calibrated on the original single OOS question (E40 speeding) at N=10. Stage 6: recalibrate against the full N=7 OOS set.

3. **OOS LLM scope-check (separate item)**: Even with recalibrated thresholds, retrieval-side caveat alone may not catch all OOS cases. Consider an LLM pre-check ("does this question's topic match the corpus's coverage?") as an additional gate before generation.

4. **Chunker section-header absorption**: Article-aware chunker sweeps `Sectie` headers into the chunk for the article that follows them on the page. This causes article 21 (dust) to stickily match trottoir/snow/cleaning queries (13 questions affected before manual correction during ground-truth review). Fixing the chunker invalidates all chunk_ids in the benchmark — deferred to a major version bump.

5. **Q27_vuur_nl mapping ambiguity**: "May I burn leaves?" mapped to no clear article in the corpus. Specific leaf-burning rules likely live in regional environmental ordinance, not municipal police. Documented but not blocking.

6. **RAGAS evaluation in hosted environment**: Re-run three-way evaluation in Colab with RAGAS faithfulness, answer_relevancy, context_precision, context_recall. Per-category RAGAS deltas — particularly on multi_hop where rerank shows biggest win — would either confirm or revise this Stage 2 decision.
