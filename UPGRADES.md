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
| 2. Hybrid BM25 | planned | — | — | Target: recover the `sluikstort` miss without regressing Hit@3. |
| 3. bge-reranker-v2-m3 | planned | — | — | Target: Hit@1 → 1.000 on current benchmark. |
| 4. Cross-lingual query expansion | planned | — | — | Needs benchmark expansion to detect gains (current set already saturates Hit@3). |
| 5. Agentic retrieval loop | planned | — | — | Latency trade-off; measure turns + cost. |
| 6. RAGAS expansion | planned | — | — | Adds faithfulness / context-precision / answer-relevance. |
