# Stage 2 — Hybrid BM25 + Dense Retrieval Plan

> Branch: `advanced-rag-upgrades`. Plan only — no code in this doc.

## Goal

Add BM25 sparse retrieval next to the existing ChromaDB dense retrieval and fuse both result lists with Reciprocal Rank Fusion. Target: recover keyword-rare terms (`sluikstort`, `Bruxell'Air`) without regressing the dense-only baseline (Hit@1 0.889, MRR 0.944).

## Files to create

| Path | Purpose |
|------|---------|
| `src/retrieval.py` | Single entry point for retrieval. Encapsulates: OpenAI embedding, ChromaDB query, BM25 query, RRF fusion, result dataclass. Both `ask.py` and `app.py` import from here. |
| `src/build_bm25_index.py` | One-shot builder: reads `data/chunks.json`, tokenizes, constructs a BM25 index, persists to disk. Mirrors the role of `embed_chunks.py` for the sparse side. |
| `data/bm25_index/` | Persisted BM25 index directory (via `bm25s.BM25.save()`). Gitignored. |

## Files to modify

| Path | Change |
|------|--------|
| `src/ask.py` | Replace the inline `openai_client.embeddings.create(...)` + `collection.query(...)` block with `retrieval.retrieve(query, top_k=TOP_K)`. Drop direct Chroma/OpenAI imports. |
| `src/app.py` | Same substitution. Preserve the streaming generator shape and the confidence-threshold check (see risk #4 below). |
| `src/benchmark.py` | Parameterize retriever mode: `dense` / `bm25` / `hybrid`. Write a combined results file `data/benchmark_stage1.json` with three runs for comparison. |
| `requirements.txt` | Add `bm25s` (recommended — numpy-backed, fast, has `save`/`load`). `rank_bm25` is an acceptable fallback but requires manual pickling and is ~10× slower on 263 chunks (still fine, but `bm25s` is future-proofing). |
| `.gitignore` | Add `data/bm25_index/`. |
| `UPGRADES.md` | After implementation: fill Stage 2 row in the results table. |

## Where hybrid retrieval plugs in

Today, `ask.py` and `app.py` each do:

```
embed query → collection.query(n_results=5) → build context block → stream answer
```

After the refactor:

```
retrieval.retrieve(query, top_k=5) → build context block → stream answer
```

`retrieval.retrieve` is the only function that touches embeddings, Chroma, or BM25. It returns a list of `RetrievalResult` with both raw scores (dense distance + BM25 score) and the fused RRF score. Callers consume it uniformly — they no longer know which retriever produced a given result.

### `RetrievalResult` shape

```
chunk_id        str            # "chunk_{i}" — same ID as in Chroma
text            str
metadata        dict           # source, language, article_number, section, page
dense_distance  float | None   # cosine distance from Chroma (None if not in dense top-N)
bm25_score      float | None   # raw BM25 score (None if not in BM25 top-N)
dense_rank      int | None     # 1-based rank from dense list
bm25_rank       int | None     # 1-based rank from BM25 list
rrf_score       float          # fused score
```

Keeping the per-retriever diagnostics on each result lets us debug why a chunk ranked where it did, and lets the confidence check in `app.py` keep using the dense distance specifically (not the fused score — see risk #4).

## ID alignment

Both indices must be built from the same `data/chunks.json` in the same order.

- ChromaDB already uses `chunk_{i}` where `i` is the index in `chunks.json`.
- BM25 builder will use the identical convention: `doc_ids = [f"chunk_{i}" for i in range(len(chunks))]`.
- `build_bm25_index.py` writes a `manifest.json` alongside the index with: chunk count, SHA-256 of `chunks.json`, build timestamp.
- `retrieval.py` on load verifies the manifest hash against the current `chunks.json` and refuses to run with a loud error if they diverge. This is the single biggest failure mode (see risk #1) and it needs a hard guard, not a warning.

## Tokenization

Multilingual corpus (FR/NL/EN). Safe default: lowercase + Unicode word split + punctuation strip. **No language-specific stemming** — Porter on FR/NL would be worse than no stemming. Applies identically to queries and documents at both build time and query time, via a single `tokenize(text)` function in `retrieval.py`. Keep the function small and exported so it's easy to test.

Query tokenization must use the same function, same normalization, or BM25 silently scores zero for words that only differ in case.

## Reciprocal Rank Fusion

### Formula

For a document `d` appearing in any of the retriever result lists:

```
RRF(d) = Σ_r  1 / (k + rank_r(d))
```

where:
- `r` ranges over retrievers (here: dense, BM25)
- `rank_r(d)` is the 1-based rank of `d` in retriever `r`'s top-N list
- if `d` is not in retriever `r`'s top-N, that term contributes 0
- `k = 60`

### Why k = 60

From Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods* (SIGIR 2009). They swept `k` across TREC collections and found 60 near-optimal; since then it has become the de-facto default in Elastic, Vespa, LangChain, and LlamaIndex.

The role of `k`: it flattens the contribution of top ranks. With `k=0`, RRF becomes pure `1/rank` and the #1 result dominates (score 1.0 vs #2 at 0.5 — a 2× gap). With `k=60`, rank 1 contributes `1/61 ≈ 0.0164` and rank 2 contributes `1/62 ≈ 0.0161` — a 2% gap. That robustness matters when one retriever is confident and the other isn't: a single high-confidence list can't monopolize the fusion.

### Retrieval width

- BM25 top-N = 20
- Dense top-N = 20
- Fuse, then return top-K = 5 (to match current `TOP_K`)

Overlap is common, so the fused pool is usually 20–30 unique chunks. Widening N further costs little (BM25 is microseconds on 263 chunks; dense is already paid for) and gives RRF more material to work with.

## Risks

1. **ID drift between BM25 and Chroma.** If `chunks.json` is regenerated but the BM25 index isn't rebuilt (or vice versa), retrievers return inconsistent chunks. *Mitigation:* SHA-256 of `chunks.json` in the BM25 manifest; `retrieval.py` raises on mismatch. Also add the hash check to `embed_chunks.py` for symmetry.
2. **Tokenizer mismatch.** If query and document tokenization drift apart (e.g., someone adds NFC normalization to one side only), BM25 scores silently collapse. *Mitigation:* single exported `tokenize()` used at both build and query time; one unit-test case with mixed diacritics/case to guard it.
3. **FR/NL sharing a BM25 index promotes the wrong language.** An EN query can exact-match a French cognate and outrank the correct EN chunk via BM25. *Mitigation:* measure on the benchmark. The fix, if it appears, isn't to split the index — it's language-filtering at query time via the query-rewriter's detected language. Don't add that complexity speculatively.
4. **Confidence threshold calibrated on cosine distance.** The 0.60 cutoff in `app.py` is tuned to cosine distance from the dense retriever. RRF scores are on a completely different scale (typically 0.01–0.03 for fused). *Mitigation:* keep the confidence check on the dense distance specifically — `RetrievalResult.dense_distance` of the top fused result (or None-handling if the top fused result wasn't in the dense top-20, which itself is a confidence signal worth warning on).
5. **Latency in the streaming path.** `chat()` in `app.py` does retrieval *inside* a generator, before the first `yield`. BM25 adds ~1–5ms on 263 chunks — negligible, but widening to N=50 or adding a reranker later will matter. *Mitigation:* no action now, flag for Stage 3 (reranker) where it actually bites.
6. **Silent per-retriever regressions masked by fusion.** Hybrid could look flat on aggregate while one retriever degrades on a subset. *Mitigation:* `benchmark.py` runs all three modes (`dense`, `bm25`, `hybrid`) and logs per-mode scores. Required to detect Simpson's-paradox-style hides.
7. **bm25s dependency.** Pulls numpy already present; low risk. If install fails on a platform, falling back to `rank_bm25` is a one-import swap because the retriever module is the only caller.

## Benchmark plan

After implementation, run `benchmark.py` in all three modes and record:

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Notes |
|------|-------|-------|-------|-----|-------|
| dense (baseline) | 0.889 | 1.000 | 1.000 | 0.944 | From `data/benchmark_baseline.json` |
| bm25 | ? | ? | ? | ? | Pure keyword; expect wins on `sluikstort`, losses on paraphrased queries |
| hybrid (RRF, k=60) | ? | ? | ? | ? | Must not regress below dense on any metric |

Hard acceptance criteria for merging Stage 2:
- Hit@3 ≥ 1.000 (no regression on the easy cases)
- Hit@1 ≥ 0.889 (no regression)
- At least one benchmark question that missed in dense-only now hits at rank 1 in hybrid, OR the BM25-exclusive "sluikstort → Article 25" case surfaces in top-5 of hybrid

If hybrid doesn't clear these, the plan is wrong, not the implementation — stop and revisit before shipping.
