# Stage 3 — BGE Reranker (bge-reranker-v2-m3) Plan

> Branch: `advanced-rag-upgrades`. Plan only — no code in this doc.

## Goal

Add a cross-encoder reranking pass on top of hybrid retrieval. Widen hybrid's candidate pool to 20, score every candidate with BAAI/bge-reranker-v2-m3, return the top 5 to the LLM. Target: recover the Hit@1/Hit@3 regressions that pure RRF fusion introduced (IEEP keyword floods, FR/NL cross-language bleed) while keeping the sluikstort win from BM25.

## Why a cross-encoder reranks better than bi-encoders retrieve

**Bi-encoders** (OpenAI `text-embedding-3-small`, the dense side today) encode the query and each document **independently** into fixed vectors, then compare with cosine similarity. This is cheap — you can pre-embed 263 chunks once and query in milliseconds — but the model never sees the query and document together, so it can't reason about how specific query terms align with specific document spans.

**Cross-encoders** (bge-reranker-v2-m3) take the `[query, document]` **pair** as a single input and produce a relevance score through full attention across both. The model can tell that "noise at night in Brussels" maps to Article 88's "prohibition on noise between 22h and 7h" rather than to an IEEP paragraph that happens to contain the words *noise*, *night*, and *Brussels*. Cost: O(N) forward passes per query instead of O(1), which is why cross-encoders are used for reranking a small candidate set, not for first-stage retrieval over a large corpus.

On 20 candidates and CPU, a 568M-param multilingual model like `bge-reranker-v2-m3` scores in well under one second — acceptable in the streaming path before the first token.

## Files to create

| Path | Purpose |
|------|---------|
| `src/rerank.py` | Exports `rerank(query, candidates, top_k=5)`. Owns the lazy model singleton. Single import point for the reranker; `retrieval.py` calls into it when enabled. |

## Files to modify

| Path | Change |
|------|--------|
| `src/retrieval.py` | Add `rerank: bool = True` kwarg to `hybrid_retrieve`. After RRF fusion, if `rerank=True`, pass the top-N fused candidates into `rerank.rerank(query, candidates, top_k=k)` and return the reranker's output. Reranker adds a new `rerank_score` field to each result dict. If `rerank=False`, behavior is unchanged. |
| `src/benchmark.py` | Run three modes: `dense`, `hybrid`, `hybrid+rerank`. Write `data/benchmark_stage2_rerank.json`. Required for isolating whether the reranker actually fixes the Stage 2 regressions. |
| `src/ask.py` / `src/app.py` | No signature change — they already call `hybrid_retrieve(query, k=TOP_K)`. Reranker is on by default. |
| `requirements.txt` | Add `sentence-transformers` and the CPU-only torch wheel (see dependencies section). |
| `.gitignore` | Add `~/.cache/huggingface/` — actually this lives in the user's home dir, not the repo, so not needed. Skip. |
| `UPGRADES.md` | Document 2GB model download on first run in a new "Environment notes" section. Fill Stage 3 row after benchmarking. |

## Rerank function skeleton

```python
# src/rerank.py
from functools import lru_cache
from typing import List, Dict, Any

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_reranker():
    """Lazy singleton. Loads once on first call, ~2GB download on first run,
    then cached under ~/.cache/huggingface/. Subsequent calls return the
    same instance — critical because model init is 2–5 seconds on CPU."""
    ...


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Score every (query, candidate.text) pair with the cross-encoder,
    sort by score descending, return top_k. Each returned dict is the
    original candidate with an added 'rerank_score' float field.
    Preserves all upstream fields (rrf_score, dense_distance, etc.) for
    diagnostics.
    """
    ...
```

```python
# src/retrieval.py — modified signature
def hybrid_retrieve(
    query: str,
    k: int = 5,
    n_candidates: int = 20,
    rerank: bool = True,
) -> List[Dict[str, Any]]:
    ...
```

## Where the model loads and why

**Module-level `@lru_cache(maxsize=1)` singleton inside `rerank.py`.**

- **Not at module import.** Importing `retrieval` (which imports `rerank`) shouldn't trigger a 2–5s model load when a caller only wants to reuse the `tokenize` helper. Lazy means "first `rerank()` call pays the cost, nothing else does."
- **Not in `hybrid_retrieve`.** If we instantiated there, we'd reload per query. `lru_cache(maxsize=1)` gives us a process-wide singleton keyed on `()` — first call loads, all subsequent calls return the same object in microseconds.
- **Why `lru_cache` over a module global with an `if _model is None` check:** semantically identical, but `lru_cache` is thread-safe and one line. The Gradio app is single-threaded today, but the CLI and benchmark invoke from different entry points — centralizing the init in one cached function means we never accidentally double-load.
- **Not in a class.** No state beyond the model handle. A bare function is simpler than a `Reranker` class with one method.

Consequence: the **first** `ask.py` run after install downloads 2GB and takes ~90s; every subsequent invocation of the same Python process has a ~3s warm-up on the first query, then microseconds to retrieve the cached instance per call.

## Dependencies

Add to `requirements.txt`:

```
sentence-transformers==3.3.1
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1
```

- **`sentence-transformers`** ships `CrossEncoder`, the class we use to load `bge-reranker-v2-m3`. It wraps HuggingFace `transformers` and handles tokenization + batched forward passes.
- **CPU-only torch** via the PyTorch index. The full GPU wheel is ~800MB larger and pulls CUDA runtimes this project doesn't need. The `--extra-index-url` line tells pip to look at the PyTorch CPU index first for `torch` while still using PyPI for everything else.
- Pinned versions because sentence-transformers minor versions have changed the `CrossEncoder` API before (e.g., `predict()` signature). Lock to a known-good pair.
- **macOS note:** on Apple Silicon, PyTorch's CPU wheel uses the Accelerate framework and runs fast enough (20 candidates in ~0.3s on an M1). MPS acceleration is available but not worth the complexity for this corpus size.

Verify install:

```
python -c "from sentence_transformers import CrossEncoder; print('ok')"
```

## First-run experience

Add to `UPGRADES.md` under a new **Environment notes** section before the Results table:

```markdown
## Environment notes

**First-run model download (Stage 3+).** The bge-reranker-v2-m3 model is
~2GB and downloads from HuggingFace Hub on first use, cached under
`~/.cache/huggingface/hub/`. Expect ~60–120s on first `python src/ask.py`
or first Gradio session. Subsequent runs load from disk in ~3s.

To pre-download before first interactive use:

    python -c "from src.rerank import _get_reranker; _get_reranker()"

Offline usage: once cached, the model works without network access.
```

Also log a one-line notice the first time the reranker loads in a given process, so users running `ask.py` don't think it hung:

```
[rerank] loading BAAI/bge-reranker-v2-m3 (first call in this process, ~3s warm / ~60s cold)...
```

## Retrieval width

- Hybrid candidate pool: **N = 20** (unchanged).
- Rerank all 20 — cross-encoder scoring cost is linear and 20 × ~40ms = ~0.8s on CPU.
- Return top **k = 5** to the LLM (unchanged).

Widening beyond 20 is Stage 4 territory; no gain expected on 263 chunks.

## Risks

1. **Model download fails or user is offline on first run.** First `ask.py` call errors out mid-question with an opaque `ConnectionError`. *Mitigation:* the one-line loading notice tells the user what's happening; README gets a "first run requires 2GB download" note; the plan documents a manual pre-download command.
2. **Latency in the streaming path.** Stage 2's plan flagged this as "not yet biting." Stage 3 makes it bite: ~0.8s reranker cost before the first token. *Mitigation:* acceptable for interactive use; document in UPGRADES.md. If it becomes a problem, batch-size tuning or FP16 halves it. Do not add complexity unless measured regressions appear.
3. **Reranker undoes the sluikstort win.** If bge-reranker trusts semantic match over keyword match as strongly as bi-encoders do, Article 14/25 could sink again. *Mitigation:* cross-encoders historically handle rare-term queries well because they see the exact token in the document, but this is an empirical claim — `benchmark.py` must run all three modes and we compare deltas per question, not just aggregates.
4. **Toggling rerank on/off inside `hybrid_retrieve` couples the two stages.** A caller who wants hybrid-without-rerank still imports the reranker module (and triggers the lazy import chain). *Mitigation:* `rerank=False` short-circuits before any reranker import is touched — use a local import inside the `if rerank:` branch, not a module-level one.
5. **lru_cache never releases the model.** On a long-running Gradio server this is fine (the model is what you want in memory). For tests, `_get_reranker.cache_clear()` can free it. *Mitigation:* documented, not coded around.
6. **Multilingual behavior on cognates.** bge-reranker-v2-m3 was trained on ~100 languages including FR and NL; it should handle the bilingual corpus without language-filtering. If it doesn't, the fix is query-time language detection — but that's the same risk #3 from Stage 2, don't add it speculatively.
7. **Torch version drift breaks sentence-transformers.** Pinning both packages costs install rigidity for predictability. Accept the trade.

## Benchmark plan

After implementation, run `benchmark.py` and record three rows:

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Notes |
|------|-------|-------|-------|-----|-------|
| dense (baseline) | 0.889 | 1.000 | 1.000 | 0.944 | From `data/benchmark_baseline.json` |
| hybrid (Stage 2) | 0.667 | 0.889 | 0.889 | 0.796 | Regressed on EN queries — IEEP keyword flood |
| hybrid + rerank (Stage 3) | ? | ? | ? | ? | Target: recover Stage 2 losses, keep sluikstort |

### Hard acceptance criteria for merging Stage 3

- **Hit@1 ≥ 0.889** (no regression vs baseline).
- **Hit@3 ≥ 1.000** (no regression vs baseline).
- **sluikstort query still lands a dumping-related Article (14, 25, 29, 30, 63) in top-5** — the Stage 2 win must not be undone.
- **MRR ≥ 0.944** (no regression).

### Soft goal

- Hit@1 → 1.000 on current 10-question benchmark. The reranker's job is to fix ranking ambiguity inside an already-good candidate set.

If hybrid+rerank doesn't clear the hard criteria, the plan is wrong, not the implementation — stop and revisit before shipping. Falling back to dense-only is cheaper than shipping a reranker that doesn't help.
