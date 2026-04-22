import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from build_bm25 import tokenize

ROOT = Path(__file__).parent.parent
CHROMA_PATH = ROOT / "chroma_db"
BM25_PATH = ROOT / "data" / "bm25_index.pkl"
COLLECTION_NAME = "brussels_regulations"
EMBED_MODEL = "text-embedding-3-small"
RRF_K = 60
CONFIDENCE_THRESHOLD = 0.60
RERANK_CONFIDENCE_THRESHOLD = 0.30

_state = {}


def _init():
    if _state:
        return
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    _state["openai"] = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _state["collection"] = chroma.get_collection(COLLECTION_NAME)
    if not BM25_PATH.exists():
        raise RuntimeError(f"BM25 index missing at {BM25_PATH}. Run: python src/build_bm25.py")
    with BM25_PATH.open("rb") as f:
        data = pickle.load(f)
    _state["bm25"] = data["bm25"]
    _state["chunk_ids"] = data["chunk_ids"]
    _state["chunks"] = data["chunks"]
    _state["id_to_idx"] = {cid: i for i, cid in enumerate(data["chunk_ids"])}


def _dense_query(query, n_candidates):
    q_emb = _state["openai"].embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    dense_res = _state["collection"].query(
        query_embeddings=[q_emb],
        n_results=n_candidates,
    )
    dense_ids = dense_res["ids"][0]
    dense_dists = dense_res["distances"][0]
    return dense_ids, dense_dists


def _bm25_query(query, n_candidates):
    tokens = tokenize(query)
    scores = _state["bm25"].get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_idx = [i for i in ranked[:n_candidates] if scores[i] > 0]
    bm25_ids = [_state["chunk_ids"][i] for i in top_idx]
    bm25_score_map = {cid: float(scores[_state["id_to_idx"][cid]]) for cid in bm25_ids}
    return bm25_ids, bm25_score_map


def _build_result(cid, dense_dist_map, dense_rank, bm25_score_map, bm25_rank, rrf_score):
    idx = _state["id_to_idx"][cid]
    chunk = _state["chunks"][idx]
    return {
        "chunk_id": cid,
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "dense_distance": dense_dist_map.get(cid),
        "dense_rank": dense_rank.get(cid),
        "bm25_score": bm25_score_map.get(cid),
        "bm25_rank": bm25_rank.get(cid),
        "rrf_score": rrf_score,
    }


def hybrid_retrieve(query, k=5, n_candidates=20, hybrid=True, rerank=True):
    _init()

    dense_ids, dense_dists = _dense_query(query, n_candidates)
    dense_rank = {cid: r + 1 for r, cid in enumerate(dense_ids)}
    dense_dist_map = dict(zip(dense_ids, dense_dists))

    if hybrid:
        bm25_ids, bm25_score_map = _bm25_query(query, n_candidates)
        bm25_rank = {cid: r + 1 for r, cid in enumerate(bm25_ids)}

        all_ids = set(dense_ids) | set(bm25_ids)
        fused = []
        for cid in all_ids:
            rrf = 0.0
            if cid in dense_rank:
                rrf += 1.0 / (RRF_K + dense_rank[cid])
            if cid in bm25_rank:
                rrf += 1.0 / (RRF_K + bm25_rank[cid])
            fused.append((cid, rrf))
        fused.sort(key=lambda x: x[1], reverse=True)
    else:
        bm25_score_map = {}
        bm25_rank = {}
        fused = [(cid, 1.0 / (RRF_K + dense_rank[cid])) for cid in dense_ids]

    candidate_count = n_candidates if rerank else k
    candidates = [
        _build_result(cid, dense_dist_map, dense_rank, bm25_score_map, bm25_rank, rrf)
        for cid, rrf in fused[:candidate_count]
    ]

    if rerank:
        from rerank import rerank as rerank_fn
        return rerank_fn(query, candidates, top_k=k)

    return candidates


def confidence_distance(results):
    if not results:
        return None
    return results[0]["dense_distance"]


def rerank_confidence(results):
    if not results:
        return None
    return results[0].get("rerank_score")


def is_low_confidence(results):
    if not results:
        return True
    rs = results[0].get("rerank_score")
    if rs is not None:
        return rs < RERANK_CONFIDENCE_THRESHOLD
    dd = results[0].get("dense_distance")
    if dd is None:
        return True
    return dd > CONFIDENCE_THRESHOLD
