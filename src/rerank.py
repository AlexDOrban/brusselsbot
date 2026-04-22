import os
import time
from functools import lru_cache
from pathlib import Path

import cohere
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
MODEL = "rerank-v3.5"
MIN_INTERVAL_S = 6.5
_last_call_ts = 0.0


@lru_cache(maxsize=1)
def _get_client():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set in .env")
    return cohere.ClientV2(api_key=api_key)


def rerank(query, candidates, top_k=5):
    if not candidates:
        return []
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < MIN_INTERVAL_S:
        time.sleep(MIN_INTERVAL_S - elapsed)
    client = _get_client()
    docs = [c["text"] for c in candidates]
    _last_call_ts = time.time()
    resp = client.rerank(
        model=MODEL,
        query=query,
        documents=docs,
        top_n=min(top_k, len(candidates)),
    )
    out = []
    for r in resp.results:
        item = dict(candidates[r.index])
        item["rerank_score"] = float(r.relevance_score)
        out.append(item)
    return out
