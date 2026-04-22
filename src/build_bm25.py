import json
import pickle
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
OUT_PATH = ROOT / "data" / "bm25_index.pkl"


def tokenize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9§]+", " ", text)
    return [t for t in text.split() if t]


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(
            {"bm25": bm25, "chunk_ids": chunk_ids, "chunks": chunks},
            f,
        )

    total_tokens = sum(len(t) for t in tokenized_corpus)
    print(f"Indexed {len(chunks)} chunks, {total_tokens:,} tokens")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
