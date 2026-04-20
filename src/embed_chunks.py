import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "brussels_regulations"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
PRICE_PER_MTOK = 0.02


def sanitize_metadata(md):
    out = {}
    for k, v in md.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        existing = client.get_collection(COLLECTION_NAME)
        count = existing.count()
    except Exception:
        existing = None
        count = 0

    if existing is not None and count > 0:
        print(f"\nWARNING: collection '{COLLECTION_NAME}' already has {count} chunks.")
        print("Re-embedding will delete the existing collection and re-run OpenAI API calls.")
        ans = input("Type 'yes' to continue: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            sys.exit(0)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    openai_client = OpenAI(api_key=api_key)

    total = len(chunks)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    start = time.time()
    total_words = 0

    for b in range(num_batches):
        lo = b * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, total)
        batch = chunks[lo:hi]
        texts = [c["text"] for c in batch]
        ids = [f"chunk_{i}" for i in range(lo, hi)]
        metadatas = [sanitize_metadata(c["metadata"]) for c in batch]

        resp = openai_client.embeddings.create(model=MODEL, input=texts)
        embeddings = [d.embedding for d in resp.data]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_words += sum(len(t.split()) for t in texts)
        print(f"Embedded batch {b + 1}/{num_batches} ({hi} chunks total)")

    elapsed = time.time() - start
    est_tokens = total_words * 1.3
    est_cost = est_tokens / 1_000_000 * PRICE_PER_MTOK

    print("\n" + "=" * 70)
    print(f"Total chunks embedded: {total}")
    print(f"Collection count:      {collection.count()}")
    print(f"Time:                  {elapsed:.2f}s")
    print(f"Estimated tokens:      {est_tokens:,.0f}")
    print(f"Estimated cost:        ${est_cost:.5f}")
    print("=" * 70)

    query = "what are the rules about noise at night"
    print(f"\nSmoke-test query: {query!r}")
    q_resp = openai_client.embeddings.create(model=MODEL, input=[query])
    q_emb = q_resp.data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for i, (doc, md, dist) in enumerate(zip(docs, metas, dists)):
        print(f"\n--- Result {i + 1} (distance={dist:.4f}) ---")
        print(f"source:  {md.get('source')}")
        print(f"lang:    {md.get('language')}")
        print(f"page:    {md.get('page')}")
        if md.get("article_number") not in (None, ""):
            print(f"article: {md.get('article_number')}")
        print(f"text[:200]: {doc[:200]}")


if __name__ == "__main__":
    main()
