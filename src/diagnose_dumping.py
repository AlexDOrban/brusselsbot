import json
import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "brussels_regulations"
EMBED_MODEL = "text-embedding-3-small"

KEYWORDS = [
    "sluikstort",
    "dépôt clandestin",
    "dépôt sauvage",
    "abandon de déchets",
    "dumping",
]

VARIANTS = [
    "sluikstort boete",
    "illegal dumping of waste fine",
    "amende dépôt clandestin de déchets",
    "abandon de déchets voie publique",
]


def show_meta_line(md):
    return (
        f"source={md.get('source')} lang={md.get('language')} "
        f"art={md.get('article_number')} page={md.get('page')} "
        f"section={md.get('section','')!r}"
    )


def keyword_search(chunks):
    print("=" * 70)
    print("(a) KEYWORD SEARCH in chunks.json")
    print("=" * 70)
    total = 0
    for kw in KEYWORDS:
        hits = [c for c in chunks if kw.lower() in c["text"].lower()]
        print(f"\n[{kw!r}] — {len(hits)} match(es)")
        for c in hits:
            total += 1
            print("  " + show_meta_line(c["metadata"]))
            print(f"  text[:300]: {c['text'][:300]}")
    print(f"\nTotal keyword matches across terms: {total}")


def top10_same_as_q2(openai_client, collection):
    print("\n" + "=" * 70)
    print("(b) EMBEDDING QUERY — top 10 for original Q2")
    print("=" * 70)
    q = "Welke boete riskeer ik voor sluikstort?"
    emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    res = collection.query(query_embeddings=[emb], n_results=10)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    print(f"Query: {q!r}")
    for rank, (d, md, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(f"\n  rank {rank}  dist={dist:.4f}  lang={md.get('language')}  "
              f"art={md.get('article_number')}  page={md.get('page')}")
        print(f"  text[:150]: {d[:150]}")


def variants_top3(openai_client, collection):
    print("\n" + "=" * 70)
    print("(c) QUERY VARIANTS — top 3 each")
    print("=" * 70)
    for q in VARIANTS:
        emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
        res = collection.query(query_embeddings=[emb], n_results=3)
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        print(f"\nVariant: {q!r}")
        for rank, (d, md, dist) in enumerate(zip(docs, metas, dists), start=1):
            print(f"  rank {rank}  dist={dist:.4f}  lang={md.get('language')}  "
                  f"art={md.get('article_number')}  page={md.get('page')}")
            print(f"  text[:150]: {d[:150]}")


def main():
    load_dotenv(ROOT / ".env")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma.get_collection(COLLECTION_NAME)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    keyword_search(chunks)
    top10_same_as_q2(openai_client, collection)
    variants_top3(openai_client, collection)


if __name__ == "__main__":
    main()
