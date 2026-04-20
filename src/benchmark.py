import json
import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

ROOT = Path(__file__).parent.parent
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "brussels_regulations"
EMBED_MODEL = "text-embedding-3-small"
RESULTS_PATH = ROOT / "data" / "benchmark_baseline.json"
TOP_K = 10

BENCHMARK = [
    {
        "question": "What are the rules about noise at night in Brussels?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 88},
        ],
        "notes": "Easy EN single-article lookup, FR/NL sources. Baseline."
    },
    {
        "question": "Mag ik 's nachts muziek maken op straat?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 88, "language": "nl"},
        ],
        "notes": "NL query, NL source should rank first."
    },
    {
        "question": "Welke boete riskeer ik voor sluikstort?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 4},
            {"source": "police_regs", "article_number": 14},
            {"source": "police_regs", "article_number": 25},
            {"source": "police_regs", "article_number": 29},
            {"source": "police_regs", "article_number": 63},
        ],
        "notes": "Multi-article synthesis. Illegal dumping is distributed across 5 articles. Any in top K = hit."
    },
    {
        "question": "How has the Brussels LEZ affected low-income households?",
        "gold_sources": [
            {"source": "ieep"},
        ],
        "notes": "Domain separation — should pull IEEP, NOT police_regs."
    },
    {
        "question": "What is the Bruxell'Air bonus?",
        "gold_sources": [
            {"source": "ieep"},
        ],
        "notes": "Specific IEEP content. Should rank IEEP chunks high."
    },
    {
        "question": "Quelles sont les règles pour l'installation d'une grue?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 54, "language": "fr"},
        ],
        "notes": "FR query, FR source, specific article. Should be rank 1."
    },
    {
        "question": "Can I put a trash container on the sidewalk?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 28},
            {"source": "police_regs", "article_number": 29},
            {"source": "police_regs", "article_number": 30},
        ],
        "notes": "EN query, FR/NL waste-collection articles. Cross-lang + multi-article."
    },
    {
        "question": "What electric vehicle incentives are available for business fleets?",
        "gold_sources": [
            {"source": "leaseplan"},
        ],
        "notes": "Leaseplan is the only source covering business EV fleets."
    },
    {
        "question": "What is the fine for speeding on the E40 highway?",
        "gold_sources": [],
        "notes": "Out-of-scope. Highway speeding is federal law, not in our corpus. Correct behavior: no hits in top K. If retriever returns high-confidence results, it's hallucinating relevance."
    },
    {
        "question": "Can my neighbour's alarm ring all night?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 97},
        ],
        "notes": "EN query, natural phrasing. Gold is Article 97 (building alarm systems)."
    },
]


def chunk_matches_gold(md, gold):
    for k, v in gold.items():
        if md.get(k) != v:
            return False
    return True


def first_gold_rank(metas, golds):
    for i, md in enumerate(metas):
        for g in golds:
            if chunk_matches_gold(md, g):
                return i + 1
    return None


def run():
    load_dotenv(ROOT / ".env")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma.get_collection(COLLECTION_NAME)

    per_question = []
    in_scope_hits = {1: [], 3: [], 5: [], 10: []}
    mrrs = []
    oos_reports = []

    for entry in BENCHMARK:
        q = entry["question"]
        golds = entry["gold_sources"]
        emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
        res = collection.query(query_embeddings=[emb], n_results=TOP_K)
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        print("=" * 70)
        print(f"Q: {q}")
        print(f"Notes: {entry['notes']}")

        top3 = []
        for i in range(min(3, len(metas))):
            md = metas[i]
            top3.append({
                "rank": i + 1,
                "source": md.get("source"),
                "article_number": md.get("article_number"),
                "language": md.get("language"),
                "distance": dists[i],
            })
            print(f"  rank {i + 1}  src={md.get('source')}  "
                  f"art={md.get('article_number')}  lang={md.get('language')}  "
                  f"dist={dists[i]:.4f}")

        if not golds:
            oos_reports.append({
                "question": q,
                "top1_distance": dists[0] if dists else None,
                "top1_source": metas[0].get("source") if metas else None,
                "top1_article": metas[0].get("article_number") if metas else None,
            })
            print(f"  [out-of-scope] top1 distance = {dists[0]:.4f}")
            per_question.append({
                "question": q,
                "gold_sources": golds,
                "out_of_scope": True,
                "top3": top3,
                "top1_distance": dists[0] if dists else None,
            })
            continue

        rank = first_gold_rank(metas, golds)
        hit = {n: (rank is not None and rank <= n) for n in (1, 3, 5, 10)}
        rr = 1.0 / rank if rank else 0.0
        for n in (1, 3, 5, 10):
            in_scope_hits[n].append(1 if hit[n] else 0)
        mrrs.append(rr)

        print(f"  first_gold_rank={rank}  "
              f"Hit@1={int(hit[1])}  Hit@3={int(hit[3])}  "
              f"Hit@5={int(hit[5])}  Hit@10={int(hit[10])}  RR={rr:.4f}")

        per_question.append({
            "question": q,
            "gold_sources": golds,
            "out_of_scope": False,
            "top3": top3,
            "first_gold_rank": rank,
            "hit_at": hit,
            "reciprocal_rank": rr,
        })

    n = len(in_scope_hits[1])
    agg = {
        "in_scope_count": n,
        "hit_at_1": sum(in_scope_hits[1]) / n if n else 0,
        "hit_at_3": sum(in_scope_hits[3]) / n if n else 0,
        "hit_at_5": sum(in_scope_hits[5]) / n if n else 0,
        "hit_at_10": sum(in_scope_hits[10]) / n if n else 0,
        "mrr": sum(mrrs) / n if n else 0,
        "out_of_scope": oos_reports,
    }

    print("\n" + "=" * 70)
    print("AGGREGATE (in-scope only)")
    print("=" * 70)
    print(f"In-scope questions: {n}")
    print(f"Hit@1:  {agg['hit_at_1']:.3f}")
    print(f"Hit@3:  {agg['hit_at_3']:.3f}")
    print(f"Hit@5:  {agg['hit_at_5']:.3f}")
    print(f"Hit@10: {agg['hit_at_10']:.3f}")
    print(f"MRR:    {agg['mrr']:.3f}")

    print("\nOUT-OF-SCOPE")
    for r in oos_reports:
        print(f"  Q: {r['question']}")
        print(f"    top1 src={r['top1_source']} art={r['top1_article']} "
              f"distance={r['top1_distance']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"per_question": per_question, "aggregate": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    run()
