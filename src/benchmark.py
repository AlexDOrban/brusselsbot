import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import hybrid_retrieve

ROOT = Path(__file__).parent.parent
TOP_K = 10

BENCHMARK = [
    {
        "id": "Q1_noise_en",
        "question": "What are the rules about noise at night in Brussels?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 88},
        ],
        "notes": "Easy EN single-article lookup, FR/NL sources. Baseline."
    },
    {
        "id": "Q2_muziek_nl",
        "question": "Mag ik 's nachts muziek maken op straat?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 88, "language": "nl"},
        ],
        "notes": "NL query, NL source should rank first."
    },
    {
        "id": "Q3_sluikstort_nl",
        "question": "Welke boete riskeer ik voor sluikstort?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 4},
            {"source": "police_regs", "article_number": 14},
            {"source": "police_regs", "article_number": 25},
            {"source": "police_regs", "article_number": 29},
            {"source": "police_regs", "article_number": 63},
        ],
        "notes": "Multi-article synthesis. Illegal dumping distributed across 5 articles. Any in top K = hit.",
        "is_sluikstort": True,
    },
    {
        "id": "Q4_lez_social",
        "question": "How has the Brussels LEZ affected low-income households?",
        "gold_sources": [
            {"source": "ieep"},
        ],
        "notes": "Domain separation — should pull IEEP, NOT police_regs."
    },
    {
        "id": "Q5_bruxellair",
        "question": "What is the Bruxell'Air bonus?",
        "gold_sources": [
            {"source": "ieep"},
        ],
        "notes": "Specific IEEP content. Should rank IEEP chunks high."
    },
    {
        "id": "Q6_grue_fr",
        "question": "Quelles sont les règles pour l'installation d'une grue?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 54, "language": "fr"},
        ],
        "notes": "FR query, FR source, specific article. Should be rank 1."
    },
    {
        "id": "Q7_trash_sidewalk",
        "question": "Can I put a trash container on the sidewalk?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 28},
            {"source": "police_regs", "article_number": 29},
            {"source": "police_regs", "article_number": 30},
        ],
        "notes": "EN query, FR/NL waste-collection articles. Cross-lang + multi-article."
    },
    {
        "id": "Q8_ev_fleet",
        "question": "What electric vehicle incentives are available for business fleets?",
        "gold_sources": [
            {"source": "leaseplan"},
        ],
        "notes": "Leaseplan is the only source covering business EV fleets."
    },
    {
        "id": "Q9_e40_oos",
        "question": "What is the fine for speeding on the E40 highway?",
        "gold_sources": [],
        "notes": "Out-of-scope. Highway speeding is federal law, not in our corpus."
    },
    {
        "id": "Q10_alarm",
        "question": "Can my neighbour's alarm ring all night?",
        "gold_sources": [
            {"source": "police_regs", "article_number": 97},
        ],
        "notes": "EN query, natural phrasing. Gold is Article 97 (building alarm systems)."
    },
]


CONFIGS = [
    {
        "label": "2A_rerank_only",
        "hybrid": False,
        "rerank": True,
        "out": "benchmark_2A_rerank_only.json",
    },
    {
        "label": "2B_hybrid_rerank",
        "hybrid": True,
        "rerank": True,
        "out": "benchmark_2B_hybrid_rerank.json",
    },
    {
        "label": "2C_baseline",
        "hybrid": False,
        "rerank": False,
        "out": "benchmark_2C_baseline.json",
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


def run_config(cfg):
    print("\n" + "#" * 70)
    print(f"# CONFIG {cfg['label']}  hybrid={cfg['hybrid']}  rerank={cfg['rerank']}")
    print("#" * 70)

    per_question = []
    in_scope_hits = {1: [], 3: [], 5: [], 10: []}
    mrrs = []
    oos_reports = []

    for entry in BENCHMARK:
        q = entry["question"]
        qid = entry["id"]
        golds = entry["gold_sources"]

        results = hybrid_retrieve(
            q, k=TOP_K, n_candidates=20,
            hybrid=cfg["hybrid"], rerank=cfg["rerank"],
        )
        metas = [r["metadata"] for r in results]
        chunk_ids = [r["chunk_id"] for r in results]

        print("=" * 70)
        print(f"[{qid}] {q}")

        top3 = []
        for i in range(min(3, len(metas))):
            md = metas[i]
            top3.append({
                "rank": i + 1,
                "chunk_id": chunk_ids[i],
                "source": md.get("source"),
                "article_number": md.get("article_number"),
                "language": md.get("language"),
            })
            print(f"  rank {i+1}  {chunk_ids[i]}  src={md.get('source')} "
                  f"art={md.get('article_number')} lang={md.get('language')}")

        record = {
            "id": qid,
            "question": q,
            "gold_sources": golds,
            "retrieved_chunk_ids": chunk_ids,
            "top3": top3,
        }

        if not golds:
            record["out_of_scope"] = True
            oos_reports.append({"id": qid, "question": q, "top1_chunk_id": chunk_ids[0] if chunk_ids else None})
            print("  [out-of-scope]")
            per_question.append(record)
            continue

        rank = first_gold_rank(metas, golds)
        hit = {n: (rank is not None and rank <= n) for n in (1, 3, 5, 10)}
        rr = 1.0 / rank if rank else 0.0
        for n in (1, 3, 5, 10):
            in_scope_hits[n].append(1 if hit[n] else 0)
        mrrs.append(rr)

        record["out_of_scope"] = False
        record["first_gold_rank"] = rank
        record["hit_at"] = hit
        record["reciprocal_rank"] = rr

        if entry.get("is_sluikstort"):
            record["sluikstort_top5"] = bool(rank is not None and rank <= 5)

        print(f"  first_gold_rank={rank}  Hit@1={int(hit[1])} Hit@3={int(hit[3])} "
              f"Hit@5={int(hit[5])} Hit@10={int(hit[10])}  RR={rr:.4f}")

        per_question.append(record)

    n = len(in_scope_hits[1])
    agg = {
        "config": cfg["label"],
        "hybrid": cfg["hybrid"],
        "rerank": cfg["rerank"],
        "in_scope_count": n,
        "hit_at_1": sum(in_scope_hits[1]) / n if n else 0,
        "hit_at_3": sum(in_scope_hits[3]) / n if n else 0,
        "hit_at_5": sum(in_scope_hits[5]) / n if n else 0,
        "hit_at_10": sum(in_scope_hits[10]) / n if n else 0,
        "mrr": sum(mrrs) / n if n else 0,
        "out_of_scope": oos_reports,
    }

    print("\n" + "=" * 70)
    print(f"AGGREGATE — {cfg['label']}")
    print("=" * 70)
    print(f"Hit@1={agg['hit_at_1']:.3f}  Hit@3={agg['hit_at_3']:.3f}  "
          f"Hit@5={agg['hit_at_5']:.3f}  Hit@10={agg['hit_at_10']:.3f}  "
          f"MRR={agg['mrr']:.3f}")

    out_path = ROOT / "data" / cfg["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"per_question": per_question, "aggregate": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {out_path}")


def run():
    for cfg in CONFIGS:
        run_config(cfg)


if __name__ == "__main__":
    run()
