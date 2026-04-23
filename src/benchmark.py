"""Three-way N=50 retrieval benchmark.

Loads benchmark_questions.json (50 entries across 6 categories), runs each
of the 3 configs (C dense-only, A rerank-only, B hybrid+rerank), writes
per-question + aggregate + per-category metrics to
data/benchmark_2{C,A,B}_*_n50.json.

Retrieval metrics only: Hit@1/3/5, MRR, OOS correctly_caveated.

RAGAS answer-quality evaluation was scoped out of this iteration due to
local hardware constraints (see data/stage2_decision_n50.md §8). The
--judge flag is retained with a single `none` choice as a forward-compat
hook; reintroducing `haiku/sonnet/opus` + a ragas_judge module would be
the minimal path to re-enable RAGAS in a hosted environment.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import hybrid_retrieve, is_low_confidence

ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
TOP_K = 10
SLUIKSTORT_ID = "Q3_sluikstort_nl"

JUDGE_CHOICES = ("none",)

CONFIGS = [
    {"label": "C", "hybrid": False, "rerank": False, "out": "benchmark_2C_baseline_n50.json"},
    {"label": "A", "hybrid": False, "rerank": True,  "out": "benchmark_2A_rerank_only_n50.json"},
    {"label": "B", "hybrid": True,  "rerank": True,  "out": "benchmark_2B_hybrid_rerank_n50.json"},
]


def first_gold_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for i, cid in enumerate(retrieved_ids):
        if cid in gold_set:
            return i + 1
    return None


def run_config(cfg, questions):
    print("\n" + "#" * 70)
    print(f"# CONFIG {cfg['label']}  hybrid={cfg['hybrid']}  rerank={cfg['rerank']}  N={len(questions)}")
    print("#" * 70)

    per_question = []
    per_cat = defaultdict(lambda: {
        "n": 0, "hits_at_1": 0, "hits_at_3": 0, "hits_at_5": 0, "rrs": [],
        "oos_correct": 0, "oos_incorrect": 0,
    })
    in_scope_hits = {1: 0, 3: 0, 5: 0}
    rrs = []
    n_in_scope = 0
    n_oos_correct = 0

    for q in questions:
        qid = q["id"]
        question = q["question"]
        category = q["category"]
        gold = q.get("expected_sources", [])
        is_oos = category == "out_of_scope"

        results = hybrid_retrieve(
            question, k=TOP_K, n_candidates=20,
            hybrid=cfg["hybrid"], rerank=cfg["rerank"],
        )
        retrieved = [r["chunk_id"] for r in results]

        record = {
            "question_id": qid,
            "question_text": question,
            "category": category,
            "language": q["language"],
            "retrieved_chunk_ids": retrieved[:5],
            "ground_truth_chunk_ids": gold,
        }

        if is_oos:
            caveated = is_low_confidence(results)
            record.update({
                "first_gold_rank": None,
                "hit_at_1": None, "hit_at_3": None, "hit_at_5": None,
                "reciprocal_rank": None,
                "correctly_caveated": bool(caveated),
            })
            per_cat[category]["n"] += 1
            if caveated:
                per_cat[category]["oos_correct"] += 1
                n_oos_correct += 1
            else:
                per_cat[category]["oos_incorrect"] += 1
            print(f"[{qid}] OOS  caveated={caveated}")
        else:
            rank = first_gold_rank(retrieved, gold)
            h1 = bool(rank is not None and rank <= 1)
            h3 = bool(rank is not None and rank <= 3)
            h5 = bool(rank is not None and rank <= 5)
            rr = 1.0 / rank if rank else 0.0
            record.update({
                "first_gold_rank": rank,
                "hit_at_1": h1, "hit_at_3": h3, "hit_at_5": h5,
                "reciprocal_rank": rr,
            })
            n_in_scope += 1
            if h1: in_scope_hits[1] += 1
            if h3: in_scope_hits[3] += 1
            if h5: in_scope_hits[5] += 1
            rrs.append(rr)

            pc = per_cat[category]
            pc["n"] += 1
            if h1: pc["hits_at_1"] += 1
            if h3: pc["hits_at_3"] += 1
            if h5: pc["hits_at_5"] += 1
            pc["rrs"].append(rr)

            print(f"[{qid}] rank={rank}  h@1={int(h1)} h@3={int(h3)} h@5={int(h5)}  RR={rr:.3f}")

        if qid == SLUIKSTORT_ID:
            record["sluikstort_top5"] = bool(record.get("first_gold_rank") is not None
                                             and record["first_gold_rank"] <= 5)

        per_question.append(record)

    n_oos = sum(1 for q in questions if q["category"] == "out_of_scope")
    aggregate = {
        "hit_at_1": in_scope_hits[1] / n_in_scope if n_in_scope else 0.0,
        "hit_at_3": in_scope_hits[3] / n_in_scope if n_in_scope else 0.0,
        "hit_at_5": in_scope_hits[5] / n_in_scope if n_in_scope else 0.0,
        "mrr": sum(rrs) / n_in_scope if n_in_scope else 0.0,
        "n_in_scope_evaluated": n_in_scope,
        "n_oos_correctly_caveated": n_oos_correct,
        "n_oos_total": n_oos,
    }

    per_category_out = {}
    for cat, pc in per_cat.items():
        n = pc["n"]
        if cat == "out_of_scope":
            per_category_out[cat] = {
                "n": n,
                "n_correctly_caveated": pc["oos_correct"],
                "n_incorrectly_answered": pc["oos_incorrect"],
            }
        else:
            per_category_out[cat] = {
                "n": n,
                "hit_at_1": pc["hits_at_1"] / n if n else 0.0,
                "hit_at_3": pc["hits_at_3"] / n if n else 0.0,
                "hit_at_5": pc["hits_at_5"] / n if n else 0.0,
                "mrr": sum(pc["rrs"]) / n if n else 0.0,
            }

    doc = {
        "config": cfg["label"],
        "hybrid": cfg["hybrid"],
        "rerank": cfg["rerank"],
        "n_questions": len(questions),
        "aggregate": aggregate,
        "per_category": per_category_out,
        "per_question": per_question,
    }

    out_path = ROOT / "data" / cfg["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"AGGREGATE — Config {cfg['label']}")
    print("=" * 70)
    print(f"in-scope N={n_in_scope}  "
          f"Hit@1={aggregate['hit_at_1']:.3f}  Hit@3={aggregate['hit_at_3']:.3f}  "
          f"Hit@5={aggregate['hit_at_5']:.3f}  MRR={aggregate['mrr']:.3f}")
    print(f"OOS correctly caveated: {n_oos_correct}/{n_oos}")
    print(f"Saved: {out_path}")


def run(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judge", default="none", choices=JUDGE_CHOICES,
        help=(
            "RAGAS judge model. Only `none` (retrieval-only, €0) is currently "
            "wired — haiku/sonnet/opus were scoped out of Stage 2 per "
            "data/stage2_decision_n50.md §8. Flag retained as forward-compat "
            "hook for hosted-env RAGAS reintroduction."
        ),
    )
    parser.add_argument("--smoke-id", default=None,
                        help="Run a single question id under Config C only (debug).")
    parser.add_argument("--config", default=None, choices=[c["label"] for c in CONFIGS],
                        help="Run only one config (C/A/B).")
    args = parser.parse_args(argv)

    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]

    if args.smoke_id:
        qs = [q for q in questions if q["id"] == args.smoke_id]
        if not qs:
            sys.exit(f"smoke-id {args.smoke_id} not found in benchmark_questions.json")
        cfg = next(c for c in CONFIGS if c["label"] == "C")
        smoke_cfg = dict(cfg)
        smoke_cfg["out"] = f"benchmark_smoke_{args.smoke_id}.json"
        run_config(smoke_cfg, qs)
        return

    configs = [c for c in CONFIGS if (args.config is None or c["label"] == args.config)]
    for cfg in configs:
        run_config(cfg, questions)


if __name__ == "__main__":
    run()
