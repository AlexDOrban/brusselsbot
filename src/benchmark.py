"""Three-way N=50 retrieval + RAGAS benchmark.

Loads benchmark_questions.json (50 entries across 6 categories), runs each of
the 3 configs (C dense-only, A rerank-only, B hybrid+rerank), and writes
per-question + aggregate + per-category metrics to
data/benchmark_2{C,A,B}_*_n50.json.

Retrieval metrics: Hit@1/3/5, MRR, OOS correctly_caveated.
Optional RAGAS metrics: faithfulness, answer_relevancy, context_precision,
context_recall. Skipped on out_of_scope questions (all four null).

Judge model: Haiku 4.5 by default (~€2-3 per full run), Sonnet/Opus opt-in.
"""
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import hybrid_retrieve, is_low_confidence

# RAGAS wrappers are imported lazily inside run()/score_ragas_batch() only
# when --judge != "none". Keeps the retrieval-only path self-contained and
# free of ragas/langchain dependencies.

JUDGE_CHOICES = ("none", "haiku", "sonnet", "opus")

ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
TOP_K = 10
SLUIKSTORT_ID = "Q3_sluikstort_nl"
GEN_MODEL = "claude-sonnet-4-6"

CONFIGS = [
    {"label": "C", "hybrid": False, "rerank": False, "out": "benchmark_2C_baseline_n50.json"},
    {"label": "A", "hybrid": False, "rerank": True,  "out": "benchmark_2A_rerank_only_n50.json"},
    {"label": "B", "hybrid": True,  "rerank": True,  "out": "benchmark_2B_hybrid_rerank_n50.json"},
]

GEN_SYSTEM = """You are BrusselsBot, answering questions from retrieved context only.

- Answer in the same language as the question. Translate source content if needed.
- If the context does not cover the question, say so plainly — do not invent facts, article numbers, or figures.
- Be concise: 2-5 sentences. Cite inline: "(Article 88, Police Regulations)" / "(Artikel 88, Politieverordening)" / "(IEEP report, page 5)" / "(Leaseplan whitepaper, page 3)".
"""


def first_gold_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for i, cid in enumerate(retrieved_ids):
        if cid in gold_set:
            return i + 1
    return None


def format_context_for_gen(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        md = r["metadata"]
        head = (f"[Source {i}] source={md.get('source')} lang={md.get('language')} "
                f"article={md.get('article_number')} page={md.get('page')}")
        blocks.append(f"{head}\n{r['text']}")
    return "\n\n".join(blocks)


def generate_answer(anthropic_client, question, results):
    """Call Claude Sonnet once on retrieved context. Returns answer string."""
    context = format_context_for_gen(results)
    user = (f"Question: {question}\n\n"
            f"Retrieved context:\n\n{context}\n\n"
            f"Answer using only the retrieved context. Cite sources inline.")
    resp = anthropic_client.messages.create(
        model=GEN_MODEL,
        max_tokens=600,
        system=GEN_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def _load_ragas():
    """Imports heavy RAGAS symbols lazily (suppressing deprecation chatter)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            answer_relevancy, context_precision, context_recall, faithfulness,
        )
    return EvaluationDataset, SingleTurnSample, evaluate, (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )


def score_ragas_batch(samples, judge_model):
    """Run RAGAS on a batch of SingleTurnSample; return list[dict] per-sample scores."""
    if not samples:
        return []
    from ragas_judge import get_ragas_embeddings, get_ragas_judge
    EvaluationDataset, _, evaluate, metrics = _load_ragas()
    judge = get_ragas_judge(judge_model)
    embeddings = get_ragas_embeddings()
    ds = EvaluationDataset(samples=samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluate(
            dataset=ds,
            metrics=list(metrics),
            llm=judge,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )
    df = result.to_pandas()
    out = []
    for _, row in df.iterrows():
        out.append({
            "faithfulness":      _safe_float(row.get("faithfulness")),
            "answer_relevancy":  _safe_float(row.get("answer_relevancy")),
            "context_precision": _safe_float(row.get("context_precision")),
            "context_recall":    _safe_float(row.get("context_recall")),
        })
    return out


def _safe_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def run_config(cfg, questions, *, judge_model, use_ragas, anthropic_client):
    print("\n" + "#" * 70)
    print(f"# CONFIG {cfg['label']}  hybrid={cfg['hybrid']}  rerank={cfg['rerank']}  "
          f"ragas={use_ragas}  N={len(questions)}")
    print("#" * 70)

    per_question = []
    per_cat = defaultdict(lambda: {
        "n": 0, "hits_at_1": 0, "hits_at_3": 0, "hits_at_5": 0, "rrs": [],
        "oos_correct": 0, "oos_incorrect": 0,
        "faithfulness": [], "answer_relevancy": [],
        "context_precision": [], "context_recall": [],
    })
    in_scope_hits = {1: 0, 3: 0, 5: 0}
    rrs = []
    n_in_scope = 0
    n_oos_correct = 0

    # First pass: retrieve + generate (for in-scope) + record retrieval metrics
    ragas_batch = []    # list of SingleTurnSample
    ragas_index = []    # parallel list of per_question indices they map back to
    _, SingleTurnSample, _, _ = _load_ragas() if use_ragas else (None, None, None, None)

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
            "answer": None,
            "ragas": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            },
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
            print(f"[{qid}] OOS  caveated={caveated}  (RAGAS skipped)")
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

            if use_ragas:
                reference = _resolve_ground_truth(q)
                answer = generate_answer(anthropic_client, question, results)
                record["answer"] = answer
                ragas_batch.append(SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=[r["text"] for r in results],
                    reference=reference,
                ))
                ragas_index.append(len(per_question))
            print(f"[{qid}] rank={rank}  h@1={int(h1)} h@3={int(h3)} h@5={int(h5)}  RR={rr:.3f}")

        if qid == SLUIKSTORT_ID:
            record["sluikstort_top5"] = bool(record.get("first_gold_rank") is not None
                                             and record["first_gold_rank"] <= 5)

        per_question.append(record)

    # Second pass: RAGAS batch
    if use_ragas and ragas_batch:
        print(f"\n[ragas] scoring {len(ragas_batch)} in-scope samples with judge={judge_model}...")
        scores = score_ragas_batch(ragas_batch, judge_model)
        for idx, s in zip(ragas_index, scores):
            per_question[idx]["ragas"].update(s)
            cat = per_question[idx]["category"]
            for k in ("faithfulness", "answer_relevancy",
                      "context_precision", "context_recall"):
                if s.get(k) is not None:
                    per_cat[cat][k].append(s[k])

    n_oos = sum(1 for q in questions if q["category"] == "out_of_scope")
    all_ragas = {k: [] for k in ("faithfulness", "answer_relevancy",
                                 "context_precision", "context_recall")}
    for r in per_question:
        rg = r.get("ragas") or {}
        for k in all_ragas:
            if rg.get(k) is not None:
                all_ragas[k].append(rg[k])

    aggregate = {
        "hit_at_1": in_scope_hits[1] / n_in_scope if n_in_scope else 0.0,
        "hit_at_3": in_scope_hits[3] / n_in_scope if n_in_scope else 0.0,
        "hit_at_5": in_scope_hits[5] / n_in_scope if n_in_scope else 0.0,
        "mrr": sum(rrs) / n_in_scope if n_in_scope else 0.0,
        "n_in_scope_evaluated": n_in_scope,
        "n_oos_correctly_caveated": n_oos_correct,
        "n_oos_total": n_oos,
        "faithfulness":      _mean(all_ragas["faithfulness"]),
        "answer_relevancy":  _mean(all_ragas["answer_relevancy"]),
        "context_precision": _mean(all_ragas["context_precision"]),
        "context_recall":    _mean(all_ragas["context_recall"]),
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
                "faithfulness":      _mean(pc["faithfulness"]),
                "answer_relevancy":  _mean(pc["answer_relevancy"]),
                "context_precision": _mean(pc["context_precision"]),
                "context_recall":    _mean(pc["context_recall"]),
            }

    doc = {
        "config": cfg["label"],
        "hybrid": cfg["hybrid"],
        "rerank": cfg["rerank"],
        "ragas_judge": judge_model if use_ragas else None,
        "gen_model": GEN_MODEL if use_ragas else None,
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
    if use_ragas:
        def f(v): return f"{v:.3f}" if v is not None else "n/a"
        print(f"faithfulness={f(aggregate['faithfulness'])}  "
              f"answer_relevancy={f(aggregate['answer_relevancy'])}  "
              f"context_precision={f(aggregate['context_precision'])}  "
              f"context_recall={f(aggregate['context_recall'])}")
    print(f"OOS correctly caveated: {n_oos_correct}/{n_oos}")
    print(f"Saved: {out_path}")


def _resolve_ground_truth(q):
    """Find the human-reviewed ground truth answer for a question, falling
    back to the Opus draft. Used as RAGAS `reference`."""
    try:
        drafts = json.loads((ROOT / "data" / "benchmark_ground_truth_draft.json").read_text())
        for d in drafts.get("drafts", []):
            if d["id"] == q["id"]:
                return d.get("ground_truth_answer_draft") or ""
    except FileNotFoundError:
        pass
    return q.get("ground_truth_answer") or ""


def cost_guardrail(questions, configs, judge_model, assume_yes):
    from ragas_judge import estimate_cost_eur
    n_in_scope = sum(1 for q in questions if q["category"] != "out_of_scope")
    est = estimate_cost_eur(n_in_scope, len(configs), judge_model)
    print(f"\nEstimated RAGAS cost: €{est:.2f}  "
          f"(judge={judge_model}, {n_in_scope} in-scope × {len(configs)} configs)")
    if assume_yes:
        print("--yes set, proceeding without prompt.")
        return True
    try:
        ans = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        ans = ""
    return ans in ("y", "yes")


def run(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judge", default="none", choices=JUDGE_CHOICES,
        help=(
            "RAGAS judge model. Cost tiers for a full 3-config × 50-question run:\n"
            "  none   (default) retrieval-only metrics, no LLM judge calls (€0)\n"
            "  haiku  Haiku 4.5  ~€2-3 per full run\n"
            "  sonnet Sonnet 4.6 ~€6-8 per full run\n"
            "  opus   Opus 4.7   ~€60+ per full run (high budget)"
        ),
    )
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip cost-guardrail prompt.")
    parser.add_argument("--smoke-id", default=None,
                        help="Run a single question id under Config C only (debug).")
    parser.add_argument("--config", default=None, choices=[c["label"] for c in CONFIGS],
                        help="Run only one config (C/A/B).")
    args = parser.parse_args(argv)

    use_ragas = args.judge != "none"
    judge_model = None
    if use_ragas:
        from ragas_judge import resolve_judge_alias
        judge_model = resolve_judge_alias(args.judge)

    load_dotenv(ROOT / ".env")
    anthropic_client = None
    if use_ragas:
        from anthropic import Anthropic
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]

    if args.smoke_id:
        qs = [q for q in questions if q["id"] == args.smoke_id]
        if not qs:
            sys.exit(f"smoke-id {args.smoke_id} not found in benchmark_questions.json")
        cfg = next(c for c in CONFIGS if c["label"] == "C")
        smoke_cfg = dict(cfg)
        smoke_cfg["out"] = f"benchmark_smoke_{args.smoke_id}.json"
        run_config(smoke_cfg, qs, judge_model=judge_model,
                   use_ragas=use_ragas, anthropic_client=anthropic_client)
        return

    configs = [c for c in CONFIGS if (args.config is None or c["label"] == args.config)]

    if use_ragas:
        if not cost_guardrail(questions, configs, judge_model, args.yes):
            print("Aborted by user. No calls made.")
            return

    for cfg in configs:
        run_config(cfg, questions, judge_model=judge_model,
                   use_ragas=use_ragas, anthropic_client=anthropic_client)


if __name__ == "__main__":
    run()
