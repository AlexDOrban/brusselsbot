import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

FILES = {
    "C": DATA / "benchmark_2C_baseline.json",
    "A": DATA / "benchmark_2A_rerank_only.json",
    "B": DATA / "benchmark_2B_hybrid_rerank.json",
}

GATES = {
    "hit_at_1": 0.889,
    "hit_at_3": 1.000,
    "mrr": 0.944,
}

SLUIKSTORT_ID = "Q3_sluikstort_nl"


def load_all():
    return {k: json.loads(p.read_text()) for k, p in FILES.items()}


GATE_TOLERANCE = 5e-4


def gate_pass(agg, key, threshold):
    return agg[key] >= threshold - GATE_TOLERANCE


def sluikstort_rec(doc):
    for r in doc["per_question"]:
        if r["id"] == SLUIKSTORT_ID:
            return r
    return None


def rank_str(rec):
    if rec is None:
        return "n/a"
    if rec.get("out_of_scope"):
        return "OOS"
    r = rec.get("first_gold_rank")
    return str(r) if r is not None else ">10"


def winner(rc, ra, rb):
    ranks = {}
    for label, rec in (("C", rc), ("A", ra), ("B", rb)):
        if rec is None or rec.get("out_of_scope"):
            continue
        r = rec.get("first_gold_rank")
        if r is not None:
            ranks[label] = r
    if not ranks:
        return "—"
    best = min(ranks.values())
    winners = [lbl for lbl, r in ranks.items() if r == best]
    return "=".join(winners) if len(winners) > 1 else winners[0]


def main():
    docs = load_all()

    qids = [r["id"] for r in docs["C"]["per_question"]]
    rows = []
    for qid in qids:
        recs = {k: next((r for r in docs[k]["per_question"] if r["id"] == qid), None)
                for k in ("C", "A", "B")}
        question = recs["C"]["question"]
        rows.append({
            "id": qid,
            "question": question,
            "C": rank_str(recs["C"]),
            "A": rank_str(recs["A"]),
            "B": rank_str(recs["B"]),
            "winner": winner(recs["C"], recs["A"], recs["B"]),
        })

    aggs = {k: docs[k]["aggregate"] for k in ("C", "A", "B")}

    def check(k):
        return {g: gate_pass(aggs[k], g, t) for g, t in GATES.items()}
    gates = {k: check(k) for k in ("C", "A", "B")}

    slk = {k: sluikstort_rec(docs[k]) for k in ("C", "A", "B")}
    slk_top5 = {
        k: bool(r and not r.get("out_of_scope") and r.get("first_gold_rank") is not None
                and r["first_gold_rank"] <= 5)
        for k, r in slk.items()
    }

    def all_gates_pass(k):
        return all(gates[k].values()) and slk_top5[k]

    a_pass = all_gates_pass("A")
    b_pass = all_gates_pass("B")
    mrr_a = aggs["A"]["mrr"]
    mrr_b = aggs["B"]["mrr"]
    delta = mrr_b - mrr_a

    if a_pass and b_pass and delta < 0.02:
        decision = "REVERT hybrid, keep reranker"
        reason = (f"A and B both pass. MRR delta (B − A) = {delta:+.4f} < 0.02. "
                  "Simpler stack wins ties — BM25 is not earning its complexity.")
    elif a_pass and b_pass and delta >= 0.02:
        decision = "KEEP hybrid, keep reranker"
        reason = (f"A and B both pass. MRR delta (B − A) = {delta:+.4f} ≥ 0.02. "
                  "BM25 contribution is measurable; retain hybrid.")
    elif a_pass and not b_pass:
        decision = "REVERT hybrid, keep reranker"
        reason = "Only A passes. Reranker alone clears the gates; BM25 fusion is net-negative."
    elif b_pass and not a_pass:
        decision = "KEEP hybrid, keep reranker"
        reason = ("Only B passes. Reranker needed BM25's candidate pool to recover. "
                  "Note: hybrid is earning its place by diversifying candidates, "
                  "not by raw rank fusion.")
    else:
        decision = "STOP — do NOT commit"
        reason = "Neither A nor B passes the acceptance gates. Investigate before proceeding."

    lines = []
    lines.append("# Stage 2 Decision — Reranker + Hybrid 3-way Evaluation")
    lines.append("")
    lines.append(f"**Decision: {decision}**")
    lines.append("")
    lines.append(f"_{reason}_")
    lines.append("")
    lines.append("## Configs")
    lines.append("")
    lines.append("| Config | hybrid | rerank | File |")
    lines.append("|--------|--------|--------|------|")
    lines.append(f"| C (baseline) | False | False | `{FILES['C'].name}` |")
    lines.append(f"| A (rerank only) | False | True | `{FILES['A'].name}` |")
    lines.append(f"| B (full stack) | True  | True  | `{FILES['B'].name}` |")
    lines.append("")
    lines.append("## Per-question ranks (gold source rank in top-10)")
    lines.append("")
    lines.append("| # | Question ID | C (baseline) | A (rerank) | B (hybrid+rerank) | Winner |")
    lines.append("|---|-------------|--------------|------------|-------------------|--------|")
    for i, r in enumerate(rows, start=1):
        lines.append(f"| {i} | {r['id']} | {r['C']} | {r['A']} | {r['B']} | {r['winner']} |")
    lines.append("")
    lines.append("> Rank values: integer rank of first gold chunk, `>10` if missed, `OOS` = out-of-scope (no gold).")
    lines.append("")
    lines.append("## Aggregate gate checks")
    lines.append("")
    lines.append("| Metric | Threshold | C | A | B |")
    lines.append("|--------|-----------|---|---|---|")
    for g, t in GATES.items():
        label = {"hit_at_1": "Hit@1 ≥ 0.889", "hit_at_3": "Hit@3 ≥ 1.000", "mrr": "MRR ≥ 0.944"}[g]
        row = [label, f"{t:.3f}"]
        for k in ("C", "A", "B"):
            v = aggs[k][g]
            mark = "✅" if gates[k][g] else "❌"
            row.append(f"{v:.3f} {mark}")
        lines.append("| " + " | ".join(row) + " |")
    row = ["sluikstort in top-5", "—"]
    for k in ("C", "A", "B"):
        row.append(f"{'✅' if slk_top5[k] else '❌'}")
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("### All-gates verdict")
    lines.append("")
    lines.append(f"- Config A (rerank only): **{'PASS' if a_pass else 'FAIL'}**")
    lines.append(f"- Config B (hybrid + rerank): **{'PASS' if b_pass else 'FAIL'}**")
    lines.append(f"- MRR(B) − MRR(A) = {delta:+.4f}")
    lines.append("")
    lines.append("## Raw aggregates")
    lines.append("")
    lines.append("| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |")
    lines.append("|--------|-------|-------|-------|--------|-----|")
    for k in ("C", "A", "B"):
        a = aggs[k]
        lines.append(f"| {k} | {a['hit_at_1']:.3f} | {a['hit_at_3']:.3f} | "
                     f"{a['hit_at_5']:.3f} | {a['hit_at_10']:.3f} | {a['mrr']:.3f} |")
    lines.append("")
    lines.append("## Q_sluikstort deep-dive (risk #3 check)")
    lines.append("")
    lines.append(f"Question: *{slk['C']['question']}*")
    lines.append("")
    lines.append("| Config | first_gold_rank | top-3 chunk IDs | top-3 (source, article, lang) |")
    lines.append("|--------|-----------------|-----------------|-------------------------------|")
    for k in ("C", "A", "B"):
        r = slk[k]
        rank = r.get("first_gold_rank") if r else None
        rank_s = str(rank) if rank is not None else ">10"
        top3 = r.get("top3", []) if r else []
        ids = ", ".join(t["chunk_id"] for t in top3)
        meta = " / ".join(f"{t['source']}·art{t['article_number']}·{t['language']}" for t in top3)
        lines.append(f"| {k} | {rank_s} | {ids} | {meta} |")
    lines.append("")
    stage1_path = DATA / "benchmark_stage1_hybrid.json"
    if stage1_path.exists():
        stage1 = json.loads(stage1_path.read_text())
        s1_slk = next((r for r in stage1["per_question"] if r["question"].startswith("Welke boete")), None)
        if s1_slk:
            s1_rank = s1_slk.get("first_gold_rank")
            lines.append(f"**Reference — Stage 1 hybrid-only (no rerank):** first_gold_rank = "
                         f"{s1_rank if s1_rank is not None else '>10'}.")
            lines.append("")
    rb = slk["B"].get("first_gold_rank") if slk["B"] else None
    ra = slk["A"].get("first_gold_rank") if slk["A"] else None
    lines.append("**Risk #3 read:** the concern was that the reranker, being a semantic cross-encoder, "
                 "would demote the BM25-surfaced sluikstort chunks and undo the Stage 1 win.")
    if rb is not None and ra is not None:
        if rb <= ra:
            lines.append(f"- Config B (hybrid+rerank) lands the gold at rank {rb}, "
                         f"vs. A (rerank only) at rank {ra}. **Risk #3 did NOT materialize** — "
                         "the reranker kept (or improved on) BM25's candidates.")
        else:
            lines.append(f"- Config B ranks gold at {rb}; A ranks it at {ra}. "
                         "**Risk #3 partially materialized** — hybrid's BM25 candidates were not "
                         "preserved by the reranker. Worth inspecting the reranker scores for this question.")
    lines.append("")
    lines.append("## Decision applied")
    lines.append("")
    lines.append(f"- A passes: **{a_pass}**")
    lines.append(f"- B passes: **{b_pass}**")
    lines.append(f"- MRR(B) − MRR(A) = {delta:+.4f} (threshold 0.02)")
    lines.append("")
    lines.append(f"→ **{decision}**")
    lines.append("")
    lines.append(f"_{reason}_")
    lines.append("")

    out = DATA / "stage2_decision.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
