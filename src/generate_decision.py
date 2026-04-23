"""Generate data/stage2_decision_n50.md from the three N=50 benchmark JSONs.

Retrieval-metrics-only (no RAGAS). Implements:
- Threshold re-anchoring to Config C at N=50 (handles both up- and
  down-revisions from the original N=10 thresholds).
- Decision matrix with simpler-wins-ties rule (MRR delta threshold 0.02)
  and per-category B-favor gap override (>0.10).
- Per-category breakdown, OOS correctness, Q7 + B-regression diagnostic.

Usage:
  venv/bin/python src/generate_decision.py     # regenerate decision doc
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

FILES = {
    "C": DATA / "benchmark_2C_baseline_n50.json",
    "A": DATA / "benchmark_2A_rerank_only_n50.json",
    "B": DATA / "benchmark_2B_hybrid_rerank_n50.json",
}

N10_GATES = {"hit_at_1": 0.889, "hit_at_3": 1.000, "mrr": 0.944}
# Back-compat alias — tests/test_decision_gates.py imports GATES.
GATES = N10_GATES
GATE_TOLERANCE = 5e-4
REANCHOR_DELTA = 0.02
MRR_TIE_THRESHOLD = 0.02
CATEGORY_GAP_THRESHOLD = 0.10

SLUIKSTORT_ID = "Q3_sluikstort_nl"
Q7_ID = "Q7_trash_sidewalk"
CATEGORIES = ["in_scope_fr", "in_scope_nl", "in_scope_en",
              "out_of_scope", "multi_hop", "cross_lingual"]


def load_all():
    return {k: json.loads(p.read_text()) for k, p in FILES.items()}


def gate_pass(value, threshold):
    return value >= threshold - GATE_TOLERANCE


def rank_str(rec):
    if rec is None:
        return "n/a"
    if rec.get("category") == "out_of_scope":
        return "OOS"
    r = rec.get("first_gold_rank")
    return str(r) if r is not None else ">10"


def by_id(doc):
    return {r["question_id"]: r for r in doc["per_question"]}


def winner(recs_by_cfg):
    ranks = {}
    for k, r in recs_by_cfg.items():
        if r is None or r.get("category") == "out_of_scope":
            continue
        rr = r.get("first_gold_rank")
        if rr is not None:
            ranks[k] = rr
    if not ranks:
        return "—"
    best = min(ranks.values())
    winners = [k for k, r in ranks.items() if r == best]
    return "=".join(winners) if len(winners) > 1 else winners[0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()
    docs = load_all()
    per_q = {k: by_id(v) for k, v in docs.items()}
    aggs = {k: v["aggregate"] for k, v in docs.items()}
    per_cat = {k: v["per_category"] for k, v in docs.items()}

    # --- threshold re-anchoring to Config C at N=50 ---
    # Baseline sets the floor: A/B must beat or match C on the harder benchmark.
    # Re-anchor symmetrically (both raise and lower). Only report significant
    # changes (|delta| > REANCHOR_DELTA) in the decision doc.
    gates = {}
    anchor_notes = []
    for metric in ("hit_at_1", "hit_at_3", "mrr"):
        c_val = aggs["C"][metric]
        old = N10_GATES[metric]
        gates[metric] = round(c_val, 3)
        if abs(c_val - old) > REANCHOR_DELTA:
            direction = "raised" if c_val > old else "lowered"
            anchor_notes.append(
                f"{metric}: {old:.3f} → {gates[metric]:.3f} "
                f"({direction} to match Config C at N=50 = {c_val:.3f}; "
                f"delta {c_val - old:+.3f})"
            )

    # --- gate checks ---
    gate_results = {k: {m: gate_pass(aggs[k][m], t) for m, t in gates.items()}
                    for k in ("C", "A", "B")}
    # sluikstort top-5
    slk = {k: per_q[k].get(SLUIKSTORT_ID) for k in ("C", "A", "B")}
    slk_top5 = {}
    for k, r in slk.items():
        if r is None or r.get("category") == "out_of_scope":
            slk_top5[k] = False
            continue
        rank = r.get("first_gold_rank")
        slk_top5[k] = bool(rank is not None and rank <= 5)

    def all_pass(k):
        return all(gate_results[k].values()) and slk_top5[k]

    a_pass = all_pass("A")
    b_pass = all_pass("B")
    mrr_a = aggs["A"]["mrr"]
    mrr_b = aggs["B"]["mrr"]
    mrr_delta = mrr_b - mrr_a

    # --- per-category Hit@1 gap in B's favor ---
    cat_gap_b_favor = []
    for cat in CATEGORIES:
        if cat == "out_of_scope":
            continue
        hb = per_cat["B"].get(cat, {}).get("hit_at_1", 0.0)
        hc = per_cat["C"].get(cat, {}).get("hit_at_1", 0.0)
        gap = hb - hc
        if gap > CATEGORY_GAP_THRESHOLD:
            cat_gap_b_favor.append((cat, gap, hb, hc))

    # --- decision ---
    if a_pass and b_pass:
        if abs(mrr_delta) < MRR_TIE_THRESHOLD and not cat_gap_b_favor:
            decision = "REVERT hybrid, keep reranker"
            reason = (f"A and B both clear Hit@K gates. |MRR(B) − MRR(A)| = "
                      f"{abs(mrr_delta):.4f} < {MRR_TIE_THRESHOLD}. No per-category "
                      f"Hit@1 gap > {CATEGORY_GAP_THRESHOLD} in B's favor. Simpler "
                      f"stack wins ties — no answer-quality signal available to "
                      f"override (RAGAS scoped out).")
        elif cat_gap_b_favor:
            decision = "KEEP hybrid, keep reranker"
            reason = (f"A and B both pass, but B shows per-category Hit@1 gap "
                      f"> {CATEGORY_GAP_THRESHOLD}: "
                      f"{', '.join(f'{c}(+{g:.2f})' for c,g,_,_ in cat_gap_b_favor)}.")
        elif mrr_delta >= MRR_TIE_THRESHOLD:
            decision = "KEEP hybrid, keep reranker"
            reason = f"A and B both pass. MRR(B) − MRR(A) = {mrr_delta:+.4f} ≥ {MRR_TIE_THRESHOLD}."
        else:
            decision = "REVERT hybrid, keep reranker"
            reason = f"A and B both pass. MRR(A) − MRR(B) = {-mrr_delta:+.4f}. A is the simpler win."
    elif b_pass and not a_pass:
        decision = "KEEP hybrid, keep reranker"
        reason = ("Only B passes. Per N=10 finding, BM25 contributes candidate-pool "
                  "diversity that the reranker depends on.")
    elif a_pass and not b_pass:
        decision = "REVERT hybrid, keep reranker"
        reason = "Only A passes. BM25 fusion is net-negative."
    else:
        decision = "STOP — investigate"
        reason = "Neither A nor B clears the acceptance gates."

    # --- Q7 + B-regressions ---
    q7_ranks = {k: per_q[k].get(Q7_ID) for k in ("C", "A", "B")}
    b_regressions = []
    for qid, rec_c in per_q["C"].items():
        rec_b = per_q["B"].get(qid)
        if rec_c is None or rec_b is None:
            continue
        if rec_c.get("category") == "out_of_scope":
            continue
        rc = rec_c.get("first_gold_rank")
        rb = rec_b.get("first_gold_rank")
        # regression = B rank worse than C (higher number = worse; None = miss)
        c_val = rc if rc is not None else 999
        b_val = rb if rb is not None else 999
        if b_val > c_val:
            b_regressions.append((qid, rc, rb))

    # ---- build document ----
    L = []
    ap = L.append

    # Section 1
    ap("# Stage 2 Decision — N=50 Three-way Evaluation (retrieval-only)")
    ap("")
    ap("## 1. Summary")
    ap("")
    ap(f"**Decision: {decision}**")
    ap("")
    ap(f"{reason}")
    ap("")
    ap(f"Retrieval-only evaluation across 50 questions in 6 categories "
       f"(in_scope_fr=10, in_scope_nl=8, in_scope_en=10, out_of_scope=7, "
       f"multi_hop=10, cross_lingual=5). RAGAS answer-quality metrics scoped out "
       f"(see §8). MRR(C)={aggs['C']['mrr']:.3f}, MRR(A)={aggs['A']['mrr']:.3f}, "
       f"MRR(B)={aggs['B']['mrr']:.3f}.")
    ap("")

    # Section 2
    ap("## 2. Per-question ranks (6 per-category sub-tables)")
    ap("")
    for cat in CATEGORIES:
        cat_recs = [r for r in docs["C"]["per_question"] if r["category"] == cat]
        if not cat_recs:
            continue
        ap(f"### {cat}  (n={len(cat_recs)})")
        ap("")
        ap("| Question ID | C | A | B | Winner |")
        ap("|-------------|---|---|---|--------|")
        for rc in cat_recs:
            qid = rc["question_id"]
            recs = {k: per_q[k].get(qid) for k in ("C", "A", "B")}
            ap(f"| {qid} | {rank_str(recs['C'])} | {rank_str(recs['A'])} | "
               f"{rank_str(recs['B'])} | {winner(recs)} |")
        ap("")
    ap("> Rank = 1-based position of first gold chunk in top-10. "
       "`>10` = not retrieved. `OOS` = out-of-scope (no gold).")
    ap("")

    # Section 3
    ap("## 3. Aggregate gate checks")
    ap("")
    ap("**Thresholds re-anchored to Config C at N=50** (baseline sets the floor; "
       "A/B must beat or match C on the harder benchmark).")
    ap("")
    if anchor_notes:
        ap("Significant changes from the original N=10 thresholds:")
        for n in anchor_notes:
            ap(f"- {n}")
    else:
        ap("All re-anchored thresholds within ±0.02 of the original N=10 values.")
    ap("")
    ap("| Metric | Threshold | C | A | B |")
    ap("|--------|-----------|---|---|---|")
    for m, t in gates.items():
        label = {"hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "mrr": "MRR"}[m]
        row = [f"{label} ≥ {t:.3f}", f"{t:.3f}"]
        for k in ("C", "A", "B"):
            v = aggs[k][m]
            mark = "✅" if gate_results[k][m] else "❌"
            row.append(f"{v:.3f} {mark}")
        ap("| " + " | ".join(row) + " |")
    row = ["sluikstort in top-5", "—"]
    for k in ("C", "A", "B"):
        row.append("✅" if slk_top5[k] else "❌")
    ap("| " + " | ".join(row) + " |")
    ap("")
    ap(f"- Config A all-gates: **{'PASS' if a_pass else 'FAIL'}**")
    ap(f"- Config B all-gates: **{'PASS' if b_pass else 'FAIL'}**")
    ap(f"- MRR(B) − MRR(A) = {mrr_delta:+.4f}")
    ap("")

    # Section 4
    ap("## 4. Per-category Hit@1 breakdown")
    ap("")
    ap("| Category | n | C Hit@1 | A Hit@1 | B Hit@1 | B−C |")
    ap("|----------|---|---------|---------|---------|-----|")
    for cat in CATEGORIES:
        if cat == "out_of_scope":
            continue
        pc_c = per_cat["C"].get(cat, {})
        pc_a = per_cat["A"].get(cat, {})
        pc_b = per_cat["B"].get(cat, {})
        if not pc_c:
            continue
        hc = pc_c.get("hit_at_1", 0.0)
        ha = pc_a.get("hit_at_1", 0.0)
        hb = pc_b.get("hit_at_1", 0.0)
        delta = hb - hc
        ap(f"| {cat} | {pc_c.get('n',0)} | {hc:.3f} | {ha:.3f} | {hb:.3f} | {delta:+.3f} |")
    ap("")
    if cat_gap_b_favor:
        ap(f"**Per-category gaps > {CATEGORY_GAP_THRESHOLD} in B's favor:**")
        for c, g, hb, hc in cat_gap_b_favor:
            ap(f"- {c}: B={hb:.3f}, C={hc:.3f}, gap=+{g:.3f}")
    else:
        ap("No per-category Hit@1 gap > 0.10 in B's favor.")
    ap("")

    # Section 5
    ap("## 5. OOS correctness (confidence caveat)")
    ap("")
    ap("| Config | correctly caveated | incorrectly answered |")
    ap("|--------|--------------------|----------------------|")
    for k in ("C", "A", "B"):
        pc = per_cat[k].get("out_of_scope", {})
        n_ok = pc.get("n_correctly_caveated", 0)
        n_bad = pc.get("n_incorrectly_answered", 0)
        ap(f"| {k} | {n_ok}/{n_ok+n_bad} | {n_bad}/{n_ok+n_bad} |")
    ap("")
    ap("> Caveat signal: dense_distance > 0.60 (Config C) or rerank_score < 0.30 (A/B).")
    ap("")

    # Section 6
    ap("## 6. Decision matrix application")
    ap("")
    ap(f"- A passes all gates: **{a_pass}**")
    ap(f"- B passes all gates: **{b_pass}**")
    ap(f"- |MRR(B) − MRR(A)| = {abs(mrr_delta):.4f}  (tie threshold {MRR_TIE_THRESHOLD})")
    ap(f"- Per-category B-favor gap > {CATEGORY_GAP_THRESHOLD}: "
       f"{'YES — ' + ', '.join(c for c,_,_,_ in cat_gap_b_favor) if cat_gap_b_favor else 'NO'}")
    ap("")
    ap(f"→ **{decision}**")
    ap("")
    ap(reason)
    ap("")

    # Section 7
    ap("## 7. Q7_trash_sidewalk + B-regression diagnostic")
    ap("")
    if all(r is not None for r in q7_ranks.values()):
        ap(f"Q7_trash_sidewalk ranks: C={rank_str(q7_ranks['C'])}, "
           f"A={rank_str(q7_ranks['A'])}, B={rank_str(q7_ranks['B'])}")
    else:
        ap("Q7_trash_sidewalk: not found in all configs.")
    ap("")
    ap(f"**All in-scope questions where B's rank is worse than C's:** "
       f"{len(b_regressions)}")
    if b_regressions:
        ap("")
        ap("| Question ID | C rank | B rank |")
        ap("|-------------|--------|--------|")
        for qid, rc, rb in b_regressions:
            rc_s = str(rc) if rc is not None else ">10"
            rb_s = str(rb) if rb is not None else ">10"
            ap(f"| {qid} | {rc_s} | {rb_s} |")
    ap("")
    if len(b_regressions) <= 1:
        ap("Pattern read: isolated outlier, likely noise.")
    elif len(b_regressions) <= 3:
        ap("Pattern read: small cluster. Inspect before dismissing.")
    else:
        ap("Pattern read: **systematic regression under B**. Investigate.")
    ap("")

    # Section 8
    ap("## 8. Methodology note — RAGAS scoped out")
    ap("")
    ap("This evaluation uses retrieval metrics (Hit@K, MRR) only. RAGAS-based "
       "answer-quality metrics (faithfulness, answer_relevancy, context_precision, "
       "context_recall) were scoped out of this iteration due to local hardware "
       "constraints — running RAGAS with adequate concurrency requires more "
       "RAM than the development machine reliably provides.")
    ap("")
    ap("The decision matrix is therefore weighted toward retrieval-side "
       "evidence. Where retrieval metrics are tied between configurations, "
       "\"simpler stack wins\" because we lack the answer-quality signal that "
       "would otherwise justify additional architectural complexity.")
    ap("")
    ap("Future work: re-run the three-way evaluation in a hosted environment "
       "(Google Colab or equivalent) with RAGAS faithfulness and "
       "answer_relevancy added. Per-category RAGAS deltas — particularly on "
       "multi_hop and cross_lingual where rerank is most likely to earn its "
       "place — would either confirm or revise the keep/revert decision "
       "documented here.")
    ap("")

    out = DATA / "stage2_decision_n50.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
