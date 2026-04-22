"""Retrieval-assisted helper for fixing flagged expected_sources.

For each question ID passed on the command line (short IDs like Q12 or full
like Q12_trottoir_fr), runs dense-only retrieval (Config C: hybrid=False,
rerank=False) at top_k=15 and prints the current (flagged) sources plus the
top-15 dense candidates. Reviewer picks the new chunk_ids manually.

Defaults to the 13 ws-flagged IDs from the Stage 5 review if none are passed.

Output: stdout AND data/fix_candidates.txt.

Does NOT modify benchmark_questions.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieval import hybrid_retrieve  # noqa: E402

QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
CHUNKS_PATH = ROOT / "data" / "chunks.json"
OUT_PATH = ROOT / "data" / "fix_candidates.txt"

DEFAULT_FLAGGED = [
    "Q12", "Q13", "Q18", "Q19", "Q22", "Q23", "Q25",
    "Q26", "Q39", "Q42", "Q44", "Q47", "Q49",
]

TOP_K = 15
BAR = "═" * 70
THIN = "─" * 70


def resolve_ids(questions, requested):
    by_full = {q["id"]: q for q in questions}
    resolved = []
    for r in requested:
        if r in by_full:
            resolved.append(by_full[r])
            continue
        prefix = r + "_"
        matches = [q for q in questions if q["id"].startswith(prefix)]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) == 0:
            print(f"warn: no match for {r}", file=sys.stderr)
        else:
            print(f"warn: ambiguous prefix {r} → {[m['id'] for m in matches]}", file=sys.stderr)
    return resolved


def short(s, n=200):
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def head(md):
    return f"{md.get('source')}, art={md.get('article_number')}, lang={md.get('language')}, page={md.get('page')}"


def render_entry(out, q, chunks_by_id):
    lines = []
    lines.append("")
    lines.append(BAR)
    lines.append(f"FIX  {q['id']}  ({q['category']} | {q['language']})")
    lines.append(f"Question: {q['question']}")
    lines.append("")
    lines.append("Current expected_sources (FLAGGED WRONG):")
    if not q.get("expected_sources"):
        lines.append("  (empty)")
    for cid in q.get("expected_sources", []):
        c = chunks_by_id.get(cid)
        if c is None:
            lines.append(f"  {cid} — (missing)")
            continue
        lines.append(f"  {cid}  ({head(c['metadata'])})")
        lines.append(f"      {short(c['text'], 120)}")
    lines.append("")
    lines.append(f"TOP-{TOP_K} DENSE CANDIDATES (lower dist = more similar):")
    results = hybrid_retrieve(q["question"], k=TOP_K, n_candidates=TOP_K,
                              hybrid=False, rerank=False)
    current_set = set(q.get("expected_sources", []))
    for i, r in enumerate(results, 1):
        cid = r["chunk_id"]
        md = r["metadata"]
        dd = r.get("dense_distance")
        dd_s = f"{dd:.4f}" if dd is not None else "n/a"
        marker = "← current" if cid in current_set else ""
        lines.append(f" [{i:2d}] {cid:<10}  dist={dd_s}  ({head(md)})  {marker}")
        lines.append(f"      {short(r['text'], 200)}")
    lines.append("")
    lines.append("SUGGESTED_NEW_SOURCES (leave blank — reviewer will choose):")
    lines.append("  []")
    block = "\n".join(lines)
    print(block)
    out.append(block)


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FLAGGED
    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    chunks = json.loads(CHUNKS_PATH.read_text())
    chunks_by_id = {f"chunk_{i}": c for i, c in enumerate(chunks)}

    resolved = resolve_ids(questions, requested)
    print(f"Helper — {len(resolved)} flagged entries, top-{TOP_K} dense each.\n")

    out_blocks = []
    for q in resolved:
        render_entry(out_blocks, q, chunks_by_id)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out_blocks) + "\n", encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
