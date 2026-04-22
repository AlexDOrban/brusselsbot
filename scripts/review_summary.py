"""Summarize data/benchmark_review_log.jsonl — what's reviewed, what's flagged,
what needs EDIT vs REWRITE vs OK.

Read-only. Stdlib only.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
LOG_PATH = ROOT / "data" / "benchmark_review_log.jsonl"


def main():
    if not QUESTIONS_PATH.exists():
        sys.exit(f"missing {QUESTIONS_PATH}")
    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    total = len(questions)

    entries = []
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception as e:
                print(f"warn: skipped bad log line: {e}")

    by_id = {}
    for e in entries:
        by_id[e["id"]] = e

    reviewed_ids = set(by_id)
    remaining = [q["id"] for q in questions if q["id"] not in reviewed_ids]

    flag_counts = Counter()
    for e in by_id.values():
        for f in e.get("flags", []):
            flag_counts[f] += 1

    action_buckets = defaultdict(list)
    flagged = []
    for q in questions:
        e = by_id.get(q["id"])
        if not e:
            continue
        action_buckets[e["action"]].append((q["id"], e))
        if e["action"] in ("edit", "rewrite"):
            flagged.append((q["id"], e))

    print(f"Review progress: {len(reviewed_ids)}/{total} reviewed, {len(remaining)} remaining")
    print(f"Flagged (edit or rewrite): {len(flagged)}")
    print()

    print("Flag counts:")
    if not flag_counts:
        print("  (none)")
    for f, n in sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {f:25s} {n}")
    print()

    for bucket, title in (("rewrite", "REWRITE (major)"),
                          ("edit",    "EDIT    (minor)"),
                          ("ok",      "OK")):
        items = action_buckets.get(bucket, [])
        print(f"{title}: {len(items)}")
        for qid, e in items:
            non_ok = [f for f in e["flags"] if f != "ok"]
            tag = ",".join(non_ok) if non_ok else "ok"
            note = e.get("notes", "").strip()
            note_s = f" — {note}" if note else ""
            print(f"  {qid}  [{tag}]{note_s}")
        print()

    if remaining:
        print(f"Remaining ({len(remaining)}):")
        for qid in remaining:
            print(f"  {qid}")


if __name__ == "__main__":
    main()
