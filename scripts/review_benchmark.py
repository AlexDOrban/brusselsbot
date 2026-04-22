"""Interactive review harness for the 50-question benchmark.

Reads data/benchmark_questions.json (wrapped as {"questions": [...]}) and
data/benchmark_ground_truth_draft.json. Prompts the reviewer per entry.
Appends one JSONL line per reviewed entry to data/benchmark_review_log.jsonl.

Resumable: on startup, already-reviewed IDs are skipped.
Read-only: never modifies questions.json or the drafts file.
Stdlib only.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
DRAFTS_PATH = ROOT / "data" / "benchmark_ground_truth_draft.json"
CHUNKS_PATH = ROOT / "data" / "chunks.json"
LOG_PATH = ROOT / "data" / "benchmark_review_log.jsonl"

BAR = "═" * 70
THIN = "─" * 70

FLAG_CODES = {
    "qu": "question_unclear",
    "ws": "wrong_expected_sources",
    "df": "draft_unfaithful",
    "wc": "wrong_category",
    "ok": "ok",
}


def parse_flags(raw):
    raw = raw.strip().lower()
    if not raw:
        return ["ok"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    flags = []
    unknown = []
    for p in parts:
        if p in FLAG_CODES:
            flags.append(FLAG_CODES[p])
        else:
            unknown.append(p)
    if unknown:
        print(f"  (ignored unknown codes: {unknown}; known: {list(FLAG_CODES)})")
    return flags or ["ok"]


def derive_action(flags):
    if "draft_unfaithful" in flags:
        return "rewrite"
    non_ok = [f for f in flags if f != "ok"]
    if non_ok:
        return "edit"
    return "ok"


def short(s, n=200):
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def load_already_reviewed():
    reviewed = set()
    if not LOG_PATH.exists():
        return reviewed
    for line in LOG_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            reviewed.add(entry["id"])
        except Exception:
            pass
    return reviewed


def append_log(entry):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        return "q"


def render(q, draft_rec, chunks_by_id, idx, total):
    print()
    print(BAR)
    print(f"[{idx}/{total}]  {q['id']}  |  {q['category']}  |  {q['language']}")
    print(THIN)
    print("QUESTION:")
    print(f"  {q['question']}")
    print()
    exp = q.get("expected_sources", [])
    print(f"EXPECTED_SOURCES ({len(exp)} chunks):")
    if not exp:
        print("  (none — out-of-scope)")
    for i, cid in enumerate(exp, 1):
        c = chunks_by_id.get(cid)
        if c is None:
            print(f"  [{i}] {cid}  (MISSING from chunks.json)")
            continue
        md = c["metadata"]
        head = f"{md.get('source')}, art={md.get('article_number')}, lang={md.get('language')}, page={md.get('page')}"
        print(f"  [{i}] {cid}  ({head})")
        print(f"      {short(c['text'], 200)}")
    print()
    print("GROUND_TRUTH_DRAFT:")
    draft = draft_rec.get("ground_truth_answer_draft", "") if draft_rec else ""
    if not draft:
        print("  (empty)")
    else:
        for line in draft.splitlines():
            print(f"  {line}")
    print()
    print(BAR)
    print("Flag codes: qu=question unclear  ws=wrong expected_sources")
    print("            df=draft unfaithful  wc=wrong category  ok=ok (default)")
    print("Commands (valid at any prompt): q=quit & save   s=skip (no log)")


def main():
    if not QUESTIONS_PATH.exists():
        sys.exit(f"missing {QUESTIONS_PATH}")
    if not DRAFTS_PATH.exists():
        sys.exit(f"missing {DRAFTS_PATH}")
    if not CHUNKS_PATH.exists():
        sys.exit(f"missing {CHUNKS_PATH}")

    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    drafts = {d["id"]: d for d in json.loads(DRAFTS_PATH.read_text())["drafts"]}
    chunks = json.loads(CHUNKS_PATH.read_text())
    chunks_by_id = {f"chunk_{i}": c for i, c in enumerate(chunks)}

    reviewed = load_already_reviewed()
    total = len(questions)
    done = len(reviewed)
    flagged = 0
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("action") in ("edit", "rewrite"):
                    flagged += 1
            except Exception:
                pass

    first_unreviewed = next(
        (i for i, q in enumerate(questions, 1) if q["id"] not in reviewed),
        None,
    )
    print(f"Benchmark review — {done}/{total} reviewed, {flagged} flagged.")
    if first_unreviewed is None:
        print("All entries already reviewed. Run scripts/review_summary.py.")
        return
    print(f"Resuming from entry {first_unreviewed} ({questions[first_unreviewed-1]['id']}).")

    for i, q in enumerate(questions, 1):
        if q["id"] in reviewed:
            continue
        render(q, drafts.get(q["id"]), chunks_by_id, i, total)

        raw_flags = ask("Flags (comma-sep, e.g. qu,ws or blank=ok): ")
        if raw_flags.strip().lower() == "q":
            print("Quit — progress saved.")
            return
        if raw_flags.strip().lower() == "s":
            print(f"  skipped {q['id']} (no log entry written)")
            continue
        flags = parse_flags(raw_flags)

        notes = ask("Notes: ")
        if notes.strip().lower() == "q":
            print("Quit — progress saved (current entry NOT logged).")
            return

        action_in = ask("[Enter]=next  [s]=skip (discard)  [q]=quit & save: ")
        cmd = action_in.strip().lower()
        if cmd == "s":
            print(f"  skipped {q['id']} (no log entry written)")
            continue

        entry = {
            "id": q["id"],
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "flags": flags,
            "notes": notes.strip(),
            "action": derive_action(flags),
        }
        append_log(entry)
        reviewed.add(q["id"])

        if cmd == "q":
            print("Quit — progress saved.")
            return

    print(f"\nDone. {len(reviewed)}/{total} entries in {LOG_PATH.name}.")


if __name__ == "__main__":
    main()
