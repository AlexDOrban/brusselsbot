"""Apply expected_sources fixes from data/source_fixes.json.

For each Q_ID in source_fixes.json:
  1. Update expected_sources in data/benchmark_questions.json
     (preserves the {"questions": [...]} wrapper).
  2. Re-draft ground_truth_answer for that ID via Claude Opus, grounded
     ONLY on the new expected_sources chunks.
  3. Replace that entry in data/benchmark_ground_truth_draft.json
     (preserves all other entries).
  4. Append a review-log entry with flags=["sources_fixed"], action="fixed".

Q_IDs may be short (Q12) or full (Q12_trottoir_fr). Short IDs must match
exactly one question by prefix.

Writes files atomically via .tmp + rename. Does NOT touch files for Q_IDs
absent from source_fixes.json.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from draft_ground_truth import SYSTEM, format_chunks, OOS_DRAFT  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from anthropic import Anthropic  # noqa: E402

QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
DRAFTS_PATH = ROOT / "data" / "benchmark_ground_truth_draft.json"
CHUNKS_PATH = ROOT / "data" / "chunks.json"
FIXES_PATH = ROOT / "data" / "source_fixes.json"
LOG_PATH = ROOT / "data" / "benchmark_review_log.jsonl"

MODEL = "claude-opus-4-7"


def atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def resolve(short_or_full, questions):
    for q in questions:
        if q["id"] == short_or_full:
            return q
    prefix = short_or_full + "_"
    matches = [q for q in questions if q["id"].startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no match for {short_or_full}")
    raise KeyError(f"ambiguous {short_or_full} → {[m['id'] for m in matches]}")


def redraft(client, q, chunks_by_id):
    if q["category"] == "out_of_scope":
        return OOS_DRAFT.get(q["language"], OOS_DRAFT["en"])
    if not q["expected_sources"]:
        return ""
    # format_chunks expects BY_ID lookup; use same signature
    # draft_ground_truth.format_chunks uses its own module-level BY_ID,
    # so call a local copy to be safe if chunks.json drifts.
    blocks = []
    for cid in q["expected_sources"]:
        c = chunks_by_id[cid]
        md = c["metadata"]
        blocks.append(
            f"[{cid}] source={md.get('source')} lang={md.get('language')} "
            f"article={md.get('article_number')} page={md.get('page')}\n{c['text']}"
        )
    context = "\n\n".join(blocks)
    user = (
        f"Question ({q['language']}): {q['question']}\n\n"
        f"Source chunks:\n\n{context}\n\n"
        f"Draft the ground-truth answer."
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=600, system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def main():
    if not FIXES_PATH.exists():
        sys.exit(f"missing {FIXES_PATH} — create it with {{Q_ID: [chunk_id, ...]}}")
    fixes = json.loads(FIXES_PATH.read_text())
    if not isinstance(fixes, dict) or not fixes:
        sys.exit(f"{FIXES_PATH} must be a non-empty object")

    questions_doc = json.loads(QUESTIONS_PATH.read_text())
    questions = questions_doc["questions"]
    drafts_doc = json.loads(DRAFTS_PATH.read_text())
    drafts = drafts_doc["drafts"]
    drafts_by_id = {d["id"]: d for d in drafts}

    chunks = json.loads(CHUNKS_PATH.read_text())
    chunks_by_id = {f"chunk_{i}": c for i, c in enumerate(chunks)}
    valid_ids = set(chunks_by_id)

    # Validate all fix inputs before calling Opus.
    resolved_fixes = []
    for key, new_sources in fixes.items():
        q = resolve(key, questions)
        if not isinstance(new_sources, list):
            sys.exit(f"{key}: expected_sources must be a list")
        for cid in new_sources:
            if cid not in valid_ids:
                sys.exit(f"{key}: unknown chunk_id {cid}")
        resolved_fixes.append((key, q, new_sources))

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in .env")
    client = Anthropic(api_key=api_key)

    log_lines = []
    for key, q, new_sources in resolved_fixes:
        old = q["expected_sources"]
        q["expected_sources"] = new_sources
        print(f"[{q['id']}] sources {old} → {new_sources}")

        try:
            new_answer = redraft(client, q, chunks_by_id)
        except Exception as e:
            sys.exit(f"Opus draft failed for {q['id']}: {type(e).__name__}: {e}")

        rec = drafts_by_id.get(q["id"])
        if rec is None:
            rec = {
                "id": q["id"],
                "category": q["category"],
                "language": q["language"],
                "question": q["question"],
                "expected_sources": new_sources,
                "ground_truth_answer_draft": new_answer,
                "error": None,
            }
            drafts.append(rec)
        else:
            rec["expected_sources"] = new_sources
            rec["ground_truth_answer_draft"] = new_answer
            rec["error"] = None
        drafts_by_id[q["id"]] = rec

        log_lines.append(json.dumps({
            "id": q["id"],
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "flags": ["sources_fixed"],
            "notes": f"sources {old} → {new_sources}; ground truth redrafted",
            "action": "fixed",
        }, ensure_ascii=False))

    atomic_write(QUESTIONS_PATH, json.dumps(questions_doc, indent=2, ensure_ascii=False) + "\n")
    atomic_write(DRAFTS_PATH, json.dumps(drafts_doc, indent=2, ensure_ascii=False) + "\n")

    with LOG_PATH.open("a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nUpdated: {QUESTIONS_PATH.name}, {DRAFTS_PATH.name}")
    print(f"Appended {len(log_lines)} log entries to {LOG_PATH.name}")


if __name__ == "__main__":
    main()
