"""Regression test for scripts/apply_source_fixes.py idempotency guard.

Asserts that when a fix's new_sources equals the question's current
expected_sources (as sets), the Opus redraft call is SKIPPED and the
log entry is flagged accordingly. Regressing this guard would silently
burn Opus budget on every re-run and invalidate downstream comparisons.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def test_idempotency_guard_skips_unchanged_sources(tmp_path, monkeypatch):
    import apply_source_fixes as asf

    # Fake corpus with one chunk that both fixes may reference.
    chunks = [{"text": "body", "metadata": {"source": "police_regs",
                                              "article_number": 1,
                                              "language": "fr", "page": 1}}]
    # Two questions:
    #   - UNCHANGED: fix's sources == current sources → guard must skip
    #   - CHANGED:   fix's sources differ from current sources → redraft must run
    questions = [
        {"id": "Q_UNCHANGED", "question": "same?",
         "language": "fr", "category": "in_scope_fr",
         "expected_sources": ["chunk_0"], "ground_truth_answer": "old"},
        {"id": "Q_CHANGED", "question": "diff?",
         "language": "fr", "category": "in_scope_fr",
         "expected_sources": [], "ground_truth_answer": "old"},
    ]
    drafts = [
        {"id": "Q_UNCHANGED", "category": "in_scope_fr", "language": "fr",
         "question": "same?", "expected_sources": ["chunk_0"],
         "ground_truth_answer_draft": "old draft", "error": None},
        {"id": "Q_CHANGED", "category": "in_scope_fr", "language": "fr",
         "question": "diff?", "expected_sources": [],
         "ground_truth_answer_draft": "old draft", "error": None},
    ]
    fixes = {
        "Q_UNCHANGED": ["chunk_0"],   # same as current — should skip
        "Q_CHANGED":   ["chunk_0"],   # different from [] — should redraft
    }

    q_path = tmp_path / "benchmark_questions.json"
    d_path = tmp_path / "benchmark_ground_truth_draft.json"
    c_path = tmp_path / "chunks.json"
    f_path = tmp_path / "source_fixes.json"
    l_path = tmp_path / "benchmark_review_log.jsonl"

    _write(q_path, {"questions": questions})
    _write(d_path, {"drafts": drafts})
    _write(c_path, chunks)
    _write(f_path, fixes)

    monkeypatch.setattr(asf, "QUESTIONS_PATH", q_path)
    monkeypatch.setattr(asf, "DRAFTS_PATH", d_path)
    monkeypatch.setattr(asf, "CHUNKS_PATH", c_path)
    monkeypatch.setattr(asf, "FIXES_PATH", f_path)
    monkeypatch.setattr(asf, "LOG_PATH", l_path)

    # Skip real Anthropic client construction.
    class _DummyClient: pass
    monkeypatch.setattr(asf, "Anthropic", lambda **kw: _DummyClient())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")

    # Count redraft calls. Stub its return so the CHANGED branch completes.
    redraft_calls = []
    def fake_redraft(client, q, chunks_by_id):
        redraft_calls.append(q["id"])
        return "new draft from stub"
    monkeypatch.setattr(asf, "redraft", fake_redraft)

    asf.main()

    # Guard must skip UNCHANGED, redraft must run for CHANGED.
    assert redraft_calls == ["Q_CHANGED"], \
        f"redraft called on wrong ids: {redraft_calls}"

    # Verify log entries — one skipped, one fixed.
    entries = [json.loads(ln) for ln in l_path.read_text().splitlines() if ln.strip()]
    by_id = {e["id"]: e for e in entries}
    assert by_id["Q_UNCHANGED"]["action"] == "skipped"
    assert "sources_unchanged" in by_id["Q_UNCHANGED"]["flags"]
    assert "regeneration_skipped" in by_id["Q_UNCHANGED"]["flags"]
    assert by_id["Q_CHANGED"]["action"] == "fixed"
    assert "sources_fixed" in by_id["Q_CHANGED"]["flags"]

    # UNCHANGED draft must be preserved; CHANGED draft must be the stub.
    d_final = {x["id"]: x for x in json.loads(d_path.read_text())["drafts"]}
    assert d_final["Q_UNCHANGED"]["ground_truth_answer_draft"] == "old draft"
    assert d_final["Q_CHANGED"]["ground_truth_answer_draft"] == "new draft from stub"
