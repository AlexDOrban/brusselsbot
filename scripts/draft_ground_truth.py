"""Use Claude Opus to draft a ground_truth_answer for each benchmark question,
grounded ONLY in the expected_sources chunks. Out-of-scope questions get a
canned refusal draft. Output goes to data/benchmark_ground_truth_draft.json
for MANUAL review before merge.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

QUESTIONS = json.loads((ROOT / "data" / "benchmark_questions.json").read_text())["questions"]
CHUNKS = json.loads((ROOT / "data" / "chunks.json").read_text())
BY_ID = {f"chunk_{i}": c for i, c in enumerate(CHUNKS)}

MODEL = "claude-opus-4-7"
OUT_PATH = ROOT / "data" / "benchmark_ground_truth_draft.json"

SYSTEM = """You are drafting a ground-truth reference answer for a retrieval-eval benchmark.

Rules:
- Answer ONLY from the provided source chunks. Never invent article numbers, figures, or facts not in the sources.
- Answer in the SAME language as the user's question. If question is in French, answer in French. Dutch → Dutch. English → English.
- Be concise: 2–5 sentences. No preamble, no "Based on the sources..." framing.
- Cite inline using the format already established: "(Article 88, Police Regulations)", "(Artikel 88, Politieverordening)", "(IEEP report, page 5)", "(Leaseplan whitepaper, page 3)".
- If multiple sources are relevant, synthesize — do not list per-source.
- Output ONLY the answer text. No JSON, no headers, no explanation."""

OOS_DRAFT = {
    "en": "This question falls outside the scope of the available sources (Brussels police regulations, IEEP LEZ case study, Leaseplan LEZ whitepaper). I cannot answer it from this corpus.",
    "fr": "Cette question dépasse le périmètre des sources disponibles (règlement de police de Bruxelles, étude de cas IEEP sur la LEZ, livre blanc Leaseplan sur la LEZ). Je ne peux pas y répondre à partir de ce corpus.",
    "nl": "Deze vraag valt buiten het bereik van de beschikbare bronnen (politieverordening Brussel, IEEP-casestudy LEZ, Leaseplan LEZ-whitepaper). Ik kan deze vraag niet beantwoorden op basis van dit corpus.",
}


def format_chunks(chunk_ids):
    blocks = []
    for cid in chunk_ids:
        c = BY_ID[cid]
        md = c["metadata"]
        head = f"[{cid}] source={md.get('source')} lang={md.get('language')} " \
               f"article={md.get('article_number')} page={md.get('page')}"
        blocks.append(f"{head}\n{c['text']}")
    return "\n\n".join(blocks)


def draft_for(client, q):
    if q["category"] == "out_of_scope":
        return OOS_DRAFT.get(q["language"], OOS_DRAFT["en"]), None
    if not q["expected_sources"]:
        return "", "no expected_sources"
    context = format_chunks(q["expected_sources"])
    user = (
        f"Question ({q['language']}): {q['question']}\n\n"
        f"Source chunks:\n\n{context}\n\n"
        f"Draft the ground-truth answer."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip(), None


def main():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)
    client = Anthropic(api_key=key)

    drafts = []
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.time()
        try:
            answer, err = draft_for(client, q)
        except Exception as e:
            answer, err = "", f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        print(f"[{i}/{len(QUESTIONS)}] {q['id']} ({q['category']}, {q['language']}) "
              f"{'OK' if not err else 'ERR'} {dt:.1f}s")
        if err:
            print(f"    {err}")
        drafts.append({
            "id": q["id"],
            "category": q["category"],
            "language": q["language"],
            "question": q["question"],
            "expected_sources": q["expected_sources"],
            "ground_truth_answer_draft": answer,
            "error": err,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"drafts": drafts}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}")
    n_err = sum(1 for d in drafts if d["error"])
    print(f"Drafted: {len(drafts) - n_err}  Errors: {n_err}")


if __name__ == "__main__":
    main()
