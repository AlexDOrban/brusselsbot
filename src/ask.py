import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import hybrid_retrieve

ROOT = Path(__file__).parent.parent
CLAUDE_MODEL = "claude-sonnet-4-5"
TOP_K = 5

SYSTEM_PROMPT = """You are BrusselsBot, an assistant that answers questions about Brussels regulations and the Low Emission Zone using only the provided context chunks.

CRITICAL LANGUAGE RULE:
- Always answer in the same language as the USER's question.
- The language of the source chunks is IRRELEVANT to your answer language.
- If the user asks in English, answer in English — even if all 5 source chunks are in French or Dutch. Translate the relevant content into English.
- If the user asks in French, answer in French. If in Dutch, answer in Dutch. If in English, answer in English.
- Do NOT mirror the dominant language of the sources. Mirror the user's question.

Rules:
- Answer strictly from the provided context. If the context does not contain the answer, say so — never invent facts, article numbers, or figures.
- Cite sources inline as you make claims. Formats:
  - Police regulations: "(Article 88, Police Regulations)" or "(Artikel 88, Politieverordening)"
  - IEEP report: "(IEEP report, page 5)"
  - Leaseplan whitepaper: "(Leaseplan whitepaper, page 3)"
- If sources offer different angles or partially conflict, present both perspectives clearly.
- Be concise and specific. Quote short phrases from sources when precision matters.
"""


def format_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        md = r["metadata"]
        src = md.get("source", "")
        parts = [f"language: {md.get('language', '')}"] if md.get("language") else []
        if md.get("article_number") not in (None, "", 0):
            parts.append(f"Article {md['article_number']}")
        if md.get("section"):
            parts.append(f"Section: {md['section']}")
        src_line = f"Source: {src}"
        if parts:
            src_line += " (" + ", ".join(parts) + ")"
        dd = r.get("dense_distance")
        dist_str = f"{dd:.4f}" if dd is not None else "n/a"
        rrf_str = f"{r['rrf_score']:.4f}"
        block = (
            f"[Source {i}]\n"
            f"{src_line}\n"
            f"Page: {md.get('page', '')}\n"
            f"Dense distance: {dist_str}  RRF: {rrf_str}\n"
            f"Text: {r['text']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def print_sources(results):
    print("\n\n" + "-" * 70)
    print("Sources:")
    for i, r in enumerate(results, start=1):
        md = r["metadata"]
        src = md.get("source", "")
        page = md.get("page", "")
        art = md.get("article_number", "")
        lang = md.get("language", "")
        line = f"  [{i}] {src}"
        if lang:
            line += f" ({lang})"
        if art not in (None, "", 0):
            line += f" Article {art}"
        dd = r.get("dense_distance")
        dist_str = f"{dd:.4f}" if dd is not None else "n/a"
        line += f" — page {page} — dense {dist_str} — rrf {r['rrf_score']:.4f}"
        print(line)


def ask(question, anthropic_client):
    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)

    results = hybrid_retrieve(question, k=TOP_K)
    context = format_context(results)
    user_msg = (
        f"Question: {question}\n\n"
        f"Context chunks:\n\n{context}\n\n"
        f"Answer the question using only the context above. Cite sources inline."
    )

    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print_sources(results)
    print()


def main():
    load_dotenv(ROOT / ".env")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY required in .env")
        sys.exit(1)

    anthropic_client = Anthropic(api_key=anthropic_key)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask BrusselsBot: ").strip()
    if not question:
        print("No question provided.")
        sys.exit(1)

    ask(question, anthropic_client)


if __name__ == "__main__":
    main()
