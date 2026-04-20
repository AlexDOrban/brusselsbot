import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from anthropic import Anthropic

ROOT = Path(__file__).parent.parent
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "brussels_regulations"
EMBED_MODEL = "text-embedding-3-small"
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


def format_context(docs, metas, dists):
    blocks = []
    for i, (doc, md, dist) in enumerate(zip(docs, metas, dists), start=1):
        src = md.get("source", "")
        parts = [f"language: {md.get('language', '')}"] if md.get("language") else []
        if md.get("article_number") not in (None, "", 0):
            parts.append(f"Article {md['article_number']}")
        if md.get("section"):
            parts.append(f"Section: {md['section']}")
        src_line = f"Source: {src}"
        if parts:
            src_line += " (" + ", ".join(parts) + ")"
        block = (
            f"[Source {i}]\n"
            f"{src_line}\n"
            f"Page: {md.get('page', '')}\n"
            f"Distance: {dist:.4f}\n"
            f"Text: {doc}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def print_sources(metas, dists):
    print("\n\n" + "-" * 70)
    print("Sources:")
    for i, (md, dist) in enumerate(zip(metas, dists), start=1):
        src = md.get("source", "")
        page = md.get("page", "")
        art = md.get("article_number", "")
        lang = md.get("language", "")
        line = f"  [{i}] {src}"
        if lang:
            line += f" ({lang})"
        if art not in (None, "", 0):
            line += f" Article {art}"
        line += f" — page {page} — distance {dist:.4f}"
        print(line)


def ask(question, openai_client, anthropic_client, collection):
    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)

    q_emb = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    context = format_context(docs, metas, dists)
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

    print_sources(metas, dists)
    print()


def main():
    load_dotenv(ROOT / ".env")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not openai_key or not anthropic_key:
        print("ERROR: OPENAI_API_KEY and ANTHROPIC_API_KEY required in .env")
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = Anthropic(api_key=anthropic_key)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma.get_collection(COLLECTION_NAME)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask BrusselsBot: ").strip()
    if not question:
        print("No question provided.")
        sys.exit(1)

    ask(question, openai_client, anthropic_client, collection)


if __name__ == "__main__":
    main()
