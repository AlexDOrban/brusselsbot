import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import hybrid_retrieve, CONFIDENCE_THRESHOLD

ROOT = Path(__file__).parent.parent
ANSWER_MODEL = "claude-sonnet-4-5"
REWRITE_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5
LOG_PATH = ROOT / "data" / "query_log.jsonl"

LOW_CONF_PREFIX = (
    "⚠️ I'm not very confident the available sources cover this — "
    "here's what I found, but verify independently."
)

REWRITE_SYSTEM = """You rewrite follow-up questions into standalone questions for a retrieval system.
Given the conversation history and the latest user message, output ONE standalone
question that captures what the user is asking, with all referents resolved
(pronouns, "that", "what about X", etc.).
Output ONLY the rewritten question. No preamble, no explanation, no quotes.
If the latest message is already standalone, output it unchanged."""

ANSWER_SYSTEM = """You are BrusselsBot, an assistant that answers questions about Brussels regulations and the Low Emission Zone using only the provided context chunks.

CRITICAL LANGUAGE RULE:
- Always answer in the same language as the USER's question.
- The language of the source chunks is IRRELEVANT to your answer language.
- If the user asks in English, answer in English — even if all source chunks are in French or Dutch. Translate the relevant content into English.
- If the user asks in French, answer in French. If in Dutch, answer in Dutch. If in English, answer in English.
- Do NOT mirror the dominant language of the sources. Mirror the user's question.

Rules:
- Answer strictly from the provided context. If the context doesn't cover the question, say exactly: "I don't have information on that in the available sources" (translated to the user's language). Do NOT fabricate facts, article numbers, or figures.
- Cite sources inline:
  - Police regulations: "(Article 88, Police Regulations)" or "(Artikel 88, Politieverordening)"
  - IEEP report: "(IEEP report, page 5)"
  - Leaseplan whitepaper: "(Leaseplan whitepaper, page 3)"
- If sources offer different angles, present both.
- Be concise and specific. Quote short phrases when precision matters.
"""

load_dotenv(ROOT / ".env")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_KEY:
    print("ERROR: ANTHROPIC_API_KEY required in .env")
    sys.exit(1)

anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)


def rewrite_query(message, history):
    if not history:
        return message
    recent = history[-4:]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    user_msg = f"Conversation so far:\n{convo}\n\nLatest user message: {message}\n\nStandalone question:"
    resp = anthropic_client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=200,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()


def format_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        md = r["metadata"]
        parts = []
        if md.get("language"):
            parts.append(f"lang={md['language']}")
        if md.get("article_number") not in (None, "", 0):
            parts.append(f"Article {md['article_number']}")
        if md.get("section"):
            parts.append(f"Section: {md['section']}")
        dd = r.get("dense_distance")
        dist_str = f"{dd:.4f}" if dd is not None else "n/a"
        rrf_str = f"{r['rrf_score']:.4f}"
        header = (
            f"[Source {i}] source={md.get('source')} "
            + " ".join(parts)
            + f" Page: {md.get('page')} Dense: {dist_str} RRF: {rrf_str}"
        )
        blocks.append(f"{header}\nText: {r['text']}")
    return "\n\n".join(blocks)


def log_interaction(record):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log error] {e}")


def chat(message, history):
    history = history or []

    rewritten = rewrite_query(message, history)
    print(f"[retrieval query] {rewritten!r}")

    results = hybrid_retrieve(rewritten, k=TOP_K)

    top1_dense = results[0]["dense_distance"] if results and results[0]["dense_distance"] is not None else 1.0
    low_confidence = top1_dense > CONFIDENCE_THRESHOLD

    context_block = format_context(results)

    history_block = ""
    if history:
        lines = [f"{m['role']}: {m['content']}" for m in history]
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    low_conf_note = ""
    if low_confidence:
        low_conf_note = (
            f'\n\nIMPORTANT: The top retrieved chunk has dense distance {top1_dense:.4f}, '
            f'which exceeds the confidence threshold of {CONFIDENCE_THRESHOLD}. '
            f'Start your response with this exact line (on its own line, preserving the emoji), '
            f'then continue with the grounded answer:\n'
            f'{LOW_CONF_PREFIX}'
        )

    user_content = (
        f"{history_block}"
        f"Retrieved context:\n\n{context_block}\n\n"
        f"User's latest question: {message}\n\n"
        f"Answer the user's latest question using only the retrieved context. "
        f"Cite sources inline.{low_conf_note}"
    )

    log_interaction({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_message": message,
        "rewritten_query": rewritten,
        "top_1_dense_distance": top1_dense,
        "low_confidence": low_confidence,
        "retrieved_sources": [
            {
                "source": r["metadata"].get("source"),
                "article_number": r["metadata"].get("article_number"),
                "language": r["metadata"].get("language"),
                "dense_distance": r.get("dense_distance"),
                "bm25_score": r.get("bm25_score"),
                "rrf_score": r["rrf_score"],
            }
            for r in results
        ],
    })

    buffer = ""
    with anthropic_client.messages.stream(
        model=ANSWER_MODEL,
        max_tokens=1024,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text in stream.text_stream:
            buffer += text
            yield buffer


def main():
    demo = gr.ChatInterface(
        fn=chat,
        title="BrusselsBot",
        description="Ask about Brussels police regulations and low-emission zones. English, French, or Dutch.",
        examples=[
            "What are the rules about noise at night?",
            "Mag ik 's nachts muziek maken op straat?",
            "Quelles sont les règles pour installer une grue?",
            "How has the LEZ affected low-income households?",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
