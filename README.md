# BrusselsBot

Multilingual conversational RAG over Brussels municipal regulations and Low Emission Zone research papers. Answers questions in English, French, or Dutch and cites specific articles and page numbers. Built on OpenAI embeddings, ChromaDB, and Claude.

## Corpus

| File | Pages | Languages | Covers |
|------|-------|-----------|--------|
| `brussels_police_regulations.pdf` | 72 | FR / NL (bilingual, two-column) | Public-space rules: noise, waste, construction, animals, alarms, fines |
| `ieep_lez_case_study.pdf` | 17 | EN | IEEP case study on social impacts of the Brussels LEZ |
| `leaseplan_lez_whitepaper.pdf` | 15 | EN | Business-fleet perspective on LEZs and EV transition |

The police regulations PDF is a bilingual FR/NL document. Each page has French on the left and Dutch on the right; the chunker splits on the first `Sectie`/`Artikel` marker per page so FR and NL articles are indexed separately and can be retrieved independently.

## Architecture

PDF parsing (pymupdf, article-aware chunking for regs, paragraph-aware ~500-word chunks for English reports) → embedding (`text-embedding-3-small`, multilingual) → ChromaDB persistent store → query rewriting (Claude Haiku, resolves follow-ups and pronouns) → retrieval (top 5, cosine) → confidence check (top-1 distance > 0.60 triggers a caveat) → answer generation (Claude Sonnet, streaming, source-cited).

1. `src/chunk_pdfs.py` — extracts and chunks the 3 PDFs into `data/chunks.json`
2. `src/embed_chunks.py` — embeds chunks into `./chroma_db` collection `brussels_regulations`
3. `src/ask.py` — single-shot CLI Q&A
4. `src/app.py` — Gradio chat UI with query rewriting, streaming, logging
5. `src/benchmark.py` — retrieval eval harness (Hit@K, MRR, out-of-scope check)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Download the 3 PDFs into `data/pdfs/`:

1. **Brussels General Police Regulations** (bilingual FR/NL, 72 pages) — save as `brussels_police_regulations.pdf`
   https://www.bruxelles.be/sites/default/files/bxl/Reglement_de_police_-_Politiereglement.pdf
2. **IEEP LEZ Case Study** (English, 17 pages) — save as `ieep_lez_case_study.pdf`
   https://ieep.eu/wp-content/uploads/2024/06/Social-aspects-of-low-emission-zones-Brussels-Capital-Region-case-study-IEEP-2024-2.pdf
3. **LeasePlan/Ayvens LEZ Whitepaper** (English, 15 pages) — save as `leaseplan_lez_whitepaper.pdf`
   https://www.leaseplan.com/-/media/ayvens/public/be/documents/whitepapers/lez-whitepaper/update-2024-final/lp_lez_belgie_2023-en.pdf

> Note: The bruxelles.be and leaseplan.com servers block bare curl requests. Either download via browser, or use curl with a browser User-Agent: `curl -L -A "Mozilla/5.0" -o <filename> <url>`.

Then:

```bash
python src/parse_pdfs.py      # sanity-check extraction
python src/chunk_pdfs.py      # → data/chunks.json
python src/embed_chunks.py    # → ./chroma_db (asks before overwriting)
python src/app.py             # → http://127.0.0.1:7860
```

Optional:

```bash
python src/benchmark.py       # retrieval scores → data/benchmark_baseline.json
python src/ask.py "What are the noise rules at night?"
```

## Baseline retrieval (10-question benchmark)

- Hit@1: 0.889 · Hit@3: 1.000 · Hit@5: 1.000 · MRR: 0.944
- Out-of-scope question (speeding on E40) top-1 distance 0.625 — above the 0.60 confidence threshold, correctly caveated.

## What I'd add next

- **Hybrid BM25 + embedding retrieval.** The NL query for "sluikstort" fails to surface Article 25 (the one chunk that literally contains the word) because semantic embedding buries it under general-fines articles. BM25 would fix this.
- **Query expansion / cross-lingual rewriting.** A NL question about illegal dumping should also retrieve the FR articles (14, 29, 63) that cover the topic under different wording. Rewrite the query into all three languages before embedding.
- **Expanded corpus.** The police regulations don't set per-offense fines for illegal dumping; that's in regional environmental law (Brussels Environment / Bruxelles-Propreté ordinances). Adding those documents would close a known answer gap.
- **Clickable source citations in the UI.** Currently citations are plain text. Linking them to the PDF page would make verification frictionless.
