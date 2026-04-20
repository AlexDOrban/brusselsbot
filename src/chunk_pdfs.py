import json
import re
import pymupdf
from pathlib import Path

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT_PATH = ROOT / "data" / "chunks.json"

DUTCH_MARKER_RE = re.compile(r'\b(Sectie|Artikel)\s')
ARTICLE_SPLIT_RE = re.compile(r'(Article|Artikel)\s+(\d+)\.')
SECTION_RE = re.compile(r'^(Section|Sectie)\s+\d+\b[^\n]*', re.MULTILINE)
PAGE_NUM_LINE_RE = re.compile(r'\n\s*\d+\s*\n?\s*$')

IEEP_HEADER_1 = "Social aspects of low emission zones: Brussels-Capital Region case study"
IEEP_HEADER_2 = "Institute for European Environmental Policy (June 2024)"


def strip_trailing_page_num(text):
    text = text.rstrip()
    lines = text.split("\n")
    while lines and lines[-1].strip().isdigit():
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def unwrap_soft_breaks(text):
    keep_prefixes = ("§", "Article", "Artikel", "Section", "Sectie")
    num_prefix_re = re.compile(r'^\d+\.')
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i == 0:
            out.append(line)
            continue
        stripped = line.lstrip()
        if stripped.startswith(keep_prefixes) or num_prefix_re.match(stripped):
            out.append("\n" + line)
        else:
            out.append(" " + line)
    merged = "".join(out)
    merged = re.sub(r'[ \t]+', ' ', merged)
    merged = re.sub(r'\n{3,}', '\n\n', merged)
    return merged.strip()


def split_articles(text, lang_marker):
    """Split a language half into (article_num, chunk_text) pairs."""
    matches = list(re.finditer(r'\b' + lang_marker + r'\s+(\d+)\.', text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        num = int(m.group(1))
        chunks.append((num, text[start:end].strip()))
    return chunks


def track_section_and_chunk(text, lang_marker, current_section):
    """Walk through text, updating section heading as we hit them, producing article chunks."""
    article_re = re.compile(r'\b' + lang_marker + r'\s+(\d+)\.')
    section_word = "Section" if lang_marker == "Article" else "Sectie"
    section_re = re.compile(r'^' + section_word + r'\s+\d+\b[^\n]*', re.MULTILINE)

    events = []
    for m in article_re.finditer(text):
        events.append(("article", m.start(), m))
    for m in section_re.finditer(text):
        events.append(("section", m.start(), m))
    events.sort(key=lambda e: e[1])

    results = []
    for i, (kind, pos, m) in enumerate(events):
        if kind == "section":
            current_section = m.group(0).strip()
            continue
        next_pos = len(text)
        for j in range(i + 1, len(events)):
            if events[j][0] == "article":
                next_pos = events[j][1]
                break
        num = int(m.group(1))
        body = text[pos:next_pos].strip()
        results.append({
            "article_number": num,
            "text": body,
            "section": current_section,
        })
    return results, current_section


def chunk_police_regs(pdf_path):
    doc = pymupdf.open(pdf_path)
    total = len(doc)
    chunks = []
    section_fr = ""
    section_nl = ""

    for page_idx in range(total):
        page_num = page_idx + 1
        if page_num <= 2 or page_num >= total - 1:
            continue
        page_text = doc[page_idx].get_text()
        page_text = page_text.replace("\xad", "").replace("\xa0", " ")

        dutch_match = DUTCH_MARKER_RE.search(page_text)
        if dutch_match:
            fr_text = page_text[:dutch_match.start()]
            nl_text = page_text[dutch_match.start():]
        else:
            fr_text = page_text
            nl_text = ""

        fr_text = strip_trailing_page_num(fr_text)
        nl_text = strip_trailing_page_num(nl_text)

        fr_articles, section_fr = track_section_and_chunk(fr_text, "Article", section_fr)
        for a in fr_articles:
            chunks.append({
                "text": unwrap_soft_breaks(a["text"]),
                "metadata": {
                    "source": "police_regs",
                    "language": "fr",
                    "article_number": a["article_number"],
                    "section": a["section"],
                    "page": page_num,
                },
            })

        nl_articles, section_nl = track_section_and_chunk(nl_text, "Artikel", section_nl)
        for a in nl_articles:
            chunks.append({
                "text": unwrap_soft_breaks(a["text"]),
                "metadata": {
                    "source": "police_regs",
                    "language": "nl",
                    "article_number": a["article_number"],
                    "section": a["section"],
                    "page": page_num,
                },
            })

    doc.close()
    return chunks


def chunk_english_pdf(pdf_path, source, skip_pages=None, strip_headers=None):
    skip_pages = skip_pages or set()
    strip_headers = strip_headers or []
    doc = pymupdf.open(pdf_path)

    page_texts = []
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num in skip_pages:
            continue
        text = doc[page_idx].get_text()
        text = text.replace("\xad", "").replace("\xa0", " ")
        for h in strip_headers:
            text = text.replace(h, "")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            page_texts.append((page_num, cleaned))
    doc.close()

    paragraphs = []
    for page_num, ptext in page_texts:
        parts = re.split(r'\n\s*\n', ptext)
        if len(parts) == 1:
            parts = re.split(r'\n(?=[A-Z])', ptext)
        for p in parts:
            p = re.sub(r'\s+', ' ', p).strip()
            if p:
                paragraphs.append((page_num, p))

    chunks = []
    target = 500
    overlap = 50
    i = 0
    while i < len(paragraphs):
        buf_words = []
        start_page = paragraphs[i][0]
        j = i
        while j < len(paragraphs) and len(buf_words) < target:
            words = paragraphs[j][1].split()
            buf_words.extend(words)
            j += 1
        text = " ".join(buf_words).strip()
        if text:
            chunks.append({
                "text": text,
                "metadata": {"source": source, "language": "en", "page": start_page},
            })
        if j >= len(paragraphs):
            break
        back_words = 0
        new_i = j
        while new_i > i + 1 and back_words < overlap:
            back_words += len(paragraphs[new_i - 1][1].split())
            new_i -= 1
        if new_i <= i:
            new_i = i + 1
        i = new_i
    return chunks


def main():
    all_chunks = []

    police_pdf = PDF_DIR / "brussels_police_regulations.pdf"
    ieep_pdf = PDF_DIR / "ieep_lez_case_study.pdf"
    leaseplan_pdf = PDF_DIR / "leaseplan_lez_whitepaper.pdf"

    all_chunks.extend(chunk_police_regs(police_pdf))
    all_chunks.extend(chunk_english_pdf(
        ieep_pdf, source="ieep",
        strip_headers=[IEEP_HEADER_1, IEEP_HEADER_2],
    ))
    all_chunks.extend(chunk_english_pdf(
        leaseplan_pdf, source="leaseplan",
        skip_pages={1, 15},
    ))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    counts = {"police_regs_fr": 0, "police_regs_nl": 0, "ieep": 0, "leaseplan": 0}
    total_words = 0
    for c in all_chunks:
        md = c["metadata"]
        if md["source"] == "police_regs":
            key = "police_regs_" + md["language"]
        else:
            key = md["source"]
        counts[key] += 1
        total_words += len(c["text"].split())

    print("=" * 70)
    print(f"Total chunks: {len(all_chunks)}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    avg = total_words / len(all_chunks) if all_chunks else 0
    print(f"Average chunk length: {avg:.1f} words")
    print(f"Saved to: {OUT_PATH}")
    print("=" * 70)

    police_sample = next((c for c in all_chunks if c["metadata"]["source"] == "police_regs"), None)
    ieep_sample = next((c for c in all_chunks if c["metadata"]["source"] == "ieep"), None)

    if police_sample:
        print("\n--- SAMPLE: police_regs ---")
        print(f"metadata: {police_sample['metadata']}")
        print(f"text:\n{police_sample['text']}")
    if ieep_sample:
        print("\n--- SAMPLE: ieep ---")
        print(f"metadata: {ieep_sample['metadata']}")
        print(f"text:\n{ieep_sample['text']}")


if __name__ == "__main__":
    main()