import json
from pathlib import Path

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"


def word_count(s):
    return len(s.split())


def show(chunk, label):
    md = chunk["metadata"]
    print(f"\n--- {label} ---")
    print(f"source:  {md.get('source')}")
    if "language" in md:
        print(f"lang:    {md['language']}")
    if "section" in md:
        print(f"section: {md['section']}")
    if "article_number" in md:
        print(f"article: {md['article_number']}")
    print(f"page:    {md.get('page')}")
    print(f"words:   {word_count(chunk['text'])}")
    print(f"text[:600]:\n{chunk['text'][:600]}")


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    fr_28 = next(
        (c for c in chunks
         if c["metadata"].get("source") == "police_regs"
         and c["metadata"].get("language") == "fr"
         and c["metadata"].get("article_number") == 28),
        None,
    )
    nl_28 = next(
        (c for c in chunks
         if c["metadata"].get("source") == "police_regs"
         and c["metadata"].get("language") == "nl"
         and c["metadata"].get("article_number") == 28),
        None,
    )

    if fr_28:
        show(fr_28, "Article 28 FR")
    else:
        print("\n[MISSING] Article 28 FR not found")
    if nl_28:
        show(nl_28, "Artikel 28 NL")
    else:
        print("\n[MISSING] Artikel 28 NL not found")

    ieep_first = next((c for c in chunks if c["metadata"].get("source") == "ieep"), None)
    if ieep_first:
        show(ieep_first, "First IEEP chunk")
    leaseplan_first = next((c for c in chunks if c["metadata"].get("source") == "leaseplan"), None)
    if leaseplan_first:
        show(leaseplan_first, "First Leaseplan chunk")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    short = [c for c in chunks if word_count(c["text"]) < 10]
    header_bleed = [c for c in chunks if "Social aspects of low emission zones" in c["text"]]
    missing_section = [
        c for c in chunks
        if c["metadata"].get("source") == "police_regs"
        and not c["metadata"].get("section")
    ]

    def report(name, offenders):
        status = "PASS" if not offenders else "FAIL"
        print(f"\n[{status}] {name} — {len(offenders)} offender(s)")
        for c in offenders[:3]:
            md = c["metadata"]
            print(f"  source={md.get('source')} page={md.get('page')} "
                  f"lang={md.get('language')} art={md.get('article_number')}")
            print(f"  text[:200]: {c['text'][:200]}")

    report("A: min 10 words per chunk", short)
    report("B: no IEEP header bleed", header_bleed)
    report("C: police_regs has section heading", missing_section)


if __name__ == "__main__":
    main()
