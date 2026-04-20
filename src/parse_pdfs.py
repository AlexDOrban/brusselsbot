import pymupdf
from pathlib import Path

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

def analyze_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    all_text = ""
    for page in doc:
        all_text += page.get_text()

    total_chars = len(all_text)
    first_page_preview = doc[0].get_text()[:500]

    doc.close()

    print("\n" + "=" * 70)
    print(f"File:             {pdf_path.name}")
    print(f"Pages:            {total_pages}")
    print(f"Total characters: {total_chars:,}")
    print("\nFirst 500 chars of page 1:")
    print("-" * 70)
    print(first_page_preview)
    print("=" * 70)

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")
    for pdf_path in pdfs:
        analyze_pdf(pdf_path)

if __name__ == "__main__":
    main()

    import pymupdf
from pathlib import Path

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

def analyze_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    all_text = ""
    for page in doc:
        all_text += page.get_text()

    total_chars = len(all_text)
    middle_idx = total_pages // 2
    first_page_preview = doc[0].get_text()[:500]
    middle_page_preview = doc[middle_idx].get_text()[:800]
    last_page_preview = doc[-1].get_text()[:500]

    doc.close()

    print("\n" + "=" * 70)
    print(f"File:             {pdf_path.name}")
    print(f"Pages:            {total_pages}")
    print(f"Total characters: {total_chars:,}")
    print(f"Chars per page:   {total_chars // total_pages:,}")

    print(f"\n--- FIRST 500 chars of page 1 ---")
    print(first_page_preview)

    print(f"\n--- FIRST 800 chars of middle page ({middle_idx + 1}) ---")
    print(middle_page_preview)

    print(f"\n--- FIRST 500 chars of last page ({total_pages}) ---")
    print(last_page_preview)
    print("=" * 70)

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")
    for pdf_path in pdfs:
        analyze_pdf(pdf_path)

if __name__ == "__main__":
    main()