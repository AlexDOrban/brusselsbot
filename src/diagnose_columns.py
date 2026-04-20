# One-off diagnostic: confirmed that police regs PDF uses two-column FR/NL layout per page (pymupdf reads left column top-to-bottom, then right). Justified the FR/NL split logic in chunk_pdfs.py.
import pymupdf
from pathlib import Path

PDF = Path(__file__).parent.parent / "data" / "pdfs" / "brussels_police_regulations.pdf"
doc = pymupdf.open(PDF)

# Print the FULL text of page 21 to see the column structure
text = doc[20].get_text()
print(f"--- FULL page 21, {len(text)} chars ---")
print(text)
doc.close()