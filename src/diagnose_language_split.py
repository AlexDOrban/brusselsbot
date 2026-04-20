# One-off diagnostic: spot-checked pages across the police regs to confirm FR is consistently in the left column and NL in the right. Used to validate the column-split assumption before committing to the chunker design.
import pymupdf
from pathlib import Path

PDF = Path(__file__).parent.parent / "data" / "pdfs" / "brussels_police_regulations.pdf"

doc = pymupdf.open(PDF)
# Print first 200 chars of a handful of pages to spot the FR -> NL transition
for page_num in [20, 35, 40, 45, 55, 65]:
    text = doc[page_num].get_text()[:200].replace("\n", " ")
    print(f"\n--- Page {page_num + 1} ---")
    print(text)
doc.close()