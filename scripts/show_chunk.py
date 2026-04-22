"""Print full text + metadata for a chunk.

Usage: venv/bin/python scripts/show_chunk.py chunk_49
Stdlib only. Read-only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: show_chunk.py <chunk_id>  (e.g. chunk_49)")
    target = sys.argv[1].strip()
    chunks = json.loads(CHUNKS_PATH.read_text())

    if target.startswith("chunk_"):
        try:
            idx = int(target[len("chunk_"):])
        except ValueError:
            sys.exit(f"bad chunk id: {target}")
    else:
        try:
            idx = int(target)
        except ValueError:
            sys.exit(f"bad chunk id: {target}")

    if not (0 <= idx < len(chunks)):
        sys.exit(f"index out of range: {idx} (corpus has {len(chunks)} chunks)")

    c = chunks[idx]
    md = c["metadata"]
    print(f"chunk_id: chunk_{idx}")
    print("metadata:")
    for k in sorted(md):
        print(f"  {k}: {md[k]}")
    print()
    print("text:")
    print(c["text"])


if __name__ == "__main__":
    main()
