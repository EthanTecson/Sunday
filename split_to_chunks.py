"""
Split a long transcript into chunked .txt files.

EDIT THESE THREE CONFIG VALUES BELOW:
- INPUT_FILE: path to the source transcript (e.g., "source_docs/CS371_lecture01.txt")
- OUTPUT_DIR: where to write the chunks (e.g., "courses/371/data/chunks/lecture1")
- LECTURE_LABEL: used in filenames (e.g., "lecture1", "lecture2", "lecture3")

Optional: adjust CHUNK_SIZE / OVERLAP if you want bigger/smaller chunks.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# ==== ⬇️⬇️⬇️ EDIT HERE ⬇️⬇️⬇️ ===============================================

INPUT_FILE = "source_docs/CS371_lecture01.txt"         # <-- change me
OUTPUT_DIR = "courses/371/data/chunks/lecture1"        # <-- change me
LECTURE_LABEL = "lecture1"                             # <-- change me

CHUNK_SIZE = 1500   # target characters per chunk (suggested: 1200–1800)
OVERLAP    = 250    # overlap in characters between consecutive chunks

# ==== ⬆️⬆️⬆️ EDIT HERE ⬆️⬆️⬆️ ===============================================


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def normalize_whitespace(text: str) -> str:
    """
    Light cleanup: collapse weird space, convert Windows newlines, trim.
    (Deliberately minimal—won't change content meaning.)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace odd whitespace (e.g., non-breaking spaces)
    text = re.sub(r"[ \t\u00A0]+", " ", text)
    # Collapse >2 newlines to exactly two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def smart_slices(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Produce start/end indices for chunks, trying to end on nice boundaries:
    1) paragraph break (double newline),
    2) sentence end (. ? !),
    3) single newline,
    otherwise hard cut at chunk_size.

    Returns a list of (start, end) index tuples.
    """
    n = len(text)
    slices = []
    start = 0
    while start < n:
        target_end = min(n, start + chunk_size)
        end = target_end

        # Try to find a boundary within [start+chunk_size-300, target_end]
        # (The 300 is a soft window we scan backwards from target_end.)
        window_start = max(start + int(chunk_size * 0.6), start)  # avoid tiny tail
        search_region_start = max(window_start, target_end - 300)

        region = text[search_region_start:target_end]

        # 1) Prefer paragraph break near the end
        para_idx = region.rfind("\n\n")
        if para_idx != -1:
            end = search_region_start + para_idx

        else:
            # 2) Sentence end
            sent_idx = max(region.rfind(". "), region.rfind("? "), region.rfind("! "))
            if sent_idx != -1:
                end = search_region_start + sent_idx + 1  # include the punctuation

            else:
                # 3) Single newline
                nl_idx = region.rfind("\n")
                if nl_idx != -1:
                    end = search_region_start + nl_idx
                else:
                    # 4) Hard cut at target_end
                    end = target_end

        # Safety: ensure end > start; if not, force progress
        if end <= start:
            end = min(n, start + chunk_size)

        slices.append((start, end))

        if end == n:
            break

        # Move start forward with overlap (but never backwards)
        start = max(end - overlap, end) if overlap > 0 else end

    return slices


def write_chunks(text: str, chunks: List[Tuple[int, int]], out_dir: str, lecture_label: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, (s, e) in enumerate(chunks, start=1):
        part = f"{i:03d}"
        filename = f"{lecture_label}-part{part}.txt"
        out_file = out_path / filename
        out_file.write_text(text[s:e].strip() + "\n", encoding="utf-8")
        print(f"Saved {out_file} ({e - s} chars)")


def main():
    print("== Sunday Chunker ==")
    print(f"Reading:   {INPUT_FILE}")
    print(f"Writing to: {OUTPUT_DIR}")
    print(f"Filenames: {LECTURE_LABEL}-partNNN.txt")
    print(f"Chunk ~{CHUNK_SIZE} chars with {OVERLAP} overlap\n")

    raw = load_text(INPUT_FILE)
    cleaned = normalize_whitespace(raw)
    slices = smart_slices(cleaned, CHUNK_SIZE, OVERLAP)

    total_chars = sum(e - s for s, e in slices)
    write_chunks(cleaned, slices, OUTPUT_DIR, LECTURE_LABEL)

    print("\n== Done ==")
    print(f"Total chunks: {len(slices)}")
    print(f"Total characters written: {total_chars}")


if __name__ == "__main__":
    main()
