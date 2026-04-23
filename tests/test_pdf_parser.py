"""
Manual spot-check for pdf_parser.py.

Run from the project root:
    python3 -m tests.test_pdf_parser

Prints every chapter transition detected in the first PAGES_TO_CHECK pages
so you can verify against the physical book.

Failures here are silent in the main pipeline and corrupt all downstream
metadata — do not proceed to chunker.py until this output looks correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PDF_PATH, CHAPTER_MAP
from src.ingest.pdf_parser import parse_pdf, compute_pdf_hash

PAGES_TO_CHECK = 126


def run():
    print(f"PDF : {PDF_PATH}")
    print(f"Hash: {compute_pdf_hash(str(PDF_PATH))}\n")

    print(f"{'=' * 70}")
    print(f"Scanning pages 3–{PAGES_TO_CHECK}  (pages 1–2 are skipped by parser)")
    print(f"Verify chapter transitions against physical book.")
    print(f"{'=' * 70}\n")

    last_chapter_num = None

    for record in parse_pdf(str(PDF_PATH)):
        if record.page_num > PAGES_TO_CHECK:
            break

        if record.chapter_num != last_chapter_num:
            expected = CHAPTER_MAP.get(record.chapter_num, {})
            print(f"\n>>> CHAPTER TRANSITION  (page {record.page_num})")
            print(f"    Detected # : {record.chapter_num}")
            print(f"    Title      : {record.chapter_title!r}")
            print(f"    Expected   : ch={record.chapter_num}  "
                  f"title={expected.get('title', '?')!r}  "
                  f"start_page={expected.get('start_page', '?')}")
            last_chapter_num = record.chapter_num

    print(f"\n{'=' * 70}")
    print("Done.")
    print("Check: correct chapter numbers, titles match CHAPTER_MAP.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
from config.settings import PDF_PATH
from src.ingest.pdf_parser import parse_pdf

for record in parse_pdf(str(PDF_PATH)):
    if record.page_num == 3:
        print(record.raw_text)
        break