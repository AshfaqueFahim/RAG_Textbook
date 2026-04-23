# RAGPROJECT

Local RAG CLI — answers US history questions from a scanned textbook PDF, with mandatory chapter/section/page citations. Runs on MacBook, no GPU. Uses OpenAI APIs (text-embedding-3-small + gpt-4o-mini).

## How to run things

```bash
# Spot-check the parser (run from project root)
python3 -m tests.test_pdf_parser

# Dump raw PyMuPDF blocks for any page (edit PAGE_RANGE first)
python3 inspect_pdf.py
```

## Current status

`pdf_parser.py` is the only completed module. Everything else (`chunker.py`, `embedder.py`, all of `store/`, `retrieval/`, `generation/`, `cli/`) is an empty stub.

**Active blocker:** The PDF at `data/raw/textbook.pdf` has garbled font encoding on most pages. Fix:
```bash
ocrmypdf --force-ocr data/raw/textbook.pdf data/raw/textbook.pdf
```
After re-OCR, verify with `python3 -m tests.test_pdf_parser` before building `chunker.py`.

## Critical non-obvious facts

**Page numbering:** `page_num = page_index - 1` in `parse_pdf`. PyMuPDF indices are 2 ahead of printed page numbers. Do not change this.

**Chapter detection:** Uses `CHAPTER_MAP` in `config/settings.py` exclusively — no PDF header parsing. `start_page` values are printed page numbers, not PyMuPDF indices.

**raw_text vs text_blocks:** `raw_text` comes from `page.get_text("text")` (PyMuPDF built-in, handles two-column layout). `text_blocks` is a separately filtered list used only for font-size detection in `detect_section_headers`. These are intentionally different.

**Image filter:** `image_bboxes` excludes images covering >50% of page area — those are the full-page background scans ocrmypdf adds. Only figure-sized images (~5–30% of page) are used for filtering.

**Section headers:** Font-size band (median+2, median+4], vowel check per word, following-line capital check. No color filter (all spans are color=0 in this PDF).

## Config

All constants in `config/settings.py`. CHAPTER_MAP is the single source of truth for chapter titles and start pages (34 chapters). `CHUNKING_VERSION` is auto-computed — never set manually. Bump `PARSER_VERSION` when `pdf_parser.py` logic changes to invalidate stale vectors.

## Plan file

Full implementation plan (chunker design, retrieval pipeline, grounding checker, prompt templates, etc.) is at:
`/Users/shirinchakder/.claude/plans/fluffy-skipping-pebble.md`
