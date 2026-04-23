import hashlib
import statistics
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

from config.settings import CHAPTER_MAP

# Pages 1–2 are front matter (cover, copyright). Skip entirely.
_SKIP_PAGES = 2
# Index begins around page 950. Nothing useful past this point.
_PAGE_LIMIT = 950


# ---------------------------------------------------------------------------
# Chapter lookup — built once from CHAPTER_MAP at import time.
# Maps every page number to its chapter number.
# For page N, chapter = highest chapter whose start_page <= N.
# ---------------------------------------------------------------------------

def _build_chapter_lookup() -> list[tuple[int, int]]:
    """
    Returns a sorted list of (start_page, chapter_num) pairs.
    Used by _chapter_for_page to do a simple linear scan.
    Only includes chapters with a valid start_page (> 0).
    """
    entries = [
        (info["start_page"], ch_num)
        for ch_num, info in CHAPTER_MAP.items()
        if info.get("start_page", 0) > 0
    ]
    entries.sort(key=lambda x: x[0])
    return entries

_CHAPTER_LOOKUP = _build_chapter_lookup()


def _chapter_for_page(page_num: int) -> int:
    """
    Return the chapter number for the given 1-indexed page number.
    Returns 0 if the page is before any known chapter start.
    """
    current = 0
    for start_page, ch_num in _CHAPTER_LOOKUP:
        if page_num >= start_page:
            current = ch_num
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PageRecord:
    def __init__(
        self,
        page_num: int,
        chapter_num: int,
        chapter_title: str,
        raw_text: str,
        blocks: list,
    ):
        self.page_num = page_num
        self.chapter_num = chapter_num
        self.chapter_title = chapter_title
        self.section_header = ""  # not detected; kept for interface compatibility
        self.raw_text = raw_text
        self.blocks = blocks  # raw PyMuPDF block list, for downstream use by chunker

    def __repr__(self):
        return (
            f"PageRecord(page={self.page_num}, ch={self.chapter_num}, "
            f"text_len={len(self.raw_text)})"
        )


# ---------------------------------------------------------------------------
# Image-overlap filter
# ---------------------------------------------------------------------------

def _is_inside_image(text_bbox: tuple, image_bboxes: list[tuple]) -> bool:
    """
    Return True if text_bbox overlaps with any bbox in image_bboxes.

    Standard rectangle overlap: two rects overlap when neither is fully to
    the left, right, above, or below the other.
      overlap iff  text.x0 < img.x1  AND  img.x0 < text.x1
                   AND  text.y0 < img.y1  AND  img.y0 < text.y1
    """
    tx0, ty0, tx1, ty1 = text_bbox
    for ix0, iy0, ix1, iy1 in image_bboxes:
        if tx0 < ix1 and ix0 < tx1 and ty0 < iy1 and iy0 < ty1:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str) -> Generator[PageRecord, None, None]:
    """
    Open the PDF lazily and yield one PageRecord per page.

    Chapter assignment uses CHAPTER_MAP start_page values directly —
    no running-header parsing. For each page, the chapter is the highest
    chapter whose start_page <= page_num. Chapter titles come from CHAPTER_MAP.

    Pages 1–2 are skipped (front matter). Stops after page 950 (index).
    Never loads more than one page into memory at a time.
    """
    path = Path(pdf_path)
    doc = fitz.open(str(path))

    try:
        for page_index in range(len(doc)):
            # PyMuPDF page indices are 2 ahead of the printed page numbers.
            page_num = page_index - 1

            if page_num <= _SKIP_PAGES:
                continue
            if page_num > _PAGE_LIMIT:
                break

            page = doc[page_index]
            page_width = page.rect.width

            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]

            # Collect bounding boxes of image blocks, but ignore full-page background
            # images. ocrmypdf embeds a full-page raster scan as an image block on
            # every processed page (bbox ≈ entire page). Including those would cause
            # every text block to overlap and get filtered out. We skip any image
            # whose area exceeds 50% of the page area — those are backgrounds, not
            # figures. Actual figures typically occupy 5–30% of the page.
            page_area = page_width * page.rect.height
            image_bboxes = [
                b["bbox"] for b in blocks
                if b.get("type") == 1
                and (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) < 0.5 * page_area
            ]

            # Keep only text blocks that don't overlap any figure image region.
            text_blocks = [
                b for b in blocks
                if b.get("type") == 0 and not _is_inside_image(b["bbox"], image_bboxes)
            ]

            # Use PyMuPDF's built-in reading order for raw text.
            raw_text = page.get_text("text")

            # Skip pages with no meaningful text — full-page maps, illustrations,
            # and other image-only pages produce empty or near-empty raw_text.
            if len(raw_text.strip()) < 50:
                continue

            chapter_num = _chapter_for_page(page_num)
            chapter_title = CHAPTER_MAP.get(chapter_num, {}).get("title", "")

            yield PageRecord(
                page_num=page_num,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                raw_text=raw_text,
                blocks=text_blocks,
            )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# PDF fingerprinting
# ---------------------------------------------------------------------------

def compute_pdf_hash(pdf_path: str) -> str:
    """
    Return a SHA-256 hex digest of the first 4 MB of the PDF.
    Reading only 4 MB keeps this fast on a 1+ GB file while still
    reliably detecting file replacement.
    """
    chunk_size = 4 * 1024 * 1024  # 4 MB
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        data = f.read(chunk_size)
        hasher.update(data)
    return hasher.hexdigest()
