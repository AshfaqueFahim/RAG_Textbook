import hashlib
import json
import re
from pathlib import Path
from typing import Generator, Optional

import tiktoken

from config.settings import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_TARGET_MAX,
    CHUNKS_PATH,
    OVERLAP_FRACTION,
    OVERLAP_FRACTION_FALLBACK,
)
from src.ingest.pdf_parser import PageRecord


# ---------------------------------------------------------------------------
# Token counting — encoder loaded once at first use
# ---------------------------------------------------------------------------

_encoder: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Chunk ID
# ---------------------------------------------------------------------------

def compute_chunk_id(chapter_num: int, page_num: int, chunk_text: str) -> str:
    """
    Deterministic 16-char hex ID: SHA-256(chapter_num|page_num|normalized_text).
    Text is whitespace-normalized only for hashing — stored text is unchanged.
    """
    normalized = re.sub(r"\s+", " ", chunk_text.strip())
    raw = f"{chapter_num}|{page_num}|{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Split-point finder
# ---------------------------------------------------------------------------

def _find_split_point(text: str, max_tokens: int) -> int:
    """
    Return the character index where text should be split so that
    text[:index] fits within max_tokens.

    Priority:
      1. Last paragraph break (\\n\\n) before the token limit
      2. Last sentence-ending punctuation (.!?) before the token limit
      3. Hard token boundary (exact max_tokens decode)
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    # Character position of the max_tokens boundary
    hard_limit_text = enc.decode(tokens[:max_tokens])
    hard_limit_pos = len(hard_limit_text)

    # 1. Last paragraph break
    para_pos = text.rfind("\n\n", 0, hard_limit_pos)
    if para_pos > 0:
        return para_pos + 2  # right part starts after the blank line

    # 2. Last sentence break (.!? followed by space or newline)
    for i in range(min(hard_limit_pos, len(text) - 1), 0, -1):
        if text[i - 1] in ".!?" and text[i] in " \n":
            return i  # right part starts at the space; caller strips whitespace

    # 3. Hard token boundary
    return hard_limit_pos


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def add_overlap(current_text: str, previous_text: str) -> str:
    """
    Prepend the last OVERLAP_FRACTION of previous_text (by token count) to
    current_text. Falls back to OVERLAP_FRACTION_FALLBACK if the primary
    fraction would push the combined text over CHUNK_MAX_TOKENS.
    Returns current_text unchanged if previous_text is empty or both
    fractions overflow.
    """
    if not previous_text:
        return current_text
    enc = _get_encoder()
    prev_tokens = enc.encode(previous_text)
    for fraction in (OVERLAP_FRACTION, OVERLAP_FRACTION_FALLBACK):
        n = max(1, int(len(prev_tokens) * fraction))
        overlap_text = enc.decode(prev_tokens[-n:])
        combined = overlap_text + "\n" + current_text
        if count_tokens(combined) <= CHUNK_MAX_TOKENS:
            return combined
    return current_text  # skip overlap rather than truncate content


# ---------------------------------------------------------------------------
# Chunk type
# ---------------------------------------------------------------------------

class Chunk:
    __slots__ = ("chunk_id", "chapter_num", "chapter_title", "page_num", "text")

    def __init__(
        self,
        chunk_id: str,
        chapter_num: int,
        chapter_title: str,
        page_num: int,
        text: str,
    ):
        self.chunk_id = chunk_id
        self.chapter_num = chapter_num
        self.chapter_title = chapter_title
        self.page_num = page_num
        self.text = text

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chapter_num": self.chapter_num,
            "chapter_title": self.chapter_title,
            "page_num": self.page_num,
            "text": self.text,
        }


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _append_chunk(chunk: Chunk, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def _rewrite_last_chunk(chunk: Chunk, path: Path) -> None:
    """Replace the last newline-terminated record in the JSONL file."""
    with open(path, "rb+") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return
        # Walk backward from (size-2) to skip the trailing \n and find
        # the \n that ends the second-to-last record.
        pos = size - 2
        while pos > 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                pos += 1  # last record starts just after this \n
                break
            pos -= 1
        # If pos reached 0 the last record is also the first; start at 0.
        f.seek(pos)
        f.write((json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n").encode("utf-8"))
        f.truncate()


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_pages(
    page_records: Generator[PageRecord, None, None],
    output_path: Path = CHUNKS_PATH,
) -> int:
    """
    Consume a PageRecord generator, split text into chunks, and write each
    chunk to output_path in JSONL append mode.

    Chunking is purely size-based (no section-header splits):
      - Accumulate pages into a buffer.
      - When buffer exceeds CHUNK_MAX_TOKENS, split at the last paragraph
        break, then sentence break, then hard token boundary before
        CHUNK_TARGET_MAX.
      - Each chunk gets a 22 % token overlap prefix from the previous chunk
        (15 % fallback if 22 % would overflow).
      - At chapter boundaries, flush the buffer. If the tail is < CHUNK_MIN_TOKENS,
        merge it backward into the previous chunk instead of creating a stub.
      - No overlap across chapter boundaries.

    Returns the number of chunks written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer_text: str = ""
    buffer_tokens: int = 0
    buffer_start_page: Optional[int] = None
    # Each entry: (page_num, start_char_index_within_buffer)
    buffer_page_markers: list[tuple[int, int]] = []

    current_chapter_num: Optional[int] = None
    current_chapter_title: str = ""
    previous_raw_text: str = ""   # last flushed chunk text WITHOUT overlap
    last_written_chunk: Optional[Chunk] = None
    total_chunks: int = 0

    # --- Helpers ---

    def _page_at(char_pos: int) -> int:
        """Return the page_num that contains char_pos in the current buffer."""
        page = buffer_start_page or 0
        for pnum, start in buffer_page_markers:
            if start <= char_pos:
                page = pnum
            else:
                break
        return page

    def _advance_markers(split_pos: int, right_start_page: int) -> None:
        """Update buffer_page_markers and buffer_start_page after a split."""
        nonlocal buffer_page_markers, buffer_start_page
        new_markers = [(p, s - split_pos) for p, s in buffer_page_markers if s > split_pos]
        # Ensure the right-hand page has a marker at position 0
        if not new_markers or new_markers[0][1] != 0:
            new_markers.insert(0, (right_start_page, 0))
        buffer_page_markers = new_markers
        buffer_start_page = right_start_page

    def _flush(text: str, ch_num: int, ch_title: str, page: int) -> Chunk:
        nonlocal total_chunks, previous_raw_text
        text_with_overlap = add_overlap(text, previous_raw_text)
        cid = compute_chunk_id(ch_num, page, text_with_overlap)
        chunk = Chunk(cid, ch_num, ch_title, page, text_with_overlap)
        _append_chunk(chunk, output_path)
        total_chunks += 1
        previous_raw_text = text  # store WITHOUT overlap for next chunk's prefix
        return chunk

    def _merge_backward(tail: str) -> None:
        nonlocal last_written_chunk, previous_raw_text
        if last_written_chunk is None:
            return
        merged = last_written_chunk.text + "\n" + tail.strip()
        new_cid = compute_chunk_id(
            last_written_chunk.chapter_num,
            last_written_chunk.page_num,
            merged,
        )
        updated = Chunk(
            new_cid,
            last_written_chunk.chapter_num,
            last_written_chunk.chapter_title,
            last_written_chunk.page_num,
            merged,
        )
        _rewrite_last_chunk(updated, output_path)
        last_written_chunk = updated
        previous_raw_text = merged

    def _reset_buffer() -> None:
        nonlocal buffer_text, buffer_tokens, buffer_start_page, buffer_page_markers
        buffer_text = ""
        buffer_tokens = 0
        buffer_start_page = None
        buffer_page_markers = []

    # --- Main loop ---

    for record in page_records:
        # Chapter boundary: flush or merge whatever is in the buffer
        if record.chapter_num != current_chapter_num:
            if buffer_text.strip():
                if buffer_tokens >= CHUNK_MIN_TOKENS:
                    last_written_chunk = _flush(
                        buffer_text,
                        current_chapter_num,
                        current_chapter_title,
                        buffer_start_page,
                    )
                elif last_written_chunk is not None:
                    _merge_backward(buffer_text)
                else:
                    # No previous chunk to merge into — save undersized stub
                    last_written_chunk = _flush(
                        buffer_text,
                        current_chapter_num,
                        current_chapter_title,
                        buffer_start_page,
                    )
            _reset_buffer()
            previous_raw_text = ""  # no overlap across chapter boundaries
            current_chapter_num = record.chapter_num
            current_chapter_title = record.chapter_title

        # Add page text to buffer
        page_text = record.raw_text.strip()
        if not page_text:
            continue

        if buffer_text:
            start_pos = len(buffer_text) + 1  # +1 for the \n separator
            buffer_text = buffer_text + "\n" + page_text
        else:
            start_pos = 0
            buffer_text = page_text
            buffer_start_page = record.page_num

        buffer_page_markers.append((record.page_num, start_pos))
        buffer_tokens = count_tokens(buffer_text)

        # Split while buffer exceeds the hard ceiling
        while buffer_tokens > CHUNK_MAX_TOKENS:
            split_pos = _find_split_point(buffer_text, CHUNK_TARGET_MAX)
            left = buffer_text[:split_pos].rstrip("\n ")
            right_raw = buffer_text[split_pos:]
            right = right_raw.lstrip("\n ")
            left_tokens = count_tokens(left)
            right_start_page = _page_at(split_pos)

            if left_tokens >= CHUNK_MIN_TOKENS:
                last_written_chunk = _flush(
                    left, current_chapter_num, current_chapter_title, buffer_start_page
                )
                lstrip_offset = len(right_raw) - len(right)
                _advance_markers(split_pos + lstrip_offset, right_start_page)
                buffer_text = right
                buffer_tokens = count_tokens(buffer_text)
            else:
                # Paragraph/sentence split was too far left — hard token split
                enc = _get_encoder()
                all_tokens = enc.encode(buffer_text)
                left_hard = enc.decode(all_tokens[:CHUNK_TARGET_MAX])
                right_hard_raw = buffer_text[len(left_hard):]
                right_hard = right_hard_raw.lstrip("\n ")
                right_hard_page = _page_at(len(left_hard))
                last_written_chunk = _flush(
                    left_hard.rstrip(),
                    current_chapter_num,
                    current_chapter_title,
                    buffer_start_page,
                )
                lstrip_offset = len(right_hard_raw) - len(right_hard)
                _advance_markers(len(left_hard) + lstrip_offset, right_hard_page)
                buffer_text = right_hard
                buffer_tokens = count_tokens(buffer_text)
                break  # avoid infinite loop on pathological input

    # Final flush
    if buffer_text.strip():
        if buffer_tokens >= CHUNK_MIN_TOKENS:
            _flush(
                buffer_text, current_chapter_num, current_chapter_title, buffer_start_page
            )
        elif last_written_chunk is not None:
            _merge_backward(buffer_text)
        else:
            # Entire input was one tiny buffer
            _flush(
                buffer_text, current_chapter_num, current_chapter_title, buffer_start_page
            )

    return total_chunks
