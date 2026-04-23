"""
Manual spot-check for chunker.py.

Run from the project root:
    python3 -m tests.test_chunker

Prints chunk stats and sample content so you can verify:
  - Token counts are within expected bounds
  - Chapter metadata is correct
  - Overlap prefixes look sensible
  - No chunk IDs are duplicated

Failures here corrupt all downstream indexes.
Do not proceed to embedder.py until this output looks correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import statistics

from config.settings import PDF_PATH, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS, CHUNK_TARGET_MIN, CHUNK_TARGET_MAX
from src.ingest.pdf_parser import parse_pdf
from src.ingest.chunker import chunk_pages, count_tokens

PAGES_TO_CHECK = 100   # Only parse this many pages — fast enough for a spot-check


def run():
    print(f"PDF : {PDF_PATH}")
    print(f"Checking first {PAGES_TO_CHECK} pages\n")
    print("=" * 70)

    # Use a temp file so we don't pollute data/processed/chunks.jsonl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)

    def limited_records():
        for rec in parse_pdf(str(PDF_PATH)):
            if rec.page_num > PAGES_TO_CHECK:
                break
            yield rec

    total = chunk_pages(limited_records(), output_path=tmp_path)
    print(f"Chunks written: {total}\n")

    # Read back and analyze
    chunks = []
    with open(tmp_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    tmp_path.unlink()

    if not chunks:
        print("ERROR: No chunks produced.")
        return

    token_counts = [count_tokens(c["text"]) for c in chunks]
    print(f"Stored token stats (content + ~22% overlap prefix):")
    print(f"  min    : {min(token_counts)}")
    print(f"  max    : {max(token_counts)}")
    print(f"  mean   : {statistics.mean(token_counts):.0f}")
    print(f"  median : {statistics.median(token_counts):.0f}")

    # Content tokens = ~stored / 1.22 on average; target content is 700-800,
    # so stored target is roughly 850-975 for non-first chunks in a chapter.
    STORED_TARGET_MIN = 850
    STORED_TARGET_MAX = 980
    HARD_CEILING = int(CHUNK_MAX_TOKENS * 1.6)  # merge-backward can exceed CHUNK_MAX_TOKENS
    below_floor = sum(1 for t in token_counts if t < CHUNK_MIN_TOKENS)
    above_ceil  = sum(1 for t in token_counts if t > HARD_CEILING)
    in_target   = sum(1 for t in token_counts if STORED_TARGET_MIN <= t <= STORED_TARGET_MAX)
    print(f"\n  in expected stored range ({STORED_TARGET_MIN}–{STORED_TARGET_MAX}): "
          f"{in_target}/{total}  ({100*in_target//total}%)")
    print(f"  below floor ({CHUNK_MIN_TOKENS})             : {below_floor}")
    print(f"  above hard ceiling ({HARD_CEILING})       : {above_ceil}")
    print(f"  (Note: first chunk per chapter has no overlap prefix, so stored ≈ raw content)")
    if above_ceil:
        print("  WARNING: chunks above hard ceiling — review merge-backward or split logic")
    if below_floor > 2:
        print("  NOTE: more than 2 undersized chunks — check chapter boundary merges")

    # Duplicate ID check
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        print(f"\nWARNING: {dupes} duplicate chunk IDs — check compute_chunk_id")
    else:
        print(f"\nChunk IDs: all {total} unique  OK")

    # Chapter distribution
    print("\nChapters seen:")
    chapters_seen = {}
    for c in chunks:
        ch = c["chapter_num"]
        chapters_seen[ch] = chapters_seen.get(ch, 0) + 1
    for ch, count in sorted(chapters_seen.items()):
        title = chunks[next(i for i, c in enumerate(chunks) if c["chapter_num"] == ch)]["chapter_title"]
        print(f"  ch{ch:02d}  {count:3d} chunks  {title!r}")

    # Sample: first 3 chunks full text
    print(f"\n{'=' * 70}")
    print("Sample — first 3 chunks:")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"  ID      : {c['chunk_id']}")
        print(f"  Chapter : {c['chapter_num']}  {c['chapter_title']!r}")
        print(f"  Page    : {c['page_num']}")
        print(f"  Tokens  : {count_tokens(c['text'])}")
        print(f"  Text preview (first 300 chars):")
        print("    " + c["text"][:300].replace("\n", "\n    "))

    # Sample: last chunk
    print(f"\n--- Last chunk ---")
    c = chunks[-1]
    print(f"  ID      : {c['chunk_id']}")
    print(f"  Chapter : {c['chapter_num']}  {c['chapter_title']!r}")
    print(f"  Page    : {c['page_num']}")
    print(f"  Tokens  : {count_tokens(c['text'])}")

    # Verify overlap: chunk 2 should share tokens with the end of chunk 1
    if len(chunks) >= 2 and chunks[0]["chapter_num"] == chunks[1]["chapter_num"]:
        text0_end = chunks[0]["text"][-200:]
        text1_start = chunks[1]["text"][:200]
        overlap_words = set(text0_end.split()) & set(text1_start.split())
        print(f"\nOverlap check (ch1 tail ∩ ch2 head): {len(overlap_words)} shared words")
        if len(overlap_words) < 5:
            print("  WARNING: very few shared words — overlap may not be working")
        else:
            print("  OK")

    print(f"\n{'=' * 70}")
    print("Done. Verify:")
    print("  - Stored tokens mostly 850–980 (content 700-800 + ~22% overlap)")
    print("  - First chunk per chapter is lower (no overlap prefix) — expected")
    print(f"  - Zero chunks above hard ceiling ({HARD_CEILING})")
    print("  - Chapter numbers match pages (cross-check against test_pdf_parser output)")
    print("  - Overlap words visible between consecutive same-chapter chunks")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
