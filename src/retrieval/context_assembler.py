import re
from pathlib import Path


def assemble_context(fused_results: list[dict]) -> tuple[str, dict]:
    """
    Sort fused results by page_num ascending, format labeled source blocks,
    build citation_map.

    Returns (context_block_string, citation_map) where:
        citation_map = {
            N: { chunk_text, chapter_num, chapter_title, section_header, page_num }
        }
    """
    sorted_results = sorted(fused_results, key=lambda r: r["page_num"])

    source_blocks = []
    citation_map = {}

    for source_num, chunk in enumerate(sorted_results, start=1):
        label = (
            f"[SOURCE {source_num} | "
            f"Chapter {chunk['chapter_num']}: {chunk['chapter_title']} | "
            f"Section: {chunk['section_header']} | "
            f"Page {chunk['page_num']}]"
        )
        source_blocks.append(f"{label}\n{chunk['text']}")
        citation_map[source_num] = {
            "chunk_text": chunk["text"],
            "chapter_num": chunk["chapter_num"],
            "chapter_title": chunk["chapter_title"],
            "section_header": chunk["section_header"],
            "page_num": chunk["page_num"],
        }

    context_block = "\n\n".join(source_blocks)
    return context_block, citation_map


def resolve_citations(
    answer_text: str,
    citation_map: dict,
    num_sources: int,
) -> tuple[str, str, list[int]]:
    """
    Parse [SOURCE N] tags from answer_text, validate each N against 1..num_sources,
    and build a programmatic Sources Used block from citation_map.

    Returns (answer_text, sources_used_block, invalid_citations).
    answer_text is returned unchanged — stripping invalid tags is the caller's job.
    """
    cited_nums = sorted(set(int(n) for n in re.findall(r"\[SOURCE (\d+)\]", answer_text)))

    invalid = [n for n in cited_nums if n < 1 or n > num_sources]
    valid = [n for n in cited_nums if 1 <= n <= num_sources]

    lines = []
    for n in valid:
        entry = citation_map[n]
        lines.append(
            f"[SOURCE {n}] Chapter {entry['chapter_num']}: {entry['chapter_title']} | "
            f"Section: {entry['section_header']} | Page {entry['page_num']}"
        )
    sources_used_block = "Sources Used:\n" + "\n".join(lines)

    return answer_text, sources_used_block, invalid


def validate_index_sync(
    chunks_path: Path,
    chroma_manager,
    bm25_manager,
) -> tuple[bool, dict]:
    """
    Compare chunk counts across chunks.jsonl, Chroma, and BM25.
    Returns (is_synced, counts_dict).

    Count-based only — same size with different IDs would pass. Catches gross
    divergence (e.g. BM25 rebuilt while Chroma was not) but not subtle ID mismatches.
    """
    jsonl_count = 0
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jsonl_count += 1

    counts = {
        "jsonl": jsonl_count,
        "chroma": chroma_manager.count(),
        "bm25": bm25_manager.corpus_size(),
    }
    is_synced = counts["jsonl"] == counts["chroma"] == counts["bm25"]
    return is_synced, counts
