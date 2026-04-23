import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config.settings import BM25_INDEX_PATH, CHUNKS_PATH


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer — consistent between build and query time."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Manager:
    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]):
        self._bm25 = bm25
        self._chunk_ids = chunk_ids  # parallel to the BM25 internal corpus order

    def corpus_size(self) -> int:
        return len(self._chunk_ids)


def build_bm25_index(chunks_path: Path = CHUNKS_PATH) -> BM25Manager:
    """
    Read all chunks from chunks_path (JSONL), sort by (page_num, chunk_id) for
    deterministic ordering, fit BM25Okapi, and return a BM25Manager instance.

    Sorting guarantees the integer-index → chunk_id mapping is reproducible
    across rebuilds even if chunks.jsonl was written in a different order.
    """
    records = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Deterministic order: page_num ascending, chunk_id as tiebreaker
    records.sort(key=lambda r: (r["page_num"], r["chunk_id"]))

    chunk_ids = [r["chunk_id"] for r in records]
    corpus = [_tokenize(r["text"]) for r in records]

    bm25 = BM25Okapi(corpus)
    return BM25Manager(bm25, chunk_ids)


def save_bm25_index(bm25_manager: BM25Manager, output_path: Path = BM25_INDEX_PATH) -> None:
    """Serialize BM25Manager to disk via pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bm25_manager, f)


def load_bm25_index(index_path: Path = BM25_INDEX_PATH) -> BM25Manager:
    """Deserialize BM25Manager from disk."""
    with open(index_path, "rb") as f:
        return pickle.load(f)


def bm25_search(
    query: str,
    bm25_manager: BM25Manager,
    top_k: int,
) -> list[dict]:
    """
    Score all chunks against query and return the top_k results.

    Each result dict has keys:
        chunk_id (str), bm25_score (float), rank (int, 1-indexed)
    """
    query_tokens = _tokenize(query)
    scores = bm25_manager._bm25.get_scores(query_tokens)

    # Pair scores with chunk_ids, sort descending, take top_k
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    return [
        {
            "chunk_id": bm25_manager._chunk_ids[idx],
            "bm25_score": float(score),
            "rank": rank + 1,
        }
        for rank, (idx, score) in enumerate(ranked)
    ]
