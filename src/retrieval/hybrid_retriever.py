from src.store.bm25_manager import bm25_search
from config.settings import (
    BM25_SEARCH_TOP_K,
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    RRF_K_CONSTANT,
    VECTOR_SEARCH_TOP_K,
)


def embed_query(
    query: str,
    openai_client,
    model_name: str = EMBEDDING_MODEL,
) -> list[float]:
    response = openai_client.embeddings.create(input=[query], model=model_name)
    return response.data[0].embedding


def rrf_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k_constant: int = RRF_K_CONSTANT,
) -> list[dict]:
    """
    Compute RRF scores from both ranked lists and return a merged list sorted by
    rrf_score descending. Chunks appearing in both lists are naturally boosted.
    Each entry: { chunk_id, rrf_score }
    """
    rrf_scores: dict[str, float] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)

    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)

    return sorted(
        [{"chunk_id": cid, "rrf_score": score} for cid, score in rrf_scores.items()],
        key=lambda x: x["rrf_score"],
        reverse=True,
    )


def hybrid_retrieve(
    query: str,
    chroma_manager,
    bm25_manager,
    openai_client,
    top_k: int = FINAL_TOP_K,
) -> list[dict]:
    """
    Embed query, fetch candidates from Chroma and BM25, fuse via RRF, fetch
    text + metadata for top-k from Chroma by ID.

    Returns list of FusedResult dicts:
        { chunk_id, rrf_score, text, chapter_num, chapter_title, section_header, page_num }
    """
    query_vector = embed_query(query, openai_client)
    vector_results = chroma_manager.similarity_search(query_vector, top_k=VECTOR_SEARCH_TOP_K)
    bm25_results = bm25_search(query, bm25_manager, top_k=BM25_SEARCH_TOP_K)

    merged = rrf_fusion(vector_results, bm25_results)
    top_entries = merged[:top_k]
    top_ids = [r["chunk_id"] for r in top_entries]
    score_by_id = {r["chunk_id"]: r["rrf_score"] for r in top_entries}

    # Fetch text + metadata for top-k IDs from Chroma in one batch call
    fetched = chroma_manager._collection.get(
        ids=top_ids,
        include=["documents", "metadatas"],
    )
    id_to_data = {
        cid: {"text": doc, "metadata": meta}
        for cid, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        )
    }

    # Preserve RRF rank order in the returned list
    fused_results = []
    for cid in top_ids:
        if cid not in id_to_data:
            continue
        data = id_to_data[cid]
        meta = data["metadata"]
        fused_results.append({
            "chunk_id": cid,
            "rrf_score": score_by_id[cid],
            "text": data["text"],
            "chapter_num": meta.get("chapter_num"),
            "chapter_title": meta.get("chapter_title", ""),
            "section_header": meta.get("section_header", ""),
            "page_num": meta.get("page_num"),
        })

    return fused_results
