import json
import os
import tempfile
from pathlib import Path

from config.settings import (
    CHUNKS_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    INGESTION_STATE_PATH,
)


def load_ingestion_state(state_path: Path = INGESTION_STATE_PATH) -> dict:
    if not state_path.exists():
        return {
            "status": "in_progress",
            "pdf_hash": None,
            "chunking_version": None,
            "embedded_chunk_count": 0,
            "bm25_chunk_count": 0,
            "last_embedded_chunk_id": None,
            "bm25_built": False,
        }
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def save_ingestion_state(state: dict, state_path: Path = INGESTION_STATE_PATH) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, state_path)
    except Exception:
        os.unlink(tmp)
        raise


def embed_batch(
    texts: list[str],
    openai_client,
    model_name: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    response = openai_client.embeddings.create(input=texts, model=model_name)
    return [item.embedding for item in response.data]


def run_ingestion(
    chunks_path: Path = CHUNKS_PATH,
    state_path: Path = INGESTION_STATE_PATH,
    chroma_manager=None,
    openai_client=None,
) -> None:
    state = load_ingestion_state(state_path)
    resume_id = state.get("last_embedded_chunk_id")

    batch: list[dict] = []

    def flush_batch() -> None:
        if not batch:
            return
        texts = [r["text"] for r in batch]
        embeddings = embed_batch(texts, openai_client)
        chunk_ids = [r["chunk_id"] for r in batch]
        metadatas = [
            {
                "chapter_num": r["chapter_num"],
                "chapter_title": r["chapter_title"],
                "section_header": r.get("section_header") or "",
                "page_num": r["page_num"],
            }
            for r in batch
        ]
        chroma_manager.upsert_chunks(chunk_ids, embeddings, metadatas, texts)
        state["embedded_chunk_count"] += len(batch)
        state["last_embedded_chunk_id"] = chunk_ids[-1]
        save_ingestion_state(state, state_path)
        batch.clear()

    with open(chunks_path, encoding="utf-8") as f:
        if resume_id is None:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch.append(json.loads(line))
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    flush_batch()
        else:
            found = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not found:
                    if record["chunk_id"] == resume_id:
                        found = True
                    continue
                batch.append(record)
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    flush_batch()

            if not found:
                raise RuntimeError(
                    "Resume checkpoint not found in chunks.jsonl. "
                    "Run with --force to re-ingest from scratch."
                )

    flush_batch()
    state["status"] = "complete"
    save_ingestion_state(state, state_path)
