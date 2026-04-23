import chromadb
from chromadb.config import Settings

from config.settings import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME


class ChromaManager:
    def __init__(self, client: chromadb.PersistentClient, collection):
        self._client = client
        self._collection = collection

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        texts: list[str],
    ) -> None:
        """Idempotent upsert — safe to re-run; existing chunk_ids are overwritten."""
        self._collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """
        Return top_k chunks by cosine similarity.

        Each result dict has keys:
            chunk_id, text, metadata (dict), cosine_score (float 0-1)
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine distance = 1 - cosine_similarity (range 0-2 for un-normalized,
            # 0-1 for L2-normalized vectors). text-embedding-3-small outputs normalized
            # vectors, so cosine_score = 1 - distance.
            output.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "cosine_score": 1.0 - dist,
            })
        return output

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        return self._collection.count()


def initialize_chroma(
    db_path: str = str(CHROMA_DB_PATH),
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> ChromaManager:
    """
    Create or load a persistent Chroma collection with cosine similarity metric.
    Safe to call multiple times — returns the existing collection if it already exists.
    """
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return ChromaManager(client, collection)
