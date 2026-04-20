"""ChromaDB vector index for semantic search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class VectorIndex:
    """Manages ChromaDB collections for product embeddings."""

    def __init__(self, llm: LLMProvider, persist_dir: str | None = None):
        self.llm = llm
        if persist_dir:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient()
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, domain: str) -> chromadb.Collection:
        if domain not in self._collections:
            self._collections[domain] = self._client.get_or_create_collection(
                name=f"products_{domain}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[domain]

    def upsert_product(
        self,
        product_id: str,
        domain: str,
        summary_text: str,
        product_name: str,
    ) -> None:
        """Embed and upsert a product summary into the vector index."""
        try:
            embedding = self.llm.embed(summary_text)
        except Exception:
            logger.exception("Failed to embed product %s", product_id)
            return

        collection = self._get_collection(domain)
        collection.upsert(
            ids=[product_id],
            embeddings=[embedding],
            documents=[summary_text],
            metadatas=[{"domain": domain, "name": product_name}],
        )

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 50,
    ) -> tuple[list[str], list[float]]:
        """Query the vector index and return (product_ids, similarity_scores).

        Similarity scores are cosine similarities in [0, 1].
        """
        collection = self._get_collection(domain)

        # Check if collection has documents
        if collection.count() == 0:
            return [], []

        try:
            query_embedding = self.llm.embed(query_text)
        except Exception:
            logger.exception("Failed to embed query")
            return [], []

        n_results = min(top_k, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []

        # ChromaDB with cosine space returns cosine distances.
        # Cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
        similarities = [max(0.0, 1.0 - d) for d in distances]

        return ids, similarities

    def clear_domain(self, domain: str) -> None:
        """Remove a domain's collection entirely."""
        collection_name = f"products_{domain}"
        try:
            self._client.delete_collection(collection_name)
            self._collections.pop(domain, None)
        except Exception:
            pass  # Collection may not exist
