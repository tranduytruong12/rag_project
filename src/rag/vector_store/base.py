"""
Vector Store — Abstract base vector store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.document import Chunk


class BaseVectorStore(ABC):
    """
    Abstract vector store client.

    Handles adding, searching, and deleting embedded chunks.
    Subclass for ChromaDB, Qdrant, Pinecone, Weaviate, etc.
    """

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Persist a list of embedded Chunks to the store.

        Args:
            chunks: Chunks with `.embedding` already populated.

        Raises:
            ValueError: If any chunk has no embedding.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for the top-k most similar chunks.

        Args:
            query_vector: Dense query embedding.
            top_k:        Number of results to return.
            filters:      Optional metadata filters (backend-specific format).

        Returns:
            List of (Chunk, similarity_score) tuples, descending by score.
        """
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all chunks belonging to a specific document.

        Args:
            document_id: The parent Document's ID.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks stored."""
        ...
