"""
Embedding — Abstract base embedder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.document import Chunk


class BaseEmbedder(ABC):
    """
    Abstract embedding client.

    Converts text (Chunks) into dense vector representations.
    Subclass for different backends: OpenAI, Sentence-Transformers, Cohere, etc.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension for this model."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of raw text strings.

        Args:
            texts: Plain-text strings to embed.

        Returns:
            List of float vectors, same length as `texts`.
        """
        ...

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed a list of Chunks, mutating each chunk's `.embedding` field in-place.

        Args:
            chunks: Chunks to embed.

        Returns:
            The same list with `.embedding` populated.
        """
        texts = [c.content for c in chunks]
        vectors = self.embed_texts(texts)
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        return chunks

    def embed_query(self, query_text: str) -> list[float]:
        """
        Embed a single query string.

        Some backends use a separate query-encoder; override as needed.
        """
        return self.embed_texts([query_text])[0]
