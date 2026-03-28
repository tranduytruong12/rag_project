"""
Chunking — Abstract base chunker.

All concrete chunkers must inherit from BaseChunker and implement `split()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.document import Chunk, Document


class BaseChunker(ABC):
    """
    Abstract text chunker.

    Converts a Document into a list of Chunks.
    Different strategies (fixed-size, recursive, semantic, …) subclass this.
    """

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """
        Split `document` into a list of Chunks.

        Args:
            document: Source Document to split.

        Returns:
            Ordered list of Chunk objects derived from the document.
        """
        ...

    def split_many(self, documents: list[Document]) -> list[Chunk]:
        """
        Split a list of documents, returning all chunks in order.

        Override for optimised batch splitting.
        """
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.split(doc))
        return chunks
