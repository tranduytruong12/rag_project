"""
Ingestion — Abstract base loader.

All concrete loaders must inherit from BaseLoader and implement `load()`.
This abstraction lets the pipeline swap backends without touching pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.document import Document


class BaseLoader(ABC):
    """
    Abstract document loader.

    Implementations should handle a specific source type
    (local files, URLs, databases, …).
    """

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """
        Load documents from `source`.

        Args:
            source: A file path, URL, database connection string, etc.

        Returns:
            A list of raw Document objects ready for chunking.

        Raises:
            ValueError: If the source is invalid or unsupported.
            IOError:    If the source cannot be read.
        """
        ...

    def load_many(self, sources: list[str]) -> list[Document]:
        """
        Load documents from multiple sources.

        Default implementation calls `load()` for each source.
        Override for batch-optimised loading.
        """
        documents: list[Document] = []
        for source in sources:
            documents.extend(self.load(source))
        return documents
