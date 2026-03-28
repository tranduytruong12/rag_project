"""
Retriever — Abstract base retriever.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.query import Query, RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract retriever.

    Converts a Query into a RetrievalResult by searching the vector store.
    """

    @abstractmethod
    def retrieve(self, query: Query) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for `query`.

        Args:
            query: The user's Query object.

        Returns:
            A RetrievalResult with ranked chunks.
        """
        ...
