"""
Reranker — Abstract base reranker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.query import RetrievalResult


class BaseReranker(ABC):
    """
    Abstract reranker.

    Rerankers take an initial RetrievalResult and reorder chunks
    using a more expensive (but more accurate) relevance model.
    """

    @abstractmethod
    def rerank(self, result: RetrievalResult) -> RetrievalResult:
        """
        Rerank chunks in `result` and return the reordered RetrievalResult.

        Args:
            result: Output from the retriever with initial ranking.

        Returns:
            A new RetrievalResult with chunks re-ordered by the reranker.
        """
        ...
