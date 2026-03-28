"""
Generator — Abstract base generator (LLM client).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas.query import RetrievalResult
from rag.schemas.response import RAGResponse


class BaseGenerator(ABC):
    """
    Abstract LLM generation client.

    Takes a user query + retrieved context chunks and produces a RAGResponse.
    Subclass for OpenAI, Anthropic, local models (Ollama), etc.
    """

    @abstractmethod
    def generate(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
    ) -> RAGResponse:
        """
        Generate an answer given the query and retrieved context.

        Args:
            query_text:       Original user query string.
            retrieval_result: Retrieved & ranked chunks that serve as context.

        Returns:
            A RAGResponse containing the generated answer and metadata.
        """
        ...
