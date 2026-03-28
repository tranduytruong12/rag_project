"""
Tests — Retriever.
"""

from __future__ import annotations

from rag.embedding.openai_embedder import OpenAIEmbedder
from rag.retriever.similarity_retriever import SimilarityRetriever
from rag.schemas.query import Query, RetrievalResult
from rag.vector_store.chroma_store import ChromaVectorStore


class TestSimilarityRetriever:
    def _make_retriever(self) -> SimilarityRetriever:
        return SimilarityRetriever(
            embedder=OpenAIEmbedder(),
            vector_store=ChromaVectorStore(),
        )

    def test_retrieve_returns_retrieval_result(self, sample_query: Query) -> None:
        retriever = self._make_retriever()
        result = retriever.retrieve(sample_query)
        assert isinstance(result, RetrievalResult)

    def test_retrieve_preserves_query_id(self, sample_query: Query) -> None:
        retriever = self._make_retriever()
        result = retriever.retrieve(sample_query)
        assert result.query_id == sample_query.id

    def test_retrieve_preserves_query_text(self, sample_query: Query) -> None:
        retriever = self._make_retriever()
        result = retriever.retrieve(sample_query)
        assert result.query_text == sample_query.text

    def test_retrieve_empty_store_returns_no_chunks(self, sample_query: Query) -> None:
        """With stub vector store, should return empty chunk list."""
        retriever = self._make_retriever()
        result = retriever.retrieve(sample_query)
        assert result.chunks == []
