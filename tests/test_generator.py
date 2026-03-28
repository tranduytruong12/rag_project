"""
Tests — Generator.
"""

from __future__ import annotations

from rag.generator.openai_generator import OpenAIGenerator
from rag.schemas.query import RetrievalResult
from rag.schemas.response import RAGResponse


class TestOpenAIGenerator:
    def test_generate_returns_rag_response(
        self,
        empty_retrieval_result: RetrievalResult,
    ) -> None:
        gen = OpenAIGenerator()
        response = gen.generate(
            query_text="What is RAG?",
            retrieval_result=empty_retrieval_result,
        )
        assert isinstance(response, RAGResponse)

    def test_generate_response_has_answer(
        self,
        empty_retrieval_result: RetrievalResult,
    ) -> None:
        gen = OpenAIGenerator()
        response = gen.generate("Tell me something", empty_retrieval_result)
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_generate_response_query_id_matches(
        self,
        empty_retrieval_result: RetrievalResult,
    ) -> None:
        gen = OpenAIGenerator()
        response = gen.generate("query", empty_retrieval_result)
        assert response.query_id == empty_retrieval_result.query_id

    def test_generate_with_context_chunks(
        self,
        retrieval_result_with_chunks: RetrievalResult,
    ) -> None:
        gen = OpenAIGenerator()
        response = gen.generate("What does the fox do?", retrieval_result_with_chunks)
        assert len(response.source_chunks) == 1
