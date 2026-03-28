"""
Tests — Pipelines (ingestion & RAG).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunking.text_splitter import FixedSizeChunker
from rag.embedding.openai_embedder import OpenAIEmbedder
from rag.generator.openai_generator import OpenAIGenerator
from rag.ingestion.file_loader import TextFileLoader
from rag.pipeline.ingestion_pipeline import IngestionPipeline, IngestionResult
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.retriever.similarity_retriever import SimilarityRetriever
from rag.schemas.response import RAGResponse
from rag.vector_store.chroma_store import ChromaVectorStore


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        loader=TextFileLoader(),
        chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=20),
        embedder=OpenAIEmbedder(),
        vector_store=ChromaVectorStore(),
    )


def _make_rag_pipeline() -> RAGPipeline:
    embedder = OpenAIEmbedder()
    vector_store = ChromaVectorStore()
    return RAGPipeline(
        retriever=SimilarityRetriever(embedder=embedder, vector_store=vector_store),
        generator=OpenAIGenerator(),
    )


# --------------------------------------------------------------------------
# IngestionPipeline tests
# --------------------------------------------------------------------------

class TestIngestionPipeline:
    def test_run_returns_ingestion_result(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("The quick brown fox. " * 30)

        result = _make_ingestion_pipeline().run(sources=[str(f)])

        assert isinstance(result, IngestionResult)
        assert result.documents_loaded >= 1
        assert result.chunks_created >= 1

    def test_missing_file_is_captured_as_error(self) -> None:
        result = _make_ingestion_pipeline().run(sources=["/no/such/file.txt"])
        assert not result.success
        assert len(result.errors) == 1

    def test_partial_failure_does_not_abort(self, tmp_path: Path) -> None:
        good = tmp_path / "good.txt"
        good.write_text("valid content " * 20)

        result = _make_ingestion_pipeline().run(
            sources=[str(good), "/nonexistent.txt"]
        )
        # good file should still be processed
        assert result.documents_loaded >= 1
        assert len(result.errors) == 1


# --------------------------------------------------------------------------
# RAGPipeline tests
# --------------------------------------------------------------------------

class TestRAGPipeline:
    def test_run_returns_rag_response(self) -> None:
        pipeline = _make_rag_pipeline()
        response = pipeline.run("What is RAG?")
        assert isinstance(response, RAGResponse)

    def test_response_has_non_empty_answer(self) -> None:
        pipeline = _make_rag_pipeline()
        response = pipeline.run("Explain retrieval-augmented generation.")
        assert len(response.answer) > 0

    def test_response_latency_is_positive(self) -> None:
        pipeline = _make_rag_pipeline()
        response = pipeline.run("test query")
        assert response.latency_ms > 0
