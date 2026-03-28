"""
conftest.py — Shared pytest fixtures.

All fixtures here are available to every test file without explicit import.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.schemas.document import Chunk, Document, DocumentSource
from rag.schemas.query import Query, RetrievalResult, RetrievedChunk


# --------------------------------------------------------------------------
# Settings override — force test environment before any imports touch .env
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force test-safe settings values regardless of local .env."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("APP_DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # suppress noisy logs in tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")


# --------------------------------------------------------------------------
# Domain object fixtures
# --------------------------------------------------------------------------

@pytest.fixture()
def sample_document() -> Document:
    return Document(
        content="The quick brown fox jumps over the lazy dog. " * 20,
        source="test_fixture",
        source_type=DocumentSource.file,
        metadata={"filename": "test.txt"},
    )


@pytest.fixture()
def sample_chunk(sample_document: Document) -> Chunk:
    return Chunk(
        document_id=sample_document.id,
        content="The quick brown fox jumps over the lazy dog.",
        chunk_index=0,
        metadata={"filename": "test.txt"},
        embedding=[0.1] * 1536,
    )


@pytest.fixture()
def sample_query() -> Query:
    return Query(text="What does the fox do?", top_k=3)


@pytest.fixture()
def empty_retrieval_result(sample_query: Query) -> RetrievalResult:
    return RetrievalResult(
        query_id=sample_query.id,
        query_text=sample_query.text,
        chunks=[],
    )


@pytest.fixture()
def retrieval_result_with_chunks(
    sample_query: Query, sample_chunk: Chunk
) -> RetrievalResult:
    return RetrievalResult(
        query_id=sample_query.id,
        query_text=sample_query.text,
        chunks=[
            RetrievedChunk(chunk=sample_chunk, score=0.95, rank=1),
        ],
    )


# --------------------------------------------------------------------------
# FastAPI test client
# --------------------------------------------------------------------------

@pytest.fixture()
def api_client() -> TestClient:
    """Return a synchronous TestClient for the FastAPI app."""
    # Import here to avoid side effects at collection time
    from rag.api.main import create_app
    app = create_app()
    return TestClient(app, raise_server_exceptions=True)
