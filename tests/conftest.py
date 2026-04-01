"""
conftest.py — Shared pytest fixtures.

All fixtures here are available to every test file without explicit import.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
import uuid
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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key") # Remember to disable this line if you want to use OpenAI
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", f"test_col_{uuid.uuid4().hex}")

    # Clear cached settings so monkeypatch takes effect for EACH test
    import rag.config.settings
    rag.config.settings._settings = None

@pytest.fixture(autouse=True)
def mock_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock out OpenAI API calls to prevent real network requests during tests."""
    
    # Mock Embeddings
    def side_effect_embed(*args, **kwargs):
        input_data = kwargs.get("input", [])
        inputs = input_data if isinstance(input_data, list) else [input_data]
        res = MagicMock()
        res.data = [MagicMock(embedding=[0.1] * 1536) for _ in inputs]
        return res
        
    mock_create_embedding = MagicMock(side_effect=side_effect_embed)
    try:
        monkeypatch.setattr("openai.resources.embeddings.Embeddings.create", mock_create_embedding)
    except Exception:
        pass # Handle if openai sdk is old/different

    # Mock Chat Completions
    mock_create_chat = MagicMock()
    res_chat = MagicMock()
    res_chat.choices = [MagicMock()]
    res_chat.choices[0].message.content = "Mocked answer"
    mock_create_chat.return_value = res_chat
    try:
        monkeypatch.setattr("openai.resources.chat.completions.Completions.create", mock_create_chat)
    except Exception:
        pass


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
