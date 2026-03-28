"""
Tests — Chunking strategies.
"""

from __future__ import annotations

import pytest

from rag.chunking.text_splitter import FixedSizeChunker, RecursiveChunker
from rag.schemas.document import Chunk, Document


def _make_doc(content: str) -> Document:
    return Document(content=content, source="test")


class TestFixedSizeChunker:
    def test_splits_short_text_into_one_chunk(self) -> None:
        doc = _make_doc("Hello world")
        chunks = FixedSizeChunker(chunk_size=512, chunk_overlap=50).split(doc)
        assert len(chunks) == 1

    def test_splits_long_text_into_multiple_chunks(self) -> None:
        long_text = "word " * 300  # ~1500 chars
        doc = _make_doc(long_text)
        chunks = FixedSizeChunker(chunk_size=200, chunk_overlap=20).split(doc)
        assert len(chunks) > 1

    def test_chunk_document_id_matches(self, sample_document: Document) -> None:
        chunks = FixedSizeChunker().split(sample_document)
        for chunk in chunks:
            assert chunk.document_id == sample_document.id

    def test_chunk_index_is_sequential(self, sample_document: Document) -> None:
        chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=10).split(sample_document)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_content_not_empty(self, sample_document: Document) -> None:
        chunks = FixedSizeChunker().split(sample_document)
        for chunk in chunks:
            assert chunk.content.strip()

    def test_overlap_raises_if_gte_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_split_many(self) -> None:
        docs = [_make_doc("Hello " * 50), _make_doc("World " * 50)]
        all_chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=10).split_many(docs)
        assert len(all_chunks) > 2


class TestRecursiveChunker:
    def test_returns_chunks(self, sample_document: Document) -> None:
        chunks = RecursiveChunker(chunk_size=200, chunk_overlap=20).split(sample_document)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
