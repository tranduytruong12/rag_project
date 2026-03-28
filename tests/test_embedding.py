"""
Tests — Embedding client.
"""

from __future__ import annotations

from rag.embedding.openai_embedder import OpenAIEmbedder
from rag.schemas.document import Chunk


class TestOpenAIEmbedder:
    def test_embed_texts_returns_correct_count(self) -> None:
        embedder = OpenAIEmbedder()
        texts = ["hello", "world", "RAG"]
        vectors = embedder.embed_texts(texts)
        assert len(vectors) == len(texts)

    def test_embed_texts_returns_correct_dimension(self) -> None:
        embedder = OpenAIEmbedder(dimension=1536)
        vectors = embedder.embed_texts(["test"])
        assert len(vectors[0]) == 1536

    def test_embed_query_returns_single_vector(self) -> None:
        embedder = OpenAIEmbedder()
        vector = embedder.embed_query("What is RAG?")
        assert isinstance(vector, list)
        assert len(vector) == embedder.dimension

    def test_embed_chunks_populates_embedding_field(self, sample_chunk: Chunk) -> None:
        embedder = OpenAIEmbedder()
        sample_chunk.embedding = None  # clear it first
        result = embedder.embed_chunks([sample_chunk])
        assert result[0].embedding is not None
        assert len(result[0].embedding) == embedder.dimension

    def test_dimension_property(self) -> None:
        embedder = OpenAIEmbedder(dimension=768)
        assert embedder.dimension == 768
