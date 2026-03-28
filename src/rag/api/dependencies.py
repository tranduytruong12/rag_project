"""
API — Dependency injection helpers.

Use FastAPI's `Depends()` to inject shared resources into route handlers.
This ensures:
  - Single instantiation of expensive objects (pipeline, vector store)
  - Easy mocking in tests (override with app.dependency_overrides)

TODO:
  - Replace stub instances with real implementations once backends are wired up
  - Add connection pool / singleton pattern for vector store
  - Add authentication dependency (OAuth2 / API-key header)
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from rag.chunking.text_splitter import FixedSizeChunker
from rag.config import Settings, get_settings
from rag.embedding.openai_embedder import OpenAIEmbedder
from rag.generator.openai_generator import OpenAIGenerator
from rag.ingestion.file_loader import TextFileLoader
from rag.pipeline.ingestion_pipeline import IngestionPipeline
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.retriever.similarity_retriever import SimilarityRetriever
from rag.vector_store.chroma_store import ChromaVectorStore


# --------------------------------------------------------------------------
# Shared singletons (created once per process)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder() -> OpenAIEmbedder:
    """Return a cached OpenAIEmbedder instance."""
    return OpenAIEmbedder()


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Return a cached ChromaVectorStore instance."""
    return ChromaVectorStore()


# --------------------------------------------------------------------------
# Pipeline factories (injected per-request but cheap after first construction)
# --------------------------------------------------------------------------

def get_ingestion_pipeline(
    settings: Settings = Depends(get_settings),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> IngestionPipeline:
    """Build and return an IngestionPipeline."""
    return IngestionPipeline(
        loader=TextFileLoader(),
        chunker=FixedSizeChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedder=embedder,
        vector_store=vector_store,
    )


def get_rag_pipeline(
    settings: Settings = Depends(get_settings),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> RAGPipeline:
    """Build and return a RAGPipeline."""
    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )
    generator = OpenAIGenerator()

    # TODO: conditionally inject reranker based on settings.reranker_enabled
    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        reranker=None,
        top_k=settings.retrieval_top_k,
    )
