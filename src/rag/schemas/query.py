"""
Schemas — Query & Retrieval Result models.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from rag.schemas.document import Chunk


class Query(BaseModel):
    """
    Represents a user's query entering the RAG pipeline.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique query identifier (useful for tracing)",
    )
    text: str = Field(..., min_length=1, description="Raw query text from the user")
    top_k: int = Field(default=5, gt=0, description="Number of chunks to retrieve")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata filters to scope retrieval",
    )


class RetrievedChunk(BaseModel):
    """
    A Chunk paired with a relevance score, produced by the retriever.
    """

    chunk: Chunk
    score: float = Field(
        description="Relevance / similarity score (unbounded, higher or lower depends on algorithm)",
    )
    rank: int = Field(ge=1, description="Rank position after retrieval (1-indexed)")


class RetrievalResult(BaseModel):
    """
    The full output of the retrieval layer for a single Query.
    """

    query_id: str = Field(..., description="ID of the originating Query")
    query_text: str = Field(..., description="Original query text")
    chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks sorted by rank",
    )

    @property
    def top_chunk(self) -> RetrievedChunk | None:
        """Return the most relevant chunk, or None if empty."""
        return self.chunks[0] if self.chunks else None
