"""
Schemas — RAG Response models.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from rag.schemas.query import RetrievedChunk


class FinishReason(str, Enum):
    """Why the LLM stopped generating."""

    stop = "stop"
    length = "length"
    error = "error"
    unknown = "unknown"


class RAGResponse(BaseModel):
    """
    The final output returned to the user after the full RAG pipeline.

    Includes the generated answer, source chunks used, and usage metadata.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique response identifier",
    )
    query_id: str = Field(..., description="ID of the originating Query")
    answer: str = Field(..., description="Generated answer from the LLM")
    source_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks used as context to generate the answer",
    )
    model_name: str = Field(default="", description="LLM model that produced the answer")
    finish_reason: FinishReason = Field(default=FinishReason.stop)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0, description="End-to-end latency in ms")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens
