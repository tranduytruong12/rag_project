"""
Schemas — Document & Chunk models.

Pydantic v2 models for the core data units flowing through the pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentSource(str, Enum):
    """Origin type of a document."""

    file = "file"
    url = "url"
    database = "database"
    unknown = "unknown"


class Document(BaseModel):
    """
    Represents a raw document before any processing.

    A Document is the atomic unit ingested from a source (file, URL, DB row, …).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier",
    )
    content: str = Field(..., description="Raw text content of the document")
    source: str = Field(default="", description="File path, URL, or table name")
    source_type: DocumentSource = Field(default=DocumentSource.unknown)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (filename, author, page, …)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not blank."""
        if not v.strip():
            raise ValueError("Document content must not be empty.")
        return v

    model_config = {"frozen": False}


class Chunk(BaseModel):
    """
    A sub-segment of a Document produced by the chunking layer.

    Chunks are the units that get embedded and stored in the vector store.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier",
    )
    document_id: str = Field(..., description="ID of the parent Document")
    content: str = Field(..., description="Text content of this chunk")
    chunk_index: int = Field(ge=0, description="Position index within the parent document")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Inherited + chunk-specific metadata",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector representation (populated by embedding layer)",
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure chunk content is not blank."""
        if not v.strip():
            raise ValueError("Chunk content must not be empty.")
        return v

    model_config = {"frozen": False}
