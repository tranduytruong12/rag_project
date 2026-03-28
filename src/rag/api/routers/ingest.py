"""
API Router — Ingestion endpoints.

POST /api/v1/ingest/file    →  ingest a single file path
POST /api/v1/ingest/batch   →  ingest a list of file paths

TODO:
  - Add multipart file upload (UploadFile) instead of raw path strings
  - Add background task processing (BackgroundTasks or Celery)
  - Add ingestion status tracking (job ID + polling endpoint)
  - Add input validation for allowed file extensions
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from rag.api.dependencies import get_ingestion_pipeline
from rag.pipeline.ingestion_pipeline import IngestionPipeline, IngestionResult

router = APIRouter()


# --------------------------------------------------------------------------
# Request / Response schemas (API-layer only — not reused in domain logic)
# --------------------------------------------------------------------------

class IngestFileRequest(BaseModel):
    source_path: str = Field(..., description="Absolute or relative path to the file to ingest")


class IngestBatchRequest(BaseModel):
    source_paths: list[str] = Field(..., min_length=1, description="List of file paths to ingest")


class IngestResponse(BaseModel):
    success: bool
    sources_processed: int
    documents_loaded: int
    chunks_stored: int
    errors: list[str]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a single file",
)
async def ingest_file(
    request: IngestFileRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """
    Load, chunk, embed, and store a single document from `source_path`.

    TODO: Replace file-path input with UploadFile for real API usage.
    """
    result: IngestionResult = pipeline.run(sources=[request.source_path])
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": result.errors},
        )
    return _to_response(result)


@router.post(
    "/ingest/batch",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest multiple files",
)
async def ingest_batch(
    request: IngestBatchRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """
    Ingest multiple documents in a single request.

    TODO: Move to background task for large batches.
    """
    result: IngestionResult = pipeline.run(sources=request.source_paths)
    return _to_response(result)


def _to_response(result: IngestionResult) -> IngestResponse:
    return IngestResponse(
        success=result.success,
        sources_processed=result.sources_processed,
        documents_loaded=result.documents_loaded,
        chunks_stored=result.chunks_stored,
        errors=result.errors,
    )
