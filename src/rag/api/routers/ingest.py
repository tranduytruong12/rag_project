"""
API Router — Ingestion endpoints.

POST /api/v1/ingest/file    →  ingest a single file path
POST /api/v1/ingest/batch   →  ingest a list of file paths
"""

import shutil
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File
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


class AsyncJobResponse(BaseModel):
    message: str
    status: str = "accepted"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.post(
    "/ingest/file",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a single file asynchronously",
)
async def ingest_file(
    request: IngestFileRequest,
    background_tasks: BackgroundTasks,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> AsyncJobResponse:
    """
    Queue a single document for ingestion.
    """
    background_tasks.add_task(pipeline.run, sources=[request.source_path])
    return AsyncJobResponse(message=f"Ingestion started for {request.source_path}")


@router.post(
    "/ingest/upload",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest a file asynchronously",
)
async def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> AsyncJobResponse:
    """
    Upload a document and queue it for ingestion.
    """
    from rag.config import get_settings
    settings = get_settings()
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    safe_filename = file.filename if file.filename else f"upload_{uuid.uuid4().hex[:8]}"
    file_path = settings.data_raw_dir / safe_filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(pipeline.run, sources=[str(file_path)])
    return AsyncJobResponse(message=f"File {safe_filename} uploaded and ingestion started.")


@router.post(
    "/ingest/batch",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest multiple files asynchronously",
)
async def ingest_batch(
    request: IngestBatchRequest,
    background_tasks: BackgroundTasks,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> AsyncJobResponse:
    """
    Queue multiple documents for ingestion.
    """
    background_tasks.add_task(pipeline.run, sources=request.source_paths)
    return AsyncJobResponse(message=f"Ingestion batch started for {len(request.source_paths)} sources")
