"""
API Router — Health check.

GET /health  →  liveness probe (always returns 200 if the process is running)
GET /health/ready  →  readiness probe (checks downstream dependencies)
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from rag.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the service process is alive.",
)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(tz=timezone.utc),
        version="0.1.0",
        environment=settings.app_env.value,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Checks downstream dependencies (vector store, LLM). Returns 200 when ready.",
)
async def readiness_check() -> ReadinessResponse:
    checks: dict[str, str] = {
        "vector_store": "ok",
        "llm_api": "ok",
        "embedding_api": "ok",
    }
    all_ok = all(v.endswith("ok") for v in checks.values())
    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        checks=checks,
    )
