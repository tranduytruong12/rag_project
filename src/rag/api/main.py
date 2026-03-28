"""
API — FastAPI application factory.

Assembles all routers and configures lifespan (startup/shutdown).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routers import health, ingest, query
from rag.config import get_settings
from rag.utils import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup: configure logging, validate settings, warm up connections.
    Shutdown: flush logs, close connections.

    TODO:
      - Initialise vector store connection pool
      - Pre-load embedding model if running local inference
      - Ping LLM API for health check
    """
    settings = get_settings()
    configure_logging()
    logger.info(
        "app_startup",
        env=settings.app_env.value,
        debug=settings.app_debug,
        vector_store=settings.vector_store_backend.value,
    )
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    """
    Construct and return the FastAPI application instance.

    Follows the Application Factory pattern so the app can be created
    multiple times in tests with fresh state.
    """
    settings = get_settings()

    app = FastAPI(
        title="RAG Project API",
        description=(
            "Production-ready RAG (Retrieval-Augmented Generation) scaffold. "
            "See /docs for interactive API documentation."
        ),
        version="0.1.0",
        debug=settings.app_debug,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ CORS
    # TODO: Tighten allow_origins in production (replace * with actual domains)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ Routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])

    return app


# Module-level app instance used by uvicorn:
#   uvicorn src.rag.api.main:app --reload
app = create_app()
