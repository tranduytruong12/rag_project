"""
Utils — Structured logging setup.

Uses structlog for structured, leveled logging.
Call `configure_logging()` once at application startup.
"""

from __future__ import annotations

import logging
import sys

import structlog

from rag.config import get_settings


def configure_logging() -> None:
    """
    Configure structlog and the stdlib root logger.

    Call this **once** at application startup (e.g. in main.py lifespan).
    After calling, use `get_logger()` to obtain module-level loggers.
    """
    settings = get_settings()
    log_level = logging.getLevelName(settings.log_level.value)

    # Configure stdlib root logger (captures uvicorn, httpx, etc.)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.app_debug:
        # Human-friendly console output in dev
        renderer: structlog.typing.Processor = structlog.dev.ConsoleRenderer()
    else:
        # JSON output in staging / production
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger.

    Usage::

        logger = get_logger(__name__)
        logger.info("document_loaded", doc_id=doc.id, source=doc.source)
    """
    return structlog.get_logger(name)
