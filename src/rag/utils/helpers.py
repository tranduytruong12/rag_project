"""
Utils — Generic helper functions.

Keep this module small. If helpers grow into a theme, extract to a new module.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


def generate_id(text: str) -> str:
    """
    Generate a deterministic short ID from a string using SHA-256.

    Useful for deduplicating documents based on content hash.

    Args:
        text: Source string to hash.

    Returns:
        First 16 hex characters of the SHA-256 hash.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def batch(items: list[Any], size: int) -> Generator[list[Any], None, None]:
    """
    Yield successive batches of `size` from `items`.

    Args:
        items: List to split into batches.
        size:  Maximum number of items per batch.

    Yields:
        Sub-lists of at most `size` items.

    Example::

        for chunk_batch in batch(chunks, 32):
            embeddings = embedder.embed(chunk_batch)
    """
    if size <= 0:
        raise ValueError("Batch size must be > 0")
    for i in range(0, len(items), size):
        yield items[i : i + size]


@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """
    Context manager that prints elapsed time in milliseconds.

    Args:
        label: Human-readable label for the timed block.

    Example::

        with timer("embedding"):
            embeddings = embedder.embed(chunks)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[timer] {label}: {elapsed_ms:.1f} ms")


def truncate_text(text: str, max_chars: int = 200) -> str:
    """
    Truncate text to `max_chars` characters, appending '…' if cut.

    Useful for logging / display without dumping entire documents.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
