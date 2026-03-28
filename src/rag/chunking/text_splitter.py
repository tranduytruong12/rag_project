"""
Chunking — Text Splitter implementations (stub).

Provides two concrete strategies:
  1. FixedSizeChunker — simple character-count windows with overlap
  2. RecursiveChunker — split on paragraph/sentence/word boundaries (stub)

TODO:
  - Implement semantic chunking (split on topic boundaries using embeddings)
  - Add token-count-aware chunking (tiktoken for OpenAI models)
  - Respect document structure (headings, tables) for structured sources
"""

from __future__ import annotations

from rag.chunking.base import BaseChunker
from rag.schemas.document import Chunk, Document
from rag.utils import get_logger

logger = get_logger(__name__)

# Separators tried in order by RecursiveChunker
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class FixedSizeChunker(BaseChunker):
    """
    Split document text into fixed-size character windows with overlap.

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Example::

        chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.split(document)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        """Split document text into fixed-size chunks."""
        text = document.content
        step = self._chunk_size - self._chunk_overlap
        chunks: list[Chunk] = []
        idx = 0

        for start in range(0, len(text), step):
            end = start + self._chunk_size
            chunk_text = text[start:end].strip()

            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=idx,
                    metadata={
                        **document.metadata,
                        "char_start": start,
                        "char_end": min(end, len(text)),
                        "chunker": "fixed_size",
                    },
                )
            )
            idx += 1

        logger.debug(
            "document_chunked",
            doc_id=document.id,
            strategy="fixed_size",
            chunk_count=len(chunks),
        )
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Split document text using a hierarchy of separators (paragraph → sentence → word).

    Tries each separator in `separators` order; falls back to the next if chunks
    are still too large.

    TODO:
      - Fully implement recursive splitting logic
      - Respect token limits (tiktoken) instead of character counts
      - Handle code blocks, tables, and lists explicitly
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or _DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Chunk]:
        """
        Recursively split document text.

        Current implementation: delegates to FixedSizeChunker as a placeholder.
        TODO: Replace with true recursive separator logic.
        """
        logger.warning(
            "recursive_chunker_stub",
            message="RecursiveChunker falls back to FixedSizeChunker. Implement recursive logic.",
        )
        fallback = FixedSizeChunker(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks = fallback.split(document)
        # Tag chunks so callers know this is still a stub
        for chunk in chunks:
            chunk.metadata["chunker"] = "recursive_stub"
        return chunks
