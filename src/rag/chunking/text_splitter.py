"""
Chunking — Text Splitter implementations.

Provides two concrete strategies:
  1. FixedSizeChunker — simple character-count windows with overlap
  2. RecursiveChunker — split on paragraph/sentence/word boundaries
"""

from __future__ import annotations
import numpy as np
from rag.chunking.base import BaseChunker
from rag.schemas.document import Chunk, Document
from rag.embedding.base import BaseEmbedder
from rag.utils import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)

# Separators tried in order by RecursiveChunker
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_SENTENCE_SEPARATORS = [". ", "! ", "? ", "\n"]

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
        """
        logger.info(
            "recursive_chunker",
            message="RecursiveChunker.",
        )
        text_splitter = RecursiveCharacterTextSplitter(
            separators=self._separators,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks_text = text_splitter.split_text(document.content)
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={
                        **document.metadata,
                        "chunker": "recursive",
                    },
                )
            )
        return chunks

class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedder: BaseEmbedder,
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        separators: list[str] | None = None,
        threshold: float = 0.8,

    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedder = embedder
        self._separators = separators or _SENTENCE_SEPARATORS
        self._threshold = threshold
        self._max_chunk_size = chunk_size*4
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    def split(self, document: Document) -> list[Chunk]:
        """
        Split document text using semantic similarity.

        Chunks are split at points where the cosine similarity between adjacent
        chunks drops below a certain threshold.
        """
        logger.info(
            "semantic_chunker",
            message="SemanticChunker.",
        )
        # Handling empty document
        if len(document.content) == 0 or document.content is None:
            logger.warning(
                "semantic_chunker",
                message="Empty document.",
            )
            return []

        sentence_splitter = RecursiveCharacterTextSplitter(
            separators=self._separators,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks_sentence = sentence_splitter.split_text(document.content)
        chunks_sentence_embedded = self._embedder.embed_texts_batched(chunks_sentence)
        current_chunk = [chunks_sentence[0]]
        chunks = []
        for i, chunk_sentence in enumerate(chunks_sentence[1:], start = 1):
            sim = self._cosine_similarity(chunks_sentence_embedded[i], chunks_sentence_embedded[i-1])
            if sim < self._threshold or sum(len(s) for s in current_chunk) >= self._max_chunk_size:
                chunks.append(
                    Chunk(
                    document_id=document.id,
                    content=" ".join(current_chunk),
                    chunk_index=len(chunks),
                    metadata={
                        **document.metadata,
                        "chunker": "semantic",
                    },
                ))
                current_chunk = [chunk_sentence]
            else:
                current_chunk.append(chunk_sentence)
        if current_chunk:
            chunks.append(Chunk(
                    document_id=document.id,
                    content=" ".join(current_chunk),
                    chunk_index=len(chunks),
                    metadata={
                        **document.metadata,
                        "chunker": "semantic",
                    },
                ))
        return chunks