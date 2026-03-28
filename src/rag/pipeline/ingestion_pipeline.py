"""
Pipeline — Ingestion Pipeline.

Orchestrates: load → chunk → embed → store.

This is the "write path" — it takes raw documents and indexes them
into the vector store so they are searchable.

Usage::

    pipeline = IngestionPipeline(
        loader=TextFileLoader(),
        chunker=FixedSizeChunker(chunk_size=512, chunk_overlap=50),
        embedder=OpenAIEmbedder(),
        vector_store=ChromaVectorStore(),
    )
    result = pipeline.run(sources=["data/raw/my_doc.txt"])
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.chunking.base import BaseChunker
from rag.config import get_settings
from rag.embedding.base import BaseEmbedder
from rag.ingestion.base import BaseLoader
from rag.schemas.document import Chunk, Document
from rag.utils import batch, get_logger
from rag.vector_store.base import BaseVectorStore

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    sources_processed: int = 0
    documents_loaded: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class IngestionPipeline:
    """
    Full ingestion pipeline: source → Document → Chunk → Embedding → VectorStore.

    Args:
        loader:       Concrete BaseLoader (file, URL, …).
        chunker:      Concrete BaseChunker.
        embedder:     Concrete BaseEmbedder.
        vector_store: Concrete BaseVectorStore.
        embed_batch_size: Number of chunks per embedding API call.
    """

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        embed_batch_size: int = 32,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._embed_batch_size = embed_batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, sources: list[str]) -> IngestionResult:
        """
        Run the full ingestion pipeline for a list of sources.

        Args:
            sources: File paths, URLs, or other source identifiers.

        Returns:
            IngestionResult summary.
        """
        result = IngestionResult()

        for source in sources:
            try:
                self._ingest_source(source, result)
            except Exception as exc:
                logger.error("ingestion_source_error", source=source, error=str(exc))
                result.errors.append(f"{source}: {exc}")

        logger.info(
            "ingestion_complete",
            documents=result.documents_loaded,
            chunks_stored=result.chunks_stored,
            errors=len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _ingest_source(self, source: str, result: IngestionResult) -> None:
        """Load, chunk, embed, and store a single source."""
        result.sources_processed += 1

        # Step 1 — Load
        documents: list[Document] = self._loader.load(source)
        result.documents_loaded += len(documents)
        logger.info("step_load", source=source, doc_count=len(documents))

        # Step 2 — Chunk
        chunks: list[Chunk] = []
        for doc in documents:
            doc_chunks = self._chunker.split(doc)
            chunks.extend(doc_chunks)
        result.chunks_created += len(chunks)
        logger.info("step_chunk", chunk_count=len(chunks))

        # Step 3 — Embed (in batches)
        all_chunks_embedded: list[Chunk] = []
        for chunk_batch in batch(chunks, self._embed_batch_size):
            embedded = self._embedder.embed_chunks(chunk_batch)
            all_chunks_embedded.extend(embedded)
        result.chunks_embedded += len(all_chunks_embedded)
        logger.info("step_embed", embedded_count=len(all_chunks_embedded))

        # Step 4 — Store
        self._vector_store.add_chunks(all_chunks_embedded)
        result.chunks_stored += len(all_chunks_embedded)
        logger.info("step_store", stored_count=len(all_chunks_embedded))
