"""
Vector Store — ChromaDB client (stub).

TODO:
  - Install `chromadb` package
  - Implement `add_chunks()` using collection.add()
  - Implement `search()` using collection.query()
  - Implement `delete_by_document_id()` using collection.delete()
  - Handle collection creation / loading in __init__
  - Add optional persistent vs in-memory mode toggle
"""

from __future__ import annotations

from rag.config import get_settings
from rag.schemas.document import Chunk
from rag.utils import get_logger
from rag.vector_store.base import BaseVectorStore

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store backed by ChromaDB.

    Stub only — no actual ChromaDB calls are made yet.

    Args:
        collection_name: Name of the Chroma collection.
        persist_dir:     Directory for persistent storage. ``None`` = in-memory.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name or settings.vector_store_collection
        self._persist_dir = persist_dir or str(settings.chroma_persist_dir)
        self._client = None      # TODO: chromadb.Client() or chromadb.PersistentClient()
        self._collection = None  # TODO: client.get_or_create_collection(self._collection_name)
        logger.info(
            "chroma_store_init_stub",
            collection=self._collection_name,
            persist_dir=self._persist_dir,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        TODO: Implement using chromadb collection.add().

        Expected logic:
            self._collection.add(
                ids=[c.id for c in chunks],
                embeddings=[c.embedding for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
        """
        logger.warning("chroma_add_chunks_stub", chunk_count=len(chunks))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        TODO: Implement using chromadb collection.query().

        Expected logic:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filters,
            )
            # Parse results["ids"], results["documents"], results["distances"]
        """
        logger.warning("chroma_search_stub", top_k=top_k)
        return []

    def delete_by_document_id(self, document_id: str) -> None:
        """
        TODO: Implement using collection.delete(where={"document_id": document_id}).
        """
        logger.warning("chroma_delete_stub", document_id=document_id)

    def count(self) -> int:
        """
        TODO: Implement using collection.count().
        """
        logger.warning("chroma_count_stub")
        return 0
