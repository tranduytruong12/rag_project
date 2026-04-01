from __future__ import annotations

import chromadb

from rag.config import get_settings
from rag.schemas.document import Chunk
from rag.utils import get_logger
from rag.vector_store.base import BaseVectorStore

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store backed by ChromaDB.

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
        self._client = chromadb.PersistentClient(path=self._persist_dir)     
        self._collection = self._client.get_or_create_collection(self._collection_name)  
        logger.info(
            "chroma_store_init",
            collection=self._collection_name,
            persist_dir=self._persist_dir,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks into vector DB
        """
        self._collection.add(
            documents=[c.content for c in chunks],
            embeddings=[c.embedding for c in chunks],
            metadatas=[{**c.metadata, "document_id": c.document_id, "chunk_index": c.chunk_index} for c in chunks],
            ids=[c.id for c in chunks],
        )
        logger.info("chroma_add_chunks_successfully", chunk_count=len(chunks))
        
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for top_k chunks similar to the query vector
        """
        results = self._collection.query(
                query_embeddings=[query_vector],
                n_results =top_k,
                where=filters,
                include=["documents", "metadatas", "distances", "embeddings"]
        )
        logger.info("chroma_search_successfully", top_k=top_k)
        return [(Chunk(id=results["ids"][0][i],
        document_id=results["metadatas"][0][i].get("document_id","unknown"),
        content=results["documents"][0][i], 
        embedding=results["embeddings"][0][i],
        chunk_index=results["metadatas"][0][i].get("chunk_index",0), 
        metadata=results["metadatas"][0][i]), # end chunk
        results["distances"][0][i]) # end tuple
                for i in range(len(results["ids"][0]))] #

    def delete_by_document_id(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        logger.info("chroma_delete_successfully", document_id=document_id)
        return None

    def document_exists(self, content_hash: str) -> bool:
        """
        Check if a document with the given content hash is already in ChromaDB.
        """
        results = self._collection.get(
            where={"content_hash": content_hash},
            limit=1,
            include=["metadatas"]
        )
        exists = len(results["ids"]) > 0
        if exists:
            logger.info("document_already_exists", content_hash=content_hash)
        return exists

    def count(self) -> int:
        count = self._collection.count()
        logger.info("chroma_count", count=count)
        return count

    def get_all_chunks(self) -> list[Chunk]:
        """
        Get all documents from the vector store
        """
        results = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(
                id=results["ids"][i],
                document_id=results["metadatas"][i].get("document_id", "unknown"),
                content=results["documents"][i],
                chunk_index=results["metadatas"][i].get("chunk_index", 0),
                metadata=results["metadatas"][i],
            )
            for i in range(len(results["ids"]))
        ]