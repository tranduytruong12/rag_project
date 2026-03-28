"""
Retriever — Similarity Retriever (stub).

TODO:
  - Wire embedder.embed_query() → real query vector
  - Wire vector_store.search() → real nearest-neighbor search
  - Add hybrid search (BM25 + dense) option
  - Add metadata pre-filtering
"""

from __future__ import annotations

from rag.embedding.base import BaseEmbedder
from rag.retriever.base import BaseRetriever
from rag.schemas.query import Query, RetrievalResult, RetrievedChunk
from rag.utils import get_logger
from rag.vector_store.base import BaseVectorStore

logger = get_logger(__name__)


class SimilarityRetriever(BaseRetriever):
    """
    Dense similarity retriever backed by vector store + embedder.

    Args:
        embedder:     Embedding client to convert queries to vectors.
        vector_store: Vector store to search against.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: Query) -> RetrievalResult:
        """
        1. Embed the query text.
        2. Search the vector store for top-k chunks.
        3. Return ranked RetrievalResult.

        TODO:
          - Replace stub zero-vector with real embedder call
          - Replace empty search results with real vector_store.search() call
        """
        logger.info("retrieval_start", query_id=query.id, top_k=query.top_k)

        # TODO: query_vector = self._embedder.embed_query(query.text)
        query_vector = self._embedder.embed_query(query.text)  # returns zeros in stub

        # TODO: raw_results = self._vector_store.search(query_vector, top_k=query.top_k, ...)
        raw_results = self._vector_store.search(
            query_vector,
            top_k=query.top_k,
            filters=query.filters or None,
        )

        retrieved_chunks = [
            RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank + 1,
            )
            for rank, (chunk, score) in enumerate(raw_results)
        ]

        logger.info("retrieval_done", query_id=query.id, result_count=len(retrieved_chunks))

        return RetrievalResult(
            query_id=query.id,
            query_text=query.text,
            chunks=retrieved_chunks,
        )
