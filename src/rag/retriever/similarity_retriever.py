"""
Retriever — Similarity Retriever.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rag.embedding.base import BaseEmbedder
from rag.retriever.base import BaseRetriever
from rag.schemas.query import Query, RetrievalResult, RetrievedChunk
from rag.utils import get_logger
from rag.vector_store.base import BaseVectorStore
from rank_bm25 import BM25Okapi
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

    def retrieve(self, query: Query, extend: bool = False) -> RetrievalResult:
        """
        1. Embed the query text.
        2. Search the vector store for top-k chunks.
        3. Return ranked RetrievalResult.
        """
        top_k = query.top_k
        if extend:
            top_k = query.top_k * 3
        
        logger.info("retrieval_start", query_id=query.id, top_k=top_k)

        query_vector = self._embedder.embed_query(query.text)  # embed query

        raw_results = self._vector_store.search(
            query_vector,
            top_k=top_k,
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

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()
 
    def sparse_retrieve(self, query: Query, extend: bool = False) -> RetrievalResult:
        """
        1. Tokenize the query text.
        2. Search the vector store for top-k chunks.
        3. Return ranked RetrievalResult.
        """
        top_k = query.top_k
        if extend:
            top_k = query.top_k * 3

        logger.info("sparse_retrieval_start", query_id=query.id, top_k=top_k)
        
        chunks = self._vector_store.get_all_chunks()
        bm25 = BM25Okapi([self.tokenize(chunk.content) for chunk in chunks])
        scores = bm25.get_scores(self.tokenize(query.text))
        top_indices = scores.argsort()[-min(len(chunks), top_k):][::-1]

        retrieved_chunks = [
            RetrievedChunk(
                chunk=chunks[idx],
                score=scores[idx],
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]

        logger.info("sparse_retrieval_done", query_id=query.id, result_count=len(retrieved_chunks))

        return RetrievalResult(
            query_id=query.id,
            query_text=query.text,
            chunks=retrieved_chunks,
        )

    def combine_and_retrieve(self, retrieved_chunks_sparse: list[RetrievedChunk], retrieved_chunks_dense: list[RetrievedChunk], rrf_k: int = 10, w: float = 0.5, top_k: int = 36) -> list[RetrievedChunk]:
        """
        Combine sparse and dense retrieval results.

        Args:
            retrieved_chunks_sparse: Sparse retrieval results.
            retrieved_chunks_dense: Dense retrieval results.
            k: Constant for rank normalization.
            w: Weight for sparse retrieval (0 <= w <= 1).

        Returns:
            Combined and re-ranked retrieval results.
        """
        scores = defaultdict(float)
        
        retrieved_chunks_sparse_dict = {retrieved_chunk.chunk.id: retrieved_chunk.chunk for retrieved_chunk in retrieved_chunks_sparse}
        retrieved_chunks_dense_dict = {retrieved_chunk.chunk.id: retrieved_chunk.chunk for retrieved_chunk in retrieved_chunks_dense}

        for retrieved_chunk in retrieved_chunks_sparse:
            scores[retrieved_chunk.chunk.id] += w / (rrf_k + retrieved_chunk.rank)
        for retrieved_chunk in retrieved_chunks_dense:
            scores[retrieved_chunk.chunk.id] += (1 - w) / (rrf_k + retrieved_chunk.rank)
        
        retrieved_chunks_info = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_chunks = []
        for rank, (idx, score) in enumerate(retrieved_chunks_info[:top_k]):
            relevant_chunk = retrieved_chunks_sparse_dict.get(idx) or retrieved_chunks_dense_dict.get(idx)
            retrieved_chunks.append(RetrievedChunk(
                chunk=relevant_chunk,
                score=score,
                rank=rank + 1,
            ))
        return retrieved_chunks

    def hybrid_retrieve(self, query: Query, k: int = 10, w: float = 0.5) -> RetrievalResult:
        """
        1. Embed the query text.
        2. Search the vector store for top-k chunks using both sparse and dense retrieval.
        3. Return ranked RetrievalResult.
        """
        logger.info("hybrid_retrieval_start", query_id=query.id, top_k=query.top_k)

        retrieved_chunks_sparse = self.sparse_retrieve(query, extend=True).chunks
        retrieved_chunks_dense = self.retrieve(query, extend=True).chunks

        retrieved_chunks = self.combine_and_retrieve(retrieved_chunks_sparse, retrieved_chunks_dense, k, w, query.top_k)     
        logger.info("hybrid_retrieval_done", query_id=query.id, result_count=len(retrieved_chunks))
        
        return RetrievalResult(
            query_id=query.id,
            query_text=query.text,
            chunks=retrieved_chunks,
        )
