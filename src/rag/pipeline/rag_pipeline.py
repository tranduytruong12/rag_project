"""
Pipeline — RAG Pipeline (query path).

Orchestrates: query → embed → retrieve → rerank → generate → RAGResponse.

This is the "read path" — it takes a user query and produces an answer
grounded in the indexed documents.

Usage::

    pipeline = RAGPipeline(
        embedder=OpenAIEmbedder(),
        retriever=SimilarityRetriever(embedder, vector_store),
        generator=OpenAIGenerator(),
        reranker=CrossEncoderReranker(),   # optional
    )
    response = pipeline.run(query_text="What is RAG?")
"""

from __future__ import annotations

import time

from rag.config import get_settings
from rag.generator.base import BaseGenerator
from rag.reranker.base import BaseReranker
from rag.retriever.base import BaseRetriever
from rag.schemas.query import Query, RetrievalResult
from rag.schemas.response import RAGResponse
from rag.utils import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG query pipeline.

    Args:
        retriever:  Concrete BaseRetriever.
        generator:  Concrete BaseGenerator (LLM client).
        reranker:   Optional concrete BaseReranker. Skipped if None.
        top_k:      Number of chunks to retrieve (overrides Query default).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        reranker: BaseReranker | None = None,
        top_k: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._reranker = reranker
        settings = get_settings()
        self._default_top_k = top_k or settings.retrieval_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query_text: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Args:
            query_text: Raw question from the user.
            top_k:      Override number of chunks to retrieve.
            filters:    Optional metadata filters passed to the retriever.

        Returns:
            RAGResponse with the generated answer and source chunks.
        """
        t_start = time.perf_counter()

        query = Query(
            text=query_text,
            top_k=top_k or self._default_top_k,
            filters=filters or {},
        )
        logger.info("rag_pipeline_start", query_id=query.id, text_preview=query_text[:80])

        # Step 1 — Retrieve
        retrieval_result: RetrievalResult = self._retriever.retrieve(query)
        logger.info("rag_step_retrieve", chunk_count=len(retrieval_result.chunks))

        # Step 2 — Rerank (optional)
        if self._reranker is not None:
            retrieval_result = self._reranker.rerank(retrieval_result)
            logger.info("rag_step_rerank", chunk_count=len(retrieval_result.chunks))

        # Step 3 — Generate
        response: RAGResponse = self._generator.generate(
            query_text=query_text,
            retrieval_result=retrieval_result,
        )

        # Patch end-to-end latency
        response.latency_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "rag_pipeline_done",
            query_id=query.id,
            latency_ms=round(response.latency_ms, 1),
            tokens=response.total_tokens,
        )
        return response
