"""
Reranker — Cross-Encoder Reranker (stub).

TODO:
  - Install `sentence-transformers` package
  - Load cross-encoder model: CrossEncoder(model_name)
  - Score (query, chunk) pairs with model.predict()
  - Re-sort chunks by new score
  - Support async / batched inference
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder
from rag.config import get_settings
from rag.reranker.base import BaseReranker
from rag.schemas.query import RetrievalResult, RetrievedChunk
from rag.utils import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using a cross-encoder model (e.g. ms-marco-MiniLM).

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        self._model = CrossEncoder(self._model_name)
        logger.info("reranker_init", model=self._model_name)

    def rerank(self, result: RetrievalResult) -> RetrievalResult:
        """
        Rerank chunks using cross-encoder scores.
        """
        pairs = [(result.query_text, c.chunk.content) for c in result.chunks]
        scores = self._model.predict(pairs)
        reranked_chunks_with_score = sorted(zip(result.chunks, scores), key=lambda x: x[1], reverse=True)
        
        # Pass-through: update ranks to reflect current order
        reranked_chunks = [
            RetrievedChunk(
                chunk=rc[0].chunk,
                score=float(rc[1]),
                rank=new_rank + 1,
            )
            for new_rank, rc in enumerate(reranked_chunks_with_score)
        ]
        return RetrievalResult(
            query_id=result.query_id,
            query_text=result.query_text,
            chunks=reranked_chunks,
        )
