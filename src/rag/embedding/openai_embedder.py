from __future__ import annotations

import openai

from rag.config import get_settings
from rag.embedding.base import BaseEmbedder
from rag.utils import batch, get_logger

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 32  # OpenAI recommends ≤ 2048 texts per call, start small


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding client backed by OpenAI (or OpenAI-compatible API).

    Args:
        model:     Model name override. Defaults to EMBEDDING_MODEL_NAME from settings.
        dimension: Expected vector dimension. Defaults to EMBEDDING_DIMENSION.
    """

    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.embedding_model_name
        self._dimension = dimension or settings.embedding_dimension
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts via the OpenAI embeddings endpoint.
        """
        logger.warning(
            "embedder_stub",
            model=self._model,
            text_count=len(texts),
            message="Returning embedding vectors. Implement OpenAI API call.",
        )
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [sentence.embedding for sentence in response.data]

    def embed_texts_batched(self, texts: list[str]) -> list[list[float]]:
        """
        Embed large text lists by splitting into batches of `_EMBED_BATCH_SIZE`.
        """
        results: list[list[float]] = []
        for text_batch in batch(texts, _EMBED_BATCH_SIZE):
            results.extend(self.embed_texts(text_batch))
        return results
