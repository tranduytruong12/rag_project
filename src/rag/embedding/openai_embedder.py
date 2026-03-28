"""
Embedding — OpenAI Embedder (stub).

TODO:
  - Install `openai` package and wire up `openai.embeddings.create()`
  - Add retry logic with exponential back-off (tenacity)
  - Add batching to respect API rate limits
  - Support async embedding via `AsyncOpenAI`
  - Cache embeddings to avoid re-computing for identical texts
"""

from __future__ import annotations

from rag.config import get_settings
from rag.embedding.base import BaseEmbedder
from rag.utils import batch, get_logger

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 32  # OpenAI recommends ≤ 2048 texts per call, start small


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding client backed by OpenAI (or OpenAI-compatible API).

    Stub only — actual API calls not yet implemented.

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
        # TODO: initialise openai.OpenAI(api_key=..., base_url=...) here

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts via the OpenAI embeddings endpoint.

        TODO: Replace stub with real API call:
            response = self._client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in response.data]
        """
        logger.warning(
            "embedder_stub",
            model=self._model,
            text_count=len(texts),
            message="Returning zero vectors. Implement OpenAI API call.",
        )
        # Stub: return zero vectors so the pipeline can run end-to-end in tests
        return [[0.0] * self._dimension for _ in texts]

    def embed_texts_batched(self, texts: list[str]) -> list[list[float]]:
        """
        Embed large text lists by splitting into batches of `_EMBED_BATCH_SIZE`.

        TODO: wire up once embed_texts() is implemented.
        """
        results: list[list[float]] = []
        for text_batch in batch(texts, _EMBED_BATCH_SIZE):
            results.extend(self.embed_texts(text_batch))
        return results
