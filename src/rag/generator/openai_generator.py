"""
Generator — OpenAI Chat Completion (stub).

TODO:
  - Install `openai` package
  - Wire up openai.ChatCompletion.create() or AsyncOpenAI.chat.completions.create()
  - Add retry logic (tenacity)
  - Support streaming responses
  - Track token usage from response object
  - Add system prompt injection from settings
"""

from __future__ import annotations

import time

from rag.config import get_settings
from rag.generator.base import BaseGenerator
from rag.prompts.templates import PromptBuilder
from rag.schemas.query import RetrievalResult
from rag.schemas.response import FinishReason, RAGResponse
from rag.utils import get_logger

logger = get_logger(__name__)


class OpenAIGenerator(BaseGenerator):
    """
    LLM generator backed by the OpenAI chat completions API.

    Stub only — API calls not yet implemented.

    Args:
        model:       Model name override. Defaults to LLM_MODEL_NAME from settings.
        temperature: Sampling temperature override.
        max_tokens:  Max tokens override.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_model_name
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._prompt_builder = PromptBuilder()
        # TODO: self._client = openai.OpenAI(api_key=settings.openai_api_key, ...)

    def generate(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
    ) -> RAGResponse:
        """
        Build prompt and call the LLM to generate an answer.

        TODO:
            messages = self._prompt_builder.build_messages(query_text, retrieval_result)
            t0 = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = response.choices[0].message.content
            usage = response.usage
        """
        logger.warning(
            "generator_stub",
            model=self._model,
            query_preview=query_text[:80],
            message="Returning stub answer. Implement OpenAI API call.",
        )

        t0 = time.perf_counter()
        # Stub: echo the query with a placeholder answer
        stub_answer = (
            f"[STUB] RAG pipeline answered: '{query_text}'\n"
            f"Context chunks available: {len(retrieval_result.chunks)}\n"
            "TODO: Connect OpenAIGenerator to the real LLM API."
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            query_id=retrieval_result.query_id,
            answer=stub_answer,
            source_chunks=retrieval_result.chunks,
            model_name=self._model,
            finish_reason=FinishReason.stop,
            prompt_tokens=0,     # TODO: fill from response.usage
            completion_tokens=0, # TODO: fill from response.usage
            latency_ms=latency_ms,
        )
