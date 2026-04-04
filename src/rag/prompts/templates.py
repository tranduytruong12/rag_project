"""
Prompts — Prompt templates & builder.

Keep all prompt strings here so they are easy to review, version, and A/B test.
No prompt logic should live in the generator or pipeline modules.

TODO:
  - Add chat-history-aware prompt (multi-turn)
  - Add citation / source attribution instruction
  - Add language / tone customisation
  - Externalise to YAML or Jinja2 templates for non-dev editing
"""

from __future__ import annotations

from rag.schemas.query import RetrievalResult

# ---------------------------------------------------------------------------
# Raw template strings
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful, precise assistant.

Rules:
- Answer in the same language as the question.
- If context is provided, answer based ONLY on the provided context. Do not make up information not present in the context. Cite the source chunk number when possible (e.g. [1], [2]).
- If no context is provided (or if you are instructed to do so), answer based on your general knowledge.
- If the context provided lacks necessary information, say so clearly.
- Keep answers concise and factual.
"""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

NO_CONTEXT_TEMPLATE = """No relevant context was found for the question below.
Please answer based on your general knowledge, or state that you cannot answer.

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Builder class
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds LLM-ready message lists from a query and retrieval result.

    Returns OpenAI-style message dicts:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._system_prompt = system_prompt

    def build_context_block(self, result: RetrievalResult) -> str:
        """
        Format retrieved chunks into a numbered context block.

        Args:
            result: RetrievalResult from the retriever.

        Returns:
            Multi-line string with numbered chunks.
        """
        if not result.chunks:
            return "(no context)"

        lines: list[str] = []
        for rc in result.chunks:
            header = f"[{rc.rank}] (score={rc.score:.3f})"
            lines.append(f"{header}\n{rc.chunk.content.strip()}")

        return "\n\n".join(lines)

    def build_messages(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
    ) -> list[dict[str, str]]:
        """
        Build the full messages list for a chat completion call.

        Args:
            query_text:       Raw query from the user.
            retrieval_result: Retrieved & ranked chunks.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        context = self.build_context_block(retrieval_result)

        if retrieval_result.chunks:
            user_content = RAG_USER_TEMPLATE.format(
                context=context,
                question=query_text,
            )
        else:
            user_content = NO_CONTEXT_TEMPLATE.format(question=query_text)

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
