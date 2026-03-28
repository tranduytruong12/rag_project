"""
API Router — Query / Chat endpoints.

POST /api/v1/query   →  single-turn Q&A
POST /api/v1/chat    →  multi-turn chat (TODO: memory not yet implemented)

TODO:
  - Add streaming response support (StreamingResponse + SSE)
  - Add conversation memory / session management for /chat
  - Add request-level top_k and filter overrides
  - Add rate limiting middleware
  - Add response caching for repeated queries
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from rag.api.dependencies import get_rag_pipeline
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.schemas.response import RAGResponse

router = APIRouter()


# --------------------------------------------------------------------------
# Request / Response schemas
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(default=5, gt=0, le=20, description="Number of chunks to retrieve")
    filters: dict = Field(default_factory=dict, description="Optional metadata filters")


class SourceChunkResponse(BaseModel):
    chunk_id: str
    content: str
    score: float
    rank: int
    metadata: dict


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    model: str
    sources: list[SourceChunkResponse]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    top_k: int = Field(default=5, gt=0, le=20)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Single-turn RAG query",
    description="Ask a question; the RAG pipeline retrieves context and generates an answer.",
)
async def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    try:
        rag_response: RAGResponse = pipeline.run(
            query_text=request.question,
            top_k=request.top_k,
            filters=request.filters or None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return _to_query_response(rag_response)


@router.post(
    "/chat",
    response_model=QueryResponse,
    summary="Multi-turn chat (stub)",
    description=(
        "Multi-turn conversational RAG. "
        "Currently uses only the last user message. "
        "TODO: Implement conversation memory."
    ),
)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """
    TODO:
      - Extract conversation history from request.messages
      - Compress/summarise history for context window management
      - Pass history to the generator's system prompt
    """
    # Stub: use only the last user message
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        None,
    )
    if not last_user_msg:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No user message found in the conversation.",
        )

    rag_response = pipeline.run(query_text=last_user_msg, top_k=request.top_k)
    return _to_query_response(rag_response)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _to_query_response(rag_response: RAGResponse) -> QueryResponse:
    sources = [
        SourceChunkResponse(
            chunk_id=rc.chunk.id,
            content=rc.chunk.content,
            score=rc.score,
            rank=rc.rank,
            metadata=rc.chunk.metadata,
        )
        for rc in rag_response.source_chunks
    ]
    return QueryResponse(
        query_id=rag_response.query_id,
        answer=rag_response.answer,
        model=rag_response.model_name,
        sources=sources,
        prompt_tokens=rag_response.prompt_tokens,
        completion_tokens=rag_response.completion_tokens,
        latency_ms=rag_response.latency_ms,
    )
