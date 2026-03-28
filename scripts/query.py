"""
Script — Query the RAG pipeline from the command line.

Usage::

    python scripts/query.py --question "What is retrieval-augmented generation?"
    python scripts/query.py --question "Summarise the documents" --top-k 10

TODO:
  - Add interactive REPL mode (--interactive)
  - Add JSON output mode for scripting
  - Add latency breakdown (retrieve / rerank / generate)
  - Add --chat flag for multi-turn mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings
from rag.embedding.openai_embedder import OpenAIEmbedder
from rag.generator.openai_generator import OpenAIGenerator
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.retriever.similarity_retriever import SimilarityRetriever
from rag.utils import configure_logging, get_logger
from rag.vector_store.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("--question", required=True, help="Question to ask the RAG pipeline.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from settings).",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()

    embedder = OpenAIEmbedder()
    vector_store = ChromaVectorStore()
    retriever = SimilarityRetriever(embedder=embedder, vector_store=vector_store)
    generator = OpenAIGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=args.top_k or settings.retrieval_top_k,
    )

    print(f"\nQuestion: {args.question}")
    print("Processing...\n")

    response = pipeline.run(query_text=args.question)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.answer)
    print()
    print(f"Model      : {response.model_name}")
    print(f"Latency    : {response.latency_ms:.1f} ms")
    print(f"Tokens     : {response.total_tokens} (prompt={response.prompt_tokens}, completion={response.completion_tokens})")
    print(f"Sources    : {len(response.source_chunks)} chunks used")
    if response.source_chunks:
        print("\nSource chunks:")
        for rc in response.source_chunks:
            preview = rc.chunk.content[:100].replace("\n", " ")
            print(f"  [{rc.rank}] score={rc.score:.3f} — {preview}…")
    print("=" * 60)


if __name__ == "__main__":
    main()
