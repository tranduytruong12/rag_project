"""
Script — Ingest documents into the vector store.

Usage::

    # Ingest a single file
    python scripts/ingest.py --source data/raw/my_doc.txt

    # Ingest all .txt files in a directory
    python scripts/ingest.py --source data/raw/ --glob "*.txt"

    # Dry run (no write to vector store)
    python scripts/ingest.py --source data/raw/ --dry-run

TODO:
  - Add support for PDF / DOCX via respective loaders
  - Add deduplication check before re-ingesting
  - Add JSON output mode for CI pipelines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src package is importable when running scripts directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking.text_splitter import FixedSizeChunker
from rag.config import get_settings
from rag.embedding.openai_embedder import OpenAIEmbedder
from rag.ingestion.file_loader import DirectoryLoader, TextFileLoader
from rag.pipeline.ingestion_pipeline import IngestionPipeline
from rag.utils import configure_logging, get_logger
from rag.vector_store.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="File path or directory path to ingest.",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern when source is a directory (default: *.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk documents but do NOT write to the vector store.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()

    source_path = Path(args.source)
    if source_path.is_dir():
        loader = DirectoryLoader(glob_pattern=args.glob)
        sources = [str(source_path)]
    else:
        loader = TextFileLoader()
        sources = [str(source_path)]

    chunker = FixedSizeChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = OpenAIEmbedder()
    vector_store = ChromaVectorStore()

    if args.dry_run:
        logger.info("dry_run_mode", message="Documents will NOT be written to the vector store.")
        # TODO: implement dry-run preview (print chunk count per doc)
        print("[DRY RUN] Ingestion skipped. Implement preview in scripts/ingest.py")
        return

    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    result = pipeline.run(sources=sources)

    print(f"\n{'=' * 50}")
    print(f"  Ingestion {'SUCCEEDED' if result.success else 'COMPLETED WITH ERRORS'}")
    print(f"{'=' * 50}")
    print(f"  Sources processed : {result.sources_processed}")
    print(f"  Documents loaded  : {result.documents_loaded}")
    print(f"  Chunks created    : {result.chunks_created}")
    print(f"  Chunks stored     : {result.chunks_stored}")
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    - {err}")
    print(f"{'=' * 50}\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
