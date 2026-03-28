"""
Ingestion — File Loader (stub).

TODO: Implement loaders for:
  - Plain text (.txt)
  - PDF (.pdf) — suggested lib: pypdf or pymupdf
  - Word (.docx) — suggested lib: python-docx
  - CSV / JSON
  - Markdown (.md)

Current implementation: reads plain text files only.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.base import BaseLoader
from rag.schemas.document import Document, DocumentSource
from rag.utils import get_logger

logger = get_logger(__name__)


class TextFileLoader(BaseLoader):
    """
    Load plain-text files (.txt, .md) from the local filesystem.

    One file → one Document.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def load(self, source: str) -> list[Document]:
        """
        Read a single text file and return it as a Document.

        Args:
            source: Absolute or relative path to the text file.

        Returns:
            A single-element list containing the Document.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path points to a directory.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise ValueError(f"Expected a file, got a directory: {path}")

        logger.info("loading_file", path=str(path))
        content = path.read_text(encoding=self._encoding)

        doc = Document(
            content=content,
            source=str(path),
            source_type=DocumentSource.file,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
            },
        )
        return [doc]


class DirectoryLoader(BaseLoader):
    """
    Load all matching files from a directory.

    TODO:
      - Add recursive option
      - Add file-type filter (e.g. only .txt, .pdf)
      - Add parallel loading with ThreadPoolExecutor
    """

    def __init__(
        self,
        glob_pattern: str = "*.txt",
        encoding: str = "utf-8",
    ) -> None:
        self._glob = glob_pattern
        self._file_loader = TextFileLoader(encoding=encoding)

    def load(self, source: str) -> list[Document]:
        """
        Load all files matching `glob_pattern` from directory `source`.

        Args:
            source: Path to the directory.

        Returns:
            List of Documents, one per matched file.
        """
        directory = Path(source)
        if not directory.is_dir():
            raise ValueError(f"Expected a directory: {directory}")

        paths = sorted(directory.glob(self._glob))
        logger.info("loading_directory", path=str(directory), file_count=len(paths))

        documents: list[Document] = []
        for path in paths:
            try:
                documents.extend(self._file_loader.load(str(path)))
            except Exception as exc:
                logger.warning("skip_file", path=str(path), error=str(exc))

        return documents
