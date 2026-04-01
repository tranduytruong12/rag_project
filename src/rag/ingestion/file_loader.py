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

import pypdf

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


class PDFLoader(BaseLoader):
    """
    Load PDF files from the local filesystem.

    Extracts text from all pages and concatenates into one Document.
    """

    def load(self, source: str) -> list[Document]:
        """
        Read a single PDF file and return it as a Document.

        Args:
            source: Absolute or relative path to the PDF file.

        Returns:
            A single-element list containing the Document.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise ValueError(f"Expected a file, got a directory: {path}")

        logger.info("loading_pdf", path=str(path))

        try:
            reader = pypdf.PdfReader(str(path))
            text_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            
            content = "\n\n".join(text_pages)
            if not content.strip():
                logger.warning("pdf_loader_empty_content", path=str(path))
                content = " "  # Provide minimal content to avoid validation error if desired, or raise

        except Exception as e:
            logger.error("pdf_loader_error", path=str(path), error=str(e))
            raise ValueError(f"Failed to read PDF file {path}: {e}") from e

        doc = Document(
            content=content,
            source=str(path),
            source_type=DocumentSource.file,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
                "num_pages": len(reader.pages),
            },
        )
        return [doc]


class DirectoryLoader(BaseLoader):
    """
    Load all matching files from a directory.

    Supported extensions currently: .txt, .md, .pdf
    """

    def __init__(
        self,
        glob_pattern: str = "*.*",
        encoding: str = "utf-8",
    ) -> None:
        self._glob = glob_pattern
        self._text_loader = TextFileLoader(encoding=encoding)
        self._pdf_loader = PDFLoader()

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
                if path.suffix.lower() == ".pdf":
                    documents.extend(self._pdf_loader.load(str(path)))
                elif path.suffix.lower() in [".txt", ".md", ".csv", ".json"]:
                    documents.extend(self._text_loader.load(str(path)))
                else:
                    logger.debug("skip_unsupported_file_type", path=str(path))
            except Exception as exc:
                logger.warning("skip_file", path=str(path), error=str(exc))

        return documents
