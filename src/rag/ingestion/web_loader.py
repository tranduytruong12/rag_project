"""
Ingestion — Web / URL Loader (stub).

TODO: Implement HTTP content fetching with:
  - httpx for async requests
  - HTML stripping (beautifulsoup4 or trafilatura)
  - robots.txt respect
  - rate limiting / retry logic
  - sitemap crawling
"""

from __future__ import annotations

import httpx
import trafilatura

from rag.ingestion.base import BaseLoader
from rag.schemas.document import Document, DocumentSource
from rag.utils import get_logger

logger = get_logger(__name__)


class WebLoader(BaseLoader):
    """
    Load documents from web URLs.

    Stub only — fetching and HTML extraction not yet implemented.

    TODO:
      - Use `httpx.AsyncClient` for async HTTP
      - Strip HTML tags (trafilatura recommended for main-content extraction)
      - Follow redirects, handle 4xx/5xx gracefully
      - Add optional authentication headers
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._timeout = timeout_seconds

    def load(self, source: str) -> list[Document]:
        """
        Fetch and parse content from `source` URL.

        Args:
            source: A fully-qualified URL (https://...).

        Returns:
            A single-element list with the page content as a Document.

        Raises:
            httpx.HTTPError: If the request fails.
            ValueError: If the content cannot be extracted or is empty.
        """
        logger.info("web_loader_fetch", url=source)

        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            response = client.get(source)
            response.raise_for_status()
            html_content = response.text

        # Use trafilatura to extract the main text content, stripping boilerplate
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )

        if not extracted_text or not extracted_text.strip():
            logger.error("web_loader_empty_content", url=source)
            raise ValueError(f"Content extraction returned empty text for URL: {source}")

        doc = Document(
            content=extracted_text.strip(),
            source=source,
            source_type=DocumentSource.url,
            metadata={
                "url": source,
                "content_length": len(extracted_text),
            },
        )
        return [doc]
