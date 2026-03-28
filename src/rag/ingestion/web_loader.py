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
            NotImplementedError: Until HTTP fetching is implemented.
        """
        logger.warning("web_loader_not_implemented", url=source)
        # TODO: replace with actual httpx + HTML parsing logic
        raise NotImplementedError(
            "WebLoader is not yet implemented. "
            "See TODO in rag/ingestion/web_loader.py for implementation guidance."
        )
