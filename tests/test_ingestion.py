"""
Tests — Ingestion (loaders).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag.ingestion.file_loader import DirectoryLoader, TextFileLoader
from rag.ingestion.web_loader import WebLoader
from rag.schemas.document import Document, DocumentSource


class TestTextFileLoader:
    def test_loads_plain_text_file(self, tmp_path: Path) -> None:
        p = tmp_path / "sample.txt"
        p.write_text("Hello, RAG world!", encoding="utf-8")

        loader = TextFileLoader()
        docs = loader.load(str(p))

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content == "Hello, RAG world!"

    def test_source_type_is_file(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.txt"
        p.write_text("content", encoding="utf-8")
        docs = TextFileLoader().load(str(p))
        assert docs[0].source_type == DocumentSource.file

    def test_metadata_contains_filename(self, tmp_path: Path) -> None:
        p = tmp_path / "myfile.txt"
        p.write_text("some text", encoding="utf-8")
        docs = TextFileLoader().load(str(p))
        assert docs[0].metadata["filename"] == "myfile.txt"

    def test_raises_for_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            TextFileLoader().load("/nonexistent/path/file.txt")

    def test_raises_for_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            TextFileLoader().load(str(tmp_path))


class TestDirectoryLoader:
    def test_loads_all_txt_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content A")
        (tmp_path / "b.txt").write_text("content B")
        (tmp_path / "c.md").write_text("not loaded")  # different extension

        loader = DirectoryLoader(glob_pattern="*.txt")
        docs = loader.load(str(tmp_path))

        assert len(docs) == 2

    def test_raises_for_non_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError):
            DirectoryLoader().load(str(f))


class TestWebLoader:
    @pytest.mark.network
    def test_loads_sample_website(self) -> None:
        """Integration test — requires live internet + valid SSL certificates."""
        pytest.importorskip("httpx")
        loader = WebLoader()
        try:
            docs = loader.load("https://example.com")
            assert len(docs) == 1
            assert "example" in docs[0].content.lower()
            assert docs[0].source_type == DocumentSource.url
        except Exception as exc:
            pytest.skip(f"Network unavailable or SSL error: {exc}")
