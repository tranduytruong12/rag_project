"""
Tests — Config / Settings.
"""

from __future__ import annotations

import pytest

from rag.config import get_settings
from rag.config.settings import AppEnv, Settings


class TestSettings:
    def test_loads_without_error(self) -> None:
        """Settings should load successfully with test env vars from conftest."""
        settings = get_settings()
        assert settings is not None

    def test_app_env_is_development(self) -> None:
        settings = get_settings()
        assert settings.app_env == AppEnv.development

    def test_chunk_overlap_less_than_chunk_size(self) -> None:
        settings = get_settings()
        assert settings.chunk_overlap < settings.chunk_size

    def test_retrieval_top_k_positive(self) -> None:
        settings = get_settings()
        assert settings.retrieval_top_k > 0

    def test_invalid_chunk_overlap_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(chunk_size=100, chunk_overlap=200)
