"""
Tests — FastAPI endpoints (health, ingest, query).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoints:
    def test_health_returns_200(self, api_client: TestClient) -> None:
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self, api_client: TestClient) -> None:
        resp = api_client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_body_has_timestamp(self, api_client: TestClient) -> None:
        resp = api_client.get("/health")
        assert "timestamp" in resp.json()

    def test_readiness_returns_200(self, api_client: TestClient) -> None:
        resp = api_client.get("/health/ready")
        assert resp.status_code == 200

    def test_readiness_has_checks(self, api_client: TestClient) -> None:
        resp = api_client.get("/health/ready")
        assert "checks" in resp.json()


class TestIngestEndpoints:
    def test_ingest_file_missing_source_returns_422(self, api_client: TestClient) -> None:
        """Missing required field → FastAPI returns 422."""
        resp = api_client.post("/api/v1/ingest/file", json={})
        assert resp.status_code == 422

    def test_ingest_file_returns_202_accepted(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/api/v1/ingest/file",
            json={"source_path": "/some/path/file.txt"},
        )
        assert resp.status_code == 202

    def test_ingest_batch_empty_list_returns_422(self, api_client: TestClient) -> None:
        resp = api_client.post("/api/v1/ingest/batch", json={"source_paths": []})
        assert resp.status_code == 422


class TestQueryEndpoints:
    def test_query_returns_200(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/api/v1/query",
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 200

    def test_query_response_has_answer(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/api/v1/query",
            json={"question": "What is RAG?"},
        )
        body = resp.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_query_empty_question_returns_422(self, api_client: TestClient) -> None:
        resp = api_client.post("/api/v1/query", json={"question": ""})
        assert resp.status_code == 422

    def test_chat_returns_200_with_user_message(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello, explain RAG."}]},
        )
        assert resp.status_code == 200

    def test_chat_no_user_message_returns_422(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "assistant", "content": "Hi"}]},
        )
        assert resp.status_code == 422
