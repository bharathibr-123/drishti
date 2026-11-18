"""API guard rails. These run without models: requests are rejected before any search happens."""

from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)  # not used as a context manager, so models are not loaded


def test_search_requires_officer_and_case_reference() -> None:
    r = client.post("/api/search", json={"query": "a man in a red shirt"})
    assert r.status_code == 400
    assert "case reference" in r.json()["detail"]


def test_empty_or_oversized_query_is_rejected() -> None:
    case = {"officer": "MYS-1", "reason": "case 1"}
    assert client.post("/api/search", json={"query": "", **case}).status_code == 422
    assert client.post("/api/search", json={"query": "x" * 301, **case}).status_code == 422


def test_raw_footage_requires_case_reference() -> None:
    r = client.get("/api/video/CAM-01")  # raw footage without a case reference
    assert r.status_code in (400, 404)


def test_watch_requires_case_reference() -> None:
    assert client.post("/api/watch", json={"query": "a man"}).status_code == 400
