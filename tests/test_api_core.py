"""Smoke tests for the FastAPI app core endpoints."""
import pytest
from fastapi.testclient import TestClient

from cineembed.api import app


@pytest.fixture(scope="module")
def client():
    # Context-managed TestClient triggers FastAPI lifespan startup/shutdown.
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["backbones_loaded"]) == {"ae_z32", "ae_z64", "ae_z128"}
    assert body["films"] == 329044
    assert "tmdb_key_configured" in body


def test_backbones_returns_three(client):
    r = client.get("/api/backbones")
    assert r.status_code == 200
    backbones = r.json()
    assert len(backbones) == 3
    ids = {b["id"] for b in backbones}
    assert ids == {"ae_z32", "ae_z64", "ae_z128"}
    preferred = [b for b in backbones if b["preferred"]]
    assert len(preferred) == 1 and preferred[0]["id"] == "ae_z32"


def test_search_inception_top_hit(client):
    r = client.get("/api/films/search", params={"q": "inception", "backbone": "ae_z32"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    titles = [r["title"] for r in rows[:3]]
    assert any("Inception" in t for t in titles)


def test_search_invalid_backbone_400(client):
    r = client.get("/api/films/search", params={"q": "x", "backbone": "bogus"})
    assert r.status_code == 422  # FastAPI Literal validation


def test_search_empty_query_returns_empty(client):
    r = client.get("/api/films/search", params={"q": "", "backbone": "ae_z32"})
    assert r.status_code == 422  # min_length=1
