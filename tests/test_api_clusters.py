"""Tests for GET /api/clusters and GET /api/clusters/{k} — Task 5.1."""
from fastapi.testclient import TestClient

from cineembed.api import app


def test_clusters_returns_21():
    with TestClient(app) as client:
        r = client.get("/api/clusters", params={"backbone": "ae_z32"})
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 21
        for c in rows:
            assert "previewFilms" in c
            assert len(c["previewFilms"]) <= 4


def test_cluster_detail_top50():
    with TestClient(app) as client:
        r = client.get("/api/clusters/0", params={"backbone": "ae_z32", "limit": 50})
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == 0
        assert "films" in body
        assert "total" in body
        assert len(body["films"]) <= 50


def test_cluster_invalid_k():
    with TestClient(app) as client:
        r = client.get("/api/clusters/99", params={"backbone": "ae_z32"})
        assert r.status_code == 422
