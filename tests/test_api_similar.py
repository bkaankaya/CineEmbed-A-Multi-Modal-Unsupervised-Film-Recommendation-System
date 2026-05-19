"""Tests for GET /api/films/{id}/similar — Task 2.6."""
from fastapi.testclient import TestClient

from cineembed.api import app


def test_similar_inception_returns_neighbors():
    with TestClient(app) as client:
        r = client.get(
            "/api/films/27205/similar",
            params={"backbone": "ae_z32", "limit": 10},
        )
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 10
        assert all(row["id"] != 27205 for row in rows)
        cosines = [row["cosine"] for row in rows]
        assert cosines == sorted(cosines, reverse=True)
        assert cosines[0] > 0.7


def test_similar_invalid_limit():
    with TestClient(app) as client:
        r = client.get(
            "/api/films/27205/similar",
            params={"backbone": "ae_z32", "limit": 999},
        )
        assert r.status_code == 422


def test_similar_unknown_film():
    with TestClient(app) as client:
        r = client.get(
            "/api/films/999999999/similar",
            params={"backbone": "ae_z32"},
        )
        assert r.status_code == 404
