"""Tests for GET /api/films/{id} — Task 2.5."""
from fastapi.testclient import TestClient

from cineembed.api import app


def test_film_detail_inception():
    with TestClient(app) as client:
        r = client.get("/api/films/27205", params={"backbone": "ae_z32"})
        assert r.status_code == 200
        f = r.json()
        assert f["id"] == 27205
        assert "Inception" in f["title"]
        assert f["tmdbStatus"] in ("ok", "missing")
        assert isinstance(f["style"], list)
        assert isinstance(f["plot"], list)
        assert isinstance(f["genres"], list)


def test_film_detail_unknown_id():
    with TestClient(app) as client:
        r = client.get("/api/films/999999999", params={"backbone": "ae_z32"})
        assert r.status_code == 404


def test_film_detail_invalid_backbone():
    with TestClient(app) as client:
        r = client.get("/api/films/27205", params={"backbone": "bogus"})
        assert r.status_code == 422
