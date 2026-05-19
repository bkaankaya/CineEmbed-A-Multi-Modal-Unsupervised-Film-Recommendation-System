"""Tests for GET /api/films/{id}/cosine-dist — Task 7.1."""
from fastapi.testclient import TestClient

from cineembed.api import app


def test_cosine_dist_inception():
    with TestClient(app) as client:
        r = client.get("/api/films/27205/cosine-dist", params={"backbone": "ae_z32", "bins": 30})
        assert r.status_code == 200
        body = r.json()
        assert len(body["bins"]) == 31  # n+1 edges
        assert len(body["counts"]) == 30
        assert len(body["top10"]) == 10
        s = body["stats"]
        for k in ("mean", "std", "min", "max", "p50", "p95"):
            assert k in s
