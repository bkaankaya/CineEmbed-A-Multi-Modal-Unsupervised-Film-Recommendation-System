from fastapi.testclient import TestClient

from cineembed.api import app


def test_gallery_returns_5x3_matrix():
    with TestClient(app) as client:
        r = client.get("/api/gallery")
        assert r.status_code == 200
        body = r.json()
        assert "queries" in body and len(body["queries"]) == 5
        assert "matrix" in body
        for q in body["queries"]:
            for bb in ("ae_z32", "ae_z64", "ae_z128"):
                cell = body["matrix"][q][bb]
                assert "query" in cell
                assert "neighbors" in cell
                assert len(cell["neighbors"]) == 5
