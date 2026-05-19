"""Tests for TMDbClient — disk cache, atomic writes, key-optional."""
from __future__ import annotations
import json
from pathlib import Path

import httpx
import pytest

from cineembed.tmdb import TMDbClient, make_poster_url, make_backdrop_url


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tmdb"
    d.mkdir()
    return d


@pytest.mark.asyncio
async def test_no_api_key_returns_none(cache_dir: Path):
    client = TMDbClient(api_key=None, cache_dir=cache_dir)
    result = await client.get_enrichment(27205)
    assert result is None
    await client.aclose()


@pytest.mark.asyncio
async def test_cache_hit_avoids_network(cache_dir: Path, monkeypatch):
    blob = {
        "movie": {"poster_path": "/abc.jpg", "tagline": "Your mind is the scene of the crime."},
        "keywords": [{"name": "neo-noir"}, {"name": "heist film"}, {"name": "dream"}],
    }
    (cache_dir / "27205.json").write_text(json.dumps(blob))

    called = {"count": 0}

    async def fake_get(_self, *_args, **_kwargs):
        called["count"] += 1
        return None

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    client = TMDbClient(api_key="dummy", cache_dir=cache_dir)
    result = await client.get_enrichment(27205)
    assert result is not None
    assert result.poster_path == "/abc.jpg"
    assert called["count"] == 0
    await client.aclose()


def test_image_url_construction():
    assert make_poster_url("/abc.jpg") == "https://image.tmdb.org/t/p/w342/abc.jpg"
    assert make_poster_url(None) is None
    assert make_backdrop_url("/x.jpg") == "https://image.tmdb.org/t/p/w1280/x.jpg"
