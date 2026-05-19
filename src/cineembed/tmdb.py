"""Async TMDb v3 client with disk-LRU cache, token bucket, and per-id dedup.

Per spec §14.5."""

from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter

log = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p"
CACHE_TTL_SEC = 30 * 24 * 60 * 60  # 30 days


@dataclass
class TmdbBlob:
    poster_path: str | None
    backdrop_path: str | None
    tagline: str | None
    keyword_names: list[str]


def make_poster_url(path: str | None) -> str | None:
    return f"{TMDB_IMG_BASE}/w342{path}" if path else None


def make_backdrop_url(path: str | None) -> str | None:
    return f"{TMDB_IMG_BASE}/w1280{path}" if path else None


class TMDbClient:
    """Singleton-style async client.

    Auth precedence: v4 access token (Bearer JWT) > v3 api_key (query param).
    Sending both is redundant; v3 key is not a JWT so it must not be used as Bearer.
    """

    def __init__(
        self,
        api_key: str | None = None,
        access_token: str | None = None,
        cache_dir: Path = Path("artifacts/cache/tmdb"),
    ):
        self.api_key = api_key or None
        self.access_token = access_token or None
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = AsyncLimiter(35, 10)
        headers: dict[str, str] = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        self._client = httpx.AsyncClient(
            base_url=TMDB_BASE,
            headers=headers,
            timeout=10.0,
        )
        self._inflight: dict[int, asyncio.Future[TmdbBlob | None]] = {}

    @property
    def key_configured(self) -> bool:
        return bool(self.access_token or self.api_key)

    def _auth_params(self) -> dict[str, str]:
        if self.access_token:
            return {}
        if self.api_key:
            return {"api_key": self.api_key}
        return {}

    def _cache_path(self, film_id: int) -> Path:
        return self.cache_dir / f"{film_id}.json"

    def _read_cache(self, film_id: int) -> TmdbBlob | None:
        p = self._cache_path(film_id)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > CACHE_TTL_SEC:
            return None
        try:
            raw = json.loads(p.read_text())
            movie = raw.get("movie", {})
            keywords = raw.get("keywords", [])
            return TmdbBlob(
                poster_path=movie.get("poster_path"),
                backdrop_path=movie.get("backdrop_path"),
                tagline=movie.get("tagline") or None,
                keyword_names=[k.get("name", "") for k in keywords if k.get("name")],
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _write_cache_atomic(self, film_id: int, movie: dict, keywords: list) -> None:
        p = self._cache_path(film_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"movie": movie, "keywords": keywords}))
        tmp.replace(p)

    async def _fetch_remote(self, film_id: int) -> TmdbBlob | None:
        if not self.key_configured:
            return None
        params = self._auth_params()
        try:
            async with self.limiter:
                movie_resp = await self._client.get(f"/movie/{film_id}", params=params)
            if movie_resp.status_code == 429:
                await asyncio.sleep(2.0)
                async with self.limiter:
                    movie_resp = await self._client.get(f"/movie/{film_id}", params=params)
            if movie_resp.status_code != 200:
                log.warning("tmdb movie/%d returned %d", film_id, movie_resp.status_code)
                return None
            movie = movie_resp.json()

            async with self.limiter:
                kw_resp = await self._client.get(f"/movie/{film_id}/keywords", params=params)
            keywords = kw_resp.json().get("keywords", []) if kw_resp.status_code == 200 else []

            self._write_cache_atomic(film_id, movie, keywords)
            return TmdbBlob(
                poster_path=movie.get("poster_path"),
                backdrop_path=movie.get("backdrop_path"),
                tagline=movie.get("tagline") or None,
                keyword_names=[k.get("name", "") for k in keywords if k.get("name")],
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.warning("tmdb fetch %d failed: %s", film_id, e)
            return None

    async def get_enrichment(self, film_id: int) -> TmdbBlob | None:
        if not self.key_configured:
            return None
        cached = self._read_cache(film_id)
        if cached is not None:
            return cached
        if film_id in self._inflight:
            return await self._inflight[film_id]
        fut: asyncio.Future[TmdbBlob | None] = asyncio.get_running_loop().create_future()
        self._inflight[film_id] = fut
        try:
            result = await self._fetch_remote(film_id)
            fut.set_result(result)
            return result
        finally:
            del self._inflight[film_id]

    async def aclose(self) -> None:
        await self._client.aclose()
