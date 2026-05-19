"""CineEmbed FastAPI sidecar — see docs/superpowers/specs/2026-05-18-frontend-backend-integration-design.md"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware

from cineembed.api_models import (
    Backbone,
    Cluster,
    ClusterDetail,
    Film,
    HealthResponse,
    Neighbor,
)
from cineembed.search import FilmSearcher
from cineembed.tmdb import TMDbClient, make_backdrop_url, make_poster_url

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INFERENCE_DIR = REPO_ROOT / "artifacts" / "inference"
BACKBONES_JSON = REPO_ROOT / "artifacts" / "backbones.json"
TMDB_CACHE_DIR = REPO_ROOT / "artifacts" / "cache" / "tmdb"
CLUSTER_OVERRIDE_PATH = INFERENCE_DIR / "cluster_names_override.json"
GALLERY_PATH = INFERENCE_DIR / "gallery.json"

BackboneId = Literal["ae_z32", "ae_z64", "ae_z128"]


class AppState:
    """Holds boot-loaded artifacts for the lifetime of the process."""

    def __init__(self) -> None:
        self.films: pd.DataFrame | None = None
        self.id_to_row: dict[int, int] = {}
        self.row_to_id: list[int] = []
        self.embeddings: dict[str, np.ndarray] = {}
        self.cluster_labels: dict[str, np.ndarray] = {}
        self.cluster_meta: dict[str, list[dict]] = {}
        self.backbones_meta: list[dict] = []
        self.searcher: FilmSearcher | None = None
        self.tmdb: TMDbClient | None = None


state = AppState()


def _load_cluster_meta(backbone: str, overrides: dict) -> list[dict]:
    raw = json.loads((INFERENCE_DIR / backbone / "cluster_meta.json").read_text())
    bb_overrides = overrides.get(backbone, {})
    for c in raw:
        key = str(c["id"])
        if key in bb_overrides:
            c["name"] = bb_overrides[key]
    return raw


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Boot sequence per spec §14.2."""
    log.info("[boot] loading backbones metadata")
    state.backbones_meta = json.loads(BACKBONES_JSON.read_text())

    log.info("[boot] loading films_master.parquet")
    state.films = pd.read_parquet(INFERENCE_DIR / "films_master.parquet", engine="pyarrow")

    log.info("[boot] building id_to_row map")
    state.row_to_id = state.films["id"].astype(int).tolist()
    state.id_to_row = {fid: i for i, fid in enumerate(state.row_to_id)}

    log.info("[boot] loading cluster name overrides")
    overrides = (
        json.loads(CLUSTER_OVERRIDE_PATH.read_text()) if CLUSTER_OVERRIDE_PATH.exists() else {}
    )

    for bb in ("ae_z32", "ae_z64", "ae_z128"):
        log.info("[boot] loading %s embeddings (mmap)", bb)
        state.embeddings[bb] = np.load(INFERENCE_DIR / bb / "embeddings.npy", mmap_mode="r")
        log.info("[boot] loading %s cluster_labels", bb)
        state.cluster_labels[bb] = np.load(INFERENCE_DIR / bb / "cluster_labels.npy")
        state.cluster_meta[bb] = _load_cluster_meta(bb, overrides)

    log.info("[boot] prewarming ae_z32 (one matmul to page-cache)")
    _ = state.embeddings["ae_z32"] @ state.embeddings["ae_z32"][0]

    log.info("[boot] building searcher")
    state.searcher = FilmSearcher(state.films)

    log.info("[boot] tmdb client")
    state.tmdb = TMDbClient(
        api_key=os.environ.get("TMDB_API_KEY"),
        access_token=os.environ.get("TMDB_ACCESS_TOKEN"),
        cache_dir=TMDB_CACHE_DIR,
    )

    log.info("[boot] ready")
    yield
    log.info("[shutdown] closing tmdb client")
    if state.tmdb:
        await state.tmdb.aclose()


app = FastAPI(title="CineEmbed API", version="1.0", lifespan=lifespan)

_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",")],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backbones_loaded=list(state.embeddings.keys()),
        films=len(state.films) if state.films is not None else 0,
        tmdb_key_configured=bool(state.tmdb and state.tmdb.key_configured),
    )


@app.get("/api/backbones", response_model=list[Backbone])
def backbones() -> list[Backbone]:
    return [Backbone(**b) for b in state.backbones_meta]


@app.get("/api/films/search", response_model=list[Film])
def search_films(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    backbone: BackboneId = "ae_z32",
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[Film]:
    """Returns Film[] with TMDb-lazy fields null (call /films/{id} to enrich)."""
    if state.searcher is None:
        raise HTTPException(503, detail="searcher not loaded")
    hits = state.searcher.search(q, limit=limit)
    return [_row_to_film(h["row_idx"], backbone) for h in hits]


def _row_to_film(row_idx: int, backbone: str, with_tmdb_blob=None) -> Film:
    """Convert a films_master row to a Film payload."""
    assert state.films is not None
    r = state.films.iloc[row_idx]
    cluster = int(state.cluster_labels[backbone][row_idx])
    year_val = r["year"]
    year = int(year_val) if pd.notna(year_val) else None

    poster_url = None
    backdrop_url = None
    tagline = None
    style: list[str] = []
    plot: list[str] = []
    status: Literal["ok", "missing"] = "missing"
    if with_tmdb_blob is not None:
        from cineembed.keywords import split_keywords

        poster_url = make_poster_url(with_tmdb_blob.poster_path)
        backdrop_url = make_backdrop_url(with_tmdb_blob.backdrop_path)
        tagline = with_tmdb_blob.tagline
        style, plot = split_keywords(with_tmdb_blob.keyword_names)
        status = "ok"

    country = str(r["country"]) if pd.notna(r["country"]) else None
    genres_val = r["genres"]
    genres = list(genres_val) if genres_val is not None else []
    return Film(
        id=int(r["id"]),
        title=str(r["title"]),
        year=year,
        rating=float(r["vote_average"]),
        votes=int(r["vote_count"]),
        genres=genres,
        country=country,
        duration=float(r["runtime"]) if pd.notna(r["runtime"]) else None,
        language=str(r["original_language"]),
        director=str(r["director_name"]),
        cluster=cluster,
        overview=str(r["overview"]) if pd.notna(r["overview"]) else None,
        time=_decade_label(year) if year else "Mixed era",
        place=country,
        poster_color=_hash_hsl(int(r["id"])),
        poster_url=poster_url,
        backdrop_url=backdrop_url,
        tagline=tagline,
        style=style,
        plot=plot,
        tmdb_status=status,
    )


def _decade_label(year: int) -> str:
    return f"{(year // 10) * 10}s"


def _hash_hsl(film_id: int) -> str:
    """Deterministic posterColor fallback."""
    h = (film_id * 2654435761) % 360
    return f"hsl({h}, 65%, 42%)"


@app.get("/api/films/{film_id}", response_model=Film)
async def film_detail(
    film_id: Annotated[int, PathParam(ge=1)],
    backbone: BackboneId = "ae_z32",
) -> Film:
    if film_id not in state.id_to_row:
        raise HTTPException(404, detail="film not found")
    row_idx = state.id_to_row[film_id]
    blob = await state.tmdb.get_enrichment(film_id) if state.tmdb else None
    return _row_to_film(row_idx, backbone, with_tmdb_blob=blob)


def _compute_cosines(film_id: int, backbone: str) -> np.ndarray:
    row = state.id_to_row[film_id]
    q = state.embeddings[backbone][row]
    return state.embeddings[backbone] @ q


@lru_cache(maxsize=50)
def _compute_cosines_cached(film_id: int, backbone: str) -> tuple:
    """lru_cache can't key ndarrays; wrap as tuple-of-one. Callers don't mutate."""
    return (_compute_cosines(film_id, backbone),)


def get_cosines(film_id: int, backbone: str) -> np.ndarray:
    return _compute_cosines_cached(film_id, backbone)[0]


@app.get("/api/films/{film_id}/similar", response_model=list[Neighbor])
async def similar(
    film_id: Annotated[int, PathParam(ge=1)],
    backbone: BackboneId = "ae_z32",
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[Neighbor]:
    if film_id not in state.id_to_row:
        raise HTTPException(404, detail="film not found")
    cosines = get_cosines(film_id, backbone)
    self_row = state.id_to_row[film_id]
    k = min(limit + 1, len(cosines) - 1)
    top_idx = np.argpartition(-cosines, k)[: k + 1]
    top_idx = top_idx[np.argsort(-cosines[top_idx])]
    # Drop self, slice to limit
    top_idx_filtered: list[int] = [int(i) for i in top_idx if i != self_row][:limit]

    # Top-5 TMDb-enriched in parallel; rest left lazy
    enrich_ids = [int(state.row_to_id[i]) for i in top_idx_filtered[:5]]
    if state.tmdb:
        blobs = await asyncio.gather(
            *(state.tmdb.get_enrichment(fid) for fid in enrich_ids),
            return_exceptions=False,
        )
    else:
        blobs = [None] * len(enrich_ids)
    blob_by_id = dict(zip(enrich_ids, blobs))

    out: list[Neighbor] = []
    for i in top_idx_filtered:
        film_id_int = int(state.row_to_id[i])
        film_payload = _row_to_film(i, backbone, with_tmdb_blob=blob_by_id.get(film_id_int))
        out.append(Neighbor(**film_payload.model_dump(), cosine=float(cosines[i])))
    return out


def _cluster_top_n_rows(backbone: str, k: int, n: int) -> list[int]:
    """Return up to n row indices in cluster k, sorted by popularity DESC."""
    assert state.films is not None
    labels = state.cluster_labels[backbone]
    mask = labels == k
    rows = np.where(mask)[0]
    if len(rows) == 0:
        return []
    pops_raw = cast(pd.Series, pd.to_numeric(state.films["popularity"].iloc[rows], errors="coerce"))
    pops = pops_raw.fillna(0)
    order = np.argsort(-np.asarray(pops.values), kind="stable")
    return rows[order[:n]].tolist()


@app.get("/api/clusters", response_model=list[Cluster])
def clusters(backbone: BackboneId = "ae_z32") -> list[Cluster]:
    meta = state.cluster_meta[backbone]
    out: list[Cluster] = []
    for c in meta:
        preview_rows = _cluster_top_n_rows(backbone, c["id"], 4)
        preview = [_row_to_film(r, backbone) for r in preview_rows]
        out.append(
            Cluster(
                id=c["id"],
                name=c["name"],
                size=c["size"],
                top_genres=c["topGenres"],
                modal_decade=c["modalDecade"],
                preview_films=preview,
            )
        )
    return out


@app.get("/api/clusters/{k}", response_model=ClusterDetail)
async def cluster_detail(
    k: Annotated[int, PathParam(ge=0, le=20)],
    backbone: BackboneId = "ae_z32",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ClusterDetail:
    meta_list = state.cluster_meta[backbone]
    c = next((c for c in meta_list if c["id"] == k), None)
    if c is None:
        raise HTTPException(404, detail="cluster not found")
    rows = _cluster_top_n_rows(backbone, k, limit)

    enrich_ids = [int(state.row_to_id[r]) for r in rows[:5]]
    if state.tmdb:
        blobs = await asyncio.gather(
            *(state.tmdb.get_enrichment(fid) for fid in enrich_ids),
            return_exceptions=False,
        )
    else:
        blobs = [None] * len(enrich_ids)
    blob_by_id = dict(zip(enrich_ids, blobs))

    films = [
        _row_to_film(r, backbone, with_tmdb_blob=blob_by_id.get(int(state.row_to_id[r])))
        for r in rows
    ]
    return ClusterDetail(
        id=c["id"],
        name=c["name"],
        size=c["size"],
        top_genres=c["topGenres"],
        modal_decade=c["modalDecade"],
        preview_films=films[:4],
        films=films,
        total=c["size"],
    )


@app.get("/api/gallery")
def gallery() -> dict:
    if not GALLERY_PATH.exists():
        raise HTTPException(503, detail="gallery.json not built; run scripts/build_gallery.py")
    return json.loads(GALLERY_PATH.read_text())


@app.get("/api/films/{film_id}/cosine-dist")
def cosine_dist(
    film_id: Annotated[int, PathParam(ge=1)],
    backbone: BackboneId = "ae_z32",
    bins: Annotated[int, Query(ge=5, le=100)] = 30,
) -> dict:
    if film_id not in state.id_to_row:
        raise HTTPException(404, detail="film not found")
    assert state.films is not None
    cosines = get_cosines(film_id, backbone)
    self_row = state.id_to_row[film_id]
    mask = np.ones_like(cosines, dtype=bool)
    mask[self_row] = False
    arr = cosines[mask]
    counts, edges = np.histogram(arr, bins=bins, range=(-1.0, 1.0))

    # top-10 from arr (which has self removed)
    top_idx = np.argpartition(-arr, 10)[:10]
    top_idx = top_idx[np.argsort(-arr[top_idx])]
    valid_rows = np.where(mask)[0]
    top10 = []
    for i in top_idx:
        orig_row = int(valid_rows[i])
        top10.append({
            "id": int(state.row_to_id[orig_row]),
            "title": str(state.films.iloc[orig_row]["title"]),
            "cosine": float(arr[i]),
        })

    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "stats": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        },
        "top10": top10,
    }
