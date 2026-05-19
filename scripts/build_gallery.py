"""Precompute gallery.json — 5 queries × 3 backbones × top-5 neighbors.

Dedupes all unique film ids before TMDb fetch per spec §5.1 task 3.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cineembed.tmdb import TMDbClient, make_poster_url, make_backdrop_url
from cineembed.keywords import split_keywords

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE = REPO_ROOT / "artifacts" / "inference"
OUT = INFERENCE / "gallery.json"
CACHE = REPO_ROOT / "artifacts" / "cache" / "tmdb"

QUERIES = [
    ("Inception", 27205),
    ("Spirited Away", 129),
    ("Shawshank", 278),
    ("Pulp Fiction", 680),
    ("Toy Story", 862),
]
BACKBONES = ["ae_z32", "ae_z64", "ae_z128"]


def hash_hsl(film_id: int) -> str:
    h = (film_id * 2654435761) % 360
    return f"hsl({h}, 65%, 42%)"


def decade_label(year) -> str:
    if pd.isna(year):
        return "Mixed era"
    return f"{(int(year) // 10) * 10}s"


async def main() -> None:
    films = pd.read_parquet(INFERENCE / "films_master.parquet")
    id_to_row = {fid: i for i, fid in enumerate(films["id"].astype(int).tolist())}

    embeddings = {bb: np.load(INFERENCE / bb / "embeddings.npy") for bb in BACKBONES}
    cluster_labels = {bb: np.load(INFERENCE / bb / "cluster_labels.npy") for bb in BACKBONES}

    matrix: dict[str, dict[str, dict]] = {}
    seen_ids: set[int] = set()
    for label, qid in QUERIES:
        matrix[label] = {}
        for bb in BACKBONES:
            row = id_to_row[qid]
            cos = embeddings[bb] @ embeddings[bb][row]
            top = np.argpartition(-cos, 6)[:6]
            top = top[np.argsort(-cos[top])]
            top = [int(t) for t in top if t != row][:5]
            matrix[label][bb] = {
                "queryId": qid,
                "neighborRows": top,
                "neighborCosines": [float(cos[t]) for t in top],
            }
            seen_ids.add(qid)
            for t in top:
                seen_ids.add(int(films.iloc[t]["id"]))

    print(f"[gallery] unique film ids: {len(seen_ids)}")

    client = TMDbClient(
        api_key=os.environ.get("TMDB_API_KEY"),
        access_token=os.environ.get("TMDB_ACCESS_TOKEN"),
        cache_dir=CACHE,
    )
    enrichment: dict[int, object | None] = {}
    if client.key_configured:
        print(f"[gallery] fetching TMDb for {len(seen_ids)} ids...")
        blobs = await asyncio.gather(
            *(client.get_enrichment(i) for i in seen_ids),
            return_exceptions=False,
        )
        enrichment = dict(zip(seen_ids, blobs))
        print(f"[gallery] TMDb successes: {sum(1 for b in blobs if b)}")
    else:
        print("[gallery] no TMDB credentials — all entries will have tmdbStatus=missing")
    await client.aclose()

    def film_payload(row_idx: int, backbone: str) -> dict:
        r = films.iloc[row_idx]
        fid = int(r["id"])
        cluster = int(cluster_labels[backbone][row_idx])
        blob = enrichment.get(fid)
        year_val = r["year"]
        year = int(year_val) if pd.notna(year_val) else None
        style: list[str] = []
        plot: list[str] = []
        if blob:
            style, plot = split_keywords(blob.keyword_names)  # type: ignore[attr-defined]
        country = str(r["country"]) if pd.notna(r["country"]) else None
        return {
            "id": fid,
            "title": str(r["title"]),
            "year": year,
            "rating": float(r["vote_average"]),
            "votes": int(r["vote_count"]),
            "genres": list(r["genres"]) if r["genres"] is not None else [],
            "country": country,
            "duration": float(r["runtime"]) if pd.notna(r["runtime"]) else None,
            "language": str(r["original_language"]),
            "director": str(r["director_name"]),
            "cluster": cluster,
            "overview": str(r["overview"]) if pd.notna(r["overview"]) else None,
            "time": decade_label(year_val),
            "place": country,
            "posterColor": hash_hsl(fid),
            "posterUrl": make_poster_url(blob.poster_path) if blob else None,  # type: ignore[attr-defined]
            "backdropUrl": make_backdrop_url(blob.backdrop_path) if blob else None,  # type: ignore[attr-defined]
            "tagline": (blob.tagline if blob else None),  # type: ignore[attr-defined]
            "style": style,
            "plot": plot,
            "tmdbStatus": "ok" if blob else "missing",
        }

    out_matrix = {}
    for label, qid in QUERIES:
        out_matrix[label] = {}
        for bb in BACKBONES:
            entry = matrix[label][bb]
            query_film = film_payload(id_to_row[qid], bb)
            neighbors = []
            for r, c in zip(entry["neighborRows"], entry["neighborCosines"]):
                f = film_payload(r, bb)
                f["cosine"] = c
                neighbors.append(f)
            out_matrix[label][bb] = {"query": query_film, "neighbors": neighbors}

    OUT.write_text(json.dumps({
        "queries": [q[0] for q in QUERIES],
        "matrix": out_matrix,
    }, ensure_ascii=False, indent=2))
    print(f"[gallery] wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
