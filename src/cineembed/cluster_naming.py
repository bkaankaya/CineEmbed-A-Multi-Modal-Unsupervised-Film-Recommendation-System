"""Cluster auto-naming heuristic per spec §5.3."""

from __future__ import annotations
from collections import Counter
from typing import Any, cast

import numpy as np
import pandas as pd

UNKNOWN_GENRE_MARKERS = {"Unknown", "unknown", "UNKNOWN", "", None}


def _decade_label(year: float) -> str | None:
    if year is None or pd.isna(year):
        return None
    decade = int(year) // 10 * 10
    return f"{decade}s"


def _top_genres(genre_lists: list[list[str]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    total = 0
    for gl in genre_lists:
        for g in gl:
            if g in UNKNOWN_GENRE_MARKERS:
                continue
            counter[g] += 1
            total += 1
    if total == 0:
        return []
    return [
        {"genre": g, "pct": round(c / total, 3)}
        for g, c in counter.most_common(3)
    ]


def _modal_decade(years: pd.Series) -> str:
    n = len(years)
    null_rate = years.isna().sum() / n if n else 1.0
    if null_rate > 0.5:
        return "Mixed era"
    decades: list[str] = []
    for y in years:
        if pd.notna(y):
            label = _decade_label(y)
            if label is not None:
                decades.append(label)
    if not decades:
        return "Mixed era"
    return Counter(decades).most_common(1)[0][0]


def auto_name_clusters(
    cluster_labels: np.ndarray,
    films: pd.DataFrame,
    k: int = 21,
) -> list[dict[str, Any]]:
    """Generate cluster_meta.json content per spec §5.3."""
    out: list[dict[str, Any]] = []
    for ci in range(k):
        mask = cluster_labels == ci
        cluster_df = films[mask]
        size = int(mask.sum())
        top_genres = _top_genres(list(cluster_df["genres"]))
        modal_decade = _modal_decade(cast(pd.Series, cluster_df["year"]))
        primary = top_genres[0]["genre"] if top_genres else "Mixed"
        name = f"{primary} · {modal_decade}"
        out.append({
            "id": ci,
            "name": name,
            "size": size,
            "topGenres": top_genres,
            "modalDecade": modal_decade,
        })
    name_count: Counter[str] = Counter(c["name"] for c in out)
    for c in out:
        if name_count[c["name"]] > 1:
            c["name"] = f'{c["name"]} (k={c["id"]})'
    return out
