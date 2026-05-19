"""Fuzzy + prefix-aware film title search over the master parquet."""

from __future__ import annotations

from typing import cast

import pandas as pd
from rapidfuzz import fuzz, process


class FilmSearcher:
    """Owns boot-time lowercase title cache + popularity float view."""

    def __init__(self, films: pd.DataFrame):
        # Required columns: id, title, popularity
        self._df = films
        self._titles_lower = films["title"].fillna("").astype(str).str.lower().tolist()
        # pd.to_numeric is typed as a union (Series | scalar); the Series-input
        # branch always returns a Series, so cast for the type checker.
        pop_series = cast(pd.Series, pd.to_numeric(films["popularity"], errors="coerce"))
        self._popularity = pop_series.fillna(0.0).tolist()

    def search(self, q: str, limit: int = 10) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []

        # Stage 1: exact prefix scan (fast path).
        # Exact-match titles get a small boost above prefix-only hits so that
        # e.g. "Inception" outranks "Inception 2" even when the latter is more
        # popular. Prefix-only hits still tiebreak on popularity then length.
        prefix_hits: list[tuple[int, float]] = []  # (row_idx, score)
        for idx, title in enumerate(self._titles_lower):
            if title == q:
                prefix_hits.append((idx, 101.0))
            elif title.startswith(q):
                prefix_hits.append((idx, 100.0))
        # Sort: score DESC, then popularity DESC, then shorter title first
        prefix_hits.sort(
            key=lambda t: (-t[1], -self._popularity[t[0]], len(self._titles_lower[t[0]]))
        )
        if len(prefix_hits) >= limit:
            return [self._row_to_dict(i) for i, _ in prefix_hits[:limit]]

        # Stage 2: fuzzy fallback
        already = {i for i, _ in prefix_hits}
        fuzzy = process.extract(
            q,
            self._titles_lower,
            scorer=fuzz.WRatio,
            limit=limit * 4,
            score_cutoff=70,
        )
        merged: list[tuple[int, float]] = list(prefix_hits)
        for _, score, idx in fuzzy:
            if idx not in already:
                merged.append((idx, float(score)))
                already.add(idx)
        # Sort: score DESC, then popularity DESC
        merged.sort(key=lambda t: (-t[1], -self._popularity[t[0]]))
        return [self._row_to_dict(i) for i, _ in merged[:limit]]

    def _row_to_dict(self, row_idx: int) -> dict:
        return {
            "id": int(self._df["id"].iloc[row_idx]),
            "title": str(self._df["title"].iloc[row_idx]),
            "row_idx": row_idx,
        }
