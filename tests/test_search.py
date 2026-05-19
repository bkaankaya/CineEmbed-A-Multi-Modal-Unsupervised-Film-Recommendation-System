"""Tests for rapidfuzz-based film search."""
import pandas as pd

from cineembed.search import FilmSearcher


def test_prefix_match():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Inception", "Interstellar", "Goodfellas"],
        "popularity": ["100", "80", "50"],
    })
    s = FilmSearcher(df)
    rows = s.search("incep", limit=5)
    assert rows[0]["id"] == 1


def test_fuzzy_fallback():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Inception", "Interstellar", "Goodfellas"],
        "popularity": ["100", "80", "50"],
    })
    s = FilmSearcher(df)
    rows = s.search("inceptin", limit=5)  # typo
    ids = [r["id"] for r in rows]
    assert 1 in ids


def test_popularity_tiebreak():
    df = pd.DataFrame({
        "id": [1, 2],
        "title": ["Inception", "Inception 2"],
        "popularity": ["50", "100"],
    })
    s = FilmSearcher(df)
    rows = s.search("inception", limit=5)
    # Exact title takes precedence even when other has higher popularity.
    assert rows[0]["title"] == "Inception"


def test_empty_query():
    df = pd.DataFrame({"id": [1], "title": ["x"], "popularity": ["1"]})
    s = FilmSearcher(df)
    assert s.search("", limit=5) == []
