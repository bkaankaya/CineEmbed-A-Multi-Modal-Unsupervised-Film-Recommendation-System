"""Tests for the cluster-naming heuristic."""
import pandas as pd
import numpy as np

from cineembed.cluster_naming import auto_name_clusters


def test_excludes_unknown_genre():
    cluster_labels = np.array([0, 0, 0, 0, 0])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "year": [1990, 1995, 2000, 2005, 2010],
        "genres": [["Unknown"], ["Unknown"], ["Drama"], ["Drama"], []],
    })
    meta = auto_name_clusters(cluster_labels, df, k=1)
    assert meta[0]["topGenres"][0]["genre"] == "Drama"


def test_modal_decade_skips_nulls():
    cluster_labels = np.array([0, 0, 0, 0])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "year": [None, None, None, 1995.0],
        "genres": [["Drama"]] * 4,
    })
    meta = auto_name_clusters(cluster_labels, df, k=1)
    assert meta[0]["modalDecade"] == "Mixed era"


def test_modal_decade_picks_dominant():
    cluster_labels = np.array([0, 0, 0, 0, 0])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "year": [1990.0, 1991.0, 1995.0, 2005.0, None],
        "genres": [["Drama"]] * 5,
    })
    meta = auto_name_clusters(cluster_labels, df, k=1)
    assert meta[0]["modalDecade"] == "1990s"


def test_disambiguator_suffix_on_collision():
    cluster_labels = np.array([0, 0, 1, 1])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "year": [1995.0, 1996.0, 1997.0, 1998.0],
        "genres": [["Drama"]] * 4,
    })
    meta = auto_name_clusters(cluster_labels, df, k=2)
    names = [c["name"] for c in meta]
    assert names[0] != names[1]
    assert "(k=" in names[0] or "(k=" in names[1]
