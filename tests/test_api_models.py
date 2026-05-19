"""Tests for Pydantic models and their camelCase JSON wire shape."""
from cineembed.api_models import Film, Neighbor, ClusterDetail, Backbone


def test_film_serializes_camel_case():
    f = Film(
        id=27205, title="Inception", year=2010, rating=8.4, votes=30000,
        genres=["Action", "Sci-Fi"], country="US", duration=148.0,
        language="en", director="Christopher Nolan", cluster=7,
        overview="A thief...", time="2010s", place="US",
        poster_color="hsl(284,60%,55%)", poster_url=None,
        backdrop_url=None, tagline=None,
        style=[], plot=[], tmdb_status="missing",
    )
    d = f.model_dump(by_alias=True)
    assert "posterColor" in d
    assert "posterUrl" in d
    assert "tmdbStatus" in d
    assert d["style"] == []
    assert d["tmdbStatus"] == "missing"


def test_film_arrays_default_empty_not_none():
    f = Film(
        id=1, title="x", year=None, rating=0.0, votes=0,
        country=None, duration=None, language="en", director="x",
        cluster=0, overview=None, time="Mixed era", place=None,
        poster_color="hsl(0,0%,50%)", poster_url=None,
        backdrop_url=None, tagline=None, tmdb_status="missing",
    )
    assert f.style == []
    assert f.plot == []
    assert f.genres == []


def test_neighbor_carries_cosine():
    n = Neighbor(
        id=1, title="x", year=None, rating=0.0, votes=0,
        country=None, duration=None, language="en", director="x",
        cluster=0, overview=None, time="Mixed era", place=None,
        poster_color="hsl(0,0%,50%)", poster_url=None,
        backdrop_url=None, tagline=None, tmdb_status="missing",
        cosine=0.92,
    )
    assert n.cosine == 0.92


def test_cluster_detail_has_total():
    cd = ClusterDetail(
        id=0, name="Drama · 1990s", size=15000,
        top_genres=[{"genre": "Drama", "pct": 0.42}],
        modal_decade="1990s", preview_films=[],
        films=[], total=15000,
    )
    assert cd.total == 15000


def test_backbone_serializes_camel_case():
    b = Backbone(
        id="ae_z32", z=32, label="AE z=32",
        genre_at_five=0.723, gnmi=0.334, preferred=True,
    )
    d = b.model_dump(by_alias=True)
    assert d["genreAtFive"] == 0.723
