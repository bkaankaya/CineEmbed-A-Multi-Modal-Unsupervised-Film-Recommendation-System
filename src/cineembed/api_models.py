"""Pydantic v2 models matching the frontend Film type with camelCase JSON
wire shape. Per spec §14.7."""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Film(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    title: str
    year: int | None
    rating: float
    votes: int
    genres: list[str] = Field(default_factory=list)
    country: str | None
    duration: float | None
    language: str
    director: str
    cluster: int
    overview: str | None
    time: str
    place: str | None
    poster_color: str = Field(alias="posterColor")
    poster_url: str | None = Field(alias="posterUrl")
    backdrop_url: str | None = Field(alias="backdropUrl")
    tagline: str | None
    style: list[str] = Field(default_factory=list)
    plot: list[str] = Field(default_factory=list)
    tmdb_status: Literal["ok", "missing"] = Field(alias="tmdbStatus")


class Neighbor(Film):
    cosine: float


class GenrePct(BaseModel):
    genre: str
    pct: float


class Cluster(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    name: str
    size: int
    top_genres: list[GenrePct] = Field(alias="topGenres")
    modal_decade: str = Field(alias="modalDecade")
    preview_films: list[Film] = Field(alias="previewFilms", default_factory=list)


class ClusterDetail(Cluster):
    films: list[Film] = Field(default_factory=list)
    total: int


class Backbone(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: Literal["ae_z32", "ae_z64", "ae_z128"]
    z: int
    label: str
    genre_at_five: float = Field(alias="genreAtFive")
    gnmi: float
    preferred: bool


class HealthResponse(BaseModel):
    status: str
    backbones_loaded: list[str]
    films: int
    tmdb_key_configured: bool


class CosineHistogramStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float


class CosineHistogramTop(BaseModel):
    id: int
    title: str
    cosine: float


class CosineHistogram(BaseModel):
    bins: list[float]
    counts: list[int]
    stats: CosineHistogramStats
    top10: list[CosineHistogramTop]
