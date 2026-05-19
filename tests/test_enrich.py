"""Tests for country enrichment of films_master.parquet."""
from __future__ import annotations
import json

import pytest

from cineembed.enrich import parse_country


@pytest.mark.parametrize("raw,expected", [
    # plain name strings
    ("Finland", "FI"),
    ("United States of America", "US"),
    ("Germany", "DE"),
    # JSON-stringified list (TMDb dump format)
    (json.dumps([{"iso_3166_1": "FI", "name": "Finland"}]), "FI"),
    (json.dumps([{"iso_3166_1": "US", "name": "United States of America"},
                 {"iso_3166_1": "CA", "name": "Canada"}]), "US"),
    # edge cases
    ("", None),
    (None, None),
    ("not a real country", None),
    (json.dumps([]), None),
])
def test_parse_country(raw, expected):
    assert parse_country(raw) == expected
