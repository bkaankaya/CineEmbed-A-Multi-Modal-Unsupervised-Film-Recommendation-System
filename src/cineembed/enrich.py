"""Robust parser for TMDb production_countries → ISO-3166 alpha-2."""

from __future__ import annotations
import json
from functools import lru_cache

import pycountry


@lru_cache(maxsize=512)
def _lookup_alpha_2(name: str) -> str | None:
    """Lookup country by name → ISO-3166 alpha-2. Cached."""
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return None


def parse_country(raw: str | None) -> str | None:
    """Parse a TMDb production_countries cell to ISO alpha-2.

    Handles two formats encountered in the wild:
    - Plain country name: "Finland", "United States of America"
    - JSON-stringified list of dicts:
      '[{"iso_3166_1": "FI", "name": "Finland"}]'

    Returns None if value is empty, null, or unresolvable.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if data and isinstance(data, list) and isinstance(data[0], dict):
                iso = data[0].get("iso_3166_1")
                if iso and len(iso) == 2:
                    return iso.upper()
                # fall back to name lookup
                name = data[0].get("name")
                if name:
                    return _lookup_alpha_2(name)
        except (json.JSONDecodeError, IndexError, KeyError):
            return None
        return None
    return _lookup_alpha_2(raw)
