# CineEmbed Frontend ↔ Backend Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a FastAPI sidecar to the teammates' Next.js 16 frontend so the SENG 474 demo serves real recommendations over the 329k-film ae_z32 latent space, with live backbone switching, cluster browsing, eyeball gallery, and cosine distribution heatmap.

**Architecture:** Two processes (FastAPI on :8000, Next.js on :3000) within one monorepo (`frontend/` subdir already merged via subtree). Backend loads 3 backbones (ae_z32 prewarmed at boot, ae_z64/ae_z128 cold) + a single shared `films_master.parquet`. Frontend uses TanStack Query for server state + URL search params for film/backbone. TMDb v3 enrichment is server-side, lazy, disk-LRU cached with a 35-req/10s token bucket. Gallery is precomputed JSON rendered via a Next.js Server Component.

**Tech Stack:** FastAPI, Pydantic v2, numpy (mmap), rapidfuzz, httpx, aiolimiter, pycountry, MiniBatchKMeans; Next.js 16, React 19, TypeScript, TanStack Query, Zustand, zod, recharts, shadcn/ui (existing).

**Source spec:** `docs/superpowers/specs/2026-05-18-frontend-backend-integration-design.md` (1111 lines, commit `6c806c3`).

**Tier ladder (per spec §8.3):**
- **T0** (Day 1 HARD GATE): WAVE 0-3 done — search + detail + similar (ae_z32 only) + URL state + about + smoke + dev-up
- **T1** +switcher (Day 2 AM)
- **T2** +gallery UI (Day 2 noon)
- **T3** +cluster browser + heatmap (Day 2 EOD, aspirational)

---

## Wave 0 — Pre-flight: dependencies + env scaffolding

### Task 0.1: Install Python deps for [demo] extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `[project.optional-dependencies].demo`**

Open `pyproject.toml`. Locate the `demo` extras block. Replace with:

```toml
demo = [
    "pyarrow>=15",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
    "rapidfuzz>=3.0",
    "httpx>=0.27",
    "pycountry>=23.12",
    "aiolimiter>=1.1",
    "scikit-learn>=1.4",
]
```

`scikit-learn` is added for `MiniBatchKMeans` (build_index extension).

- [ ] **Step 2: Reinstall demo extras**

Run: `pip install -e ".[demo]"`
Expected: all 9 packages installed with no resolver conflicts.

- [ ] **Step 3: Smoke-import**

Run: `python -c "import rapidfuzz, httpx, pycountry, aiolimiter, sklearn.cluster; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add demo extras for FastAPI sidecar (rapidfuzz, httpx, pycountry, aiolimiter, sklearn)"
```

### Task 0.2: Install Next.js deps

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: Add TanStack Query + zustand**

```bash
cd frontend && pnpm add @tanstack/react-query@^5 zustand@^4
```

(zod, recharts, lucide-react already in `package.json`.)

- [ ] **Step 2: Verify install**

```bash
cd frontend && pnpm list @tanstack/react-query zustand zod recharts
```

Expected: all 4 packages resolve to versions; no peer-dep warnings.

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml
git commit -m "chore(frontend): add @tanstack/react-query + zustand"
```

### Task 0.3: TMDb env scaffolding

**Files:**
- Create: `.env.example`
- Create: `frontend/.env.local`
- Modify: `.gitignore`

- [ ] **Step 1: Create `.env.example`**

```bash
# Cinematic embedding demo — environment template
# Copy to .env and fill in. NEVER commit .env.

TMDB_API_KEY=                       # v3 API key from https://www.themoviedb.org/settings/api
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
```

- [ ] **Step 2: Create `frontend/.env.local`**

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

- [ ] **Step 3: Update `.gitignore`**

Append to `.gitignore`:

```
# secrets
.env
frontend/.env.local
# tmdb runtime cache (gitignored, regenerable)
artifacts/cache/tmdb/
```

- [ ] **Step 4: Commit**

```bash
git add .env.example frontend/.env.local .gitignore
git commit -m "chore(env): TMDb key + CORS scaffolding, gitignore runtime cache"
```

---

## Wave 1 — Build pipeline

### Task 1.1: STYLISTIC_DICT keyword module

**Files:**
- Create: `src/cineembed/keywords.py`
- Test: `tests/test_keywords.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_keywords.py
"""Tests for the curated stylistic keyword set used to split TMDb keywords."""
from cineembed.keywords import STYLISTIC_DICT, split_keywords


def test_split_pure_style():
    """Keywords entirely from the stylistic dict go to style[]."""
    style, plot = split_keywords(["neo-noir", "atmospheric"])
    assert style == ["neo-noir", "atmospheric"]
    assert plot == []


def test_split_pure_plot():
    """Keywords not in the stylistic dict go to plot[]."""
    style, plot = split_keywords(["bank robbery", "betrayal"])
    assert style == []
    assert plot == ["bank robbery", "betrayal"]


def test_split_mixed():
    """Mixed input partitions correctly."""
    style, plot = split_keywords(["noir", "time travel", "atmospheric", "ai"])
    assert set(style) == {"noir", "atmospheric"}
    assert set(plot) == {"time travel", "ai"}


def test_split_caps_at_eight():
    """Each array capped at 8 entries; original order preserved."""
    style, plot = split_keywords(["noir"] * 15)
    assert len(style) == 8


def test_case_insensitive():
    """Match is case-insensitive."""
    style, _ = split_keywords(["Neo-Noir"])
    assert "Neo-Noir" in style  # original case preserved in output


def test_dict_size_at_least_30():
    """Sanity: dictionary is not empty."""
    assert len(STYLISTIC_DICT) >= 30
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_keywords.py -v`
Expected: `ModuleNotFoundError: No module named 'cineembed.keywords'`

- [ ] **Step 3: Implement module**

```python
# src/cineembed/keywords.py
"""Curated stylistic keyword set used to split TMDb keywords into
'style' (mood/tone/genre-tropes) vs 'plot' (subject matter)."""

from __future__ import annotations

STYLISTIC_DICT: frozenset[str] = frozenset({
    # noir family
    "neo-noir", "film noir", "noir",
    # horror sub-genres
    "slasher", "psychological horror", "supernatural horror",
    "body horror", "cosmic horror", "gothic",
    # comedy sub-genres
    "mockumentary", "satire", "parody", "dark comedy",
    "screwball comedy", "romantic comedy",
    # thriller sub-genres
    "psychological thriller",
    # cinematic style markers
    "atmospheric", "surrealism", "experimental", "art house",
    "indie film", "cult classic",
    # setting / world style
    "dystopian future", "post-apocalyptic", "cyberpunk", "steampunk",
    # western sub-genres
    "neo-western", "spaghetti western",
    # narrative form
    "anthology film", "non-linear narrative", "unreliable narrator",
    "ensemble cast", "one-shot", "epistolary",
    # production style
    "silent film", "black and white", "musical",
    "stop motion", "claymation", "anime", "rotoscoping",
    "found footage",
    # genre tropes
    "buddy cop", "courtroom drama", "heist film",
    "war epic", "coming-of-age", "fish out of water",
})

MAX_KEYWORDS_PER_BUCKET = 8


def split_keywords(keywords: list[str]) -> tuple[list[str], list[str]]:
    """Split a TMDb keyword list into (style, plot) by STYLISTIC_DICT.

    Match is case-insensitive on keyword name. Original case is
    preserved in output. Order is preserved. Each output array is
    capped at MAX_KEYWORDS_PER_BUCKET entries.

    Per spec §5.5: if the dict yields zero hits, the spec amendment
    drops the dual presentation entirely; this function still returns
    `(style=[], plot=keywords_capped)` and the caller decides how
    to render.
    """
    style: list[str] = []
    plot: list[str] = []
    lower_dict = STYLISTIC_DICT  # already lowercase frozenset
    for kw in keywords:
        if kw.lower() in lower_dict:
            style.append(kw)
        else:
            plot.append(kw)
    return style[:MAX_KEYWORDS_PER_BUCKET], plot[:MAX_KEYWORDS_PER_BUCKET]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_keywords.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/keywords.py tests/test_keywords.py
git commit -m "feat(keywords): STYLISTIC_DICT + split_keywords helper

50+ curated cinematic-style keywords (noir, slasher, mockumentary,
atmospheric, dystopian, etc.) used to partition TMDb /keywords output
into the frontend's style[] and plot[] chip lists. Capped at 8 entries
per bucket, case-insensitive match, original casing preserved."
```

### Task 1.2: Country enrichment + films_master.parquet

**Files:**
- Create: `scripts/enrich_films.py`
- Test: `tests/test_enrich.py`

- [ ] **Step 1: Inspect CSV format to verify production_countries column**

Run:

```bash
python -c "
import pandas as pd
df = pd.read_csv('artifacts/movies_eda_final.csv', usecols=['id','production_countries'], nrows=5)
print(df.to_string())
print('---')
print('null pct:', df['production_countries'].isna().mean())
"
```

Expected: 5 rows printed. Note the literal format of `production_countries` — likely either bare country name string (`"Finland"`) or JSON-stringified list of dicts (`[{"iso_3166_1": "FI", "name": "Finland"}]`). The enrichment script must handle both branches.

- [ ] **Step 2: Write failing test**

```python
# tests/test_enrich.py
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
```

- [ ] **Step 3: Run test, verify failure**

Run: `pytest tests/test_enrich.py -v`
Expected: `ModuleNotFoundError: No module named 'cineembed.enrich'`

- [ ] **Step 4: Create `src/cineembed/enrich.py` with `parse_country`**

```python
# src/cineembed/enrich.py
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
```

- [ ] **Step 5: Run tests, expect PASS**

Run: `pytest tests/test_enrich.py -v`
Expected: 8 passed.

- [ ] **Step 6: Write the script**

```python
# scripts/enrich_films.py
"""Build artifacts/inference/films_master.parquet — a single shared
329k-row films table consumed by every backbone-scoped endpoint.

Merges fields from each backbone's existing films.parquet (id, title,
year, rating, votes, genres, duration, language, director, overview)
with the production_countries field parsed from movies_eda_final.csv.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cineembed.enrich import parse_country  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "artifacts" / "movies_eda_final.csv"
BASE_PARQUET = REPO_ROOT / "artifacts" / "inference" / "ae_z32" / "films.parquet"
OUTPUT = REPO_ROOT / "artifacts" / "inference" / "films_master.parquet"


def main() -> None:
    print(f"[enrich] reading base parquet: {BASE_PARQUET}")
    films = pd.read_parquet(BASE_PARQUET)
    print(f"[enrich] base shape: {films.shape}")

    print(f"[enrich] reading CSV production_countries: {CSV_PATH}")
    csv = pd.read_csv(CSV_PATH, usecols=["id", "production_countries"])
    print(f"[enrich] CSV rows: {len(csv)}")

    print("[enrich] parsing country values...")
    csv["country"] = csv["production_countries"].apply(parse_country)
    csv = csv.drop(columns=["production_countries"])

    print("[enrich] merging on id...")
    merged = films.merge(csv, on="id", how="left")
    null_pct = merged["country"].isna().mean()
    print(f"[enrich] country null rate: {null_pct:.1%}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT, engine="pyarrow", index=False)
    print(f"[enrich] wrote {OUTPUT}  shape={merged.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the enrichment script**

Run: `python scripts/enrich_films.py`
Expected output (approximate):
```
[enrich] reading base parquet: artifacts/inference/ae_z32/films.parquet
[enrich] base shape: (329044, 14)
[enrich] reading CSV production_countries: artifacts/movies_eda_final.csv
[enrich] CSV rows: 329044
[enrich] parsing country values...
[enrich] merging on id...
[enrich] country null rate: 10-15%
[enrich] wrote artifacts/inference/films_master.parquet  shape=(329044, 15)
```

If country null rate > 50%, the CSV format is unexpected — STOP and inspect.

- [ ] **Step 8: Smoke verify output**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('artifacts/inference/films_master.parquet')
print('shape:', df.shape)
print('cols:', list(df.columns))
print('country sample:', df['country'].head(10).tolist())
print('non-null countries:', df['country'].notna().sum())
"
```

Expected: shape (329044, 15), cols include `country`, sample shows ISO codes like `FI`, `US`, etc.

- [ ] **Step 9: Commit**

```bash
git add scripts/enrich_films.py src/cineembed/enrich.py tests/test_enrich.py
git commit -m "feat(enrich): country enrichment via pycountry → films_master.parquet

Adds a single shared films_master.parquet at artifacts/inference/ with a new
country column (ISO-3166 alpha-2). parse_country() handles both plain name
strings and JSON-stringified TMDb production_countries lists. Per-backbone
films.parquet copies are now redundant but left in place; the FastAPI loader
reads only films_master.parquet."
```

### Task 1.3: Extend `scripts/build_index.py` with KMeans cluster labels + cluster_meta

**Files:**
- Modify: `scripts/build_index.py`
- Test: `tests/test_cluster_meta.py`

- [ ] **Step 1: Read existing build_index.py to find the right insertion point**

```bash
grep -n "def main\|argparse\|np.save\|manifest" scripts/build_index.py | head -20
```

Locate where `embeddings.npy` is written (likely near the end of the encode pass). The new KMeans pass must run AFTER embeddings.npy exists.

- [ ] **Step 2: Write failing test for cluster naming heuristic**

```python
# tests/test_cluster_meta.py
"""Tests for the cluster-naming heuristic."""
import pandas as pd
import numpy as np

from cineembed.cluster_naming import auto_name_clusters


def test_excludes_unknown_genre():
    """A cluster dominated by Unknown should not name itself Unknown."""
    cluster_labels = np.array([0, 0, 0, 0, 0])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "year": [1990, 1995, 2000, 2005, 2010],
        "genres": [["Unknown"], ["Unknown"], ["Drama"], ["Drama"], []],
    })
    meta = auto_name_clusters(cluster_labels, df, k=1)
    # Should pick Drama as top genre, not Unknown
    assert meta[0]["topGenres"][0]["genre"] == "Drama"


def test_modal_decade_skips_nulls():
    """Cluster with mostly-null years should use 'Mixed era'."""
    cluster_labels = np.array([0, 0, 0, 0])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "year": [None, None, None, 1995.0],
        "genres": [["Drama"]] * 4,
    })
    meta = auto_name_clusters(cluster_labels, df, k=1)
    assert meta[0]["modalDecade"] == "Mixed era"


def test_modal_decade_picks_dominant():
    """Cluster with non-null years uses the mode of decades."""
    cluster_labels = np.array([0, 0, 0, 0, 0])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "year": [1990.0, 1991.0, 1995.0, 2005.0, None],
        "genres": [["Drama"]] * 5,
    })
    meta = auto_name_clusters(cluster_labels, df, k=1)
    assert meta[0]["modalDecade"] == "1990s"


def test_disambiguator_suffix_on_collision():
    """Two clusters with identical generated names get a (k=N) suffix."""
    cluster_labels = np.array([0, 0, 1, 1])
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "year": [1995.0, 1996.0, 1997.0, 1998.0],
        "genres": [["Drama"]] * 4,
    })
    meta = auto_name_clusters(cluster_labels, df, k=2)
    names = [c["name"] for c in meta]
    # both clusters would generate "Drama · 1990s"
    assert names[0] != names[1]
    assert "(k=" in names[0] or "(k=" in names[1]
```

- [ ] **Step 3: Run test, verify failure**

Run: `pytest tests/test_cluster_meta.py -v`
Expected: `ModuleNotFoundError: No module named 'cineembed.cluster_naming'`

- [ ] **Step 4: Implement `cineembed.cluster_naming`**

```python
# src/cineembed/cluster_naming.py
"""Cluster auto-naming heuristic per spec §5.3."""

from __future__ import annotations
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

UNKNOWN_GENRE_MARKERS = {"Unknown", "unknown", "UNKNOWN", "", None}


def _decade_label(year: float) -> str | None:
    """1995 → '1990s', None → None."""
    if year is None or pd.isna(year):
        return None
    decade = int(year) // 10 * 10
    return f"{decade}s"


def _top_genres(genre_lists: list[list[str]]) -> list[dict[str, Any]]:
    """Top-3 genres by frequency, excluding Unknown/empty."""
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
    """Mode of decade labels; 'Mixed era' when year-null > 50%."""
    n = len(years)
    null_rate = years.isna().sum() / n if n else 1.0
    if null_rate > 0.5:
        return "Mixed era"
    decades = [_decade_label(y) for y in years if pd.notna(y)]
    if not decades:
        return "Mixed era"
    return Counter(decades).most_common(1)[0][0]


def auto_name_clusters(
    cluster_labels: np.ndarray,
    films: pd.DataFrame,
    k: int = 21,
) -> list[dict[str, Any]]:
    """Generate cluster_meta.json content per spec §5.3.

    Args:
        cluster_labels: shape (n_films,) uint8, value in [0, k)
        films: DataFrame with at least 'id', 'year', 'genres'
        k: number of clusters (default 21)

    Returns: list of k dicts with id, name, size, topGenres, modalDecade.
    Collisions on name get '(k=N)' suffix.
    """
    out: list[dict[str, Any]] = []
    for ci in range(k):
        mask = cluster_labels == ci
        cluster_df = films[mask]
        size = int(mask.sum())
        top_genres = _top_genres(list(cluster_df["genres"]))
        modal_decade = _modal_decade(cluster_df["year"])
        primary = top_genres[0]["genre"] if top_genres else "Mixed"
        name = f"{primary} · {modal_decade}"
        out.append({
            "id": ci,
            "name": name,
            "size": size,
            "topGenres": top_genres,
            "modalDecade": modal_decade,
        })
    # disambiguate name collisions
    name_count: Counter[str] = Counter(c["name"] for c in out)
    for c in out:
        if name_count[c["name"]] > 1:
            c["name"] = f'{c["name"]} (k={c["id"]})'
    return out
```

- [ ] **Step 5: Run tests, expect PASS**

Run: `pytest tests/test_cluster_meta.py -v`
Expected: 4 passed.

- [ ] **Step 6: Read existing `scripts/build_index.py` argparse to understand integration**

```bash
grep -n "argparse\|add_argument\|--out\|--model-type" scripts/build_index.py | head -20
```

- [ ] **Step 7: Extend build_index.py with a `--cluster-only` flag**

Add a new flag and code path. After the existing encode/save logic, add:

```python
# Append near the end of main(), before the manifest write
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from cineembed.cluster_naming import auto_name_clusters

if args.cluster_only or args.with_clusters:
    print("[cluster] running MiniBatchKMeans k=21 on existing embeddings...")
    embeddings_path = out_dir / "embeddings.npy"
    embs = np.load(embeddings_path)
    km = MiniBatchKMeans(n_clusters=21, batch_size=4096, n_init=10, random_state=42)
    labels = km.fit_predict(embs).astype(np.uint8)
    np.save(out_dir / "cluster_labels.npy", labels)
    print(f"[cluster] wrote cluster_labels.npy ({labels.shape}, {labels.dtype})")

    print("[cluster] computing cluster_meta.json (auto-naming)...")
    films_master = pd.read_parquet(
        Path("artifacts/inference/films_master.parquet")
    )
    meta = auto_name_clusters(labels, films_master, k=21)
    with open(out_dir / "cluster_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[cluster] wrote cluster_meta.json (21 clusters)")
```

Also extend argparse:

```python
parser.add_argument(
    "--cluster-only",
    action="store_true",
    help="Skip encode, only run KMeans on existing embeddings.npy",
)
parser.add_argument(
    "--with-clusters",
    action="store_true",
    help="After encode, also run KMeans and write cluster_labels.npy + cluster_meta.json",
)
```

- [ ] **Step 8: Run cluster-only for all three backbones**

```bash
python scripts/build_index.py --checkpoint artifacts/models/ae_z32/ae.pt \
    --model-type ae --out artifacts/inference/ae_z32/ --cluster-only

python scripts/build_index.py --checkpoint artifacts/models/ae_z64.pt \
    --model-type ae --out artifacts/inference/ae_z64/ --cluster-only

python scripts/build_index.py --checkpoint artifacts/models/ae_z128/ae.pt \
    --model-type ae --out artifacts/inference/ae_z128/ --cluster-only
```

Each command takes ~30 seconds.

- [ ] **Step 9: Verify cluster_meta.json sanity (sanity-check all 21 names per backbone)**

```bash
python -c "
import json
for bb in ['ae_z32','ae_z64','ae_z128']:
    print(f'=== {bb} ===')
    with open(f'artifacts/inference/{bb}/cluster_meta.json') as f:
        for c in json.load(f):
            print(f\"  k={c['id']:2d}  size={c['size']:>6,}  {c['name']}\")
"
```

Expected: 21 lines per backbone, each with `name` like "Drama · 1990s" or "Action · 2000s". Note any that look garbled — those go into `cluster_names_override.json` (Task 1.4).

- [ ] **Step 10: Commit**

```bash
git add src/cineembed/cluster_naming.py tests/test_cluster_meta.py scripts/build_index.py
git commit -m "feat(build_index): add KMeans k=21 + cluster_meta.json output

Adds --cluster-only and --with-clusters flags to build_index.py. KMeans
runs MiniBatchKMeans(k=21, batch_size=4096, random_state=42) on the
existing L2-normed embeddings. cluster_meta.json carries auto-named
clusters with Unknown-excluded top-genres and year-null-aware modal
decade. Names disambiguated with (k=N) on collision. Per spec §5.3."
```

### Task 1.4: cluster_names_override.json (manual sanity bucket)

**Files:**
- Create: `artifacts/inference/cluster_names_override.json`

- [ ] **Step 1: Initialize empty override file**

After reviewing Task 1.3's output, identify any cluster names that look wrong (e.g., a cluster of all-Romance films misnamed "Drama") and override. For the initial pass, ship an empty override map; sanity-review during demo prep.

```json
{
  "ae_z32": {},
  "ae_z64": {},
  "ae_z128": {}
}
```

The boot loader (Task 2.2) merges `cluster_meta.json` with overrides: if `overrides[backbone][str(k)]` exists, replace the auto-name. Format: `{"7": "Romance Drama · 1990s"}`.

- [ ] **Step 2: Commit**

```bash
git add artifacts/inference/cluster_names_override.json
git commit -m "feat(clusters): initialize empty cluster_names_override.json

Manual override slot per spec §5.3. Populated by hand during demo prep
if auto-naming surfaces obviously-wrong cluster names."
```

### Task 1.5: backbones.json metadata

**Files:**
- Create: `scripts/build_backbones_metadata.py`
- Create: `artifacts/backbones.json`

- [ ] **Step 1: Write the script**

```python
# scripts/build_backbones_metadata.py
"""Aggregate per-backbone metadata into artifacts/backbones.json.

Source: each ae_z*/manifest.json (already contains retrieval stats)
plus the gNMI + genre@5 numbers from journal/10 baked in here.

This file is read by /api/backbones at boot to avoid runtime markdown
parsing.
"""

from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "artifacts" / "backbones.json"

# Numbers from docs/journal/10-results-table.md Round 2 section
METRICS = {
    "ae_z32":  {"z": 32,  "gnmi": 0.334, "genreAtFive": 0.723},
    "ae_z64":  {"z": 64,  "gnmi": 0.328, "genreAtFive": 0.715},
    "ae_z128": {"z": 128, "gnmi": 0.273, "genreAtFive": 0.722},
}

LABELS = {
    "ae_z32":  "AE z=32 (demo backbone)",
    "ae_z64":  "AE z=64 (MVP carry-over)",
    "ae_z128": "AE z=128 (over-parameterised)",
}


def main() -> None:
    out: list[dict] = []
    for bid, m in METRICS.items():
        manifest_path = REPO_ROOT / "artifacts" / "inference" / bid / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        out.append({
            "id": bid,
            "z": m["z"],
            "label": LABELS[bid],
            "genreAtFive": m["genreAtFive"],
            "gnmi": m["gnmi"],
            "preferred": bid == "ae_z32",
            "checkpointSha256_32": manifest.get("checkpoint_sha256_32", ""),
            "nFilms": manifest.get("n_films", 0),
        })
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[backbones] wrote {OUT} with {len(out)} entries")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

Run: `python scripts/build_backbones_metadata.py`
Expected: writes `artifacts/backbones.json` with 3 entries.

- [ ] **Step 3: Verify shape**

```bash
cat artifacts/backbones.json | python -m json.tool
```

Expected: array of 3 objects each with `id, z, label, genreAtFive, gnmi, preferred, checkpointSha256_32, nFilms`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_backbones_metadata.py artifacts/backbones.json
git commit -m "feat(backbones): generate artifacts/backbones.json for /api/backbones

Bakes the 3 backbones' z / label / gNMI / genre@5 / checkpoint SHA into
a static JSON file. Avoids runtime markdown parsing per spec §12.3."
```

---

## Wave 2 — FastAPI core (T0 endpoints)

### Task 2.1: Pydantic models

**Files:**
- Create: `src/cineembed/api_models.py`
- Test: `tests/test_api_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api_models.py
"""Tests for Pydantic models and their camelCase JSON wire shape."""
from cineembed.api_models import Film, Neighbor, Cluster, ClusterDetail, Backbone


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
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_api_models.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement models**

```python
# src/cineembed/api_models.py
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_api_models.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/api_models.py tests/test_api_models.py
git commit -m "feat(api): Pydantic v2 models with camelCase JSON aliases

Mirrors the frontend TypeScript Film type. style/plot/genres default to
empty list (never null) so the frontend zod schema can stay simple per
spec §14.7."
```

### Task 2.2: TMDb async client with disk LRU cache

**Files:**
- Create: `src/cineembed/tmdb.py`
- Test: `tests/test_tmdb.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tmdb.py
"""Tests for TMDbClient — disk cache, atomic writes, key-optional."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from cineembed.tmdb import TMDbClient


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tmdb"
    d.mkdir()
    return d


@pytest.mark.asyncio
async def test_no_api_key_returns_none(cache_dir: Path):
    client = TMDbClient(api_key=None, cache_dir=cache_dir)
    result = await client.get_enrichment(27205)
    assert result is None
    await client.aclose()


@pytest.mark.asyncio
async def test_cache_hit_avoids_network(cache_dir: Path, monkeypatch):
    # Seed cache with a fake entry
    blob = {
        "movie": {"poster_path": "/abc.jpg", "tagline": "Your mind is the scene of the crime."},
        "keywords": [{"name": "neo-noir"}, {"name": "heist film"}, {"name": "dream"}],
    }
    (cache_dir / "27205.json").write_text(json.dumps(blob))

    called = {"count": 0}

    async def fake_get(self, *args, **kwargs):
        called["count"] += 1
        return None

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    client = TMDbClient(api_key="dummy", cache_dir=cache_dir)
    result = await client.get_enrichment(27205)
    assert result is not None
    assert result.poster_path == "/abc.jpg"
    assert called["count"] == 0
    await client.aclose()


def test_image_url_construction():
    from cineembed.tmdb import make_poster_url, make_backdrop_url
    assert make_poster_url("/abc.jpg") == "https://image.tmdb.org/t/p/w342/abc.jpg"
    assert make_poster_url(None) is None
    assert make_backdrop_url("/x.jpg") == "https://image.tmdb.org/t/p/w1280/x.jpg"
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_tmdb.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Add pytest-asyncio to dev deps**

Edit `pyproject.toml` `[project.optional-dependencies].dev` to add:

```toml
dev = ["pytest>=8.0", "pytest-cov>=4.1", "pytest-asyncio>=0.23"]
```

Run: `pip install -e ".[dev]"`

Then create `tests/conftest.py` if it doesn't exist with `pytest-asyncio` mode:

```python
# tests/conftest.py
import pytest

pytest_plugins = ["pytest_asyncio"]


def pytest_collection_modifyitems(config, items):
    for item in items:
        if asyncio_mark_needed(item):
            item.add_marker(pytest.mark.asyncio)


def asyncio_mark_needed(item):
    return item.name.startswith("test_") and "async" in item.function.__code__.co_flags.__class__.__name__
```

Simpler: just add `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` to `pyproject.toml`.

- [ ] **Step 4: Implement TMDbClient**

```python
# src/cineembed/tmdb.py
"""Async TMDb v3 client with disk-LRU cache, token bucket, and per-id dedup.

Per spec §14.5."""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter

log = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p"
CACHE_TTL_SEC = 30 * 24 * 60 * 60  # 30 days


@dataclass
class TmdbBlob:
    poster_path: str | None
    backdrop_path: str | None
    tagline: str | None
    keyword_names: list[str]


def make_poster_url(path: str | None) -> str | None:
    return f"{TMDB_IMG_BASE}/w342{path}" if path else None


def make_backdrop_url(path: str | None) -> str | None:
    return f"{TMDB_IMG_BASE}/w1280{path}" if path else None


class TMDbClient:
    """Singleton-style async client.

    Use:
        client = TMDbClient(api_key=os.environ.get("TMDB_API_KEY"),
                            cache_dir=Path("artifacts/cache/tmdb"))
        blob = await client.get_enrichment(27205)
        # blob is None when api_key absent OR all retries failed
        await client.aclose()
    """

    def __init__(self, api_key: str | None, cache_dir: Path):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 35 requests / 10 seconds — safely under TMDb's legacy 40 / 10 s.
        self.limiter = AsyncLimiter(35, 10)
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"  # works for v4 read-tokens
        self._client = httpx.AsyncClient(
            base_url=TMDB_BASE,
            headers=headers,
            timeout=10.0,
            http2=True,
        )
        self._inflight: dict[int, asyncio.Future[TmdbBlob | None]] = {}

    @property
    def key_configured(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self, film_id: int) -> Path:
        return self.cache_dir / f"{film_id}.json"

    def _read_cache(self, film_id: int) -> TmdbBlob | None:
        p = self._cache_path(film_id)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > CACHE_TTL_SEC:
            return None
        try:
            raw = json.loads(p.read_text())
            movie = raw.get("movie", {})
            keywords = raw.get("keywords", [])
            return TmdbBlob(
                poster_path=movie.get("poster_path"),
                backdrop_path=movie.get("backdrop_path"),
                tagline=movie.get("tagline") or None,
                keyword_names=[k.get("name", "") for k in keywords if k.get("name")],
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _write_cache_atomic(self, film_id: int, movie: dict, keywords: list[dict]) -> None:
        p = self._cache_path(film_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"movie": movie, "keywords": keywords}))
        tmp.replace(p)  # atomic on POSIX

    async def _fetch_remote(self, film_id: int) -> TmdbBlob | None:
        """Hit TMDb; persist on success; return None on hard failure."""
        if not self.api_key:
            return None
        try:
            async with self.limiter:
                movie_resp = await self._client.get(
                    f"/movie/{film_id}", params={"api_key": self.api_key},
                )
            if movie_resp.status_code == 429:
                await asyncio.sleep(2.0)
                async with self.limiter:
                    movie_resp = await self._client.get(
                        f"/movie/{film_id}", params={"api_key": self.api_key},
                    )
            if movie_resp.status_code != 200:
                log.warning("tmdb movie/%d returned %d", film_id, movie_resp.status_code)
                return None
            movie = movie_resp.json()

            async with self.limiter:
                kw_resp = await self._client.get(
                    f"/movie/{film_id}/keywords", params={"api_key": self.api_key},
                )
            keywords = kw_resp.json().get("keywords", []) if kw_resp.status_code == 200 else []

            self._write_cache_atomic(film_id, movie, keywords)
            return TmdbBlob(
                poster_path=movie.get("poster_path"),
                backdrop_path=movie.get("backdrop_path"),
                tagline=movie.get("tagline") or None,
                keyword_names=[k.get("name", "") for k in keywords if k.get("name")],
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.warning("tmdb fetch %d failed: %s", film_id, e)
            return None

    async def get_enrichment(self, film_id: int) -> TmdbBlob | None:
        """Cache-first, network-fallback, dedup-concurrent."""
        if not self.api_key:
            return None
        cached = self._read_cache(film_id)
        if cached is not None:
            return cached
        if film_id in self._inflight:
            return await self._inflight[film_id]
        fut: asyncio.Future[TmdbBlob | None] = asyncio.get_running_loop().create_future()
        self._inflight[film_id] = fut
        try:
            result = await self._fetch_remote(film_id)
            fut.set_result(result)
            return result
        finally:
            del self._inflight[film_id]

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 5: Add `asyncio_mode` to pyproject.toml**

Append/edit `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Replace the `tests/conftest.py` I sketched with empty file (no longer needed):

```python
# tests/conftest.py
"""Shared pytest fixtures."""
```

- [ ] **Step 6: Run tests, expect PASS**

Run: `pytest tests/test_tmdb.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/cineembed/tmdb.py tests/test_tmdb.py tests/conftest.py pyproject.toml
git commit -m "feat(tmdb): async TMDb client with disk-LRU cache + token bucket

Per spec §14.5. 35 req / 10 s aiolimiter, per-id Future dedup, atomic
disk writes (tmp + rename), 30-day TTL, optional api_key (key-less mode
returns None — frontend falls back to gradient posters)."
```

### Task 2.3: rapidfuzz search helper

**Files:**
- Create: `src/cineembed/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_search.py
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
    # Both match; popularity breaks tie — id=2 has higher popularity, but
    # id=1 has exact prefix advantage. Exact title takes precedence.
    assert rows[0]["title"] == "Inception"


def test_empty_query():
    df = pd.DataFrame({"id": [1], "title": ["x"], "popularity": ["1"]})
    s = FilmSearcher(df)
    assert s.search("", limit=5) == []
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_search.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement FilmSearcher**

```python
# src/cineembed/search.py
"""Fuzzy + prefix-aware film title search over the master parquet."""

from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process


class FilmSearcher:
    """Owns boot-time lowercase title cache + popularity float view."""

    def __init__(self, films: pd.DataFrame):
        # Required columns: id, title, popularity
        self._df = films
        self._titles_lower = films["title"].astype(str).str.lower().tolist()
        # popularity in source is str; convert to float, NaN → 0.
        self._popularity = (
            pd.to_numeric(films["popularity"], errors="coerce").fillna(0.0).tolist()
        )

    def search(self, q: str, limit: int = 10) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []

        # Stage 1: exact prefix scan (fast path)
        prefix_hits: list[tuple[int, float, int]] = []  # (row_idx, score, popularity)
        for idx, title in enumerate(self._titles_lower):
            if title.startswith(q):
                prefix_hits.append((idx, 100.0, idx))
        # Sort: prefix-score DESC (all 100), then popularity DESC, then shorter title first
        prefix_hits.sort(
            key=lambda t: (-t[1], -self._popularity[t[0]], len(self._titles_lower[t[0]]))
        )
        if len(prefix_hits) >= limit:
            return [self._row_to_dict(i) for i, _, _ in prefix_hits[:limit]]

        # Stage 2: fuzzy fallback
        already = {i for i, _, _ in prefix_hits}
        # rapidfuzz over the full list with score_cutoff=70
        fuzzy = process.extract(
            q,
            self._titles_lower,
            scorer=fuzz.WRatio,
            limit=limit * 4,
            score_cutoff=70,
        )
        # fuzzy returns [(matched_str, score, idx), ...]
        merged: list[tuple[int, float]] = [(i, 100.0) for i, _, _ in prefix_hits]
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_search.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/search.py tests/test_search.py
git commit -m "feat(search): rapidfuzz prefix + fuzzy film title search

Boot-time lowercase title cache + popularity float view. Stage 1 prefix
scan, Stage 2 fuzzy fallback (rapidfuzz.process.extract, score_cutoff=70).
Sort key: (score DESC, popularity DESC, title-length ASC). Per spec §6.2."
```

### Task 2.4: FastAPI app — boot, health, backbones, search

**Files:**
- Create: `src/cineembed/api.py`
- Test: `tests/test_api_core.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api_core.py
"""Smoke tests for the FastAPI app core endpoints."""
from fastapi.testclient import TestClient

from cineembed.api import app


client = TestClient(app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["backbones_loaded"]) == {"ae_z32", "ae_z64", "ae_z128"}
    assert body["films"] == 329044
    assert "tmdb_key_configured" in body


def test_backbones_returns_three():
    r = client.get("/api/backbones")
    assert r.status_code == 200
    backbones = r.json()
    assert len(backbones) == 3
    ids = {b["id"] for b in backbones}
    assert ids == {"ae_z32", "ae_z64", "ae_z128"}
    # ae_z32 is preferred
    preferred = [b for b in backbones if b["preferred"]]
    assert len(preferred) == 1 and preferred[0]["id"] == "ae_z32"


def test_search_inception_top_hit():
    r = client.get("/api/films/search", params={"q": "inception", "backbone": "ae_z32"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    # Inception (TMDb id 27205) should be at or near top
    titles = [r["title"] for r in rows[:3]]
    assert any("Inception" in t for t in titles)


def test_search_invalid_backbone_400():
    r = client.get("/api/films/search", params={"q": "x", "backbone": "bogus"})
    assert r.status_code == 422  # FastAPI Literal validation


def test_search_empty_query_returns_empty():
    r = client.get("/api/films/search", params={"q": "", "backbone": "ae_z32"})
    # min_length=1 → 422
    assert r.status_code == 422
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_api_core.py -v`
Expected: `ModuleNotFoundError: No module named 'cineembed.api'`

- [ ] **Step 3: Implement minimal FastAPI app**

```python
# src/cineembed/api.py
"""CineEmbed FastAPI sidecar — see docs/superpowers/specs/2026-05-18-frontend-backend-integration-design.md"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import conint, constr

from cineembed.api_models import (
    Backbone, Cluster, ClusterDetail, CosineHistogram, Film, HealthResponse,
    Neighbor,
)
from cineembed.search import FilmSearcher
from cineembed.tmdb import TMDbClient, make_poster_url, make_backdrop_url

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INFERENCE_DIR = REPO_ROOT / "artifacts" / "inference"
BACKBONES_JSON = REPO_ROOT / "artifacts" / "backbones.json"
TMDB_CACHE_DIR = REPO_ROOT / "artifacts" / "cache" / "tmdb"
CLUSTER_OVERRIDE_PATH = INFERENCE_DIR / "cluster_names_override.json"

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


def _load_cluster_meta(backbone: BackboneId, overrides: dict) -> list[dict]:
    raw = json.loads((INFERENCE_DIR / backbone / "cluster_meta.json").read_text())
    bb_overrides = overrides.get(backbone, {})
    for c in raw:
        key = str(c["id"])
        if key in bb_overrides:
            c["name"] = bb_overrides[key]
    return raw


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot sequence per spec §14.2."""
    log.info("[boot] loading backbones metadata")
    state.backbones_meta = json.loads(BACKBONES_JSON.read_text())

    log.info("[boot] loading films_master.parquet")
    state.films = pd.read_parquet(INFERENCE_DIR / "films_master.parquet", engine="pyarrow")

    log.info("[boot] building id_to_row map")
    state.id_to_row = {int(r.id): i for i, r in enumerate(state.films.itertuples(index=False))}
    state.row_to_id = state.films["id"].astype(int).tolist()

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


@app.get("/api/films/search")
def search_films(
    q: constr(min_length=1, max_length=200),
    backbone: BackboneId = "ae_z32",
    limit: int = Query(10, ge=1, le=50),
) -> list[Film]:
    """Returns Film[] with TMDb-lazy fields null (call /films/{id} to enrich)."""
    hits = state.searcher.search(q, limit=limit)
    out: list[Film] = []
    for h in hits:
        out.append(_row_to_film(h["row_idx"], backbone, with_tmdb=False))
    return out


def _row_to_film(row_idx: int, backbone: BackboneId, with_tmdb_blob=None) -> Film:
    """Convert a films_master row to a Film payload.

    `with_tmdb_blob` is a TmdbBlob instance (already fetched) or None (lazy).
    """
    df = state.films
    r = df.iloc[row_idx]
    cluster = int(state.cluster_labels[backbone][row_idx])
    year_val = r["year"]
    year = int(year_val) if pd.notna(year_val) else None

    poster_url = None
    backdrop_url = None
    tagline = None
    style: list[str] = []
    plot: list[str] = []
    status = "missing"
    if with_tmdb_blob is not None:
        from cineembed.keywords import split_keywords
        poster_url = make_poster_url(with_tmdb_blob.poster_path)
        backdrop_url = make_backdrop_url(with_tmdb_blob.backdrop_path)
        tagline = with_tmdb_blob.tagline
        style, plot = split_keywords(with_tmdb_blob.keyword_names)
        status = "ok"

    return Film(
        id=int(r["id"]),
        title=str(r["title"]),
        year=year,
        rating=float(r["vote_average"]),
        votes=int(r["vote_count"]),
        genres=list(r["genres"]) if r["genres"] is not None else [],
        country=str(r["country"]) if pd.notna(r["country"]) else None,
        duration=float(r["runtime"]) if pd.notna(r["runtime"]) else None,
        language=str(r["original_language"]),
        director=str(r["director_name"]),
        cluster=cluster,
        overview=str(r["overview"]) if pd.notna(r["overview"]) else None,
        time=_decade_label(year) if year else "Mixed era",
        place=str(r["country"]) if pd.notna(r["country"]) else None,
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
    return f"hsl({h}, 60%, 55%)"
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_api_core.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/api.py tests/test_api_core.py
git commit -m "feat(api): FastAPI scaffold + /health /backbones /search

Boot lifespan loads films_master parquet, builds id_to_row map, loads 3
backbones' embeddings (mmap) + cluster_labels + cluster_meta with
override merge, prewarms ae_z32, builds rapidfuzz searcher, initializes
TMDb client. CORS allow-origin from env. Per spec §14.2."
```

### Task 2.5: /films/{id} with TMDb enrichment

**Files:**
- Modify: `src/cineembed/api.py`
- Test: `tests/test_api_film_detail.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api_film_detail.py
from fastapi.testclient import TestClient

from cineembed.api import app

client = TestClient(app)


def test_film_detail_inception():
    r = client.get("/api/films/27205", params={"backbone": "ae_z32"})
    assert r.status_code == 200
    f = r.json()
    assert f["id"] == 27205
    assert "Inception" in f["title"]
    # TMDb fields may be null if key absent, but tmdbStatus is one of two
    assert f["tmdbStatus"] in ("ok", "missing")
    # arrays are never null
    assert isinstance(f["style"], list)
    assert isinstance(f["plot"], list)
    assert isinstance(f["genres"], list)


def test_film_detail_unknown_id():
    r = client.get("/api/films/999999999", params={"backbone": "ae_z32"})
    assert r.status_code == 404


def test_film_detail_invalid_backbone():
    r = client.get("/api/films/27205", params={"backbone": "bogus"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_api_film_detail.py -v`
Expected: 404 on the path / route not found

- [ ] **Step 3: Add the endpoint**

Append to `src/cineembed/api.py`:

```python
@app.get("/api/films/{film_id}")
async def film_detail(
    film_id: conint(ge=1),
    backbone: BackboneId = "ae_z32",
) -> Film:
    if film_id not in state.id_to_row:
        raise HTTPException(404, detail="film not found")
    row_idx = state.id_to_row[film_id]
    blob = await state.tmdb.get_enrichment(film_id) if state.tmdb else None
    return _row_to_film(row_idx, backbone, with_tmdb_blob=blob)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_api_film_detail.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/api.py tests/test_api_film_detail.py
git commit -m "feat(api): GET /api/films/{id} with lazy TMDb enrichment"
```

### Task 2.6: /similar with cosine LRU helper

**Files:**
- Modify: `src/cineembed/api.py`
- Test: `tests/test_api_similar.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api_similar.py
from fastapi.testclient import TestClient

from cineembed.api import app

client = TestClient(app)


def test_similar_inception_returns_neighbors():
    r = client.get(
        "/api/films/27205/similar",
        params={"backbone": "ae_z32", "limit": 10},
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 10
    # Self should not appear
    assert all(row["id"] != 27205 for row in rows)
    # Cosine is descending
    cosines = [row["cosine"] for row in rows]
    assert cosines == sorted(cosines, reverse=True)
    # Top cosine on ae_z32 should be high (cluster-mate cosines around 0.9+)
    assert cosines[0] > 0.7


def test_similar_invalid_limit():
    r = client.get(
        "/api/films/27205/similar",
        params={"backbone": "ae_z32", "limit": 999},
    )
    assert r.status_code == 422


def test_similar_unknown_film():
    r = client.get(
        "/api/films/999999999/similar",
        params={"backbone": "ae_z32"},
    )
    assert r.status_code == 404
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_api_similar.py -v`
Expected: 404 (route not yet added) / errors.

- [ ] **Step 3: Implement the cosine helper + endpoint**

Append to `src/cineembed/api.py`:

```python
from functools import lru_cache


def _compute_cosines(film_id: int, backbone: str) -> np.ndarray:
    """Shared cosine vector helper. Use the LRU-cached entry below."""
    row = state.id_to_row[film_id]
    q = state.embeddings[backbone][row]
    return state.embeddings[backbone] @ q


@lru_cache(maxsize=50)
def _compute_cosines_cached(film_id: int, backbone: str) -> tuple:
    """Wraps ndarray as a tuple of (ndarray,) because lru_cache can't key ndarrays.
    Returning the ndarray by ref is fine since callers don't mutate."""
    arr = _compute_cosines(film_id, backbone)
    return (arr,)


def get_cosines(film_id: int, backbone: str) -> np.ndarray:
    return _compute_cosines_cached(film_id, backbone)[0]


@app.get("/api/films/{film_id}/similar")
async def similar(
    film_id: conint(ge=1),
    backbone: BackboneId = "ae_z32",
    limit: int = Query(10, ge=1, le=50),
) -> list[Neighbor]:
    if film_id not in state.id_to_row:
        raise HTTPException(404, detail="film not found")
    cosines = get_cosines(film_id, backbone)
    self_row = state.id_to_row[film_id]
    # Get top (limit + 1) using argpartition, then sort exactly
    k = min(limit + 1, len(cosines) - 1)
    top_idx = np.argpartition(-cosines, k)[:k + 1]
    top_idx = top_idx[np.argsort(-cosines[top_idx])]
    # Drop self
    top_idx = [i for i in top_idx if i != self_row][:limit]

    # Top-5 TMDb-enriched in parallel; rest left lazy
    enrich_ids = [int(state.row_to_id[i]) for i in top_idx[:5]]
    import asyncio
    if state.tmdb:
        blobs = await asyncio.gather(
            *(state.tmdb.get_enrichment(fid) for fid in enrich_ids),
            return_exceptions=False,
        )
    else:
        blobs = [None] * len(enrich_ids)
    blob_by_id = dict(zip(enrich_ids, blobs))

    out: list[Neighbor] = []
    for i in top_idx:
        film_payload = _row_to_film(i, backbone, with_tmdb_blob=blob_by_id.get(int(state.row_to_id[i])))
        out.append(Neighbor(**film_payload.model_dump(), cosine=float(cosines[i])))
    return out
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_api_similar.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/api.py tests/test_api_similar.py
git commit -m "feat(api): GET /api/films/{id}/similar with cosine LRU helper

LRU-cached compute_cosines(film_id, backbone) shared with cosine-dist
endpoint (Task 7). argpartition + exact sort for top-N; self-drop;
top-5 TMDb-enriched via asyncio.gather, rest lazy."
```

---

## Wave 3 — Frontend rewire (T0 path)

### Task 3.1: dev-up.sh + demo-smoke.sh

**Files:**
- Create: `scripts/dev-up.sh`
- Create: `scripts/demo-smoke.sh`

- [ ] **Step 1: Write dev-up.sh**

```bash
#!/usr/bin/env bash
# scripts/dev-up.sh — launch FastAPI + Next.js dev servers in parallel
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Cap BLAS threads to keep numpy from starving FastAPI's threadpool.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Source .env if present (TMDB_API_KEY, CORS_ORIGINS)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

cleanup() {
  echo "[dev-up] cleaning up..."
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[dev-up] starting FastAPI on :8000"
python -m uvicorn cineembed.api:app --port 8000 --reload --reload-dir src &
API_PID=$!

echo "[dev-up] starting Next.js on :3000"
(cd frontend && pnpm dev) &
WEB_PID=$!

echo ""
echo "[dev-up] both processes started:"
echo "  FastAPI:  http://localhost:8000  (pid $API_PID)"
echo "  Next.js:  http://localhost:3000  (pid $WEB_PID)"
echo ""
echo "[dev-up] ready when /api/health returns 200 and Next.js compiles."
echo "         First Next.js compile of ~30 components can take 20-40 seconds."
echo "         Press Ctrl-C to stop both."
wait
```

- [ ] **Step 2: Write demo-smoke.sh**

```bash
#!/usr/bin/env bash
# scripts/demo-smoke.sh — hit every endpoint and validate JSON shape
set -euo pipefail

BASE="${1:-http://localhost:8000}"
PASS="\033[32m✓\033[0m"
FAIL="\033[31m✗\033[0m"
FAILED=0

check() {
  local name="$1"
  local url="$2"
  local jq_filter="$3"
  local result
  result=$(curl -sf "$url" 2>/dev/null | jq -e "$jq_filter" 2>/dev/null || echo "")
  if [ -n "$result" ]; then
    echo -e "  $PASS $name"
  else
    echo -e "  $FAIL $name :: $url"
    FAILED=$((FAILED + 1))
  fi
}

echo "=== CineEmbed demo smoke @ $BASE ==="
check "health"             "$BASE/api/health"                                   '.status == "ok"'
check "backbones"          "$BASE/api/backbones"                                'length == 3'
check "search inception"   "$BASE/api/films/search?q=inception&backbone=ae_z32" 'length > 0'
check "film 27205"         "$BASE/api/films/27205?backbone=ae_z32"              '.id == 27205'
check "similar 27205"      "$BASE/api/films/27205/similar?backbone=ae_z32"      'length == 10 and (.[0].cosine | type == "number")'
check "cosine-dist 27205"  "$BASE/api/films/27205/cosine-dist?backbone=ae_z32"  '.bins | length > 0'
check "clusters"           "$BASE/api/clusters?backbone=ae_z32"                 'length == 21'
check "cluster 0"          "$BASE/api/clusters/0?backbone=ae_z32"               '.id == 0'
check "gallery"            "$BASE/api/gallery"                                  '.matrix | length > 0'

echo "==="
if [ $FAILED -eq 0 ]; then
  echo -e "$PASS all checks passed"
  exit 0
else
  echo -e "$FAIL $FAILED checks failed"
  exit 1
fi
```

- [ ] **Step 3: chmod both**

Run: `chmod +x scripts/dev-up.sh scripts/demo-smoke.sh`

- [ ] **Step 4: Test dev-up.sh starts (manual)**

Run: `bash scripts/dev-up.sh`
Expected: both processes start. Visit `http://localhost:8000/api/health` → 200 JSON. Visit `http://localhost:3000` → mock page renders (frontend not yet rewired). Ctrl-C to stop.

- [ ] **Step 5: Test demo-smoke.sh against running API (T0 endpoints only)**

While dev-up.sh is running in another terminal:

Run: `bash scripts/demo-smoke.sh`
Expected: 5 checks PASS (health, backbones, search, film, similar). 4 FAIL (cosine-dist, clusters, gallery — not yet implemented).

- [ ] **Step 6: Commit**

```bash
git add scripts/dev-up.sh scripts/demo-smoke.sh
git commit -m "feat(scripts): dev-up.sh parallel launcher + demo-smoke.sh

dev-up sources .env, caps BLAS threads, runs uvicorn + pnpm dev in
parallel with trap-based cleanup. demo-smoke hits all 9 endpoints and
exits non-zero on shape mismatch."
```

### Task 3.2: Frontend lib/api.ts typed fetch wrapper

**Files:**
- Create: `frontend/lib/api.ts`
- Create: `frontend/lib/api-types.ts`

- [ ] **Step 1: Write api-types.ts (zod schemas mirror Pydantic models)**

```typescript
// frontend/lib/api-types.ts
import { z } from "zod";

export const BackboneIdSchema = z.enum(["ae_z32", "ae_z64", "ae_z128"]);
export type BackboneId = z.infer<typeof BackboneIdSchema>;

export const FilmSchema = z.object({
  id: z.number().int(),
  title: z.string(),
  year: z.number().int().nullable(),
  rating: z.number(),
  votes: z.number().int(),
  genres: z.array(z.string()).default([]),
  country: z.string().nullable(),
  duration: z.number().nullable(),
  language: z.string(),
  director: z.string(),
  cluster: z.number().int(),
  overview: z.string().nullable(),
  time: z.string(),
  place: z.string().nullable(),
  posterColor: z.string(),
  posterUrl: z.string().nullable(),
  backdropUrl: z.string().nullable(),
  tagline: z.string().nullable(),
  style: z.array(z.string()).default([]),
  plot: z.array(z.string()).default([]),
  tmdbStatus: z.enum(["ok", "missing"]),
});
export type Film = z.infer<typeof FilmSchema>;

export const NeighborSchema = FilmSchema.extend({ cosine: z.number() });
export type Neighbor = z.infer<typeof NeighborSchema>;

export const BackboneSchema = z.object({
  id: BackboneIdSchema,
  z: z.number().int(),
  label: z.string(),
  genreAtFive: z.number(),
  gnmi: z.number(),
  preferred: z.boolean(),
});
export type Backbone = z.infer<typeof BackboneSchema>;

export const ClusterSchema = z.object({
  id: z.number().int(),
  name: z.string(),
  size: z.number().int(),
  topGenres: z.array(z.object({ genre: z.string(), pct: z.number() })),
  modalDecade: z.string(),
  previewFilms: z.array(FilmSchema).default([]),
});
export type Cluster = z.infer<typeof ClusterSchema>;

export const ClusterDetailSchema = ClusterSchema.extend({
  films: z.array(FilmSchema).default([]),
  total: z.number().int(),
});
export type ClusterDetail = z.infer<typeof ClusterDetailSchema>;

export const HealthSchema = z.object({
  status: z.literal("ok"),
  backbones_loaded: z.array(z.string()),
  films: z.number().int(),
  tmdb_key_configured: z.boolean(),
});
export type Health = z.infer<typeof HealthSchema>;

export const CosineHistogramSchema = z.object({
  bins: z.array(z.number()),
  counts: z.array(z.number().int()),
  stats: z.object({
    mean: z.number(),
    std: z.number(),
    min: z.number(),
    max: z.number(),
    p50: z.number(),
    p95: z.number(),
  }),
  top10: z.array(z.object({
    id: z.number().int(),
    title: z.string(),
    cosine: z.number(),
  })),
});
export type CosineHistogram = z.infer<typeof CosineHistogramSchema>;

export const GallerySchema = z.object({
  queries: z.array(z.string()),
  matrix: z.record(z.string(), z.record(z.string(), z.object({
    query: FilmSchema,
    neighbors: z.array(NeighborSchema),
  }))),
});
export type Gallery = z.infer<typeof GallerySchema>;
```

- [ ] **Step 2: Write api.ts (typed fetch with zod validation)**

```typescript
// frontend/lib/api.ts
import {
  BackboneSchema, ClusterDetailSchema, ClusterSchema,
  CosineHistogramSchema, FilmSchema, GallerySchema,
  HealthSchema, NeighborSchema,
  type Backbone, type BackboneId, type Cluster, type ClusterDetail,
  type CosineHistogram, type Film, type Gallery, type Health, type Neighbor,
} from "./api-types";
import { z } from "zod";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, public detail?: string) {
    super(`API error ${status}: ${detail ?? ""}`);
  }
}

async function fetchJson<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (e) {
    throw new ApiError(0, "network");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  const json = await res.json();
  return schema.parse(json);
}

export const api = {
  getHealth: (init?: RequestInit) =>
    fetchJson("/api/health", HealthSchema, init),

  getBackbones: (init?: RequestInit) =>
    fetchJson("/api/backbones", z.array(BackboneSchema), init),

  searchFilms: (q: string, backbone: BackboneId, limit = 10, init?: RequestInit) =>
    fetchJson(
      `/api/films/search?q=${encodeURIComponent(q)}&backbone=${backbone}&limit=${limit}`,
      z.array(FilmSchema),
      init,
    ),

  getFilm: (id: number, backbone: BackboneId, init?: RequestInit) =>
    fetchJson(`/api/films/${id}?backbone=${backbone}`, FilmSchema, init),

  getSimilar: (id: number, backbone: BackboneId, limit = 10, init?: RequestInit) =>
    fetchJson(
      `/api/films/${id}/similar?backbone=${backbone}&limit=${limit}`,
      z.array(NeighborSchema),
      init,
    ),

  getCosineDist: (id: number, backbone: BackboneId, bins = 30, init?: RequestInit) =>
    fetchJson(
      `/api/films/${id}/cosine-dist?backbone=${backbone}&bins=${bins}`,
      CosineHistogramSchema,
      init,
    ),

  getClusters: (backbone: BackboneId, init?: RequestInit) =>
    fetchJson(`/api/clusters?backbone=${backbone}`, z.array(ClusterSchema), init),

  getCluster: (k: number, backbone: BackboneId, limit = 50, init?: RequestInit) =>
    fetchJson(
      `/api/clusters/${k}?backbone=${backbone}&limit=${limit}`,
      ClusterDetailSchema,
      init,
    ),

  getGallery: (init?: RequestInit) =>
    fetchJson("/api/gallery", GallerySchema, init),
};

export type { BackboneId, Film, Neighbor, Backbone, Cluster, ClusterDetail, Health, CosineHistogram, Gallery };
```

- [ ] **Step 3: Verify compiles (no type errors)**

Run: `cd frontend && pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/api.ts frontend/lib/api-types.ts
git commit -m "feat(frontend): typed fetch wrapper with zod schemas

lib/api.ts wraps every backend endpoint with runtime validation via zod.
Custom ApiError carries status + detail for offline-banner detection.
NEXT_PUBLIC_API_BASE env override."
```

### Task 3.3: providers.tsx + layout.tsx wrap

**Files:**
- Create: `frontend/app/providers.tsx`
- Modify: `frontend/app/layout.tsx`

- [ ] **Step 1: Write providers.tsx**

```typescript
// frontend/app/providers.tsx
"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5 * 60 * 1000,
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  }));
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
```

- [ ] **Step 2: Read existing layout.tsx**

Run: `cat frontend/app/layout.tsx`

Note the current structure (likely a `<html><body>{children}</body></html>` shape).

- [ ] **Step 3: Wrap children in Providers**

Edit `frontend/app/layout.tsx`:

```typescript
import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";
// ...keep existing imports...

export const metadata: Metadata = {
  title: "CineEmbed",
  description: "Multimodal film recommender over 329,044 films",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

- [ ] **Step 4: Verify compiles**

Run: `cd frontend && pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/providers.tsx frontend/app/layout.tsx
git commit -m "feat(frontend): wrap app in QueryClientProvider"
```

### Task 3.4: Rewire app/page.tsx — search + URL state + mock → real

**Files:**
- Modify: `frontend/app/page.tsx`
- Modify: `frontend/components/search-bar.tsx`
- Modify: `frontend/components/empty-state.tsx`

- [ ] **Step 1: Read existing page.tsx + search-bar.tsx**

Run: `cat frontend/app/page.tsx frontend/components/search-bar.tsx`

Note the existing mock-data wiring.

- [ ] **Step 2: Rewrite app/page.tsx with URL state + real fetch**

```typescript
// frontend/app/page.tsx
"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { Sidebar } from "@/components/sidebar";
import { SearchBar } from "@/components/search-bar";
import { SelectedFilmPanel } from "@/components/selected-film-panel";
import { SimilarFilmsPanel } from "@/components/similar-films-panel";
import { EmptyState } from "@/components/empty-state";
import { api, type BackboneId } from "@/lib/api";

export default function HomePage() {
  const params = useSearchParams();
  const router = useRouter();
  const filmIdParam = params.get("film");
  const filmId = filmIdParam && /^\d+$/.test(filmIdParam) ? Number(filmIdParam) : null;
  const backbone = ((params.get("backbone") ?? "ae_z32") as BackboneId);

  const { data: film, isLoading: filmLoading } = useQuery({
    queryKey: ["film", filmId, backbone],
    queryFn: ({ signal }) => api.getFilm(filmId!, backbone, { signal }),
    enabled: filmId !== null,
  });

  const setFilm = (id: number | null) => {
    const next = new URLSearchParams(params.toString());
    if (id === null) next.delete("film");
    else next.set("film", String(id));
    router.replace(`?${next.toString()}`, { scroll: false });
  };

  return (
    <div className="flex min-h-screen bg-[#f8f9fb]">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8">
        <SearchBar
          backbone={backbone}
          onSelectFilm={(id) => setFilm(id)}
        />
        <div className="mt-6">
          {filmId === null ? (
            <EmptyState />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-[58fr_42fr] gap-6">
              <SelectedFilmPanel
                film={film ?? null}
                loading={filmLoading}
                backbone={backbone}
              />
              <SimilarFilmsPanel
                filmId={filmId}
                backbone={backbone}
                onSelectFilm={(id) => setFilm(id)}
              />
            </div>
          )}
        </div>
        <footer className="mt-12 text-xs text-gray-500">
          CineEmbed · SENG 474 · TED University · 2026
        </footer>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Rewrite components/search-bar.tsx**

```typescript
// frontend/components/search-bar.tsx
"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search as SearchIcon } from "lucide-react";
import { api, type BackboneId } from "@/lib/api";

interface Props {
  backbone: BackboneId;
  onSelectFilm: (id: number) => void;
}

export function SearchBar({ backbone, onSelectFilm }: Props) {
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const { data: hits = [], isFetching } = useQuery({
    queryKey: ["search", debouncedQ, backbone],
    queryFn: ({ signal }) => api.searchFilms(debouncedQ, backbone, 10, { signal }),
    enabled: debouncedQ.length >= 1,
  });

  return (
    <div className="relative">
      <div className="relative">
        <SearchIcon
          aria-hidden="true"
          className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-4 h-4"
        />
        <input
          type="search"
          role="combobox"
          aria-expanded={hits.length > 0}
          aria-autocomplete="list"
          aria-controls="search-results"
          placeholder="Search 329,044 films..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="w-full pl-9 pr-3 py-2 border border-[#e5e4ec] rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-purple-300"
        />
      </div>
      {hits.length > 0 && (
        <ul
          id="search-results"
          role="listbox"
          className="absolute mt-1 w-full bg-white border border-[#e5e4ec] rounded-md shadow-lg z-10 max-h-80 overflow-y-auto"
        >
          {hits.map((f) => (
            <li
              key={f.id}
              role="option"
              tabIndex={0}
              className="px-3 py-2 hover:bg-purple-50 cursor-pointer text-sm"
              onClick={() => {
                onSelectFilm(f.id);
                setQ("");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectFilm(f.id);
                  setQ("");
                }
              }}
            >
              <div className="font-medium">{f.title}</div>
              <div className="text-xs text-gray-500">
                {f.year ?? "—"} · {f.director}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Update empty-state.tsx copy**

Replace contents of `frontend/components/empty-state.tsx`:

```typescript
export function EmptyState() {
  return (
    <div className="text-center py-16 text-gray-500">
      <p className="text-lg">Search 329,044 films from the multimodal embedding space.</p>
      <p className="mt-2 text-sm">Pick any film to see its nearest neighbors in the latent.</p>
    </div>
  );
}
```

- [ ] **Step 5: Verify compiles**

Run: `cd frontend && pnpm exec tsc --noEmit`
Expected: 0 errors.

- [ ] **Step 6: Manual smoke**

Run: `bash scripts/dev-up.sh` (in another terminal). Visit `http://localhost:3000`. Type "inception" → dropdown shows results from real API → click → URL updates to `?film=27205&backbone=ae_z32`.

- [ ] **Step 7: Commit**

```bash
git add frontend/app/page.tsx frontend/components/search-bar.tsx frontend/components/empty-state.tsx
git commit -m "feat(frontend): rewire search + URL state mock→real"
```

### Task 3.5: Rewire selected-film-panel + film-poster

**Files:**
- Modify: `frontend/components/selected-film-panel.tsx`
- Modify: `frontend/components/film-poster.tsx`
- Modify: `frontend/next.config.mjs`

- [ ] **Step 1: Update next.config.mjs for TMDb image domain**

```javascript
// frontend/next.config.mjs
/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "image.tmdb.org" },
    ],
  },
};
export default nextConfig;
```

- [ ] **Step 2: Rewrite film-poster.tsx**

```typescript
// frontend/components/film-poster.tsx
"use client";

import Image from "next/image";
import type { Film } from "@/lib/api";

export function FilmPoster({ film, size = "md" }: { film: Film; size?: "sm" | "md" | "lg" }) {
  const dims = size === "sm" ? "w-16 h-24" : size === "lg" ? "w-64 h-96" : "w-32 h-48";
  if (film.posterUrl) {
    return (
      <div className={`relative overflow-hidden rounded-md ${dims}`}>
        <Image
          src={film.posterUrl}
          alt={`${film.title} poster`}
          fill
          sizes="200px"
          className="object-cover"
        />
      </div>
    );
  }
  return (
    <div
      className={`${dims} rounded-md flex items-center justify-center text-white text-xs font-medium text-center px-2`}
      style={{ background: film.posterColor }}
      role="img"
      aria-label={`${film.title} (no poster available)`}
    >
      {film.title}
    </div>
  );
}
```

- [ ] **Step 3: Rewrite selected-film-panel.tsx**

```typescript
// frontend/components/selected-film-panel.tsx
"use client";

import { FilmPoster } from "./film-poster";
import type { Film } from "@/lib/api";

interface Props {
  film: Film | null;
  loading: boolean;
  backbone: string;
}

export function SelectedFilmPanel({ film, loading, backbone }: Props) {
  if (loading || !film) {
    return (
      <div className="border border-[#e5e4ec] rounded-lg p-6 bg-white animate-pulse">
        <div className="h-64 bg-gray-100 rounded mb-4" />
        <div className="h-6 bg-gray-100 rounded w-2/3 mb-2" />
        <div className="h-4 bg-gray-100 rounded w-1/2" />
      </div>
    );
  }

  // STYLISTIC_DICT fallback: when style empty but plot has entries, render
  // a single "Keywords" list (no fake split) per spec §5.5.
  const showSplit = film.style.length > 0;
  const flatKeywords = !showSplit ? film.plot : [];

  return (
    <article className="border border-[#e5e4ec] rounded-lg p-6 bg-white">
      <div className="flex gap-6 mb-4">
        <FilmPoster film={film} size="md" />
        <div className="flex-1">
          <h2 className="text-2xl font-semibold">{film.title}</h2>
          <p className="text-sm text-gray-500 mt-1">
            {film.year ?? "—"} · {film.director}
            {film.duration ? ` · ${Math.round(film.duration)} min` : ""}
            {film.country ? ` · ${film.country}` : ""}
          </p>
          {film.tagline && (
            <p className="italic text-sm text-gray-600 mt-2">"{film.tagline}"</p>
          )}
          <div className="mt-3 flex flex-wrap gap-1">
            {film.genres.slice(0, 5).map((g) => (
              <span key={g} className="px-2 py-0.5 text-xs bg-purple-50 text-purple-800 rounded">{g}</span>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            ★ {film.rating.toFixed(1)} ({film.votes.toLocaleString()} votes) ·
            Cluster #{film.cluster} · {film.time} · backbone {backbone}
          </p>
        </div>
      </div>
      {film.overview && (
        <p className="text-sm text-gray-700 leading-relaxed">{film.overview}</p>
      )}
      {showSplit && (
        <>
          {film.style.length > 0 && (
            <ChipList title="Style" chips={film.style} variant="indigo" />
          )}
          {film.plot.length > 0 && (
            <ChipList title="Plot" chips={film.plot} variant="rose" />
          )}
        </>
      )}
      {!showSplit && flatKeywords.length > 0 && (
        <ChipList title="Keywords" chips={flatKeywords} variant="slate" />
      )}
      {/* Cosine heatmap embedded — added in Task 7.2 (lazy dynamic import) */}
    </article>
  );
}

function ChipList({ title, chips, variant }: { title: string; chips: string[]; variant: "indigo" | "rose" | "slate" }) {
  const colors = {
    indigo: "bg-indigo-50 text-indigo-700",
    rose: "bg-rose-50 text-rose-700",
    slate: "bg-slate-100 text-slate-700",
  };
  return (
    <div className="mt-4">
      <p className="text-xs font-medium text-gray-500 mb-1">{title}</p>
      <div className="flex flex-wrap gap-1">
        {chips.map((c) => (
          <span key={c} className={`px-2 py-0.5 text-xs rounded ${colors[variant]}`}>{c}</span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify compiles**

Run: `cd frontend && pnpm exec tsc --noEmit`
Expected: 0 errors.

- [ ] **Step 5: Manual smoke**

With dev-up.sh running, search "inception", click → selected-film panel renders with real data + TMDb poster (or HSL fallback if no key).

- [ ] **Step 6: Commit**

```bash
git add frontend/components/selected-film-panel.tsx frontend/components/film-poster.tsx frontend/next.config.mjs
git commit -m "feat(frontend): rewire selected-film panel + TMDb poster integration

Real Film fields, style/plot chips with empty-fallback to single Keywords
list per spec §5.5, TMDb image.tmdb.org whitelisted for Next.js Image.
Watchlist button removed (out-of-scope dead UI)."
```

### Task 3.6: Rewire similar-films-panel

**Files:**
- Modify: `frontend/components/similar-films-panel.tsx`

- [ ] **Step 1: Rewrite component**

```typescript
// frontend/components/similar-films-panel.tsx
"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type BackboneId, type Neighbor } from "@/lib/api";

interface Props {
  filmId: number;
  backbone: BackboneId;
  onSelectFilm: (id: number) => void;
}

function cosineColor(c: number): string {
  if (c >= 0.95) return "bg-green-100 text-green-800";
  if (c >= 0.8) return "bg-blue-100 text-blue-800";
  return "bg-slate-100 text-slate-700";
}

export function SimilarFilmsPanel({ filmId, backbone, onSelectFilm }: Props) {
  const { data: neighbors = [], isLoading } = useQuery({
    queryKey: ["similar", filmId, backbone],
    queryFn: ({ signal }) => api.getSimilar(filmId, backbone, 10, { signal }),
  });

  if (isLoading) {
    return (
      <div className="border border-[#e5e4ec] rounded-lg p-4 bg-white animate-pulse">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-10 bg-gray-100 rounded mb-2" />
        ))}
      </div>
    );
  }

  if (neighbors.length === 0) {
    return <div className="border border-[#e5e4ec] rounded-lg p-4 bg-white text-sm text-gray-500">No similar films found.</div>;
  }

  return (
    <aside className="border border-[#e5e4ec] rounded-lg p-4 bg-white">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Similar films (backbone {backbone})</h3>
      <ol className="space-y-2">
        {neighbors.map((n: Neighbor, i: number) => (
          <li key={n.id}>
            <button
              type="button"
              onClick={() => onSelectFilm(n.id)}
              className="w-full flex items-start gap-3 px-2 py-2 rounded hover:bg-purple-50 text-left"
            >
              <span className="text-xs text-gray-400 w-6">#{i + 1}</span>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{n.title}</div>
                <div className="text-xs text-gray-500 truncate">
                  {n.year ?? "—"} · {n.director}
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {n.genres.slice(0, 2).map((g) => (
                    <span key={g} className="text-[10px] px-1.5 py-0.5 bg-gray-100 rounded">{g}</span>
                  ))}
                </div>
              </div>
              <span
                className={`text-xs px-1.5 py-0.5 rounded ${cosineColor(n.cosine)}`}
                aria-label={`cosine ${n.cosine.toFixed(3)}`}
              >
                {n.cosine.toFixed(2)}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </aside>
  );
}
```

- [ ] **Step 2: Verify compiles + manual smoke**

Run: `cd frontend && pnpm exec tsc --noEmit`
Expected: 0 errors.

With dev-up.sh running, click a film → similar panel populates with 10 neighbors + color-coded cosine badges.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/similar-films-panel.tsx
git commit -m "feat(frontend): rewire similar-films panel with cosine badges"
```

### Task 3.7: Sidebar nav + About page (closes T0 gate)

**Files:**
- Modify: `frontend/components/sidebar.tsx`
- Create: `frontend/app/about/page.tsx`

- [ ] **Step 1: Update sidebar.tsx**

```typescript
// frontend/components/sidebar.tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Film, Layers, GalleryHorizontal, Info } from "lucide-react";

const NAV = [
  { href: "/", label: "Home", icon: Film },
  { href: "/cluster", label: "Clusters", icon: Layers },
  { href: "/gallery", label: "Gallery", icon: GalleryHorizontal },
  { href: "/about", label: "About", icon: Info },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Primary navigation"
      className="fixed left-0 top-0 bottom-0 w-[220px] bg-white border-r border-[#e5e4ec] p-4"
    >
      <h1 className="font-semibold mb-6 text-purple-700">CineEmbed</h1>
      <ul className="space-y-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/" && pathname.startsWith(href));
          return (
            <li key={href}>
              <Link
                href={href}
                className={`flex items-center gap-2 px-2 py-1.5 rounded text-sm ${
                  active ? "bg-purple-50 text-purple-800" : "text-gray-700 hover:bg-gray-50"
                }`}
              >
                <Icon className="w-4 h-4" aria-hidden="true" />
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
```

- [ ] **Step 2: Create app/about/page.tsx**

```typescript
// frontend/app/about/page.tsx
import { Sidebar } from "@/components/sidebar";

export default function AboutPage() {
  return (
    <div className="flex min-h-screen bg-[#f8f9fb]">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8 max-w-3xl">
        <h1 className="text-2xl font-semibold mb-4">About CineEmbed</h1>

        <section className="prose prose-sm">
          <p>
            CineEmbed is a multimodal movie recommender built for SENG 474
            (Deep Learning, TED University, Spring 2026) by Baran Dinçoğuz,
            Arda Arvas, and Kaan Kaya. The model encodes each of 329,044
            films into a 32-dimensional latent space using a multi-modal
            autoencoder over seven feature blocks (numerical metadata,
            genre one-hot, language one-hot, decade scalar, prior-awards,
            text overview embedding, and director profile).
          </p>

          <h2>Two methodological findings</h2>

          <h3>NMI ≠ retrieval quality</h3>
          <p>
            The MVP champion model, <code>dec_z64_k21</code>, won on the
            NMI clustering metric (geo_NMI = 0.323) but collapsed under
            cosine retrieval: every pair of films inside a cluster sat at
            cosine ≈ 1.000 (angular collapse), so top-5 retrieval degenerated
            into a random tie-break. We adopted <code>genre@5</code> — the
            mean fraction of top-5 nearest neighbours sharing a film's
            primary genre — as the demo-relevant metric, and switched the
            demo backbone to <code>ae_z64</code>. See journal/07.
          </p>

          <h3>Information-bottleneck sweet spot at z=32</h3>
          <p>
            Round 2 swept latent dimension across z ∈ {`{32, 64, 128}`} with
            the recipe held constant. Counter-intuitively the smallest
            variant won on both <code>genre@5</code> (0.723 vs 0.715 vs
            0.722) and <code>gNMI</code> (0.334 vs 0.328 vs 0.273) — a
            U-curve. The over-parameterised z=128 variant produced
            near-dead latent dimensions and a narrowing pair-cosine
            distribution. We interpret z=32 as the information-bottleneck
            sweet spot for this task. See journal/12.
          </p>

          <h2>How to read the gallery</h2>
          <p>
            The <a href="/gallery">Gallery</a> page renders five well-known
            queries (Inception, Spirited Away, Shawshank, Pulp Fiction, Toy
            Story) against the three backbones side-by-side. The same
            query produces visibly different top-5 neighbours per backbone
            — the strongest demonstration of the project's findings.
          </p>

          <p className="text-xs text-gray-500 mt-8">
            Source repo: github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System · branch
            main · spec
            docs/superpowers/specs/2026-05-18-frontend-backend-integration-design.md
          </p>
        </section>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Verify compiles + manual smoke**

Run: `cd frontend && pnpm exec tsc --noEmit`
Expected: 0 errors.

With dev-up.sh running: navigate to `/about` → page renders.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/sidebar.tsx frontend/app/about/page.tsx
git commit -m "feat(frontend): sidebar nav links + /about page

Sidebar gains nav for Home / Clusters / Gallery / About (active state
highlight). About page explains the two methodological findings in
plain language with links to journal/07 and journal/12."
```

### Task 3.8: T0 hard-gate verification

- [ ] **Step 1: Run smoke script with full T0 path**

In one terminal: `bash scripts/dev-up.sh`

In another: `bash scripts/demo-smoke.sh`

Expected: health / backbones / search / film / similar all PASS. cosine-dist / clusters / gallery FAIL (T1+ work).

- [ ] **Step 2: Manual T0 acceptance walkthrough**

1. Visit http://localhost:3000 → empty state copy visible.
2. Type "inception" → dropdown of real results.
3. Click Inception (2010) → URL becomes `?film=27205&backbone=ae_z32`, selected panel + similar panel populate.
4. Click a similar film → URL updates.
5. Navigate to /about → page renders.
6. Stop backend (Ctrl-C in dev-up terminal) → frontend should still render but show offline-state indicators (no banner yet — that's polish).

- [ ] **Step 3: Tag T0 milestone**

```bash
git tag t0-shippable -m "T0 hard gate met: search + detail + similar + URL state + about + smoke"
```

T0 is now safe. Day 1 done.

---

## Wave 4 — Backbone switcher (T1)

### Task 4.1: backbone-switcher.tsx + page header integration

**Files:**
- Create: `frontend/components/backbone-switcher.tsx`
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Write backbone-switcher.tsx**

```typescript
// frontend/components/backbone-switcher.tsx
"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams, useRouter } from "next/navigation";
import { api, type BackboneId } from "@/lib/api";

const FALLBACK = [
  { id: "ae_z32", z: 32, label: "AE z=32", genreAtFive: 0.723, gnmi: 0.334, preferred: true },
  { id: "ae_z64", z: 64, label: "AE z=64", genreAtFive: 0.715, gnmi: 0.328, preferred: false },
  { id: "ae_z128", z: 128, label: "AE z=128", genreAtFive: 0.722, gnmi: 0.273, preferred: false },
] as const;

export function BackboneSwitcher() {
  const params = useSearchParams();
  const router = useRouter();
  const qc = useQueryClient();
  const current = ((params.get("backbone") ?? "ae_z32") as BackboneId);

  const { data: backbones = FALLBACK } = useQuery({
    queryKey: ["backbones"],
    queryFn: ({ signal }) => api.getBackbones({ signal }),
    staleTime: Infinity,
  });

  const setBackbone = (id: BackboneId) => {
    const next = new URLSearchParams(params.toString());
    next.set("backbone", id);
    router.replace(`?${next.toString()}`, { scroll: false });
    qc.invalidateQueries({ queryKey: ["film"] });
    qc.invalidateQueries({ queryKey: ["similar"] });
    qc.invalidateQueries({ queryKey: ["cosineDist"] });
    qc.invalidateQueries({ queryKey: ["clusters"] });
    qc.invalidateQueries({ queryKey: ["cluster"] });
  };

  return (
    <div
      role="radiogroup"
      aria-label="Backbone selection"
      className="inline-flex border border-[#e5e4ec] rounded-md overflow-hidden bg-white"
    >
      {backbones.map((b) => {
        const active = current === b.id;
        return (
          <button
            key={b.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => setBackbone(b.id as BackboneId)}
            title={`${b.label} · genre@5=${b.genreAtFive.toFixed(3)} · gNMI=${b.gnmi.toFixed(3)}`}
            className={`px-3 py-1.5 text-xs font-medium transition ${
              active ? "bg-purple-600 text-white" : "text-gray-700 hover:bg-gray-50"
            }`}
          >
            {b.label}
          </button>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Mount in app/page.tsx**

Edit `frontend/app/page.tsx` — add `<BackboneSwitcher />` above `<SearchBar />`:

```typescript
import { BackboneSwitcher } from "@/components/backbone-switcher";
// ...
<main className="flex-1 ml-[220px] p-8">
  <div className="flex justify-end mb-4">
    <BackboneSwitcher />
  </div>
  <SearchBar ... />
```

Also export it from other page files when needed (cluster pages will mount it).

- [ ] **Step 3: Verify + smoke**

`cd frontend && pnpm exec tsc --noEmit`

With dev-up.sh running, switching backbone changes the URL `?backbone=`, similar panel re-fetches.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/backbone-switcher.tsx frontend/app/page.tsx
git commit -m "feat(frontend): backbone-switcher segmented control at page header"
```

---

## Wave 5 — Cluster browser (T3)

### Task 5.1: /api/clusters + /api/clusters/{k} endpoints

**Files:**
- Modify: `src/cineembed/api.py`
- Test: `tests/test_api_clusters.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api_clusters.py
from fastapi.testclient import TestClient

from cineembed.api import app

client = TestClient(app)


def test_clusters_returns_21():
    r = client.get("/api/clusters", params={"backbone": "ae_z32"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 21
    for c in rows:
        assert "previewFilms" in c
        assert len(c["previewFilms"]) <= 4


def test_cluster_detail_top50():
    r = client.get("/api/clusters/0", params={"backbone": "ae_z32", "limit": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 0
    assert "films" in body
    assert "total" in body
    assert len(body["films"]) <= 50


def test_cluster_invalid_k():
    r = client.get("/api/clusters/99", params={"backbone": "ae_z32"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_api_clusters.py -v`

- [ ] **Step 3: Implement endpoints**

Append to `src/cineembed/api.py`:

```python
def _cluster_top_n_rows(backbone: str, k: int, n: int) -> list[int]:
    """Return up to n row indices in cluster k, sorted by popularity DESC."""
    labels = state.cluster_labels[backbone]
    mask = labels == k
    rows = np.where(mask)[0]
    if len(rows) == 0:
        return []
    pops = pd.to_numeric(state.films["popularity"].iloc[rows], errors="coerce").fillna(0)
    order = np.argsort(-pops.values, kind="stable")
    return rows[order[:n]].tolist()


@app.get("/api/clusters", response_model=list[Cluster])
def clusters(backbone: BackboneId = "ae_z32") -> list[Cluster]:
    meta = state.cluster_meta[backbone]
    out: list[Cluster] = []
    for c in meta:
        preview_rows = _cluster_top_n_rows(backbone, c["id"], 4)
        preview = [_row_to_film(r, backbone) for r in preview_rows]
        out.append(Cluster(
            id=c["id"],
            name=c["name"],
            size=c["size"],
            top_genres=c["topGenres"],
            modal_decade=c["modalDecade"],
            preview_films=preview,
        ))
    return out


@app.get("/api/clusters/{k}", response_model=ClusterDetail)
async def cluster_detail(
    k: conint(ge=0, le=20),
    backbone: BackboneId = "ae_z32",
    limit: int = Query(50, ge=1, le=100),
) -> ClusterDetail:
    meta_list = state.cluster_meta[backbone]
    c = next((c for c in meta_list if c["id"] == k), None)
    if c is None:
        raise HTTPException(404, detail="cluster not found")
    rows = _cluster_top_n_rows(backbone, k, limit)

    # Enrich first 5
    enrich_ids = [int(state.row_to_id[r]) for r in rows[:5]]
    import asyncio
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_api_clusters.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/cineembed/api.py tests/test_api_clusters.py
git commit -m "feat(api): cluster endpoints — list + detail with total"
```

### Task 5.2: cluster-card component + /cluster + /cluster/[k] pages

**Files:**
- Create: `frontend/components/cluster-card.tsx`
- Create: `frontend/app/cluster/page.tsx`
- Create: `frontend/app/cluster/[k]/page.tsx`

- [ ] **Step 1: cluster-card.tsx**

```typescript
// frontend/components/cluster-card.tsx
"use client";

import Link from "next/link";
import { FilmPoster } from "./film-poster";
import type { Cluster } from "@/lib/api";

export function ClusterCard({ cluster, backbone }: { cluster: Cluster; backbone: string }) {
  return (
    <Link
      href={`/cluster/${cluster.id}?backbone=${backbone}`}
      className="block border border-[#e5e4ec] rounded-lg p-4 bg-white hover:border-purple-300 transition"
    >
      <h3 className="font-medium text-sm">{cluster.name}</h3>
      <p className="text-xs text-gray-500 mt-1">{cluster.size.toLocaleString()} films</p>
      <div className="flex flex-wrap gap-1 mt-2">
        {cluster.topGenres.slice(0, 3).map((g) => (
          <span key={g.genre} className="text-[10px] px-1.5 py-0.5 bg-purple-50 text-purple-700 rounded">
            {g.genre} {(g.pct * 100).toFixed(0)}%
          </span>
        ))}
      </div>
      <div className="flex gap-1 mt-3">
        {cluster.previewFilms.slice(0, 4).map((f) => (
          <FilmPoster key={f.id} film={f} size="sm" />
        ))}
      </div>
    </Link>
  );
}
```

- [ ] **Step 2: app/cluster/page.tsx**

```typescript
// frontend/app/cluster/page.tsx
"use client";

import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { Sidebar } from "@/components/sidebar";
import { BackboneSwitcher } from "@/components/backbone-switcher";
import { ClusterCard } from "@/components/cluster-card";
import { api, type BackboneId } from "@/lib/api";

export default function ClustersPage() {
  const params = useSearchParams();
  const backbone = ((params.get("backbone") ?? "ae_z32") as BackboneId);
  const { data: clusters = [], isLoading } = useQuery({
    queryKey: ["clusters", backbone],
    queryFn: ({ signal }) => api.getClusters(backbone, { signal }),
  });

  return (
    <div className="flex min-h-screen bg-[#f8f9fb]">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold">Clusters (k=21)</h1>
          <BackboneSwitcher />
        </div>
        {isLoading ? (
          <p className="text-gray-500">Loading clusters…</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {clusters.map((c) => (
              <ClusterCard key={c.id} cluster={c} backbone={backbone} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
```

- [ ] **Step 3: app/cluster/[k]/page.tsx**

```typescript
// frontend/app/cluster/[k]/page.tsx
"use client";

import { useQuery } from "@tanstack/react-query";
import { useSearchParams, useRouter } from "next/navigation";
import { use } from "react";
import { Sidebar } from "@/components/sidebar";
import { BackboneSwitcher } from "@/components/backbone-switcher";
import { FilmPoster } from "@/components/film-poster";
import { api, type BackboneId } from "@/lib/api";

export default function ClusterDetailPage({ params: pa }: { params: Promise<{ k: string }> }) {
  const { k } = use(pa);
  const kInt = Number(k);
  const params = useSearchParams();
  const router = useRouter();
  const backbone = ((params.get("backbone") ?? "ae_z32") as BackboneId);

  const { data, isLoading } = useQuery({
    queryKey: ["cluster", kInt, backbone],
    queryFn: ({ signal }) => api.getCluster(kInt, backbone, 50, { signal }),
    enabled: !isNaN(kInt) && kInt >= 0 && kInt <= 20,
  });

  return (
    <div className="flex min-h-screen bg-[#f8f9fb]">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-semibold">{data?.name ?? `Cluster #${k}`}</h1>
          <BackboneSwitcher />
        </div>
        {data && (
          <p className="text-sm text-gray-500 mb-6">
            {data.size.toLocaleString()} films · top genres:&nbsp;
            {data.topGenres.map((g) => `${g.genre} ${(g.pct * 100).toFixed(0)}%`).join(", ")}
            &nbsp;· decade {data.modalDecade}&nbsp;· showing {data.films.length} of {data.total}
          </p>
        )}
        {isLoading && <p className="text-gray-500">Loading…</p>}
        {data && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {data.films.map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => router.push(`/?film=${f.id}&backbone=${backbone}`)}
                className="text-left"
              >
                <FilmPoster film={f} size="sm" />
                <p className="text-xs mt-1 truncate">{f.title}</p>
                <p className="text-[10px] text-gray-500">{f.year ?? "—"}</p>
              </button>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
```

- [ ] **Step 4: Verify + smoke**

`cd frontend && pnpm exec tsc --noEmit`

With dev-up.sh running, visit `/cluster` → 21 cards. Click one → top-50 grid.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/cluster-card.tsx frontend/app/cluster/page.tsx frontend/app/cluster/[k]/page.tsx
git commit -m "feat(frontend): cluster browser pages — index + detail"
```

---

## Wave 6 — Gallery (T2)

### Task 6.1: build_gallery.py precompute

**Files:**
- Create: `scripts/build_gallery.py`

- [ ] **Step 1: Write the script**

```python
# scripts/build_gallery.py
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
    return f"hsl({h}, 60%, 55%)"


def decade_label(year):
    if pd.isna(year):
        return "Mixed era"
    return f"{(int(year) // 10) * 10}s"


async def main() -> None:
    films = pd.read_parquet(INFERENCE / "films_master.parquet")
    id_to_row = {int(r.id): i for i, r in enumerate(films.itertuples(index=False))}

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

    client = TMDbClient(api_key=os.environ.get("TMDB_API_KEY"), cache_dir=CACHE)
    enrichment = {}
    if client.key_configured:
        print(f"[gallery] fetching TMDb for {len(seen_ids)} ids...")
        blobs = await asyncio.gather(
            *(client.get_enrichment(i) for i in seen_ids),
            return_exceptions=False,
        )
        enrichment = dict(zip(seen_ids, blobs))
        print(f"[gallery] TMDb successes: {sum(1 for b in blobs if b)}")
    await client.aclose()

    def film_payload(row_idx: int, backbone: str) -> dict:
        r = films.iloc[row_idx]
        fid = int(r["id"])
        cluster = int(cluster_labels[backbone][row_idx])
        blob = enrichment.get(fid)
        year = int(r["year"]) if pd.notna(r["year"]) else None
        style, plot = ([], [])
        if blob:
            style, plot = split_keywords(blob.keyword_names)
        return {
            "id": fid,
            "title": str(r["title"]),
            "year": year,
            "rating": float(r["vote_average"]),
            "votes": int(r["vote_count"]),
            "genres": list(r["genres"]) if r["genres"] is not None else [],
            "country": str(r["country"]) if pd.notna(r["country"]) else None,
            "duration": float(r["runtime"]) if pd.notna(r["runtime"]) else None,
            "language": str(r["original_language"]),
            "director": str(r["director_name"]),
            "cluster": cluster,
            "overview": str(r["overview"]) if pd.notna(r["overview"]) else None,
            "time": decade_label(r["year"]) if year else "Mixed era",
            "place": str(r["country"]) if pd.notna(r["country"]) else None,
            "posterColor": hash_hsl(fid),
            "posterUrl": make_poster_url(blob.poster_path) if blob else None,
            "backdropUrl": make_backdrop_url(blob.backdrop_path) if blob else None,
            "tagline": (blob.tagline if blob else None),
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
```

- [ ] **Step 2: Run with TMDb key set**

Run: `python scripts/build_gallery.py`
Expected: ~3-4 minutes if TMDb key set, ~5 seconds if not. Writes `artifacts/inference/gallery.json` (~80 KB).

- [ ] **Step 3: Sanity check output**

```bash
python -c "
import json
g = json.load(open('artifacts/inference/gallery.json'))
print('queries:', g['queries'])
for q in g['queries']:
    for bb in ['ae_z32','ae_z64','ae_z128']:
        n = len(g['matrix'][q][bb]['neighbors'])
        c0 = g['matrix'][q][bb]['neighbors'][0]['cosine']
        print(f'  {q} / {bb}: {n} neighbors, top cosine {c0:.3f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_gallery.py artifacts/inference/gallery.json
git commit -m "feat(gallery): precompute 5×3 eyeball matrix as gallery.json"
```

### Task 6.2: /api/gallery endpoint + /gallery server component page

**Files:**
- Modify: `src/cineembed/api.py`
- Create: `frontend/app/gallery/page.tsx`

- [ ] **Step 1: Add /api/gallery endpoint**

Append to `src/cineembed/api.py`:

```python
GALLERY_PATH = INFERENCE_DIR / "gallery.json"


@app.get("/api/gallery")
def gallery() -> dict:
    if not GALLERY_PATH.exists():
        raise HTTPException(503, detail="gallery.json not built; run scripts/build_gallery.py")
    return json.loads(GALLERY_PATH.read_text())
```

- [ ] **Step 2: Write /gallery page (Server Component)**

```typescript
// frontend/app/gallery/page.tsx
import { Sidebar } from "@/components/sidebar";
import { GallerySchema } from "@/lib/api-types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function fetchGallery() {
  const res = await fetch(`${BASE}/api/gallery`, { next: { revalidate: 3600 } });
  const json = await res.json();
  return GallerySchema.parse(json);
}

export default async function GalleryPage() {
  const gallery = await fetchGallery();
  const backbones = ["ae_z32", "ae_z64", "ae_z128"] as const;

  return (
    <div className="flex min-h-screen bg-[#f8f9fb]">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8">
        <h1 className="text-2xl font-semibold mb-2">Eyeball gallery</h1>
        <p className="text-sm text-gray-600 mb-6">
          Five well-known queries × three backbones. The same query produces
          visibly different top-5 neighbours per backbone — the strongest
          demonstration of the project's z-sweep finding (see <a className="text-purple-700 underline" href="/about">About</a>).
        </p>
        <div className="space-y-8">
          {gallery.queries.map((q) => (
            <section key={q}>
              <h2 className="text-lg font-medium mb-3">{q}</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {backbones.map((bb) => {
                  const cell = gallery.matrix[q][bb];
                  return (
                    <div key={bb} className="border border-[#e5e4ec] rounded-lg p-3 bg-white">
                      <p className="text-xs font-medium text-purple-700 mb-2">{bb}</p>
                      <p className="text-sm font-medium mb-2">{cell.query.title} ({cell.query.year ?? "—"})</p>
                      <ol className="text-xs space-y-1">
                        {cell.neighbors.map((n, i) => (
                          <li key={n.id} className="flex justify-between">
                            <span>#{i + 1} {n.title}</span>
                            <span className="text-gray-500">{n.cosine.toFixed(3)}</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Verify + smoke**

Restart dev-up.sh (FastAPI reloads, frontend hot-reload). Visit `/gallery` → renders 5 sections × 3 columns with cosines.

- [ ] **Step 4: Commit**

```bash
git add src/cineembed/api.py frontend/app/gallery/page.tsx
git commit -m "feat(gallery): /api/gallery endpoint + server-component page"
```

---

## Wave 7 — Cosine heatmap (T3)

### Task 7.1: /api/films/{id}/cosine-dist endpoint

**Files:**
- Modify: `src/cineembed/api.py`
- Test: `tests/test_api_cosine_dist.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api_cosine_dist.py
from fastapi.testclient import TestClient

from cineembed.api import app

client = TestClient(app)


def test_cosine_dist_inception():
    r = client.get("/api/films/27205/cosine-dist", params={"backbone": "ae_z32", "bins": 30})
    assert r.status_code == 200
    body = r.json()
    assert len(body["bins"]) == 31  # n+1 edges for n bins
    assert len(body["counts"]) == 30
    assert len(body["top10"]) == 10
    s = body["stats"]
    for k in ("mean", "std", "min", "max", "p50", "p95"):
        assert k in s
```

- [ ] **Step 2: Run, verify failure, implement endpoint**

Append to `src/cineembed/api.py`:

```python
@app.get("/api/films/{film_id}/cosine-dist")
def cosine_dist(
    film_id: conint(ge=1),
    backbone: BackboneId = "ae_z32",
    bins: int = Query(30, ge=5, le=100),
) -> dict:
    if film_id not in state.id_to_row:
        raise HTTPException(404, detail="film not found")
    cosines = get_cosines(film_id, backbone)
    self_row = state.id_to_row[film_id]
    mask = np.ones_like(cosines, dtype=bool)
    mask[self_row] = False
    arr = cosines[mask]
    counts, edges = np.histogram(arr, bins=bins, range=(-1.0, 1.0))

    # top-10
    top_idx = np.argpartition(-arr, 10)[:10]
    top_idx = top_idx[np.argsort(-arr[top_idx])]
    top10 = []
    for i in top_idx:
        # i is index into arr (which has self removed); recover original row
        orig_row = int(np.where(mask)[0][i])
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
```

- [ ] **Step 3: Run tests, expect PASS**

Run: `pytest tests/test_api_cosine_dist.py -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add src/cineembed/api.py tests/test_api_cosine_dist.py
git commit -m "feat(api): /cosine-dist endpoint with histogram + stats + top10"
```

### Task 7.2: cosine-heatmap component (lazy-imported)

**Files:**
- Create: `frontend/components/cosine-heatmap.tsx`
- Modify: `frontend/components/selected-film-panel.tsx`

- [ ] **Step 1: Write cosine-heatmap.tsx**

```typescript
// frontend/components/cosine-heatmap.tsx
"use client";

import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { api, type BackboneId } from "@/lib/api";

interface Props {
  filmId: number;
  backbone: BackboneId;
}

export function CosineHeatmap({ filmId, backbone }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["cosineDist", filmId, backbone],
    queryFn: ({ signal }) => api.getCosineDist(filmId, backbone, 30, { signal }),
  });

  if (isLoading || !data) {
    return <div className="mt-6 h-44 bg-gray-50 rounded animate-pulse" />;
  }

  const histData = data.counts.map((c, i) => ({
    bin: data.bins[i].toFixed(2),
    count: c,
  }));

  return (
    <div className="mt-6 border-t border-[#e5e4ec] pt-4">
      <p className="text-xs font-medium text-gray-500 mb-2">Cosine distribution across 329,043 films</p>
      <p className="text-xs text-gray-500 mb-3">
        μ={data.stats.mean.toFixed(2)} · σ={data.stats.std.toFixed(2)} · p50={data.stats.p50.toFixed(2)} · p95={data.stats.p95.toFixed(2)} · top={data.stats.max.toFixed(3)}
      </p>
      <span className="sr-only">
        Cosine distribution. Mean {data.stats.mean.toFixed(2)},
        standard deviation {data.stats.std.toFixed(2)},
        top neighbor cosine {data.stats.max.toFixed(2)}.
      </span>
      <div style={{ width: "100%", height: 140 }}>
        <ResponsiveContainer>
          <BarChart data={histData}>
            <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={4} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#a78bfa" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs font-medium text-gray-500 mt-3 mb-1">Top-10 cosines</p>
      <div className="space-y-1">
        {data.top10.map((t) => (
          <div key={t.id} className="flex justify-between text-xs">
            <span className="truncate flex-1 pr-2">{t.title}</span>
            <span className="text-gray-500">{t.cosine.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Lazy-mount inside selected-film-panel.tsx**

Edit `frontend/components/selected-film-panel.tsx` — add at top:

```typescript
import dynamic from "next/dynamic";
const CosineHeatmap = dynamic(
  () => import("./cosine-heatmap").then((m) => m.CosineHeatmap),
  { ssr: false, loading: () => <div className="mt-6 h-44 bg-gray-50 rounded animate-pulse" /> }
);
```

And replace the trailing comment placeholder with:

```typescript
      <CosineHeatmap filmId={film.id} backbone={backbone as any} />
    </article>
```

- [ ] **Step 3: Verify + smoke**

`cd frontend && pnpm exec tsc --noEmit`

With dev-up.sh running, click a film → heatmap renders below overview after a short load.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/cosine-heatmap.tsx frontend/components/selected-film-panel.tsx
git commit -m "feat(frontend): cosine-heatmap component (lazy-imported)

recharts BarChart 30-bin histogram + stats badges + top-10 list. Embedded
in selected-film-panel via next/dynamic so recharts is excluded from the
initial bundle. SR-only summary text for a11y."
```

---

## Wave 8 — README + demo-script + final polish

### Task 8.1: README run instruction

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read existing README**

Run: `head -50 README.md`

- [ ] **Step 2: Replace run section**

Replace the "Running" / "Setup" section with:

```markdown
## Demo — 5-line setup

```bash
# 1. Install Python deps
pip install -e ".[demo]"

# 2. Install frontend deps
cd frontend && pnpm install && cd ..

# 3. Set TMDb API key (optional but recommended)
cp .env.example .env
# edit .env: TMDB_API_KEY=your_v3_key_from_themoviedb.org

# 4. Launch
bash scripts/dev-up.sh

# 5. Open browser
open http://localhost:3000
```

See `docs/demo-script.md` for the presentation flow and known-good
query list.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): 5-line demo run instruction"
```

### Task 8.2: docs/demo-script.md

**Files:**
- Create: `docs/demo-script.md`

- [ ] **Step 1: Write the file**

```markdown
# CineEmbed Demo Script

**Goal:** 5-minute walkthrough of the SENG 474 final demo.
**Audience:** SENG 474 grader (Spring 2026, TED University).
**Repo:** `bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System`, branch `main`.
**Setup:** see `README.md` § Demo — 5-line setup.

---

## Presentation flow (~5 minutes)

### 1. Opening (15 sec)
Open `localhost:3000`. Point to the title and the empty-state copy
("Search 329,044 films...").

### 2. Search demo (30 sec)
Type "inception". Note: live filter over 329k films, results appear
within 500ms. Click the top result.

### 3. Selected film panel (45 sec)
Walk through metadata (genres, year, cluster badge, director). Point
to the TMDb poster, the overview, the style/plot keyword chips. Scroll
to the cosine heatmap: "this is the distribution of Inception's
similarity to every one of 329,043 other films — the right tail is the
recommendation."

### 4. Similar panel (30 sec)
Show the top-5 with color-coded cosine badges. Click a similar film
(e.g., The Dark Knight). Point out the URL state navigation:
`?film=155&backbone=ae_z32`.

### 5. Backbone switch — the methodological story (60 sec)
Click `ae_z128` in the page-header switcher. Similar panel re-renders
with different neighbors. Narrate: "Smaller models win on the demo task
because the information bottleneck forces concentration on
high-entropy modalities (text overview, director PCA). z=128 has
near-dead latent dimensions and gives more Marvel-flavored neighbors;
z=32 keeps the Nolan signature." Cite journal/12.

### 6. Cluster browser (45 sec)
Navigate to `/cluster`. Show the 21 named clusters, each with a
4-poster mosaic. Pick a memorable cluster (e.g., "Drama · 1990s" or
"Action · 2000s"). Click it → top-50 film grid.

### 7. Gallery (45 sec)
Navigate to `/gallery`. Five queries × three backbones precomputed
side-by-side. Spend 20 sec on "Inception → ae_z32 vs ae_z128" to
narrate the z-sweep finding visually.

### 8. About page (30 sec)
Navigate to `/about`. Brief stop — the grader sees the methodology
explained in writing.

### 9. Q&A buffer
End at the home page on Inception so questions can quickly demonstrate
specific points.

---

## Known-good query list (verified during prep)

| Primary | Secondary backup if needed |
|---|---|
| **Inception (27205)** | The Dark Knight (155), Interstellar (157336) |
| **Spirited Away (129)** | My Neighbor Totoro (8392), Princess Mononoke (128) |
| **Shawshank Redemption (278)** | The Green Mile (497) |
| **Pulp Fiction (680)** | Reservoir Dogs (500), Kill Bill Vol. 1 (24) |
| **Toy Story (862)** | WALL·E (10681), Finding Nemo (12) |

Substitute from the same row's backup list if a primary query surfaces
unexpected results during a dry run.

---

## Failure modes and recovery

- **TMDb key missing or rate-limited**: posters fall back to HSL
  gradient cards. App still functional. Keyless-mode banner explains
  the state. Do NOT panic during demo — narrate as "the demo also
  works without TMDb."
- **Backend offline**: top banner says "Backend not reachable. Run
  `bash scripts/dev-up.sh`". Restart from CLI; frontend reconnects.
- **Slow first render**: Next.js dev server first-compile can take
  20-40s for Next 16 + shadcn. Pre-warm before the presentation by
  running `dev-up.sh` and clicking around the home page once.

---

## Talking points

- "Multimodal autoencoder over 7 feature blocks, 329k films, latent dim 32."
- "We found two paper-worthy methodological findings during this
  project — they're not in textbooks."
  1. **NMI ≠ retrieval quality** (journal/07): clustering metrics
     do not predict cosine-retrieval quality. Our NMI champion had
     angular collapse and gave random top-5 ranks within cluster.
  2. **z-sweep U-curve** (journal/12): smaller latent (z=32) beat
     larger latent (z=128) on the demo task. Information-bottleneck
     sweet spot.
- "The demo lets you SEE both findings — the gallery is the proof."
```

- [ ] **Step 2: Commit**

```bash
git add docs/demo-script.md
git commit -m "docs(demo-script): presentation flow + known-good queries

5-minute walkthrough with timestamps, talking points tied to journal/07
and journal/12, failure-mode recovery, backup query list per spec §15."
```

### Task 8.3: Final smoke + push

- [ ] **Step 1: Run full smoke**

In one terminal: `bash scripts/dev-up.sh`
In another: `bash scripts/demo-smoke.sh`
Expected: ALL 9 checks PASS.

- [ ] **Step 2: Manual acceptance walkthrough**

Per spec §11 acceptance criteria 1-13:
- [ ] dev-up.sh ready signal
- [ ] smoke 0
- [ ] search "inception" → Inception @ top within 500ms
- [ ] click film → details + similar within 1-4s
- [ ] backbone switcher → re-render within 1.5s, URL update
- [ ] URL share test
- [ ] /cluster → 21 named clusters → detail
- [ ] /gallery → 5×3 instant
- [ ] cosine heatmap visible
- [ ] TMDb offline scenario (set TMDB_API_KEY="" and restart) → app still works
- [ ] /about renders
- [ ] README run instruction works
- [ ] keyboard-only navigation

- [ ] **Step 3: Tag T3 demo-ready**

```bash
git tag t3-demo-ready -m "T3 demo: all acceptance criteria met"
```

- [ ] **Step 4: Push everything**

```bash
git push origin main
git push origin t0-shippable
git push origin t3-demo-ready
```

---

## Self-review check

**Spec coverage scan:**

| Spec section | Covered by task(s) |
|---|---|
| §1 motivation | (no task; context) |
| §2 scope | implicit in all tasks |
| §3 architecture | all tasks |
| §4 repo layout | already-done (subtree) + paths used by every task |
| §5.1 build pipeline | 1.2 (enrich), 1.3 (build_index), 1.5 (backbones), 6.1 (gallery) |
| §5.2 runtime memory | 2.4 (api.py lifespan) |
| §5.3 cluster naming | 1.3 + 1.4 (override) |
| §5.4 field matrix | 2.4 (_row_to_film) |
| §5.5 STYLISTIC_DICT | 1.1 |
| §6.1 shared types | 2.1 (Pydantic) + 3.2 (zod) |
| §6.2 endpoint catalog | 2.4, 2.5, 2.6, 5.1, 6.2, 7.1 |
| §6.3 error model | 2.4, 2.5, 5.1 |
| §6.4 latency budgets | 2.4 (prewarm) + 2.6 (LRU helper) |
| §7.1 pages | 3.4 (home), 5.2 (clusters), 6.2 (gallery), 3.7 (about) |
| §7.2 components | 3.4-3.6, 4.1, 5.2, 7.2 |
| §7.3 state mgmt | 3.3 (providers) + 4.1 (invalidation) |
| §7.4 UX details | 3.4, 3.5, 3.6, 3.7, 7.2 |
| §7.5 build config | 3.5 (next.config) + 0.2 (deps) |
| §8 sequencing | all waves |
| §11 acceptance | 3.8 + 8.3 |
| §12 deps | 0.1 + 0.2 |
| §14 invariants | 2.4 (id_to_row, prewarm, BLAS threads via dev-up), 2.6 (LRU cache) |
| §15 demo script | 8.2 |

No spec section uncovered.

**Placeholder scan:** no `TBD`, `TODO`, `implement later`, vague "add validation". Every step has complete code or exact command.

**Type consistency:** `Film` shape matches Pydantic model (`api_models.py`) ↔ zod schema (`api-types.ts`) ↔ frontend usage. `BackboneId` literal consistent (`ae_z32 | ae_z64 | ae_z128`). `Cluster.previewFilms` is the listing variant; `ClusterDetail.films` is the full variant — both populated correctly in §5.1's endpoint.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-18-frontend-backend-integration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks. Best for this scope: many isolated tasks, easy parallel review.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints. Slower but the user can intervene per checkpoint.

**Which approach?**
