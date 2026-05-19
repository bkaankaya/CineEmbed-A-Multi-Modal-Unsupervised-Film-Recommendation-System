# Frontend ↔ Backend Integration — Design Spec

**Date:** 2026-05-18
**Status:** APPROVED (brainstorm) — AMENDED 2026-05-18 PM after Codex adversarial review
**Deadline:** 2026-05-20 (Wednesday)
**Cross-ref:** ADR `0001-modeling-hybrid-architecture.md` D14, D15;
`docs/superpowers/specs/2026-05-16-web-app-demo-design.md`;
`docs/journal/12-z-sweep-ae-z32-discovery.md`

## Amendment summary — 2026-05-18 PM (post-Codex review)

After a full adversarial review by Codex (session
`019e3c53-b50c-77a1-9660-8f6650d2d4c0`), the spec was hardened across
12 dimensions. Major changes incorporated:

- **Scope target moved from "T3 by Day 2" to "hybrid: T0 hard-gate end of
  Day 1, extras opportunistic on Day 2"** (§8). T3 stays aspirational but
  the demo is rollback-safe at the T0 gate.
- **Backend invariants** spelled out in a new §14: `id_to_row_idx`
  mapping at boot, ae_z32 prewarm, concurrency model (sync routes /
  `run_in_threadpool`, BLAS thread cap), atomic disk-LRU cache with TTL
  + size bounds, Pydantic validators with field caps.
- **API contract fixes**: `limit` consistent (replaces `top`), `Cluster`
  detail returns `{...Cluster, films, total}` with `limit ∈ [1,100]`,
  TMDb upstream failure returns `200` + `tmdb_status: "missing"` flag
  (not `502`).
- **Film type**: `style`, `plot` are always `string[]` (never null,
  empty `[]` when no TMDb data); `posterUrl`/`backdropUrl`/`tagline`
  nullable. Backend constructs full TMDb CDN URLs
  (`https://image.tmdb.org/t/p/w342{path}`).
- **Frontend**: URL state for `film` + `backbone` (`?film=27205&backbone=ae_z32`),
  backbone-switcher moved from sidebar nav into main content's top
  segmented bar, `/gallery` rendered as Next.js server component for
  static-fast first paint, `CosineHeatmap` lazy-imported via
  `next/dynamic`. New `/about` page added in-scope.
- **TMDb client**: global async token bucket ≤35 req/10s + 429
  backoff, per-id semaphore prevents duplicate concurrent fetches,
  cache TTL 30 days + max ~50k entries, dedup across gallery fetches.
- **Cluster naming**: exclude Unknown / empty genres (37% null rate per
  journal/07) from top-genre computation; manual override JSON for
  collisions; "Mixed era" when year-null rate exceeds 50% in a cluster.
- **Data quality**: `production_countries` parse fixed (it's a string
  field, not array-indexable); style/plot fall back to a single
  "Keywords" chip list when STYLISTIC_DICT yields zero hits (no fake
  split).
- **Deliverables added**: `scripts/demo-smoke.sh`, `docs/demo-script.md`
  (presentation flow + known-good queries), `app/about/page.tsx`
  (in-app methodology blurb).
- **Latency budget**: hot/cold split, ae_z32 first-touch prewarm.

Three Codex suggestions were modified in implementation detail rather
than accepted verbatim:

1. **Scope** — Codex urged a hard cut to T1+gallery; we keep T3 as a
   ceiling but insert a T0 hard gate at end of Day 1 so the demo is
   never at risk.
2. **Style/plot empty fallback** — Codex offered two options
   ("style unavailable" or label as "keywords"); we drop the
   dual-chip presentation entirely when STYLISTIC_DICT yields zero
   and render a single "Keywords" list. Cleaner UX, honest data.
3. **`/cosine-dist` endpoint** — Codex offered to merge into
   `/similar?includeDist=true` or share cosines via cache; we keep the
   endpoint separate (SRP) and add a server-side LRU cache on
   `compute_cosines(id, backbone)` (size 50) to amortize repeated scans.

All other Codex findings (HIGH and MEDIUM) were accepted as written.

## 1. Motivation

The model side of CineEmbed is locked: `ae_z32` is the demo backbone (ADR D15),
inference indices exist for all three z-sweep variants under
`artifacts/inference/ae_z{32,64,128}/`. Teammates have produced a Next.js 16
frontend with shadcn/ui components on the `frontend-ui` branch of the fork
`bkaankaya/CineEmbed-Multimodal-Movie-Embeddings`, currently driven by
mock data (8 films, hand-curated similarity map).

This spec connects the two: a FastAPI sidecar serves the embeddings to the
Next.js frontend, the frontend is rewired from mock to real data, and the
demo gains four ML-revealing features that make the project's
methodological findings visceral to a grader — backbone switcher,
cluster browser, eyeball gallery, and per-film cosine distribution
heatmap.

## 2. Scope

**In scope:**

- Monorepo merge of teammates' `frontend-ui` branch as `frontend/` subtree.
- New `src/cineembed/api.py` (FastAPI) serving 8 endpoints.
- Live runtime switching across all three backbones (`ae_z32`, `ae_z64`, `ae_z128`).
- TMDb lazy-fetched enrichment with disk-LRU cache for posters, keywords, tagline.
- Three new frontend pages: home (modified), `/cluster/[k]` (new),
  `/gallery` (new).
- Four new frontend components: `backbone-switcher`, `cosine-heatmap`,
  `cluster-card`, `lib/api.ts` (typed fetch wrapper).
- One-shot build scripts for index extension (KMeans labels +
  cluster auto-naming) and gallery precompute.
- `scripts/dev-up.sh` parallel-launcher for the two processes.

**Out of scope (deferred to future work):**

- Cloud deployment (Vercel / Render). Demo is local-only.
- Authentication / multi-user / rate limiting beyond CORS.
- Persistence layer (database). All data is in-memory or file-based.
- Trailer playback, cast lists, review scraping.
- Production observability (Sentry, OpenTelemetry).

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  👤 Browser (localhost:3000)                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/JSON (fetch)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  🌐 Next.js (:3000) — frontend/ subdir                       │
│    app/page.tsx          (home: search + details + similar)  │
│    app/cluster/page.tsx  (21-card grid)                      │
│    app/cluster/[k]/page.tsx (cluster detail)                 │
│    app/gallery/page.tsx  (eyeball 5×3 matrix)                │
│                                                              │
│    components/  (modify 7 existing + 4 new)                  │
│    lib/api.ts   (typed fetch + zod schemas)                  │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/JSON (CORS allow-origin localhost:3000)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  🐍 FastAPI (:8000) — src/cineembed/api.py                   │
│                                                              │
│  Boot-time load (per backbone × 3):                          │
│    embeddings.npy → np.ndarray (329044, z)                   │
│    cluster_labels.npy → np.ndarray (329044,) uint8           │
│    films_master.parquet → pd.DataFrame (shared across 3)     │
│                                                              │
│  Endpoints:                                                  │
│    GET /api/health                                           │
│    GET /api/backbones                                        │
│    GET /api/films/search                                     │
│    GET /api/films/{id}                                       │
│    GET /api/films/{id}/similar                               │
│    GET /api/films/{id}/cosine-dist                           │
│    GET /api/clusters, /api/clusters/{k}                      │
│    GET /api/gallery                                          │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTPS (server-side only)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  🎬 TMDb API (external)                                       │
│    /movie/{id} → posters, tagline, runtime, vote_average     │
│    /movie/{id}/keywords → semantic tags                      │
│                                                              │
│  Disk-backed LRU cache at artifacts/cache/tmdb/{id}.json     │
│  Pre-warmed: top-2000 popularity films (optional, ~15 min)   │
└─────────────────────────────────────────────────────────────┘
```

**Two processes, one repo, one start command (`scripts/dev-up.sh`).** CORS
restricted to `http://localhost:3000` in dev; configurable via env var.

## 4. Repository layout

```
CineEmbed-/                          ← bizim repo, branch main
├── src/cineembed/
│   ├── api.py                       ← NEW · FastAPI app
│   ├── search.py                    ← NEW · rapidfuzz helper
│   ├── keywords.py                  ← NEW · STYLISTIC_DICT (~50 entries)
│   ├── tmdb.py                      ← NEW · TMDb client + disk LRU cache
│   ├── backbone.py, data.py, ...    ← existing
│   └── wandb_integration.py
├── frontend/                        ← NEW · subtree from bkaankaya frontend-ui
│   ├── app/
│   │   ├── page.tsx                 ← modified (mock → real)
│   │   ├── cluster/page.tsx         ← NEW
│   │   ├── cluster/[k]/page.tsx     ← NEW
│   │   ├── gallery/page.tsx         ← NEW
│   │   ├── providers.tsx            ← NEW (TanStack Query)
│   │   ├── layout.tsx, globals.css
│   ├── components/
│   │   ├── backbone-switcher.tsx    ← NEW
│   │   ├── cosine-heatmap.tsx       ← NEW
│   │   ├── cluster-card.tsx         ← NEW
│   │   ├── search-bar, selected-film-panel, similar-films-panel,
│   │     sidebar, empty-state, film-poster, theme-provider, ui/ (shadcn)
│   ├── lib/
│   │   ├── api.ts                   ← NEW (typed fetch + zod)
│   │   ├── utils.ts                 ← existing
│   └── package.json
├── artifacts/
│   ├── models/ae_z{32,64,128}/      ← existing
│   ├── inference/
│   │   ├── films_master.parquet     ← NEW (single shared 329k-row table, country enriched)
│   │   ├── gallery.json             ← NEW (precomputed 5-query × 3-backbone)
│   │   └── ae_z{32,64,128}/         ← per-backbone inference dir
│   │       ├── embeddings.npy       ← existing
│   │       ├── cluster_labels.npy   ← NEW (KMeans k=21 labels per backbone)
│   │       ├── cluster_meta.json    ← NEW (21-cluster summary: name, top genres, decade)
│   │       └── manifest.json        ← existing (legacy films.parquet per-dir is removed)
│   └── cache/tmdb/                  ← NEW (gitignored, runtime LRU JSON cache)
├── scripts/
│   ├── build_index.py               ← extend (KMeans labels, cluster_meta)
│   ├── enrich_films.py              ← NEW (CSV → master parquet)
│   ├── build_gallery.py             ← NEW (5-query × 3-backbone precompute)
│   ├── warm_tmdb_cache.py           ← NEW (optional)
│   └── dev-up.sh                    ← NEW (uvicorn + pnpm dev parallel)
├── docs/                            ← existing journal, adr, specs
└── pyproject.toml                   ← [demo] extras append: rapidfuzz, httpx, slowapi
```

`frontend/` lands via `git subtree add --prefix=frontend <fork-url>
frontend-ui --squash`. History is squashed into one commit; future pulls
via `git subtree pull` if teammates push changes.

## 5. Data plane

### 5.1 Build-time pipeline (one-shot, ~25 min)

| # | Script | Action | Output | Runtime |
|---|---|---|---|---|
| 1 | `scripts/enrich_films.py` | Read `artifacts/movies_eda_final.csv`. **`production_countries` is a string column** — it is the first country name (e.g. `"Finland"`, `"United States of America"`) OR a JSON-stringified list of `{iso_3166_1, name}` dicts depending on the upstream TMDb dump. Parse robustly: if `value.startswith('[')`, `json.loads` and take `[0]["iso_3166_1"]`; otherwise treat as a country name and look up the ISO-3166 alpha-2 via `pycountry.countries.lookup(value).alpha_2`; on lookup failure store null. Write enriched DataFrame as `artifacts/inference/films_master.parquet` (single shared table). | `films_master.parquet` (329k rows, 15 cols; `country` is ISO alpha-2 nullable string) | ~30-60 sec |
| 2 | `scripts/build_index.py` (extend) | (existing) encode films through backbone, L2-norm, write `embeddings.npy` — this is the **slow** step (~5-8 min per backbone if rebuilding from scratch). (new) After encode, KMeans k=21 on the L2-normed embedding via `MiniBatchKMeans(batch_size=4096, n_init=10)` — this is **fast** (~30 sec per backbone). Write `cluster_labels.npy` (uint8). Compute per-cluster top-3 genres (excluding `Unknown` and empty-genre rows) + modal decade (or `"Mixed era"` when year-null rate exceeds 50% in that cluster) → `cluster_meta.json` with field `manualOverride: false`. | `ae_z{32,64,128}/{embeddings.npy, cluster_labels.npy, cluster_meta.json, manifest.json}` | encode-and-cluster from scratch ~6-9 min × 3 = ~25 min; if `embeddings.npy` already present (this project's state), KMeans-only ~30 sec × 3 = ~1.5 min |
| 2.5 | `scripts/build_backbones_metadata.py` (NEW) | Write static `artifacts/backbones.json` aggregating each `ae_z*/manifest.json` retrieval metrics + the gNMI / genre@5 numbers from `docs/journal/10-results-table.md`. Consumed by `/api/backbones` at boot. Avoids runtime markdown parsing. | `artifacts/backbones.json` (~1 KB, 3 entries) | ~1 sec |
| 3 | `scripts/build_gallery.py` | Hardcoded 5 query film TMDb ids (Inception 27205, Spirited Away 129, Shawshank 278, Pulp Fiction 680, Toy Story 862). For each (query, backbone) pair, compute top-5 neighbors via cosine. **Then dedupe all unique film ids across the entire 15-pair matrix BEFORE any TMDb call** (5 unique query films + up to ~50 unique neighbors ≈ ~55 unique ids). Fetch TMDb with 2 endpoints × ~55 ids ≈ ~110 calls under the shared token bucket. Persist as `gallery.json` referencing the enriched lookup map (matrix entries cite ids, the lookup map carries enriched Film objects). Idempotent: a partial gallery from a prior interrupted run is reused. | `artifacts/inference/gallery.json` (~80 KB) | ~3-4 min at 35 req / 10 sec rate cap |
| 4 | `scripts/warm_tmdb_cache.py` (optional) | Top-2000 films by `popularity` across union of 3 backbones. Dedup, then 2 endpoints × 2000 = ~4000 calls under the same global token bucket. Supports `--refresh` flag to rebuild stale entries. | `artifacts/cache/tmdb/{id}.json` for ~2000 ids (~10 MB) | ~20 min at 35 req / 10 sec |

### 5.2 Runtime memory layout (FastAPI boot, ~5-15 sec)

| Backbone | embeddings.npy | cluster_labels | Total per backbone |
|---|---:|---:|---:|
| ae_z32 | 42 MB | 0.3 MB | 42 MB |
| ae_z64 | 80 MB | 0.3 MB | 80 MB |
| ae_z128 | 168 MB | 0.3 MB | 168 MB |
| **Sum** | **290 MB** | **1 MB** | **~291 MB** |

Plus `films_master.parquet` once: ~80 MB. Total RAM at idle: **~370 MB**.

`embeddings.npy` loaded with `np.load(mmap_mode='r')` so the OS pages it
in on demand (no upfront cost; per-backbone cold first-touch ≈
33-123ms as measured empirically — see §6.4). `films_master.parquet`
is **materialized** into an in-memory `pd.DataFrame` at boot via
`pd.read_parquet(..., engine="pyarrow")`. Pyarrow is the parser, not
a lazy layer — the table lives in RAM after boot (~80 MB). This is
the spec's intended behavior; an earlier draft incorrectly described
it as "lazy."

### 5.3 Cluster auto-naming heuristic

For each KMeans cluster k ∈ [0, 20]:

1. **Compute genre frequency excluding `Unknown` and empty-genre films**
   (~37% of the corpus lacks a primary genre per journal/07; including
   them would make every cluster look genre-Unknown-dominated).
2. Top-3 genres with their coverage %: e.g., `[("Drama", 0.42),
   ("Romance", 0.18), ("Comedy", 0.12)]`.
3. **Modal decade computed on rows with non-null `year`**. If the
   year-null rate within the cluster exceeds 50%, use `"Mixed era"`
   instead of a decade label (year is null for ~7.6% of the corpus,
   per local parquet inspection, but cluster-level skew is possible).
4. Optional: dominant language only if >70% of cluster films share one
   language; omit otherwise.
5. Template: `"{Top1Genre} · {ModalDecade}{lang_suffix?}"`.

**Collisions are expected** because Drama is the dominant genre across
several clusters (per journal/01). The disambiguator suffix `" (k=N)"`
is applied automatically when two clusters generate the same name, but
**all 21 names MUST be sanity-reviewed by a human** before the demo —
manual overrides go into `artifacts/inference/cluster_names_override.json`
(one entry per backbone, keyed by `{backbone}.{k}`). The boot loader
merges auto-names with overrides; the override JSON ships in git.

Naming is deterministic per backbone (different KMeans seeds across
backbones give different labels — this is correct, the clusters are
genuinely different).

### 5.4 Film type field matrix (final)

| Frontend field | Source | Timing |
|---|---|---|
| `id` | parquet `id` | build |
| `title` | parquet `title` | build |
| `year` | parquet `year` | build |
| `rating` | parquet `vote_average` (renamed) | build |
| `votes` | parquet `vote_count` (renamed) | build |
| `genres` | parquet `genres` | build |
| `duration` | parquet `runtime` (renamed) | build |
| `language` | parquet `original_language` (renamed) | build |
| `director` | parquet `director_name` (renamed) | build |
| `overview` | parquet `overview` | build |
| `country` | parquet `country` (enriched from CSV) | build |
| `cluster` | `cluster_labels[backbone][row]` | request |
| `time` | derived: `decade_label(year)` → `"1990s"` | request |
| `place` | alias of `country` | request |
| `posterColor` | derived: `hash_hsl(id)` → `"hsl(284, 60%, 55%)"` | request |
| `posterUrl` | TMDb `poster_path` joined to base CDN: `f"https://image.tmdb.org/t/p/w342{path}"` (backend constructs full URL) | request (LRU) |
| `backdropUrl` | TMDb `backdrop_path` joined to `"https://image.tmdb.org/t/p/w1280"` | request (LRU) |
| `tagline` | TMDb `tagline` (nullable) | request (LRU) |
| `style` | `tmdb_keywords ∩ STYLISTIC_DICT`, capped at 8 entries, original order preserved | request (LRU) |
| `plot` | `tmdb_keywords \ STYLISTIC_DICT`, capped at 8 entries, original order preserved | request (LRU) |
| `tmdbStatus` | `"ok"` when TMDb fetch succeeded (even partially), `"missing"` on failure | request |

### 5.5 STYLISTIC_DICT (style vs plot split)

`src/cineembed/keywords.py` defines a curated set of cinematic-style
keywords. Examples:

```python
STYLISTIC_DICT = frozenset({
    "neo-noir", "film noir", "slasher", "mockumentary", "found footage",
    "anthology film", "satire", "parody", "dark comedy", "screwball comedy",
    "romantic comedy", "psychological thriller", "psychological horror",
    "supernatural horror", "body horror", "cosmic horror", "atmospheric",
    "dystopian future", "post-apocalyptic", "cyberpunk", "steampunk",
    "surrealism", "gothic", "noir", "neo-western", "spaghetti western",
    "cult classic", "indie film", "art house", "experimental",
    "musical", "stop motion", "claymation", "anime", "rotoscoping",
    "silent film", "black and white", "one-shot", "epistolary",
    "non-linear narrative", "unreliable narrator", "ensemble cast",
    "buddy cop", "courtroom drama", "heist film", "war epic",
    "coming-of-age", "fish out of water",
    # ~50 total
})
```

Words checked case-insensitive against TMDb keyword names. Anything in
the dict → `style[]`; anything else → `plot[]`.

**Empty `style[]` fallback rule (revised post-Codex):** if the
STYLISTIC_DICT yields zero hits on a given film's TMDb keywords, **do
NOT redistribute plot entries into style**. That move would fabricate a
"stylistic insight" the data does not actually support. Instead:

- Backend returns `style = []`, `plot = []`, **and** sets a derived
  client-side field `keywordsFallback = true` (computed in `lib/api.ts`
  Zod transformer when `style.length === 0 && plot.length > 0` AND
  `tmdbStatus === "ok"`).
- Frontend `selected-film-panel.tsx` renders ONE chip list labelled
  **"Keywords"** (not "Style" / "Plot" split) when the fallback flag
  fires, populated from the raw TMDb keywords (capped at 8). Honest
  presentation, no fake categorization.

The split presentation is only used when STYLISTIC_DICT genuinely
matches at least one keyword.

## 6. API contract

### 6.1 Shared types

```typescript
type Film = {
  id: number;
  title: string;
  year: number | null;
  rating: number;
  votes: number;
  genres: string[];
  country: string | null;          // ISO-3166 alpha-2 (e.g. "US"), or null if unknown
  duration: number | null;
  language: string;
  director: string;
  cluster: number;                 // 0-20 — depends on requested backbone
  overview: string | null;
  time: string;                    // derived "1990s" or "Mixed era"
  place: string | null;            // human-readable, alias of country lookup
  posterColor: string;             // deterministic HSL fallback

  // TMDb fields — always present in schema, populated when cache hit or fetch succeeds.
  // Arrays are NEVER null (use [] when no data) so the zod runtime check stays simple.
  // Scalar URL/tagline fields are nullable.
  posterUrl: string | null;        // FULL CDN URL e.g. "https://image.tmdb.org/t/p/w342/abc.jpg"
  backdropUrl: string | null;      // FULL CDN URL "/t/p/w1280" base
  tagline: string | null;
  style: string[];                 // TMDb keywords ∩ STYLISTIC_DICT, capped at 8
  plot: string[];                  // TMDb keywords \ STYLISTIC_DICT, capped at 8
  tmdbStatus: "ok" | "missing";    // "missing" when TMDb fetch failed, fields above null/[]
};

type Neighbor = Film & { cosine: number };

type Cluster = {
  id: number;                      // 0-20
  name: string;                    // e.g. "Drama · 1990s", or "Drama · 1990s (k=7)" on collision
  size: number;
  topGenres: { genre: string; pct: number }[];  // empty/"Unknown" genres excluded
  modalDecade: string;             // "1990s" or "Mixed era" when year-null > 50%
  previewFilms: Film[];            // top-4 by popularity, no TMDb (fast listing)
};

type ClusterDetail = Cluster & {
  films: Film[];                   // top-{limit} by popularity; first 5 TMDb-enriched, rest TMDb-lazy
  total: number;                   // size of cluster (films.length may be less due to limit)
};

type Backbone = {
  id: "ae_z32" | "ae_z64" | "ae_z128";
  z: 32 | 64 | 128;
  label: string;                   // "AE z=32 (demo)"
  genreAtFive: number;             // from artifacts/backbones.json (NOT runtime-parsed markdown)
  gnmi: number;
  preferred: boolean;              // ae_z32 true, others false
};
```

**Note on arrays-never-null**: zod schemas should declare
`z.array(z.string())` (no nullable), backed by a Pydantic model with
`default_factory=list`. Frontend code can rely on `film.style.length > 0`
checks without null guards. This is enforced even when `tmdbStatus ===
"missing"` — arrays stay `[]`.

**Note on `previewFilms` vs `films`**: `previewFilms` is used by the
21-cluster listing endpoint (kept light, no TMDb), while `films` is used
by the cluster-detail endpoint (TMDb-enriched first 5 in parallel). The
two field names disambiguate so a junior implementer doesn't conflate
them.

### 6.2 Endpoint catalog

**`GET /api/health`** → status string + loaded backbones.

**`GET /api/backbones`** → array of `{id, z, label, genre_at_5, gnmi}` with
metrics drawn from manifests + journal/10.

**`GET /api/films/search?q={str}&backbone={id}&limit={int=10}`** → `Film[]`.
Algorithm: lowercase prefix scan first; if results < limit, `rapidfuzz.process.extract`
with `score_cutoff=70`; merge, dedupe, sort by `(prefix_score, popularity)` desc.
TMDb-lazy fields are `null`.

**`GET /api/films/{id}?backbone={id}`** → `Film`. Includes `cluster`,
`time`, `place`, `posterColor`. Triggers TMDb enrichment on cache miss
(2 calls: `/movie/{id}` + `/movie/{id}/keywords`). **TMDb upstream
failure returns `200` with `tmdbStatus: "missing"` and empty arrays /
null URLs — never `502`** (the base Film from parquet is always
producible).

**`GET /api/films/{id}/similar?backbone={id}&limit={int=10}`** →
`Neighbor[]`. Computes `cosines = embeddings[backbone] @ q_emb` (via
internal `compute_cosines(id, backbone)` helper with LRU cache size 50),
drops self, returns top-`limit`. Top-5 are TMDb-enriched in parallel; rest
left tmdb-lazy (`tmdbStatus: "missing"` until `/api/films/{id}` is hit).
**Query param `limit` (1-50, default 10).** Old `top` name is removed.

**`GET /api/films/{id}/cosine-dist?backbone={id}&bins={int=30}`** →
`{bins, counts, stats, top10}`. Used by the cosine-heatmap component.
`stats` includes `{mean, std, min, max, p50, p95}` over the (329044 − 1)
cosines. `top10` mirrors first 10 of `/similar` (`{id, title, cosine}`).
The implementation calls the same `compute_cosines(id, backbone)` helper
(LRU cache shared with `/similar`), so a typical home-page click triggers
only one vector scan even if both endpoints fire. Param `bins ∈ [5, 100]`.

**`GET /api/clusters?backbone={id}`** → `Cluster[]` (21 entries; each
carries `previewFilms` = top-4 by popularity; TMDb-lazy fields stay
empty / null to keep the listing fast).

**`GET /api/clusters/{k}?backbone={id}&limit={int=50}`** →
`ClusterDetail` (`Cluster` plus `films` = top-{limit} by popularity, plus
`total` = cluster size). `limit ∈ [1, 100]`, default 50; first 5 films
TMDb-enriched in parallel, rest TMDb-lazy. `k ∈ [0, 20]` validated as
Pydantic `conint`.

**`GET /api/gallery`** → precomputed 5-query × 3-backbone matrix from
`artifacts/inference/gallery.json`. Pure file read; no live computation.

### 6.3 Error model

- `400` invalid backbone → `{detail: "backbone must be one of: ae_z32, ae_z64, ae_z128"}`
- `400` invalid query param (`q` too long, `limit` out of range, etc.) →
  Pydantic's default validation error shape (`detail` field with location)
- `404` film / cluster not found → `{detail: "<resource> not found"}`
- `503` backbone not loaded → `{detail: "backbone <id> not loaded; check /api/health"}`

**No `502` for TMDb upstream failure.** Endpoints that produce a base
Film (search, /films/{id}, /similar, /clusters) ALWAYS return `200` with
`tmdbStatus: "missing"` on TMDb failure. The only `502` use would be if
TMDb were a hard dependency for a response shape, which it never is in
this spec.

No auth. CORS allow-origin set = `{http://localhost:3000,
http://127.0.0.1:3000}` (env override `CORS_ORIGINS` for grader machine
with a different hostname). Wildcard `*` is NOT used even in dev — the
risk-register entry suggesting it was reverted post-Codex.

### 6.4 Latency budgets

Hot / cold separation per backbone. "Cold" = first request after boot for
that backbone before mmap page-cache is warm. Empirical cold-dot timings
on a MacBook Air M2 measured directly against the real files:

| Endpoint | Cold (first hit) | Hot (subsequent) | Notes |
|---|---:|---:|---|
| `/health`, `/backbones` | <5ms | <5ms | Memory-only |
| `/search` (prefix only) | 50ms | 20ms | Boot-time lowercase title cache used |
| `/search` (fuzzy fallback) | 350ms | 250ms | rapidfuzz score_cutoff=70 |
| `/films/{id}` (TMDb cache miss) | 600ms | 600ms | 2 TMDb calls dominate |
| `/films/{id}` (TMDb cache hit) | 5ms | 5ms | Disk read |
| `/films/{id}/similar` (ae_z32) | ~33ms + TMDb | ~10ms + TMDb | mmap first-touch warmup |
| `/films/{id}/similar` (ae_z64) | ~67ms + TMDb | ~10ms + TMDb | |
| `/films/{id}/similar` (ae_z128) | ~123ms + TMDb | ~10ms + TMDb | |
| `/films/{id}/cosine-dist` | shares cosines with `/similar` | <10ms additional | LRU helper |
| `/clusters` | <5ms | <5ms | All in-RAM |
| `/clusters/{k}` (TMDb misses) | 600-1200ms | 100ms | TMDb-bound; first 5 films enriched |
| `/gallery` | 3-5ms | 3-5ms | Static JSON file read |

**Mitigation for cold latency**: at boot, run a single forward
matmul on each backbone with a dummy query to prewarm the page cache.
ae_z32 gets prewarmed unconditionally (it's the default backbone). The
other two are prewarmed lazily on first request, with a one-time
~70-130ms hit.

## 7. Frontend changes

### 7.1 Pages

| Path | State | Notes |
|---|---|---|
| `app/page.tsx` | MODIFY | Mock → real fetch via TanStack Query. URL state: `?film={id}&backbone={id}` via `useSearchParams` + `useRouter().push(..., { scroll: false })`. Top of content area has segmented `backbone-switcher` (not in sidebar — see §7.2). Selected panel embeds cosine-heatmap. The existing `Watchlist` button (out-of-spec dead UI) is REMOVED from `selected-film-panel`. |
| `app/cluster/page.tsx` | NEW | 21 cluster card grid (3 cols × 7 rows responsive), each card shows name + size + top-3 genres + 4-poster mosaic. Reads `?backbone=` from URL; preserves it in the link to `/cluster/[k]`. |
| `app/cluster/[k]/page.tsx` | NEW | Cluster detail: hero stats + top-50 films grid (popularity sort). Reads `?backbone=`. Renders pagination chip ("showing 50 of 15,720") from `total`. |
| `app/gallery/page.tsx` | NEW | **Rendered as a React Server Component** (no client-side fetch). Reads `artifacts/inference/gallery.json` at request time on the server, renders the 5×3 matrix HTML directly. On desktop: 5 row × 3 col grid; on mobile (`<md`): tabs by query (5 query tabs, each shows 3 backbone columns vertically). Bottom: narrative blurb linking to `/about`. |
| `app/about/page.tsx` | NEW | In-app methodology blurb (~300 words). Sections: "What CineEmbed is" + "The two methodological findings" (NMI ≠ retrieval, z-sweep U-curve) + "How to read the gallery" + cite journal docs. Static page, no API calls. |
| `app/providers.tsx` | NEW | Wraps children in `QueryClientProvider` (TanStack Query) + Zustand backbone-store provider. |
| `app/layout.tsx` | MODIFY | Wrap content in `<Providers>`. |
| `app/api/health/route.ts` (optional bridge) | NEW | Optional. Proxies `/api/health` from FastAPI to allow Next.js to surface offline-banner without CORS preflight. Skip if direct fetch works. |

**URL state hygiene:**

- `backbone` defaults to `ae_z32` when absent.
- `film` is validated as a positive integer; invalid → ignored.
- Replacing state uses `router.replace` (not push) for query-only
  changes to avoid history pollution on every search keystroke.
- Cluster pages preserve `?backbone=` on cross-page navigation.
- The `/gallery` page does NOT read `?backbone=` (it shows all 3 at
  once). The `/about` page reads no params.

### 7.2 Components

**MODIFY (existing 7):**

- `search-bar.tsx` — hit `/api/films/search?limit=10`, 300ms debounce, render real `Film[]`. Proper a11y: `role="combobox"` on input, `role="listbox"` on dropdown, arrow-key navigation, `Escape` clears + closes, focus management on selection. Use shadcn `<Command>` primitive which handles combobox semantics out of the box.
- `selected-film-panel.tsx` — render Film fields (genres, country/place chips, time, director, rating, votes, language, duration, cluster badge, overview, tagline, posterUrl/posterColor fallback). Render style/plot chips ONLY when STYLISTIC_DICT yielded ≥1 hit; otherwise single "Keywords" chip list (per §5.5 fallback). Embed `<CosineHeatmap filmId={...} backbone={...}/>` lazy-imported via `next/dynamic`. **Remove the `Watchlist` button** (out-of-scope dead UI, was in mock).
- `similar-films-panel.tsx` — fetch `/api/films/{id}/similar?limit=10`, render `Neighbor[]` with cosine score badge per row, color-scaled (green ≥0.95, blue 0.8-0.95, gray <0.8). On click, update URL `?film={n.id}`.
- `film-poster.tsx` — accept `Film`, `src={film.posterUrl ?? hslGradient(film.posterColor)}`. Use Next.js `<Image />` with `image.tmdb.org` in `images.remotePatterns`. Backend already constructs the full URL; frontend just renders.
- `sidebar.tsx` — nav links (Home / Clusters / Gallery / **About** — new). NO backbone switcher here; switcher is in the page-content top bar (see backbone-switcher entry below).
- `empty-state.tsx` — copy updated: "Search 329,044 films from the multimodal embedding space."
- `theme-provider.tsx` — unchanged.

**NEW (4):**

- `backbone-switcher.tsx` — **segmented control rendered at the top of
  the main content area** (not in the sidebar — Codex/§4.2 amendment).
  Three options pulled from `/api/backbones`. Tooltip per option:
  `"z=32 · genre@5=0.723 · gNMI=0.334"`. onChange: (a) write
  `?backbone={id}` via `router.replace`, (b) invalidate
  `queryClient.invalidateQueries(['film'], ['similar'], ['cosineDist'],
  ['clusters'])`. Tab-key and arrow-key navigable.
- `cosine-heatmap.tsx` — **lazy-loaded via `next/dynamic`** to keep
  recharts out of initial bundle. Fetches `/api/films/{id}/cosine-dist`.
  Renders recharts BarChart (30 bins) + secondary mini-bar row for
  top-10 cosines + stats badges (μ, σ, p95). Height ~180px. ARIA
  live-region "summary" text for screen readers (e.g., "Cosine
  distribution over 329,043 films. Mean 0.30, std 0.30, top neighbor
  0.99.").
- `cluster-card.tsx` — used by `app/cluster/page.tsx`. Display: name,
  size with comma-grouping, top-3 genre chips with %, 4-poster
  mosaic. Loading skeleton during fetch.
- `lib/api.ts` — typed fetch wrapper with zod schemas. Methods:
  `getFilms`, `getFilm`, `getSimilar`, `getCosineDist`, `getClusters`,
  `getCluster`, `getGallery` (read-only static JSON via Next.js
  server-component, not part of client API surface), `getBackbones`.
  Base URL from `NEXT_PUBLIC_API_BASE` (server-only env var if
  fetched from server components). Error handler maps fetch failures
  + 5xx to a typed `ApiError` consumed by the offline-banner. Per-call
  abort signal exposed for TanStack Query.

### 7.3 State management

**Server state:** TanStack Query. **Page-scoped UI state:** URL search
params + small Zustand store for transient UI flags (panel open/close,
search dropdown focus).

URL state pattern:

```typescript
// Read URL state in any page or component
import { useSearchParams, useRouter } from "next/navigation";

const searchParams = useSearchParams();
const router = useRouter();
const film = searchParams.get("film") ? Number(searchParams.get("film")) : null;
const backbone = (searchParams.get("backbone") ?? "ae_z32") as BackboneId;

// Write — replace (no history push) for transient changes
const setBackbone = (next: BackboneId) => {
  const params = new URLSearchParams(searchParams.toString());
  params.set("backbone", next);
  router.replace(`?${params.toString()}`, { scroll: false });
  queryClient.invalidateQueries({ queryKey: ["film"] });
  queryClient.invalidateQueries({ queryKey: ["similar"] });
  queryClient.invalidateQueries({ queryKey: ["cosineDist"] });
  queryClient.invalidateQueries({ queryKey: ["clusters"] });
};
```

TanStack Query usage:

```typescript
const { data: film, isLoading, error } = useQuery({
  queryKey: ["film", id, backbone],
  queryFn: ({ signal }) => api.getFilm(id, backbone, { signal }),
  staleTime: 5 * 60_000,        // 5 min
  retry: 1,
  enabled: id !== null,
});
```

**Backbone scope invalidation**: on backbone change, every query key
shaped `[name, ..., backbone]` is invalidated and refetches. The
gallery query is NOT invalidated on backbone change (it shows all 3
backbones at once).

**No optimistic updates** in this scope — single-user demo, all data
flows server-to-client. Frontend only renders.

### 7.4 UX details

- **Skeleton loaders** on film panel + similar panel + cosine-heatmap +
  cluster grid + cluster detail (shadcn `<Skeleton>`).
- **Backbone-switch transition**: 200ms opacity fade on the changing
  panels → spinner overlay → re-render with new data. Tooltip on each
  segment shows the backbone's gNMI + genre@5 metrics.
- **Cosine score color scale**: green ≥0.95 / blue 0.8-0.95 / gray <0.8.
  ARIA label includes the numeric value for screen readers.
- **Poster fallback**: gradient card with title centered, background
  `hsl(...)` from `posterColor`. Used whenever `posterUrl === null`.
- **API-offline banner**: detect via fetch error catch in `lib/api.ts`
  AND via `/api/health` polled every 30s (TanStack Query
  `refetchInterval`). Banner copy: "Backend not reachable. Run
  `bash scripts/dev-up.sh` from the project root." Distinct copy for
  network/CORS error vs `503`.
- **Keyless-mode banner**: when any successful API response carries
  `tmdbStatus: "missing"` AND `/api/health` returns
  `{tmdb_key_configured: false}`, show a one-time dismissable banner:
  "TMDb posters disabled (no API key configured). The app uses gradient
  fallbacks. Set `TMDB_API_KEY` in `.env` to enable posters." Prevents
  the missing posters from looking broken during a key-less demo.
- **Accessibility (minimum bar)**:
  - Combobox semantics on search bar (shadcn `<Command>`).
  - Keyboard nav: Tab through interactive elements, Enter to select,
    Escape to dismiss popovers, arrow keys in lists/segments.
  - Visible focus rings on all focusable elements.
  - Color contrast ≥ 4.5:1 for text on backgrounds (verify shadcn
    default theme passes).
  - `<CosineHeatmap>` includes a hidden `<p>` summary for screen
    readers (e.g., "Cosine distribution over 329,043 films. Mean 0.30,
    standard deviation 0.30, top neighbor 0.99.").
  - All decorative icons have `aria-hidden="true"`; semantic icons get
    `aria-label`.
- **Mobile responsive**:
  - Desktop ≥ `md` breakpoint: full layout per mockups.
  - Mobile (`<md`): sidebar collapses to top-bar hamburger; selected +
    similar panels stack vertically; gallery 5×3 matrix collapses to
    tabs by query (5 query tabs each showing 3-backbone vertical
    stack); cluster card grid wraps to single column.
  - The eyeball gallery comparison is genuinely hard to do on mobile —
    presentation note in `app/about/page.tsx` advises desktop for the
    full experience.
- **Animation polish**: shadcn defaults (fade-in 150ms, slide 200ms).
  Avoid CSS bouncing animations during data load — distracting.
- **Empty states**: `/search` with no query → centered "Search 329,044
  films from the multimodal embedding space."; `/cluster/[k]` with
  empty cluster → "This cluster is empty — pick another."; `/similar`
  with self-only neighbors → unreachable (always 9 returned).

### 7.5 Build config

- `next.config.mjs`: `images.remotePatterns` add `image.tmdb.org`.
- `package.json`: add `@tanstack/react-query`. Existing deps already cover
  `zod`, `recharts`, `lucide-react`.
- `.env.local`: `NEXT_PUBLIC_API_BASE=http://localhost:8000`.

## 8. Implementation sequencing

### 8.1 Day 1 — Backend + monorepo + minimal frontend rewire (~10h)

**Wave 1 · Repo + build (~2h):**
- 1.1 **DONE 2026-05-18 PM** — `git subtree add --prefix=frontend
  frontend-remote/frontend-ui --squash` was already executed as a prep
  step (commits `7c08a8e` + `194b82a`). Periodic updates via
  `git frontend-pull` alias. Implementation skips this step;
  future runs idempotent — if `frontend/` exists, just verify and
  proceed.
- 1.2 `scripts/enrich_films.py`: `production_countries` parse with
  string-or-JSON branch → ISO alpha-2 → `films_master.parquet`.
- 1.3 `scripts/build_index.py` extend: KMeans labels + cluster_meta
  JSON (with Unknown-excluded top-genres + year-null-aware modal
  decade) for all 3 backbones. **Skip re-encode if embeddings.npy is
  current**; only run KMeans.
- 1.4 `src/cineembed/keywords.py`: STYLISTIC_DICT (~50 entries).
- 1.5 `scripts/build_backbones_metadata.py`: write `artifacts/backbones.json`.

**Wave 2 · FastAPI core (~4h):**
- 2.1 `src/cineembed/api.py` scaffold: lifespan loader (3 backbones + master parquet), CORS middleware
- 2.2 `/api/health`, `/api/backbones`
- 2.3 `/api/films/search` (rapidfuzz + popularity sort)
- 2.4 `/api/films/{id}` (lazy TMDb 2-call, disk cache `src/cineembed/tmdb.py`, style/plot split)
- 2.5 `/api/films/{id}/similar` (numpy matmul, top-N, TMDb top-5 parallel)

**Wave 3 · Frontend rewire (~3h):**
- 3.1 `scripts/dev-up.sh` (uvicorn + pnpm dev parallel)
- 3.2 `frontend/lib/api.ts` typed wrapper + zod schemas
- 3.3 `app/providers.tsx` + TanStack Query setup
- 3.4 rewire search-bar, selected-film-panel, similar-films-panel
- 3.5 film-poster: TMDb URL + HSL fallback + Next Image config

**Day 1 gate:** "Minimal demo" shipable. Search → click → details + similar with TMDb posters. ae_z32 only. Commit + push.

### 8.2 Day 2 — Extras (~9h)

**Wave 4 · Backbone switcher (~1.5h):**
- 4.1 `backbone-switcher.tsx`
- 4.2 Lift selectedBackbone state, propagate
- 4.3 TanStack query invalidation on change

**Wave 5 · Cluster browser (~3h):**
- 5.1 `/api/clusters`, `/api/clusters/{k}` endpoints
- 5.2 `app/cluster/page.tsx` (21-card grid)
- 5.3 `app/cluster/[k]/page.tsx` (detail)
- 5.4 `cluster-card.tsx`

**Wave 6 · Gallery (~2h):**
- 6.1 `scripts/build_gallery.py` (5 query × 3 backbone × top-5 precompute)
- 6.2 `/api/gallery` endpoint
- 6.3 `app/gallery/page.tsx`

**Wave 7 · Cosine heatmap (~2.5h):**
- 7.1 `/api/films/{id}/cosine-dist` endpoint
- 7.2 `cosine-heatmap.tsx` (recharts BarChart + stats)
- 7.3 Embed in selected-film-panel

**Day 2 gate:** "Full demo" shipable. T3 tier. Commit + push.

### 8.3 Graceful degradation tiers (revised — T0 is the demo-day floor)

| Tier | What ships | Mandatory by | Cut if cut | Hours cumulative |
|---|---|---|---|---:|
| **T0 Minimal** | search + detail + similar (ae_z32 only) + URL state + about page + smoke script + dev-up.sh | **HARD GATE: end of Day 1** | switcher, clusters, gallery, heatmap | ~11h |
| **T1 +Switcher** | T0 + live backbone switching (top segmented control + URL `?backbone=`) | Day 2 morning | clusters, gallery (precomputed JSON still pre-built but no UI), heatmap | ~13h |
| **T2 +Gallery UI** | T1 + `/gallery` server-component page | Day 2 noon | clusters, heatmap | ~15h |
| **T3 Full** | T2 + cluster browser + cosine heatmap | Day 2 EOD (aspirational) | — | ~20h |

**Hard gate at end of Day 1**: T0 MUST be committed, pushed, and demo-able
before any T1+ work begins. If T0 slips into Day 2, T2/T3 are cut without
discussion. If T0 ships Day 1, the team can chase T1/T2/T3 opportunistically
without putting the demo at risk.

This is a revision of Codex's "drop T3, target T1+gallery" — we keep T3
as an aspiration but enforce hard-gate discipline. Best of both: the
demo is rollback-safe at T0, ambition layers on top without risking the
deadline.

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| TMDb API key delayed | Med | Code works without it — TMDb-lazy fields null OK, frontend degrades. Demo runs even key-less. |
| Teammates frontend-ui parallel edits → merge conflict | Med | `git subtree pull` periodic; PR-based coordination; squash protocol contains drift. |
| rapidfuzz search latency >500ms | Low-Med | Pre-compute lowercase title cache at boot; score_cutoff=80; first-char bucketing fallback. |
| 3-backbone load >30s boot | Low | numpy mmap=True; pyarrow lazy parquet. One-time cost, demo boots once. |
| TanStack Query learning curve | Low | Fallback raw useEffect (+200 LOC but known territory). |
| Cluster auto-naming collisions | Med | Disambiguator suffix `(k=N)`; manual override JSON for outliers. |
| CORS misconfig on grader machine | Low | Allow-origin set = `{http://localhost:3000, http://127.0.0.1:3000}`; override via env `CORS_ORIGINS` for grader machine. NEVER `*` even in dev. |

## 10. Open questions (resolved 2026-05-18 PM)

1. **TMDb API key:** user is registering and will provide. Code paths
   are key-optional (graceful degrade with keyless-mode banner per §7.4).
   Required for T2 gallery posters; T0 functional without it.
2. **Frontend repo update cadence:** subtree merge **complete** (commits
   `7c08a8e` + `194b82a`). Future updates pulled with the configured
   `git frontend-pull` alias. Cadence: on-demand if teammates push to
   their `frontend-ui` branch.
3. **Deploy target:** local-only per scope cut (§2).
4. **Cluster naming:** auto-heuristic with Unknown / null handling (per
   §5.3). Manual sanity-review required for all 21 names before demo;
   overrides go into `cluster_names_override.json`.
5. **Eyeball gallery queries:** fixed 5 (Inception 27205, Spirited Away
   129, Shawshank 278, Pulp Fiction 680, Toy Story 862) per journal/12.
   Plus 3 backup queries listed in `docs/demo-script.md` in case
   primary queries surface oddities during practice run.

## 11. Acceptance criteria (revised)

The integration is "done" when:

1. `bash scripts/dev-up.sh` brings up FastAPI health endpoint in <10s
   and the Next.js dev server in <40s (first compile of Next.js 16 +
   shadcn tree can be 20-30s; that's the realistic budget). The script
   itself prints a "ready when both report green" line.
2. `scripts/demo-smoke.sh` (new) hits every endpoint and validates JSON
   shape: `/health`, `/backbones`, `/search?q=inception`, `/films/27205`,
   `/films/27205/similar`, `/films/27205/cosine-dist`, `/clusters`,
   `/clusters/0`, `/gallery`. Exits 0 on success. Run before each commit.
3. Search "inception" returns Inception (2010, TMDb id 27205) at top
   within 500ms.
4. Click film → details panel populated within 1s; TMDb posters visible
   within 2s (cache hit) or 4s (cache miss with 2 TMDb round-trips).
   Similar panel populated within 2s.
5. Backbone switcher: change to ae_z128 → similar panel re-renders with
   different neighbors within 1.5s (allows for cold mmap first hit).
   URL updates to `?backbone=ae_z128` and reload preserves state.
6. **URL share test**: copy URL with `?film=27205&backbone=ae_z32`, open
   in new tab, the same film+backbone view loads.
7. "Browse Clusters" → 21 named clusters visible; click one → top-50
   films grid renders. Names show no `Unknown` and no obvious gibberish.
8. "Gallery" page renders the 5×3 matrix instantly (server-component
   first-paint; no client-side fetch waterfall).
9. Cosine heatmap visible in selected-film panel: histogram + top-10
   bars + stats badges. Cosine values match `/api/films/{id}/similar`
   top-10 (data consistency).
10. **TMDb offline scenario**: stop network → app still functional;
    posters replaced by HSL gradient cards; keyless-mode banner visible.
    No console errors.
11. **About page** (`/about`) explains the project + the two
    methodological findings in plain English, with links to journal/07
    and journal/12.
12. `README.md` updated with 5-line run instruction (clone, install Python
    deps, install frontend deps, run `dev-up.sh`, open browser). `docs/demo-script.md`
    has the presentation flow with known-good query list.
13. **Keyboard-only acceptance**: tab through home page from search bar
    to selected film to similar film to backbone switcher — all
    actions reachable, focus visible.

## 12. Dependencies & references

**Python deps to add to `[demo]` extras:**

- `rapidfuzz>=3.0` — fuzzy search C-backed
- `httpx>=0.27` — async TMDb client (uses HTTP/2)
- `pycountry>=23.12` — country name → ISO-3166 alpha-2 for enrichment
- `aiolimiter>=1.1` — global async token bucket for TMDb rate limiting
- (existing: `pyarrow`, `fastapi`, `uvicorn[standard]`, `pydantic`)

**Frontend deps to add:**

- `@tanstack/react-query` — server state management (cross-endpoint
  cache invalidation on backbone switch is the deciding factor — SWR
  would not handle this as cleanly)
- `zustand` — small client-side store for UI-only state (panel
  open/close, search dropdown focus). NOT for server state.

**Referenced artifacts:**

- `artifacts/inference/ae_z{32,64,128}/embeddings.npy` (existing)
- `artifacts/inference/ae_z{32,64,128}/films.parquet` (existing per-dir copies — the build will replace these with a single shared `artifacts/inference/films_master.parquet`)
- `artifacts/movies_eda_final.csv` (source for country enrichment)
- `artifacts/models/ae_z{32,64,128}/ae.pt` (existing, not loaded at runtime)

**Referenced documentation:**

- `docs/journal/07-retrieval-vs-nmi-discovery.md` — first methodological finding
- `docs/journal/12-z-sweep-ae-z32-discovery.md` — second methodological finding, gallery source
- `docs/journal/10-results-table.md` — backbone metrics for switcher tooltips
- `docs/adr/0001-modeling-hybrid-architecture.md` D14 (web demo pivot), D15 (ae_z32 lock)
- `docs/superpowers/specs/2026-05-16-web-app-demo-design.md` — superseded by this spec for the runtime architecture; backbone selection still locked there

## 13. Out-of-spec (future work)

**Note**: the following items were moved INTO scope post-Codex review and
are no longer deferred: `app/about/page.tsx`, `scripts/demo-smoke.sh`,
`docs/demo-script.md`. They are demo-blocking enough that deferral
would risk grading. See §11 acceptance criteria.

Remaining out-of-scope:

- Cloud deployment (Vercel + Render / Fly.io). Deferred.
- TMDb full pre-bake (all 329k films). Too expensive; lazy + LRU is enough.
- Trailer playback (`/videos` endpoint). Out of demo scope.
- z=16 stretch ablation as a 4th backbone in the switcher. Optional bonus.
- Rate limiting (slowapi). Demo scope = single user.
- Telemetry / Sentry / structured logging. Demo scope.
- Multi-tenant / authentication. Demo scope.
- Database persistence (Postgres / SQLite). Demo scope.
- SSR for everything (only `/gallery` and `/about` use server
  components; client pages stay client-fetched for now).

---

## 14. Implementation invariants (added post-Codex review)

Non-obvious correctness constraints. MUST be honored across every
backend endpoint and frontend hook.

### 14.1 ID ↔ row index mapping

`embeddings.npy` is indexed by **row position**, not by film id.
`films_master.parquet` carries `id` as a column. Build the lookup once
at boot:

```python
films = pd.read_parquet("artifacts/inference/films_master.parquet")
id_to_row = {int(row.id): i for i, row in enumerate(films.itertuples(index=False))}
row_to_id = films["id"].tolist()
```

Usage:

- `/films/{id}` → `row = id_to_row[id]` → `films.iloc[row]`.
- `/similar` → `q_emb = embeddings[id_to_row[id]]` → matmul →
  `argpartition` returns row indices → map back via `row_to_id`.

**Never** rely on the DataFrame's default integer index as row
position; it becomes meaningless after any sort/filter. Use the
`id_to_row` map and `films.iloc[row]` exclusively.

### 14.2 Boot sequence

```
[boot] cineembed.api startup lifespan:
  t0: load artifacts/backbones.json                      ~1ms
  t1: load artifacts/inference/films_master.parquet      ~3-5s (dominant)
  t2: build id_to_row map                                ~200ms
  t3: per backbone in ["ae_z32", "ae_z64", "ae_z128"]:
        load embeddings.npy with mmap_mode='r'           instant
        load cluster_labels.npy                          ~10ms
        load cluster_meta.json (with overrides merged)   ~5ms
  t4: PREWARM ae_z32 — one matmul with embeddings[0]     ~33ms
  t5: build lowercase title cache for /search            ~500ms
  t6: ready
  total: ~5-8 seconds boot to readiness
```

ae_z64 and ae_z128 are NOT prewarmed (saves ~200ms boot). First
request to either pays the one-time cold-mmap cost (~70-130ms).

### 14.3 Concurrency model

FastAPI is async by default. CPU-bound numpy work MUST NOT block the
event loop:

- **Synchronous `def` route handlers** for CPU-bound endpoints
  (`/similar`, `/cosine-dist`, `/clusters`, `/clusters/{k}`). FastAPI
  auto-wraps sync handlers in a threadpool.
- **`async def` route handlers** for I/O-bound endpoints
  (`/films/{id}` when TMDb is touched).
- **Cap BLAS threads** via env vars in `scripts/dev-up.sh`:
  `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
  Prevents numpy from oversubscribing the CPU per matmul.

### 14.4 Cosine compute helper with LRU cache

```python
from functools import lru_cache

@lru_cache(maxsize=50)
def _compute_cosines_cached(film_id: int, backbone: str) -> np.ndarray:
    row = id_to_row[film_id]
    q = embeddings[backbone][row]                # (z,)
    return embeddings[backbone] @ q              # (329044,)
```

Called by both `/similar` and `/cosine-dist`. A typical home-page
click triggers both within ~50ms — the second call is a cache hit
(~free). 50-entry cache ≈ 65 MB RAM, bounded.

### 14.5 TMDb client

`src/cineembed/tmdb.py` exports a single async `TMDbClient` used by
all endpoints:

```python
class TMDbClient:
    def __init__(self, api_key: str | None, cache_dir: Path):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.limiter = AsyncLimiter(35, 10)          # 35 req / 10 s
        self.client = httpx.AsyncClient(http2=True, timeout=10.0)
        self._inflight: dict[int, asyncio.Future] = {}

    async def get_film(self, id: int) -> TmdbBlob | None:
        # 1. If id in _inflight, return shared Future.
        # 2. Else disk cache lookup (with TTL 30 days).
        # 3. Else acquire semaphore + limiter.
        # 4. httpx.get /movie/{id} and /movie/{id}/keywords concurrently.
        # 5. Atomic write to cache: tmp + rename.
        # 6. On 429: exponential backoff + retry once. On 4xx other:
        #    persist negative marker for 1 day, return None.
```

- **Atomic disk writes**: `tmp_path.rename(final_path)` is atomic on POSIX.
- **Max cache size**: ~50k entries (~50k × 5KB = 250MB). LRU evict by mtime.
- **TTL**: 30 days. Stale entries refetched lazily.
- **API key optional**: absent → all calls return `None` immediately
  (no network attempt). `/health` exposes `tmdb_key_configured: bool`.
- **Per-id de-dup**: prevents two concurrent client requests for the
  same film from triggering two TMDb calls.

### 14.6 Param validators

Every input gets a constraint via Pydantic `Query(...)`:

| Param | Constraint |
|---|---|
| `q` | `min_length=1`, `max_length=200` |
| `limit` for `/clusters/{k}` | `ge=1`, `le=100` |
| `limit` for `/search`, `/similar` | `ge=1`, `le=50` |
| `bins` | `ge=5`, `le=100` |
| `k` (cluster id, path) | `ge=0`, `le=20`, `conint` |
| `id` (film id, path) | `ge=1`, `conint` |
| `backbone` | `Literal["ae_z32", "ae_z64", "ae_z128"]` |

All caps server-side regardless of frontend behavior.

### 14.7 Pydantic models (camelCase JSON wire shape)

```python
# src/cineembed/api_models.py
class Film(BaseModel):
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

    model_config = ConfigDict(populate_by_name=True)


class Neighbor(Film):
    cosine: float


class Cluster(BaseModel):
    id: int
    name: str
    size: int
    top_genres: list[dict] = Field(alias="topGenres")
    modal_decade: str = Field(alias="modalDecade")
    preview_films: list[Film] = Field(alias="previewFilms")


class ClusterDetail(Cluster):
    films: list[Film]
    total: int


class Backbone(BaseModel):
    id: Literal["ae_z32", "ae_z64", "ae_z128"]
    z: int
    label: str
    genre_at_five: float = Field(alias="genreAtFive")
    gnmi: float
    preferred: bool
```

Aliases convert snake_case Python → camelCase JSON. Frontend zod
schemas mirror the camelCase shape.

---

## 15. Demo script and known-good queries

A new file `docs/demo-script.md` carries the SENG 474 presentation flow:

1. **Opening (15 sec)** — open `localhost:3000`, point to title and
   the brief intro on the home empty state.
2. **Search demo (30 sec)** — type "inception" → emphasize live
   filter over 329k films. Click top result.
3. **Selected film panel (45 sec)** — walk metadata (genres, year,
   cluster badge), TMDb poster + overview + style/plot keyword chips,
   scroll to cosine heatmap ("distribution of this film's similarity
   to all 329k others — the right tail is the recommendation").
4. **Similar panel (30 sec)** — top-5 with color-coded cosines;
   click a similar film, point out URL state navigation.
5. **Backbone switch (60 sec) — the methodological story** — switch
   to ae_z128, similar panel re-renders with different neighbors.
   "ae_z32 (smaller model) gives stronger director-aware groupings;
   z-sweep U-curve, journal/12."
6. **Cluster browser (45 sec)** — `/cluster` → 21 auto-named clusters
   → pick one → top-50 films grid.
7. **Gallery (45 sec)** — `/gallery` → 5 queries × 3 backbones
   matrix → narrate the Inception → Dark Knight @ cos=0.991 on z=32
   vs Marvel mix on z=128.
8. **About page (30 sec)** — brief stop at `/about`.
9. **Q&A buffer**.

**Known-good query list** (verified during prep):

| Primary | Secondary backups |
|---|---|
| Inception (27205) | The Dark Knight (155), Interstellar (157336) |
| Spirited Away (129) | My Neighbor Totoro (8392), Princess Mononoke (128) |
| Shawshank Redemption (278) | The Green Mile (497) |
| Pulp Fiction (680) | Reservoir Dogs (500), Kill Bill Vol. 1 (24) |
| Toy Story (862) | WALL·E (10681), Finding Nemo (12) |

Substitute from the same row's backup list if a primary query
surfaces unexpected results during a dry run.
