# Web App Demo — Design Spec

**Date:** 2026-05-16
**Status:** APPROVED
**Deadline:** 2026-05-20
**Cross-ref:** ADR `0001-modeling-hybrid-architecture.md` D14;
`docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`;
`docs/journal/12-z-sweep-ae-z32-discovery.md`

## Amendment — 2026-05-17 (PM) — Demo backbone swapped to `ae_z32`

After completion of the Round 2 z-sweep (z ∈ {32, 64, 128}), the demo
backbone is **swapped from `ae_z64` to `ae_z32`**. New deployment artifact
path: `artifacts/inference/ae_z32/{embeddings.npy, films.parquet, manifest.json}`.
Checkpoint: `artifacts/models/ae_z32/ae.pt`.

**Selection criterion unchanged:** `genre@5` retrieval quality, with Occam
tiebreaker on latent size when within ±0.01.

| Backbone | gNMI | `genre@5` mean | `pair_cos_std` | `dim_std_min` | Verdict |
|---|---:|---:|---:|---:|---|
| **`ae_z32`** | **0.334** | **0.723** | 0.301 | 0.117 | **SELECTED** — best gNMI, cleanest dim utilisation |
| `ae_z64` (previous) | 0.328 | 0.715 | 0.299 | 0.062 | superseded |
| `ae_z128` | 0.273 | 0.722 | 0.289 | **0.025** | over-parameterised; near-dead dim; gNMI collapse |

Three-way Round 2 result revealed a U-curve in z: the genre clustering
metric `gNMI` peaks at z=32 (0.334), drops slightly at z=64 (0.328), and
collapses 6 points at z=128 (0.273). `ae_z128`'s `genre@5` (0.722)
matches z=32 within noise — but `dim_std_min = 0.025` shows a near-dead
latent dimension, and the pair-cosine spread is narrowing (early-warning
of angular-collapse like `dec_z64_k21`). z=32 is the information-bottleneck
sweet spot for this task.

**Implications for the demo:**

- **Inference RAM** drops ~2× on the embedding matrix: 329 044 × 32 ×
  float32 ≈ **42 MB** instead of ≈ 80 MB for z=64.
- **Cosine search step ~2× faster** (32-dim matmul vs 64-dim) — affects
  `/similar` endpoint p95 latency, but already <50 ms target either way.
- **API contract unchanged.** The latent dimension is internal — clients
  see only the `[{id, title, score}]` JSON shape.
- **Teammates (backend / frontend):** new path under `artifacts/inference/ae_z32/`
  is the deployment target. Embedding `.npy` dtype + shape unchanged
  except for the trailing dim (32 vs 64). `films.parquet` schema identical.
  Manifest schema identical (`embedding_dim` field reflects new value).

**Open WANDB key incident (`commits/17d6fbb`)** is unrelated to this swap
and remains unrevoked per user decision.

See `docs/journal/12-z-sweep-ae-z32-discovery.md` §9 for the full sweet-
spot analysis. The earlier 2026-05-17 (AM) amendment below (DEC → AE
pivot) still applies — this swap is the second-stage refinement within
the AE family.

---

## Amendment — 2026-05-17 (AM) — DEC → AE pivot (superseded by the PM swap above)

The demo backbone is **`artifacts/models/ae_z64.pt`** (multi-modal AE at z=64,
KMeans-evaluated geo_NMI=0.309). This is **not** the highest-NMI model: the
NMI champion `dec_z64_k21` (geo_NMI=0.323) was disqualified on the basis of a
retrieval-task evaluation introduced after Round 1.

**Selection criterion: `genre@5` retrieval quality, not clustering NMI.**

| Backbone | geo_NMI | `genre@5` mean | `genre@5` median | Notes |
|---|---:|---:|---:|---|
| `dec_z64_k21` | **0.323** | 0.557 | 0.600 | Angular collapse — all in-cluster pairs cos≈1.000 |
| **`ae_z64`** | 0.309 | **0.714** | **0.800** | Smooth manifold; coherent eyeball top-5 |

Root cause: DEC's clustering objective pulls intra-cluster vectors to the same
direction (cluster centroid). Cosine ranking inside a cluster degenerates to a
random tie-break. AE's reconstruction objective preserves a smooth latent
manifold whose angular gradient is what cosine top-k actually needs.

**Eyeball confirmation** (10 well-known queries, `--eyeball` mode of
`scripts/build_index.py`): AE gave a coherent Inception → Interstellar/Dunkirk
Nolan grouping, Toy Story → Pixar+WALL·E animation grouping, Spirited Away →
Princess Mononoke/Kiki Studio Ghibli grouping. DEC returned tied-at-1.000
randomly-ordered same-cluster films for every query.

**Implication for Round 2:** the z-sweep now runs in the AE family (`ae_z32`,
`ae_z128`), not the DEC family — see `2026-05-16-two-round-modeling-strategy.md`
amendment of the same date.

**Implication for the report:** the AE-vs-DEC retrieval divergence is itself a
finding — "clustering NMI does not predict recommender quality; intra-cluster
angular collapse penalises cosine retrieval even when NMI is highest". This
goes into the report's analysis section.

---

## 1. Motivation

The SENG 474 final deliverable is a **working web app**, not just a report.
A live cosine-similarity recommender over the 64-dim latent embeddings is the
clearest possible empirical demonstration that the multi-modal architecture
produced a meaningful representation — vastly more compelling than a static
NMI table.

The demo is intentionally minimal: one model checkpoint (Round-2 winner),
one numpy file of pre-computed embeddings, three REST endpoints, a single
static HTML page. Everything runs on a laptop with no GPU.

---

## 2. API contract

FastAPI app exposed at `cineembed.api:app`. JSON over HTTP. All endpoints
read from in-RAM artifacts loaded at startup.

### 2.1 `GET /api/films/search`

Query-string search by title prefix / substring.

**Request:**

```
GET /api/films/search?q=blade&limit=10
```

| Param | Type | Default | Notes |
|---|---|---|---|
| `q` | str (required) | — | Case-insensitive substring match on title |
| `limit` | int | 10 | Max results |

**Response (200):**

```json
{
  "query": "blade",
  "results": [
    {"id": 78,    "title": "Blade",          "year": 1998},
    {"id": 36586, "title": "Blade Runner",   "year": 1982},
    {"id": 335984, "title": "Blade Runner 2049", "year": 2017}
  ]
}
```

### 2.2 `GET /api/films/{id}/similar`

Top-N nearest neighbours by cosine over the L2-normalized z=64 latent.

**Request:**

```
GET /api/films/335984/similar?top=5
```

| Param | Type | Default | Notes |
|---|---|---|---|
| `id` | int (path) | — | TMDb id, must exist in `films.parquet` |
| `top` | int (query) | 5 | One of 2 / 5 / 10 (UI tier) |

**Response (200):**

```json
{
  "query_id": 335984,
  "query_title": "Blade Runner 2049",
  "results": [
    {"id": 78, "title": "Blade Runner", "year": 1982, "similarity": 0.913},
    {"id": 1726, "title": "Iron Man", "year": 2008, "similarity": 0.802},
    ...
  ]
}
```

`similarity` is the cosine in [-1, 1]; for L2-normalized embeddings this is
the dot product.

**Errors:**

- `404` if `id` not in `films.parquet`.

### 2.3 `GET /api/films/random`

Random films, used as UI cold-start and a "surprise me" affordance.

**Request:**

```
GET /api/films/random?n=12
```

| Param | Type | Default | Notes |
|---|---|---|---|
| `n` | int | 12 | Max 50 |

**Response (200):**

```json
{
  "results": [
    {"id": 12345, "title": "...", "year": 1993},
    ...
  ]
}
```

---

## 3. Inference architecture

### 3.1 Pre-compute step (`scripts/build_index.py`)

**Contract:** backbone-agnostic. Selects via `--model-type {ae,dec,vae,backbone}`
and rebuilds the matching head shape before extracting the backbone. The script:

1. Loads `artifacts/feature_matrix.npz` (329044, 564).
2. Builds a fresh `MultiModalBackbone` + the matching head (AE/DEC/VAE) with the
   stored hyperparameters (`--latent-dim`, `--hidden-dim`, `--n-clusters`).
3. Loads `state_dict` from the supplied checkpoint into the head, then extracts
   `head.backbone`.
4. Encodes all 329k rows in batches (no gradient, CPU is ≤1s wall-clock).
5. L2-normalizes each row of the resulting (329044, 64) latent.
6. Saves `<out>/embeddings.npy` — float32 (329044, 64) ≈ 80 MB.
7. Joins on `movies_eda_final.csv` to materialize
   id/title/year/director/genres/overview/popularity/vote_*/runtime/lang
   and saves `<out>/films.parquet`.
8. Optionally (`--retrieval-eval`) reports `genre@k` mean / median / std on a
   random query subset and a random-pair-cosine sanity histogram (mean / std /
   range) — both written into `<out>/manifest.json` along with the
   checkpoint SHA-256.
9. Optionally (`--eyeball`) prints top-5 similar films for a curated set of
   well-known query titles, also persisted to the manifest.

```bash
# Demo backbone — chosen by genre@5, not NMI (see amendment)
python scripts/build_index.py \
  --checkpoint artifacts/models/ae_z64.pt \
  --model-type ae \
  --out artifacts/inference/ae_z64/ \
  --retrieval-eval --eyeball
```

**Manifest schema** (`<out>/manifest.json`):

```json
{
  "schema_version": 1,
  "created_unix_seconds": 1778968093,
  "checkpoint": "artifacts/models/ae_z64.pt",
  "checkpoint_sha256_32": "e7326ef3...",
  "model_type": "ae",
  "latent_dim": 64,
  "hidden_dim": 128,
  "n_clusters": null,
  "n_films": 329044,
  "embedding_dim": 64,
  "normalization": "L2",
  "distance_metric": "cosine (dot product after L2 normalization)",
  "retrieval": { "k": 5, "genre_at_k_mean": 0.714, "...": "..." },
  "eyeball":  [ { "query": "Inception", "neighbors": [ ... ] }, ... ]
}
```

### 3.2 Runtime cosine search

The API loads `embeddings.npy` and `films.parquet` once at startup:

```python
E = np.load("artifacts/inference/embeddings.npy")   # (329044, 64), L2-normed
films = pd.read_parquet("artifacts/inference/films.parquet")
```

For `/similar?top=N`:

```python
q = E[idx_of_id]            # (64,)
sims = E @ q                # (329044,)
top_idx = np.argpartition(-sims, top + 1)[:top + 1]  # exclude self
top_idx = top_idx[np.argsort(-sims[top_idx])]
```

Measured locally: 329k × 64 matmul + topk completes in **<10 ms** on a
modern laptop (numpy + BLAS). No FAISS, no annoy, no vector DB — the
problem size genuinely fits in RAM with millisecond latency.

Memory footprint: 329044 × 64 × 4 B = 80.3 MB for embeddings,
~25 MB for the parquet (titles dominate). Total ≈ 110 MB resident.

---

## 4. Frontend tier

Minimal static HTML + vanilla JS. No framework, no build step.

- Single page at `/` served from the `/static/` mount.
- Search box → calls `/api/films/search` on debounced input.
- Selecting a search result calls `/api/films/{id}/similar?top=N`.
- Top-N selector: radio group for 2 / 5 / 10.
- Result cards: title, year, similarity score.
- The `frontend-design` skill is used to keep the visual layer above the
  generic-AI-aesthetic floor — distinctive, production-grade, no template
  smell.

**Posters: DEFERRED — open decision.** Each row in `films.parquet` carries
a TMDb `id`; the natural option is `GET https://api.themoviedb.org/3/movie/{id}`
on-demand (with browser-side caching) once a TMDb API key is in place. Decision
deferred behind the model + REST priority.

---

## 5. Deployment

**Local dev:**

```bash
uvicorn cineembed.api:app --reload
# → http://localhost:8000
```

- Static files served from `cineembed/static/` via FastAPI's
  `StaticFiles` mount at `/static/`.
- `/` redirects to `/static/index.html`.
- `embeddings.npy` + `films.parquet` paths are configurable via
  `CINEEMBED_INFERENCE_DIR` env var (default `artifacts/inference/`).

**No production deployment in scope.** The local-dev command is the
demo experience; if the course evaluation requires a hosted URL, the
~100 MB artifact + tiny stateless API can be deployed to any free PaaS
(Fly, Render, Hugging Face Spaces) — out of scope for this spec.

---

## 6. Open decisions

- Poster source (TMDb on-demand vs none) — deferred to after the API +
  frontend land.
- Search ranking (substring match vs fuzzy / typo-tolerant) — current
  spec is substring, sufficient for the demo.
- Whether to expose the genre / language / decade labels in the response
  payload — yes if frontend cards need them; trivial addition.

---

## 7. Acceptance

- [ ] `scripts/build_index.py` produces `embeddings.npy` (329044, 64) and
      `films.parquet` from the Round-2 winner checkpoint.
- [ ] `cineembed.api:app` starts under `uvicorn` and serves all three
      endpoints with the contract above.
- [ ] `/api/films/{id}/similar` returns in <50 ms p95 on a laptop.
- [ ] Static UI works end-to-end: search → select → see top-N similar.
- [ ] The demo runs without network access (no TMDb call) when posters
      are deferred.
