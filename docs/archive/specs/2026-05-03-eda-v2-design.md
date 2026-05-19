# CineEmbed — `eda_v2.ipynb` Design

**Status:** Approved (2026-05-03)
**Author:** Baran Dinçoğuz (with Claude)
**Course:** SENG 474 — Deep Learning · TED University · Spring 2026
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya

---

## 1. Overview

The current EDA notebook (`Deep_Learning_EDA_ipynb_adlı_not_defterinin_kopyası.ipynb`) is comprehensive but contains 13 issues that, if left in place, would degrade the unsupervised models (AE/VAE/DEC) downstream. The largest issues are: temporal leakage in director-award features, a `vote_average` median imputation that injects an artificial cluster of zeros, a genre missing-handling inconsistency between slides and code, modality-scale imbalance (text 384-dim vs. other 62-dim), and English-only sentence embeddings on a multilingual catalogue.

This document specifies a new notebook, `eda_v2.ipynb`, that:

1. Applies all 13 fixes (critical, high, medium, low — all categories) without altering the original notebook.
2. Adopts a **pipeline-first architecture** — pure functions defined at the top, execution and visualisation below — so that the same code can be reused with a one-hour refactor when the modelling phase begins.
3. Caches the two expensive steps (text embedding, full feature matrix) so that warm-start runs complete in under two minutes.
4. Produces a clean set of artifacts (`feature_matrix.npz`, `text_embeddings.npy`, `movies_eda_final.csv`, scalers, metadata, version JSON) ready to be consumed by the AE/VAE/DEC training notebooks.

The original notebook is preserved untouched so that the figures referenced by the existing project-proposal slides remain valid.

---

## 2. Goals and Non-Goals

### Goals
- Resolve all 13 EDA issues identified during the audit (Section 4).
- Produce a deterministic, cached feature pipeline whose output is stable across runs (byte-identical `feature_matrix.npz`).
- Make the modality blocks contribute comparably to the input variance, eliminating the 6× text-embedding dominance.
- Restore multilingual coverage by switching to `paraphrase-multilingual-MiniLM-L12-v2`.
- Provide artifacts in formats that the AE/VAE/DEC training notebooks can load with two lines of code.

### Non-Goals
- Building the AE/VAE/DEC models themselves (separate spec, follows this one).
- Building a REST API or front-end (explicitly out of scope per user direction).
- Refactoring the original EDA notebook (kept untouched as a reference).
- Modifying the source CSVs.

---

## 3. Architecture

### 3.1 Notebook Skeleton (six top-level sections)

```
§1  Setup & Reproducibility
    • imports, sentence-transformers, umap
    • seed_everything(42) — np, random, torch (CUDA included)
    • CONFIG block — every magic number lives here
      (top_n_genre=20, top_n_lang=30, q99_threshold, scaler_mode, ...)

§2  Pipeline Function Definitions (pure, deterministic)
    §2.1 Data layer       — load + merge + slug-normalize
    §2.2 Awards layer     — temporal-aware aggregation (regex word-boundary)
    §2.3 Feature eng.     — numerical, genre, language, decade, awards
    §2.4 Embedding        — multilingual + cache
    §2.5 Assembly         — modality-aware scaling + matrix build
    §2.6 Sanity helpers   — assert_shape, assert_no_nan, assert_value_range,
                            assert_unit_norm

§3  Pipeline Execution
    • Calls the §2 functions in sequence
    • Block-level assertions after each engineer_* call
    • Final-stage assertions after assemble_feature_matrix
    • Sanity print spot-checks (awards merge rate, has_vote count, modality
      variance ratio, etc.)

§4  EDA Visualizations (slide-aligned)
    §4.1 Data quality        §4.5 Modality balance
    §4.2 Distributions       §4.6 Text embedding (PCA, 2D)
    §4.3 Correlation         §4.7 Clusterability (full PCA)
    §4.4 Categorical         §4.8 Variance thresholding
    • Original 19 figures preserved (slide cross-references intact)
    • +4 new figures: vote_average imputation impact, modality scaling
      before/after, awards temporal cutoff, multilingual coverage

§5  Persistence — Artifacts (see §3.4 below for the file table)

§6  Pipeline Import Bridge for Modeling Phase
    • The §2 functions can be copied into pipeline.py with a one-hour refactor
    • models_*.ipynb then does: from pipeline import build_feature_matrix
```

**Key principle:** every function in §2 is **idempotent and side-effect free**. Same input ⇒ same output. This is what makes caching safe and the unit-level debugging in §3 reliable.

### 3.2 Function Signatures

| Layer | Function | Purpose |
|---|---|---|
| §2.1 | `load_csvs(paths) -> (details, casting, awards)` | Load three CSVs with correct separators and dtypes |
| §2.1 | `normalize_director_name(name) -> str` | NFKD accent strip → lowercase → "Last, First" → "First Last" → whitespace collapse |
| §2.1 | `merge_details_casting(details, casting) -> df` | id-join; carries both `director_name` and `director_name_norm` |
| §2.2 | `aggregate_awards_temporal(awards, films) -> awards_per_film` | Per-film aggregation of director awards limited to `award.year ≤ film.release_year`; uses regex `\bOscar\b`, `\bPalme\b` |
| §2.3 | `engineer_numerical(df, config) -> block_df` | Q99 clip + log for popularity & vote_count; runtime cap [10,300] + min-max; **vote_average imputation: compute `imputed_value = df.loc[vote_count > 0, 'vote_average'].mean()` (a single scalar derived from the data, ~6.0), then fill NaN/missing with that scalar before min-max**; emits `has_vote = (vote_count > 0)`, `has_engagement = (popularity > 0) | (vote_count > 0)` flags |
| §2.3 | `engineer_genres(df, top_n=20) -> block_df` | Empty `genres_list` → `['Unknown']`; multi-hot encoding; emits `has_genre` flag |
| §2.3 | `engineer_languages(df, top_n=30) -> block_df` | Top-N + 'other'; one-hot encoding |
| §2.3 | `engineer_decade(df) -> block_df` | For films with a parseable `release_date`: `decade_norm = (decade − 1900) / 130 ∈ [0, 1]`. **For films without a release_date: `decade_norm = 0.0` AND `has_release_date = 0`** — the model can learn to ignore decade when the flag is off, instead of treating decade=0 as "year 1900" |
| §2.3 | `engineer_director_awards(df) -> block_df` | log1p of prior_* award columns from §2.2 |
| §2.4 | `compute_text_embeddings(overviews, *, model_name, cache_path, batch_size) -> ndarray` | (n, 384) embeddings; cache hit if (model_name, n_films, total_chars) hash matches; multilingual model |
| §2.5 | `apply_modality_aware_scaling(blocks) -> (scaled_blocks, scalers)` | StandardScaler for numerical/decade/awards; no scaling for genre/language one-hot; L2-normalize for text rows |
| §2.5 | `assemble_feature_matrix(blocks) -> (X, feature_names)` | Block order: numerical · genre · language · decade · awards · text. Final shape: (329044, **D**) where `D = 6 + 22 + 31 + 2 + 6 + 384 = 451` (see §3.5 for breakdown) |
| §2.6 | `assert_shape`, `assert_no_nan`, `assert_value_range`, `assert_unit_norm` | Fail-fast helpers used after every block |

### 3.3 Execution Flow (§3 of the notebook)

```python
# 1. Load + merge
details, casting, awards = load_csvs(PATHS)
details, casting = map(normalize_director_in_df, [details, casting])
df = merge_details_casting(details, casting)

# 2. Temporal-aware awards aggregation (fixes #1, #9)
awards_per_film = aggregate_awards_temporal(awards, df)
df = df.merge(awards_per_film, on='id', how='left').fillna({...:0})

# 3. Feature blocks
blocks = {
    'numerical': engineer_numerical(df, CONFIG),
    'genre':     engineer_genres(df, top_n=20),
    'language':  engineer_languages(df, top_n=30),
    'decade':    engineer_decade(df),
    'awards':    engineer_director_awards(df),
}

# 4. Embedding (cached: cold ~30-60 min on T4, warm < 5 s)
blocks['text'] = compute_text_embeddings(
    df['overview'].fillna(''),
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    cache_path=Path('artifacts/text_embeddings.npy'),
)

# 5. Modality-aware scaling (fix #4)
blocks_scaled, scalers = apply_modality_aware_scaling(blocks)

# 6. Assemble + sanity
X, feature_names = assemble_feature_matrix(blocks_scaled)
assert_shape(X, (len(df), 451), 'X')   # 6+22+31+2+6+384 — see §3.5
assert_no_nan(X, 'X')
```

### 3.4 Artifacts and Directory Layout

```
deep learning movie project/
├── Deep_Learning_EDA_ipynb_adlı_not_defterinin_kopyası.ipynb   # untouched
├── eda_v2.ipynb                                                  # this spec
├── SENG474_Presentation_Team3.pptx                               # untouched
├── data/
│   ├── AllMoviesDetailsCleaned.csv
│   ├── AllMoviesCastingRaw.csv
│   └── 220k_awards_by_directors.csv
└── artifacts/
    ├── feature_matrix.npz             # X: (329044, 451) — model input
    ├── feature_matrix_raw.npz         # block-wise dict, pre-scaling — for ablation
    ├── text_embeddings.npy            # (329044, 384) — cache
    ├── text_embeddings.meta.json      # cache invalidation hash
    ├── movies_eda_final.csv           # all encoded columns included — human-readable
    ├── scalers.pkl                    # per-block sklearn scaler instances
    ├── feature_metadata.json          # column names, dtypes, value ranges
    ├── pipeline_version.json          # seed, library versions, MD5 hash
    └── figures/                       # 23 PNGs (original 19 + 4 new)
```

| File | Format | Used by |
|---|---|---|
| `feature_matrix.npz` | `np.savez_compressed(X, feature_names)` | Modelling notebooks (AE/VAE/DEC input) |
| `feature_matrix_raw.npz` | block dict, pre-scaling | Ablation: "scaling impact" comparison |
| `text_embeddings.npy` | `(n, 384)` raw | Optional text-only DEC experiment |
| `movies_eda_final.csv` | `df_master + all_blocks` joined | Cluster-result interpretation (back to titles/genres) |
| `scalers.pkl` | `{'numerical': StandardScaler, ...}` | New-film inference |
| `feature_metadata.json` | `[{name, dtype, block, range}, ...]` | Latent-space interpretability |
| `pipeline_version.json` | seed + versions + MD5 | Reproducibility audit; ensures models load identical X |

### 3.5 Feature Matrix Dimension Breakdown

The original notebook actually computes 446 dimensions (4 + 20 + 31 + 1 + 6 + 384) — the project-proposal slides round up to 447, but the notebook cell-5 output is the source of truth at 446. The fixes in Section 4 add `has_*` flags and an `Unknown` genre category, yielding **451 dimensions** in `eda_v2`:

| Block | Dim | Composition |
|---|---:|---|
| numerical | 6 | `log_popularity`, `log_vote_count`, `runtime_norm`, `vote_average_norm`, `has_vote`, `has_engagement` |
| genre | 22 | top-20 multi-hot + `genre_Unknown` + `has_genre` |
| language | 31 | top-30 + `lang_other` |
| decade | 2 | `decade_norm`, `has_release_date` |
| awards | 6 | `prior_log_total_wins`, `prior_log_total_nominations`, `prior_log_oscar_wins`, `prior_log_oscar_nominations`, `prior_log_palme_wins`, `prior_log_palme_nominations` |
| text | 384 | sentence embedding (multilingual) |
| **Total** | **451** | |

If a future change alters this breakdown, only the assertion in §3.3 step 6 and this table need updating — the rest of the pipeline is dimension-agnostic.

---

## 4. The 13 Fixes — Placement and Verification

| # | Severity | Fix | Where | Verification |
|---|---|---|---|---|
| 1 | 🔴 | Awards temporal leak | §2.2 `aggregate_awards_temporal` | A 1990 film gets 0 from a 2015 award |
| 2 | 🔴 | `vote_average` imputation error | §2.3 `engineer_numerical` (mean of voted-only + `has_vote` flag) | No artificial spike at 0 in `vote_average_norm` histogram |
| 3 | 🔴 | Genre missing inconsistency | §2.3 `engineer_genres` (empty → `['Unknown']` + `has_genre`) | `df['genre_Unknown'].sum() ≈ 121k`; `has_genre=0` matches |
| 4 | 🟠 | Modality scale imbalance | §2.5 `apply_modality_aware_scaling` (text L2, others Standard) | `var(text) / var(others) ∈ [0.5, 2.0]` |
| 5 | 🟠 | Multilingual embedding | §2.4 `compute_text_embeddings` (model swap) | Cross-lingual cosine sim spot-checks ("tutku" ↔ "passion" > 0.6) |
| 6 | 🟠 | Decade scale mismatch | §2.3 `engineer_decade` (`(d−1900)/130 → [0,1]`) | `decade_norm.max() ≤ 1.0` |
| 7 | 🟠 | `vote_count` clip missing | §2.3 `engineer_numerical` (Q99 clip + log) | `log_vote_count` skewness < 2 (was 28) |
| 8 | 🟡 | Director-name string match | §2.1 `normalize_director_name` | The notebook computes both rates side by side: `baseline_rate = match rate using raw director_name` (the original notebook's behaviour) and `normalized_rate = match rate using director_name_norm`. Pass criterion: `normalized_rate − baseline_rate ≥ 5 percentage points` |
| 9 | 🟡 | Oscar/Palme substring false positive | §2.2 (regex `\bOscar\b`) | "Oscar Wilde Award" no longer counted in `oscar_*` |
| 10 | 🟡 | Engagement-zero flag | §2.3 `engineer_numerical` (`has_engagement`) | Flag count matches popularity-or-vote-count > 0 |
| 11 | 🟢 | Reproducibility | §1 `seed_everything(42)` | Two consecutive runs produce identical `feature_matrix_md5` |
| 12 | 🟢 | Embedding cache | §2.4 + §5 persistence | Second run of §2.4 < 5 seconds |
| 13 | 🟢 | `movies_eda_final.csv` missing columns | §5 persistence (concat raw df + non-text blocks) | Saved CSV shape `(329044, ~97)`: original 30 raw/derived columns + 67 encoded columns from numerical (6) + genre (22) + language (31) + decade (2) + awards (6) blocks. **Text embedding (384 dims) intentionally excluded — kept in `text_embeddings.npy` for size and readability** |

---

## 5. Caching Strategy

Two expensive steps cached:

| Step | Cold cost | Cache file | Format |
|---|---|---|---|
| Text embedding | ~30-60 min (Colab T4) | `artifacts/text_embeddings.npy` | numpy |
| Final feature matrix | ~5-10 s | `artifacts/feature_matrix.npz` | numpy compressed |

Cache invalidation key for embeddings: MD5 of `f"{model_name}|{n_films}|{total_overview_chars}"`. When any of these change, the cache is regenerated automatically. Manual invalidation: `rm artifacts/text_embeddings.npy`.

The other steps complete in under five seconds and are not cached — caching them would add complexity for negligible gain.

---

## 6. Reproducibility

`seed_everything(42)` in §1 sets:
- `random.seed(42)`
- `np.random.seed(42)`
- `os.environ['PYTHONHASHSEED'] = '42'`
- `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

`pipeline_version.json` records, on every run: seed, UTC timestamp, library versions (numpy, pandas, sentence-transformers, torch, umap-learn), model name, n_films, feature_dim, and the MD5 hash of `feature_matrix.npz`. Reproducibility is verified by running the notebook twice and checking the MD5 hashes match.

---

## 7. Error Handling

Three assertion levels, fail-fast philosophy:

| Level | Where | Example | On failure |
|---|---|---|---|
| Block | After each `engineer_*` | `assert_no_nan(numerical_block)` | `AssertionError` — cell stops with the offending function named |
| Modality | After `apply_modality_aware_scaling` | `var(text)/var(others) ∈ [0.3, 3.0]` | Warning + log; pipeline continues (soft) |
| Final | After `assemble_feature_matrix` | `X.shape == (329044, 451)`, `np.isnan(X).sum() == 0` | `AssertionError` — pipeline stops |

A sanity-print cell at the end of §3 surfaces the headline numbers (awards merge rate, has_vote count, has_genre count, modality variance ratio, log_vote_count skewness) so the operator can eyeball them without scrolling through assertions.

---

## 8. Success Criteria

The notebook is "done" when **all** of the following pass:

| Criterion | Verification |
|---|---|
| All 13 fixes applied | Every "Verification" row in Section 4 passes |
| Reproducible | `feature_matrix_md5` identical across two consecutive runs |
| Cache works | Second run completes in under two minutes |
| Artifacts complete | All seven files in `artifacts/` plus `figures/` exist |
| Modality balance | `var(text) / var(others) ∈ [0.5, 2.0]` |
| Sanity print silent | No warnings; values within expected ranges |

---

## 9. Out of Scope (Explicit)

- AE/VAE/DEC training (separate spec).
- REST API and front-end (project descoped to ML core for now).
- Refactor of the original EDA notebook (preserved as-is).
- Replacing the source CSVs or merging additional datasets.
- Building a CI pipeline or automated tests beyond inline assertions.

---

## 10. Cold-Start vs. Warm-Start Runtime

| Run type | Estimated time | What happens |
|---|---|---|
| Cold (first ever) | 30-60 min | All steps run; embedding hits the model; artifacts written |
| Warm (cache present) | 1-2 min | Embedding loaded from `.npy`; everything else recomputed cheaply |
| Re-render only | < 30 s | Only §4 visualisation cells re-executed |

---

## 11. Bridge to the Modelling Phase

Once `eda_v2.ipynb` succeeds, the modelling notebooks load the artifacts directly:

```python
import numpy as np
data = np.load('artifacts/feature_matrix.npz')
X, feature_names = data['X'], data['feature_names']
# → AE/VAE/DEC training begins
```

If the modelling phase needs to call the pipeline programmatically (e.g., for a held-out subset), the §2 functions are copied verbatim into a `pipeline.py` module — a one-hour refactor that is **out of scope** for this spec but trivial because every §2 function is already pure.
