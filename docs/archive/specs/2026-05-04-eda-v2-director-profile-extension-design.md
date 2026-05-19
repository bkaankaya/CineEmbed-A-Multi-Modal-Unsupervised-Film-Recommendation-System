# CineEmbed — `eda_v2.ipynb` Director-Profile Extension Design

**Status:** Draft (2026-05-04)
**Author:** Baran Dinçoğuz (with Claude)
**Course:** SENG 474 — Deep Learning · TED University · Spring 2026
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya
**Extends:** [`2026-05-03-eda-v2-design.md`](2026-05-03-eda-v2-design.md)

---

## 1. Overview

The approved `eda_v2.ipynb` design produces a `(329044, 451)` feature matrix from five film-level modalities (numerical, genre, language, decade, awards) plus a 384-dim multilingual sentence embedding of the film overview. Three new director-centric CSV files have since been added to `data/` — Wikipedia bios for ~589 acclaimed directors and a director-language frequency table covering ~85k directors. This document specifies a **sixth modality**, `director_profile`, that incorporates these signals while preserving every architectural decision in the parent spec.

The extension:

1. Adds a `director_profile` block (113 dims) built from director Wikipedia bios (PCA-reduced multilingual embeddings) plus the director's dominant language and country.
2. Reuses the existing pipeline-first architecture — every new step is a pure, idempotent function in §2 of the notebook.
3. Reuses the existing multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) so bio embeddings live in the same vector space as overview embeddings.
4. Reuses the existing missing-handling pattern (zero-vector + `has_*` flag) so no rows are dropped.
5. Caches the new embedding step the same way the parent spec caches overview embeddings.
6. Increases the final feature matrix from `(329044, 451)` to `(329044, 564)`.

The parent spec remains the source of truth for everything not modified here.

---

## 2. Goals and Non-Goals

### Goals
- Add a `director_profile` modality (113 dims) without dropping any rows from the 329,044-film dataset.
- Capture director-level semantic signal from Wikipedia bios via PCA-reduced multilingual sentence embeddings (64 dims).
- Capture director-level filmography signal — dominant working language (31 dims) and derived country (16 dims).
- Use the same multilingual sentence-transformer model as the parent spec so the two text spaces are mutually comparable.
- Keep every architectural property of the parent spec: pipeline-first §2 functions, fail-fast assertions, deterministic caching, reproducibility.

### Non-Goals
- AE/VAE/DEC training (unchanged from parent spec — out of scope).
- REST API / front-end (unchanged — out of scope).
- Refactoring the original EDA notebook (unchanged — kept as-is).
- Using `900_acclaimed_directors_awards.csv` (wide-format, no per-event year → temporal-leak risk).
- Using `spielberg_awards.csv` (single-director sample / sanity-check file only).
- Replacing the existing temporal-aware `220k_awards_by_directors.csv` aggregation.

---

## 3. Data Sources

### 3.1 Used by this extension

| File | Rows | Role |
|---|---:|---|
| `500 favorite directors_with wikipedia summary.csv` | 589 | Director Wikipedia bio text → multilingual embedding → PCA(64) |
| `MostCommonLanguageByDirector.csv` | 85,882 | Director × language × film-count → dominant working language per director |
| `language to country.csv` | 94 | ISO language code → ISO country code lookup |

### 3.2 Excluded (with rationale)

| File | Reason |
|---|---|
| `900_acclaimed_directors_awards.csv` | Wide-format, no per-event year. Cannot apply temporal-aware filtering → reintroduces the leakage that fix #1 of the parent spec eliminated. The existing `220k_awards_by_directors.csv` (per-event with year) covers the same award space. |
| `spielberg_awards.csv` | Single-director sample (332 rows for Spielberg only). Not representative; treated as a worked example, not a data source. |

---

## 4. Architecture

### 4.1 New modality structure

`director_profile` is a single block composed of five sub-components:

| Sub-component | Dim | Source | Encoding |
|---|---:|---|---|
| `bio_text_pca` | 64 | Wikipedia bio of director | multilingual sentence embedding (384) → PCA(64) |
| `has_director_bio` | 1 | derived | binary flag — 1 if bio available, 0 otherwise |
| `dominant_lang` | 31 | `MostCommonLanguageByDirector` | top-30 + 'other' one-hot |
| `country_region` | 16 | `dominant_lang` → `language to country` lookup | top-15 + 'other' one-hot |
| `has_director_lang` | 1 | derived | binary flag — 1 if director appears in language table |
| **Total** | **113** | | |

**Broadcast:** Profile data is per-director; the feature matrix is per-film. Each row of `df` inherits its director's profile via left-join on `director_name_norm` (the existing normalized key). All films of the same director share identical `director_profile` rows.

### 4.2 New `§2` functions (pipeline-first, pure)

Added to the existing notebook §2 layer; signatures consistent with parent-spec conventions.

| Layer | Function | Purpose |
|---|---|---|
| §2.1 | `load_director_profile_csvs(paths) -> (bios_df, langs_df, lang_to_country_df)` | Load three CSVs with correct separators; auto-detect comma vs semicolon |
| §2.1 | `normalize_director_in_profile_dfs(bios_df, langs_df) -> (bios_df, langs_df)` | Apply existing `normalize_director_name` to add `director_name_norm` column to both |
| §2.4 | `compute_director_bio_embeddings(bios_df, *, model_name, cache_path, batch_size) -> ndarray` | (n_dirs, 384) cached embeddings; cache key = MD5(`model_name + n_directors + total_bio_chars`) |
| §2.4 | `pca_reduce_director_bios(emb, n_components=64, random_state=42) -> (ndarray, PCA)` | Returns reduced (n_dirs, 64) and the fitted PCA object |
| §2.3 | `engineer_director_profile(df, bios_df, bio_pca, langs_df, lang_to_country_df, *, top_n_lang=30, top_n_country=15) -> block_df` | Build full 113-dim block via left-join on `director_name_norm`; emit `has_*` flags; zero-vector for missing |

Every new function is **pure and idempotent** — same inputs always produce same outputs.

### 4.3 Modified `§2.5` function

`apply_modality_aware_scaling` extended to handle the new sub-blocks. Scaling rules:

| Sub-block | Scaling |
|---|---|
| `bio_text_pca` (64) | L2-normalize per row (same as overview text) |
| `dominant_lang` (31) | bypass (already 0/1 one-hot) |
| `country_region` (16) | bypass (already 0/1 one-hot) |
| `has_director_bio` (1) | bypass (binary flag) |
| `has_director_lang` (1) | bypass (binary flag) |

### 4.4 Updated execution flow (§3)

```python
# 1-3. Existing: load, name normalize, awards temporal merge — UNCHANGED
# 4. Existing: 5 blocks (numerical, genre, language, decade, awards) — UNCHANGED
# 5. Existing: overview text embedding — UNCHANGED
blocks['text_overview'] = compute_text_embeddings(df['overview'].fillna(''), ...)

# === NEW: director profile pipeline ===
# 6. Load director-profile CSVs and normalize names
bios_df, langs_df, lang_to_country_df = load_director_profile_csvs(PATHS)
bios_df, langs_df = normalize_director_in_profile_dfs(bios_df, langs_df)

# 7. Bio embedding + PCA (cached)
bio_emb_full = compute_director_bio_embeddings(
    bios_df,
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    cache_path=Path('artifacts/director_bio_embeddings.npy'),
)
bio_pca, pca_director = pca_reduce_director_bios(bio_emb_full, n_components=64)

# 8. Build the director_profile block (broadcast + missing flags)
blocks['director_profile'] = engineer_director_profile(
    df, bios_df, bio_pca, langs_df, lang_to_country_df,
    top_n_lang=30, top_n_country=15,
)
# === END NEW ===

# 9. Modality-aware scaling (extended)
blocks_scaled, scalers = apply_modality_aware_scaling(blocks)

# 10. Assemble + assert
X, feature_names = assemble_feature_matrix(blocks_scaled)
assert_shape(X, (len(df), 564), 'X')   # was 451
assert_no_nan(X, 'X')
```

### 4.5 Updated feature matrix dim breakdown

This table replaces the parent spec's §3.5 table:

| Block | Dim | Composition |
|---|---:|---|
| numerical | 6 | (unchanged from parent) |
| genre | 22 | (unchanged) |
| language | 31 | (unchanged, **film-level**) |
| decade | 2 | (unchanged) |
| awards | 6 | (unchanged, temporal-aware) |
| text_overview | 384 | (unchanged, multilingual) |
| **director_profile** | **113** | bio_text_pca (64) + has_director_bio (1) + dominant_lang (31) + country_region (16) + has_director_lang (1) |
| **Total** | **564** | |

### 4.6 Updated artifacts

Added to the parent spec's `artifacts/` directory:

```
artifacts/
├── ... (existing 7 files from parent spec)
├── director_bio_embeddings.npy           # raw (589, 384) cache
├── director_bio_embeddings.meta.json     # cache-invalidation hash
├── director_bio_pca.pkl                  # fitted PCA(64)
├── director_profile_metadata.json        # n_dirs_covered, top languages, top countries
└── figures/                              # +5 new PNGs (28 total, was 23)
```

`feature_matrix.npz` and `movies_eda_final.csv` are still produced — only their shapes change (564 dims and an extra ~50 columns respectively).

---

## 5. Missing Handling

Mirrors the parent spec's FR-D2 zero-vector pattern. **No film is dropped.**

| Case | Sub-blocks affected | Strategy |
|---|---|---|
| Director not in 589-bio list | `bio_text_pca` (64) | Zero-vector; `has_director_bio = 0` |
| Director not in `MostCommonLanguageByDirector` | `dominant_lang` (31), `country_region` (16) | All-zero one-hot; `has_director_lang = 0` |
| Director's dominant language not in `lang_to_country` lookup | `country_region` (16) only | All-zero one-hot for country; `dominant_lang` still encoded; `has_director_lang = 1` (lang exists, country doesn't) |
| `director_name` missing on the film | All sub-blocks | All zeros; both flags = 0 |

The flags are themselves features — when `has_director_bio = 0`, the model can learn to ignore the (zero) bio dims for that row, exactly as it does for `has_overview = 0` in the parent spec.

---

## 6. Cache Strategy

Adds one new cached step. Total cached steps: 2 (parent: overview embedding) + 1 (this: director-bio embedding) = 3.

| Step | Cold cost | Warm cost | Cache file | Invalidation key |
|---|---|---|---|---|
| Director bio embedding | ~30 s (CPU) / ~5 s (T4) | < 2 s | `artifacts/director_bio_embeddings.npy` | MD5(`model_name + n_directors + total_bio_chars`) |
| Director-bio PCA fit | ~1 s | always re-fit (cheap) | not cached | n/a |

Manual invalidation: `rm artifacts/director_bio_embeddings.npy`.

The PCA fit itself is not cached — it takes under a second on 589 vectors and re-fitting on every run guarantees the projection matches the embedding cache version.

---

## 7. Sanity Checks and Assertions

Added to the existing block / modality / final assertion stack.

| Level | Assertion | Expected | On failure |
|---|---|---|---|
| Block | `bio_emb_full.shape == (n_dirs_with_bio, 384)` and `n_dirs_with_bio ≥ 580` | Bio CSV is ~589 rows, parsing should succeed for nearly all | `AssertionError` |
| Block | `pca_director.explained_variance_ratio_.sum() ≥ 0.75` | 64 PCs typically capture 80%+ on sentence-transformer embeddings | `AssertionError` |
| Block | `df['has_director_bio'].mean() ≥ 0.10` | At least 10% film-level coverage (acclaimed directors are prolific) | warn — see §10 mitigation |
| Block | `df['has_director_lang'].mean() ≥ 0.95` | Language CSV covers 85k directors → very high film-level coverage | warn |
| Block | `(blocks['director_profile'].shape[1] == 113)` | Total sub-block dim correct | `AssertionError` |
| Block | Spot-check (conditional, skip if either director absent from the 589-bio list): pick two directors from the same broad cluster present in the bios — e.g., (Tarantino, Scorsese) or (Kaurismäki, Bergman) — and assert `cosine_sim` between their `bio_pca` rows > 0.2 | Same broad cluster → semantic neighborhood preserved | warn (skip if pairs missing) |
| Modality | `var(text_overview ∪ bio_text_pca) / var(others) ∈ [0.5, 2.5]` | Upper bound relaxed from 2.0 to 2.5 (text now 448 dims vs 384) | warn |
| Final | `X.shape == (329044, 564)` | New target dim | `AssertionError` |
| Final | Two consecutive runs → identical `feature_matrix_md5` | Determinism preserved | `AssertionError` |

The end-of-§3 sanity-print cell gains four new headline numbers: bio coverage %, lang coverage %, PCA explained variance %, and bio variance / overview variance ratio.

---

## 8. New Visualizations (§4)

The parent spec produces 23 figures. This extension adds **5 more** (total: 28).

| # | Figure | Slot | Purpose |
|---|---|---|---|
| F1 | `director_bio_coverage.png` | §4.1 (Data quality) | Stacked bar: `has_director_bio` distribution overall and by decade. Justifies the new modality is not mostly-empty. |
| F2 | `director_bio_pca_scree.png` | §4.6 (Text emb section) | Cumulative explained variance vs. # PCs. Marks 64 with the achieved %. Justifies the dim choice. |
| F3 | `director_bio_pca_2d_scatter.png` | §4.6 | First 2 PCs of PCA(64), colored by director country. Visual evidence the embedding captures meaningful structure. |
| F4 | `director_lang_vs_film_lang.png` | §4.4 (Categorical) | Confusion-style heatmap: director's dominant lang × film's original lang. Justifies the new feature isn't redundant with the existing film-language block (off-diagonal mass should be ≥ 5%). |
| F5 | `modality_balance_v2.png` | §4.5 (replaces parent's modality balance figure) | 6 modalities (added director_profile) — variance contribution and effective dims. |

---

## 9. Updated Success Criteria

The parent spec's six criteria stay; three new criteria are added:

| Criterion | Verification |
|---|---|
| (Parent's six criteria) | unchanged |
| Director bio film-level coverage ≥ 10% | F1 + sanity print |
| Director bio PCA explained variance ≥ 75% (at 64 PCs) | F2 + assertion |
| Director-lang ≠ film-lang in ≥ 5% of rows | F4 (off-diagonal mass) |

If the third criterion fails — i.e., the director's dominant language matches the film's original language for >95% of rows — then `dominant_lang` and `country_region` are redundant with the existing `language` block and should be dropped. The notebook will print a clear `INFO` line and continue; the dim breakdown table is the source of truth so this is recoverable post-hoc.

---

## 10. Risk and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bio film-level coverage falls below 10% | Low — acclaimed directors are very prolific (each averages 15+ films) | Block has weak signal | Sanity print warns; the `has_director_bio` flag is itself a useful signal even when coverage is low |
| Director name normalization fails to match Wikipedia rows to casting rows | Medium — Wikipedia titles can have parentheticals, comma forms | Bio coverage drops below threshold | Existing `normalize_director_name` already handles "Last, First" and accent stripping; new spot-check assertion compares baseline vs normalized match rates (must improve by ≥ 5pp, mirroring parent fix #8) |
| Director-lang and film-lang are highly correlated | Medium — most directors work primarily in one language | Block is partially redundant | Off-diagonal verification in F4; if redundant, drop sub-block (criterion in §9) |
| Modality balance constraint fails | Low — bio_pca is only 64 dims and L2-normalized | Text dominance | Upper bound relaxed to 2.5; if exceeded, fall back plan: drop the bio_pca block and keep only language/country (29 dims) |
| Wikipedia bios contain extraneous noise (markup, references) | Medium — CSV preview shows lowercased/cleaned text but with garbled IPA | PCA components capture noise, not semantics | F3 visual check (country-grouping in PCA space); if noise-dominated, light cleanup pass before embedding (out of scope unless triggered) |

---

## 11. Out of Scope (Explicit)

- AE/VAE/DEC training (separate spec).
- Wide-format `900_acclaimed_directors_awards.csv` (excluded — temporal leak).
- `spielberg_awards.csv` (excluded — sample file).
- REST API and front-end (unchanged from parent).
- Refactor of the original EDA notebook (unchanged).
- Augmenting Wikipedia bios with external scraping or text cleanup beyond what the CSV already provides.

---

## 12. Bridge to the Modelling Phase

The contract with the AE/VAE/DEC notebooks is unchanged in form — only the second dimension grows:

```python
import numpy as np
data = np.load('artifacts/feature_matrix.npz')
X, feature_names = data['X'], data['feature_names']
# X.shape == (329044, 564)   — was (329044, 451)
```

If a downstream notebook needs only the original 5 modalities + overview text (e.g., for an ablation comparing "with vs. without director profile"), it can slice via `feature_names`:

```python
mask = ~np.array([fn.startswith('dir_') for fn in feature_names])
X_no_director = X[:, mask]   # shape (329044, 451) — matches parent spec
```

This makes the "director profile improves clustering" ablation a one-line slice, rather than re-running the pipeline.
