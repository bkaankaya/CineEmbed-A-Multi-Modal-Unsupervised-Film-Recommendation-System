# `eda_v2.ipynb` Director-Profile Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `eda_v2.ipynb` pipeline with a sixth modality (`director_profile`, 113 dims) so the final feature matrix grows from `(329044, 451)` to `(329044, 564)`. Adds a director Wikipedia bio embedding (PCA-reduced to 64 dims) plus the director's dominant working language and country, with no row drops.

**Architecture:** Pure-function additions to the existing `§2` pipeline layer of `eda_v2.ipynb`. Five new functions (one per pipeline stage), one cached embedding step, one extension to `apply_modality_aware_scaling`, three new execution-flow steps in `§3`, five new figures in `§4`, and three new artifact files. Reuses the parent spec's multilingual sentence-transformer model so director bios live in the same vector space as film overviews. Notebook-friendly TDD: each function has an inline test cell with synthetic mock data immediately after its definition.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn (`PCA`, `StandardScaler`), `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, already loaded in parent §2.4), matplotlib, seaborn. Runs anywhere the parent notebook runs (Google Colab T4 GPU or local CPU).

**Reference:** [`docs/superpowers/specs/2026-05-04-eda-v2-director-profile-extension-design.md`](../specs/2026-05-04-eda-v2-director-profile-extension-design.md)

---

## Prerequisites

This plan assumes the parent `eda_v2.ipynb` (per `docs/superpowers/plans/2026-05-03-eda-v2-implementation.md`) is **fully implemented and runs end-to-end** with output `(329044, 451)`. Specifically the following must already be defined in §2 of the notebook:

- `§2.1` — `load_csvs`, `normalize_director_name`, `merge_details_casting`
- `§2.2` — `aggregate_awards_temporal`
- `§2.3` — `engineer_numerical`, `engineer_genres`, `engineer_languages`, `engineer_decade`, `engineer_director_awards`
- `§2.4` — `compute_text_embeddings` (returns `(n, 384)` and uses the multilingual model from CONFIG)
- `§2.5` — `apply_modality_aware_scaling`, `assemble_feature_matrix`
- `§2.6` — `assert_shape`, `assert_no_nan`, `assert_value_range`, `assert_unit_norm`

If any of these are missing, finish the parent plan first.

---

## File Structure

| Path | Status | Purpose |
|---|---|---|
| `eda_v2.ipynb` | MODIFY | Add ~12 new cells; modify CONFIG and 2 existing cells |
| `data/500 favorite directors_with wikipedia summary.csv` | EXISTS | Director Wikipedia bios (no header, semicolon-separated, 589 rows, 84 empty bios) |
| `data/MostCommonLanguageByDirector.csv` | EXISTS | Director × language × film-count (header, comma, ~85k rows, leading-whitespace director names) |
| `data/language to country.csv` | EXISTS | Lang code → country code (header, semicolon, 93 rows) |
| `artifacts/director_bio_embeddings.npy` | NEW | Cache: `(~505, 384)` raw bio embeddings |
| `artifacts/director_bio_embeddings.meta.json` | NEW | Cache invalidation MD5 hash |
| `artifacts/director_bio_pca.pkl` | NEW | Fitted `sklearn.decomposition.PCA` (64 components) |
| `artifacts/director_profile_metadata.json` | NEW | Coverage metrics; top-N lang/country vocabularies |
| `artifacts/figures/director_bio_coverage.png` | NEW | F1 |
| `artifacts/figures/director_bio_pca_scree.png` | NEW | F2 |
| `artifacts/figures/director_bio_pca_2d_scatter.png` | NEW | F3 |
| `artifacts/figures/director_lang_vs_film_lang.png` | NEW | F4 |
| `artifacts/figures/modality_balance_v2.png` | NEW | F5 (replaces parent's `modality_balance.png`) |

---

## Notebook Testing Pattern

Same pattern as the parent plan:

1. **Add a test cell** below where the new function will go. The cell creates a small synthetic DataFrame, calls the function, and asserts on the output. Run it → fails with `NameError` because the function isn't defined yet.
2. **Add the function cell** with the full implementation. Run it.
3. **Re-run the test cell** → passes.
4. **Commit**.

The §3 spot-check cell at the bottom of the pipeline serves as the integration test against the real 329k-film dataset.

---

## Task 1 — Update §1 CONFIG with new paths and knobs

**Files:** Modify `eda_v2.ipynb` cell `§1 Setup & Reproducibility`.

- [ ] **Step 1.1: Open the §1 cell. Locate the `CONFIG['paths']` block and append three new entries.**

Existing block:
```python
'paths': {
    'details': Path('data/AllMoviesDetailsCleaned.csv'),
    'casting': Path('data/AllMoviesCastingRaw.csv'),
    'awards':  Path('data/220k_awards_by_directors.csv'),
},
```

Replace with:
```python
'paths': {
    'details':            Path('data/AllMoviesDetailsCleaned.csv'),
    'casting':            Path('data/AllMoviesCastingRaw.csv'),
    'awards':             Path('data/220k_awards_by_directors.csv'),
    'director_bios':      Path('data/500 favorite directors_with wikipedia summary.csv'),
    'director_langs':     Path('data/MostCommonLanguageByDirector.csv'),
    'lang_to_country':    Path('data/language to country.csv'),
},
```

- [ ] **Step 1.2: Append the new feature-engineering knobs to CONFIG (after `top_n_languages`).**

Add these lines:
```python
# Director profile (extension)
'top_n_director_country': 15,
'bio_pca_n_components':   64,
'bio_embedding_cache':    Path('artifacts/director_bio_embeddings.npy'),
'bio_embedding_meta':     Path('artifacts/director_bio_embeddings.meta.json'),
'bio_pca_cache':          Path('artifacts/director_bio_pca.pkl'),
'director_profile_meta':  Path('artifacts/director_profile_metadata.json'),
```

- [ ] **Step 1.3: Run the §1 cell. Verify it still prints `✅ §1 Setup complete` and that the new paths exist.**

Add a one-line sanity check at the bottom of the cell (just before the print statements):
```python
for k in ('director_bios', 'director_langs', 'lang_to_country'):
    assert CONFIG['paths'][k].exists(), f"missing data file: {CONFIG['paths'][k]}"
```

Expected: cell completes silently before printing. If any path is missing, the assertion names the file.

- [ ] **Step 1.4: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): add director-profile paths and knobs to CONFIG"
```

---

## Task 2 — `load_director_profile_csvs` (new §2.1 function)

**Files:** Modify `eda_v2.ipynb`, add two new cells immediately after the existing `merge_details_casting` cell in §2.1.

- [ ] **Step 2.1: Insert a TEST CELL (markdown header + code).**

Markdown:
```markdown
### Test — load_director_profile_csvs
```

Code:
```python
# Synthetic mini-CSVs in a tmp dir to assert load behavior
import tempfile, shutil
_tmp = Path(tempfile.mkdtemp())

(_tmp / 'bios.csv').write_text(
    'Steven Spielberg;is an american filmmaker known for jaws and e.t.\n'
    'Aki Kaurismäki;is a finnish film director.\n'
    'Empty Director;\n',
    encoding='utf-8',
)
(_tmp / 'langs.csv').write_text(
    'director_name,original_language,nb\n'
    '\tSteven Spielberg,en,30\n'           # leading tab in name
    'Steven Spielberg,fr,1\n'              # second lang for same director
    'Aki Kaurismäki,fi,15\n',
    encoding='utf-8',
)
(_tmp / 'l2c.csv').write_text(
    'Language;Country\nen;USA\nfi;FIN\nfr;FRA\n',
    encoding='utf-8',
)

bios, langs, l2c = load_director_profile_csvs({
    'director_bios':   _tmp / 'bios.csv',
    'director_langs':  _tmp / 'langs.csv',
    'lang_to_country': _tmp / 'l2c.csv',
})

# 1. Bios: 3 rows in raw, 2 should remain after dropping empty bios
assert list(bios.columns) == ['director_name', 'bio'], bios.columns.tolist()
assert len(bios) == 2, f"expected 2 non-empty bios, got {len(bios)}"
assert 'Empty Director' not in bios['director_name'].values

# 2. Langs: leading tab stripped from director_name
assert (langs['director_name'] == '\tSteven Spielberg').sum() == 0
assert (langs['director_name'] == 'Steven Spielberg').sum() == 2

# 3. lang_to_country: 3 rows, columns lower-cased
assert list(l2c.columns) == ['language', 'country']
assert len(l2c) == 3

shutil.rmtree(_tmp)
print("✅ load_director_profile_csvs: all 3 assertions passed")
```

- [ ] **Step 2.2: Run the test cell. Confirm it fails with `NameError: name 'load_director_profile_csvs' is not defined`.**

- [ ] **Step 2.3: Insert the FUNCTION CELL above the test cell.**

```python
def load_director_profile_csvs(paths: dict) -> tuple:
    """Load the three director-profile CSVs with their idiosyncratic formats.

    Args:
        paths: dict with keys 'director_bios', 'director_langs', 'lang_to_country'.

    Returns:
        (bios_df, langs_df, lang_to_country_df). Director names are leading/trailing
        whitespace-stripped. Empty bios are dropped from bios_df.
    """
    # 1. Bios — no header, semicolon-separated, two columns
    bios = pd.read_csv(
        paths['director_bios'],
        sep=';', header=None,
        names=['director_name', 'bio'],
        dtype=str, encoding='utf-8',
    )
    bios['director_name'] = bios['director_name'].str.strip()
    bios['bio'] = bios['bio'].fillna('').str.strip()
    bios = bios[bios['bio'].str.len() > 0].reset_index(drop=True)

    # 2. Langs — header, comma-separated; some director names have a leading TAB
    langs = pd.read_csv(paths['director_langs'], sep=',', dtype={'nb': int})
    langs['director_name'] = langs['director_name'].str.strip()
    langs = langs.dropna(subset=['director_name', 'original_language']).reset_index(drop=True)

    # 3. Lang-to-country — header, semicolon-separated; lower-case columns for consistency
    l2c = pd.read_csv(paths['lang_to_country'], sep=';')
    l2c.columns = [c.strip().lower() for c in l2c.columns]
    l2c['language'] = l2c['language'].str.strip()
    l2c['country']  = l2c['country'].str.strip()

    return bios, langs, l2c
```

- [ ] **Step 2.4: Re-run the function cell, then re-run the test cell. Confirm it prints `✅ load_director_profile_csvs: all 3 assertions passed`.**

- [ ] **Step 2.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): add load_director_profile_csvs with inline test"
```

---

## Task 3 — `normalize_director_in_profile_dfs` (new §2.1 function)

**Files:** Modify `eda_v2.ipynb`, add two new cells after Task 2.

- [ ] **Step 3.1: Insert TEST CELL.**

```python
# Test: applies the existing normalize_director_name to add a director_name_norm column
bios = pd.DataFrame({'director_name': ['Aki Kaurismäki', 'Spielberg, Steven', 'Wes  Anderson']})
langs = pd.DataFrame({
    'director_name': ['Aki Kaurismäki', 'Steven Spielberg'],
    'original_language': ['fi', 'en'],
    'nb': [15, 30],
})

bios_n, langs_n = normalize_director_in_profile_dfs(bios, langs)

# director_name_norm column added
assert 'director_name_norm' in bios_n.columns
assert 'director_name_norm' in langs_n.columns

# Same director resolves to the same normalized form across both dfs
spielberg_in_bios  = bios_n.loc[bios_n['director_name'] == 'Spielberg, Steven',  'director_name_norm'].iloc[0]
spielberg_in_langs = langs_n.loc[langs_n['director_name'] == 'Steven Spielberg', 'director_name_norm'].iloc[0]
assert spielberg_in_bios == spielberg_in_langs, f"{spielberg_in_bios!r} != {spielberg_in_langs!r}"

# Accents stripped, casefolded, double-space collapsed
kaurismaki_norm = bios_n.loc[bios_n['director_name'] == 'Aki Kaurismäki', 'director_name_norm'].iloc[0]
assert 'ä' not in kaurismaki_norm and kaurismaki_norm == kaurismaki_norm.lower()
wes_norm = bios_n.loc[bios_n['director_name'] == 'Wes  Anderson', 'director_name_norm'].iloc[0]
assert '  ' not in wes_norm

print("✅ normalize_director_in_profile_dfs: all 4 assertions passed")
```

- [ ] **Step 3.2: Run the test cell. Confirm it fails with `NameError`.**

- [ ] **Step 3.3: Insert FUNCTION CELL above the test cell.**

```python
def normalize_director_in_profile_dfs(bios_df: pd.DataFrame, langs_df: pd.DataFrame
) -> tuple:
    """Add a `director_name_norm` column to both dataframes using the existing
    normalize_director_name (defined in parent §2.1).

    Returns:
        (bios_df_with_norm, langs_df_with_norm) — copies, originals untouched.
    """
    bios_out  = bios_df.copy()
    langs_out = langs_df.copy()
    bios_out['director_name_norm']  = bios_out['director_name'].map(normalize_director_name)
    langs_out['director_name_norm'] = langs_out['director_name'].map(normalize_director_name)
    return bios_out, langs_out
```

- [ ] **Step 3.4: Re-run function cell, then test cell. Confirm `✅ normalize_director_in_profile_dfs: all 4 assertions passed`.**

- [ ] **Step 3.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): add normalize_director_in_profile_dfs with inline test"
```

---

## Task 4 — `compute_director_bio_embeddings` (cached, new §2.4 function)

**Files:** Modify `eda_v2.ipynb`, add two new cells immediately after the existing `compute_text_embeddings` cell in §2.4.

- [ ] **Step 4.1: Insert TEST CELL.**

```python
# Test: 5 mini-bios → cold call returns (5, 384); warm call hits the cache instantly
import time

_tmp_cache = Path('artifacts/_test_bio_emb.npy')
_tmp_meta  = Path('artifacts/_test_bio_emb.meta.json')
for p in (_tmp_cache, _tmp_meta):
    if p.exists():
        p.unlink()

mini_bios = pd.DataFrame({
    'director_name': [f'Director_{i}' for i in range(5)],
    'bio': [
        'is an american filmmaker known for action movies.',
        'is a french auteur known for slow-paced cinema.',
        'is a japanese animator famous for studio ghibli.',
        'is a british director of art-house thrillers.',
        'is a turkish-german filmmaker working in europe.',
    ],
})

t0 = time.time()
emb_cold = compute_director_bio_embeddings(
    mini_bios, model_name=CONFIG['embedding_model'],
    cache_path=_tmp_cache, meta_path=_tmp_meta,
    batch_size=8,
)
cold_dt = time.time() - t0

assert emb_cold.shape == (5, 384), emb_cold.shape
assert _tmp_cache.exists() and _tmp_meta.exists()

# Warm call — same inputs, should hit cache and skip the model
t0 = time.time()
emb_warm = compute_director_bio_embeddings(
    mini_bios, model_name=CONFIG['embedding_model'],
    cache_path=_tmp_cache, meta_path=_tmp_meta,
    batch_size=8,
)
warm_dt = time.time() - t0

assert np.allclose(emb_cold, emb_warm), "cache hit returned different embeddings"
assert warm_dt < cold_dt, f"warm not faster than cold: cold={cold_dt:.2f}s warm={warm_dt:.2f}s"

# Cleanup
_tmp_cache.unlink(); _tmp_meta.unlink()
print(f"✅ compute_director_bio_embeddings: cold={cold_dt:.1f}s, warm={warm_dt:.3f}s, shapes match")
```

- [ ] **Step 4.2: Run test cell. Confirm `NameError`.**

- [ ] **Step 4.3: Insert FUNCTION CELL above the test cell.**

```python
def compute_director_bio_embeddings(
    bios_df: pd.DataFrame,
    *,
    model_name: str,
    cache_path: Path,
    meta_path: Path,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed director Wikipedia bios with a multilingual sentence-transformer.

    Same pattern as parent's `compute_text_embeddings`: cache key is MD5 of
    (model_name, n_directors, total_bio_chars). On cache hit the .npy is loaded
    and the model is skipped entirely.

    Args:
        bios_df: must have a 'bio' column; row order defines embedding order.

    Returns:
        ndarray of shape (len(bios_df), 384).
    """
    # 1. Compute cache key from inputs
    n = len(bios_df)
    total_chars = int(bios_df['bio'].str.len().sum())
    key_str = f"{model_name}|{n}|{total_chars}"
    expected_md5 = hashlib.md5(key_str.encode('utf-8')).hexdigest()

    # 2. Cache hit?
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('md5') == expected_md5 and meta.get('shape', [None])[0] == n:
            emb = np.load(cache_path)
            print(f"   [cache HIT] director bio embeddings: {emb.shape} loaded from {cache_path.name}")
            return emb

    # 3. Cold path — compute embeddings
    from sentence_transformers import SentenceTransformer
    print(f"   [cache MISS] computing {n} director bio embeddings (model={model_name})...")
    model = SentenceTransformer(model_name)
    emb = model.encode(
        bios_df['bio'].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # 4. Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    meta_path.write_text(json.dumps({
        'md5': expected_md5,
        'shape': list(emb.shape),
        'model_name': model_name,
        'n_directors': n,
        'total_chars': total_chars,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    return emb
```

- [ ] **Step 4.4: Re-run function cell, then test cell. Confirm `✅` line; cold should be a few seconds, warm under 0.1s.**

- [ ] **Step 4.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): add compute_director_bio_embeddings with cache + inline test"
```

---

## Task 5 — `pca_reduce_director_bios` (new §2.4 function)

**Files:** Modify `eda_v2.ipynb`, add two new cells after Task 4.

- [ ] **Step 5.1: Insert TEST CELL.**

```python
# Test: reduce a (20, 384) synthetic embedding to 8 PCs
np.random.seed(0)
fake_emb = np.random.randn(20, 384).astype(np.float32)

reduced, pca_obj = pca_reduce_director_bios(fake_emb, n_components=8, random_state=42)

assert reduced.shape == (20, 8), reduced.shape
assert hasattr(pca_obj, 'explained_variance_ratio_')
assert len(pca_obj.explained_variance_ratio_) == 8
# Random data → no single PC captures most variance, but the sum should be > 0
assert pca_obj.explained_variance_ratio_.sum() > 0
# Re-running with the SAME random_state gives the same projection (numerical sign flip OK)
reduced2, _ = pca_reduce_director_bios(fake_emb, n_components=8, random_state=42)
assert np.allclose(np.abs(reduced), np.abs(reduced2))

print(f"✅ pca_reduce_director_bios: shape ok; "
      f"explained_var_sum={pca_obj.explained_variance_ratio_.sum():.3f}")
```

- [ ] **Step 5.2: Run test cell. Confirm `NameError`.**

- [ ] **Step 5.3: Insert FUNCTION CELL above the test cell.**

```python
def pca_reduce_director_bios(
    emb: np.ndarray,
    n_components: int = 64,
    random_state: int = 42,
) -> tuple:
    """Reduce director-bio embeddings via PCA.

    The fitted PCA object is returned alongside the reduced array so it can be
    persisted to disk (re-applying it to held-out directors at inference time).

    Returns:
        (reduced_array of shape (n_dirs, n_components), fitted PCA instance).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(emb).astype(np.float32)
    return reduced, pca
```

- [ ] **Step 5.4: Re-run function cell, then test cell. Confirm `✅` line.**

- [ ] **Step 5.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): add pca_reduce_director_bios with inline test"
```

---

## Task 6 — `engineer_director_profile` (new §2.3 function — main builder)

**Files:** Modify `eda_v2.ipynb`, add two new cells after Task 5 (function lives in §2.3, but is most easily added next to its dependencies in §2.4 — pick the location that matches the rest of `engineer_*` functions in your notebook).

This is the largest task. The function broadcasts director-level data to film-level rows via left-join, builds five sub-blocks, and returns a `(n_films, 113)` DataFrame.

- [ ] **Step 6.1: Insert TEST CELL.**

```python
# Test: 4 synthetic films, 3 directors (1 in bios, 2 in langs, 1 missing both)
df_films = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'director_name':      ['Spielberg', 'Kaurismaki', 'Kaurismaki', 'NobodyKnown'],
    'director_name_norm': ['spielberg', 'kaurismaki', 'kaurismaki', 'nobodyknown'],
})
bios_df = pd.DataFrame({
    'director_name':      ['Spielberg'],
    'bio':                ['mock bio'],
    'director_name_norm': ['spielberg'],
})
bio_pca = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)   # (1, 4) for test simplicity
langs_df = pd.DataFrame({
    'director_name':      ['Spielberg', 'Kaurismaki'],
    'original_language':  ['en', 'fi'],
    'nb':                 [30, 15],
    'director_name_norm': ['spielberg', 'kaurismaki'],
})
l2c = pd.DataFrame({'language': ['en', 'fi'], 'country': ['USA', 'FIN']})

block = engineer_director_profile(
    df_films, bios_df, bio_pca, langs_df, l2c,
    top_n_lang=2, top_n_country=2, bio_pca_dim=4,
)

# 1. Shape: (4 films, 4 bio_pca + 1 has_bio + 3 lang [en, fi, other] + 3 country [USA, FIN, other] + 1 has_lang) = 12
assert block.shape == (4, 12), block.shape

# 2. has_director_bio flag: only Spielberg row has bio
assert block['has_director_bio'].tolist() == [1, 0, 0, 0]

# 3. has_director_lang flag: Spielberg + Kaurismaki rows yes, NobodyKnown row no
assert block['has_director_lang'].tolist() == [1, 1, 1, 0]

# 4. bio_pca cols match for Spielberg, zero-vector for others
assert list(block.iloc[0][['dir_bio_pca_0', 'dir_bio_pca_1', 'dir_bio_pca_2', 'dir_bio_pca_3']]) == [0.1, 0.2, 0.3, 0.4]
assert list(block.iloc[1][['dir_bio_pca_0', 'dir_bio_pca_1', 'dir_bio_pca_2', 'dir_bio_pca_3']]) == [0.0, 0.0, 0.0, 0.0]

# 5. dominant_lang one-hot: Spielberg=en, Kaurismaki=fi, NobodyKnown=all-zero
assert block.loc[0, 'dir_lang_en'] == 1 and block.loc[0, 'dir_lang_fi'] == 0
assert block.loc[1, 'dir_lang_en'] == 0 and block.loc[1, 'dir_lang_fi'] == 1
assert block.loc[3, ['dir_lang_en', 'dir_lang_fi', 'dir_lang_other']].sum() == 0

# 6. country one-hot derived from language
assert block.loc[0, 'dir_country_USA'] == 1
assert block.loc[1, 'dir_country_FIN'] == 1

print(f"✅ engineer_director_profile: shape={block.shape}, all 6 assertions passed")
```

- [ ] **Step 6.2: Run test cell. Confirm `NameError`.**

- [ ] **Step 6.3: Insert FUNCTION CELL above the test cell.**

```python
def engineer_director_profile(
    df: pd.DataFrame,
    bios_df: pd.DataFrame,
    bio_pca: np.ndarray,
    langs_df: pd.DataFrame,
    lang_to_country_df: pd.DataFrame,
    *,
    top_n_lang: int = 30,
    top_n_country: int = 15,
    bio_pca_dim: int = 64,
) -> pd.DataFrame:
    """Build the per-film director_profile feature block by broadcasting director-level
    signals via left-join on `director_name_norm`.

    Sub-blocks (in column order):
      - dir_bio_pca_0..(D-1)        : PCA-reduced bio embedding
      - has_director_bio            : 1 if director's bio was embedded, 0 otherwise
      - dir_lang_<top-N> + dir_lang_other : one-hot of director's dominant filmography language
      - dir_country_<top-N> + dir_country_other : one-hot of country derived from dominant language
      - has_director_lang           : 1 if director appears in MostCommonLanguageByDirector

    Args:
        df: film-level frame; must have 'director_name_norm'.
        bios_df: must have 'director_name_norm' (one row per director with non-empty bio).
        bio_pca: (len(bios_df), bio_pca_dim) — order matches bios_df rows.
        langs_df: director_name_norm × original_language × nb (multiple rows allowed per director).
        lang_to_country_df: 'language', 'country' columns.

    Returns:
        DataFrame (len(df), 1 + bio_pca_dim + (top_n_lang + 1) + (top_n_country + 1) + 1) dims.
        Index aligned with df.index.
    """
    n_films = len(df)
    assert bio_pca.shape == (len(bios_df), bio_pca_dim), (
        f"bio_pca shape {bio_pca.shape} doesn't match (n_bios={len(bios_df)}, dim={bio_pca_dim})")

    # ─── Sub-block 1: bio_text_pca + has_director_bio ────────────────────────
    bio_lookup = pd.DataFrame(
        bio_pca,
        columns=[f'dir_bio_pca_{i}' for i in range(bio_pca_dim)],
        index=bios_df['director_name_norm'].values,
    )
    bio_lookup = bio_lookup.groupby(level=0).first()  # dedupe in case of duplicate norm names

    bio_block = df[['director_name_norm']].merge(
        bio_lookup, left_on='director_name_norm', right_index=True, how='left',
    )
    bio_cols = [f'dir_bio_pca_{i}' for i in range(bio_pca_dim)]
    has_bio = bio_block[bio_cols[0]].notna().astype(int)
    bio_block[bio_cols] = bio_block[bio_cols].fillna(0.0)

    # ─── Sub-block 2: dominant_lang one-hot ──────────────────────────────────
    # Per director: pick row with max nb (ties → first)
    langs_dom = (
        langs_df.sort_values('nb', ascending=False)
        .drop_duplicates(subset='director_name_norm', keep='first')
        [['director_name_norm', 'original_language']]
        .rename(columns={'original_language': 'dominant_lang'})
    )

    # Top-N over the dominant_lang values seen across directors
    top_langs = (langs_dom['dominant_lang'].value_counts().head(top_n_lang).index.tolist())
    langs_dom['dominant_lang_grouped'] = langs_dom['dominant_lang'].where(
        langs_dom['dominant_lang'].isin(top_langs), other='other'
    )

    lang_block = df[['director_name_norm']].merge(
        langs_dom[['director_name_norm', 'dominant_lang_grouped', 'dominant_lang']],
        on='director_name_norm', how='left',
    )
    has_lang = lang_block['dominant_lang_grouped'].notna().astype(int)

    # One-hot — initialize all expected columns to ensure consistent width
    lang_one_hot_cols = [f'dir_lang_{l}' for l in top_langs] + ['dir_lang_other']
    lang_one_hot = pd.DataFrame(0, index=df.index, columns=lang_one_hot_cols, dtype=int)
    for l in top_langs + ['other']:
        col = f'dir_lang_{l}'
        lang_one_hot[col] = (lang_block['dominant_lang_grouped'] == l).astype(int).values

    # ─── Sub-block 3: country_region one-hot ─────────────────────────────────
    # Map dominant_lang (raw, not grouped) → country via lookup
    l2c_map = dict(zip(lang_to_country_df['language'], lang_to_country_df['country']))
    country = lang_block['dominant_lang'].map(l2c_map)

    top_countries = country.dropna().value_counts().head(top_n_country).index.tolist()
    country_grouped = country.where(country.isin(top_countries), other='other')

    country_one_hot_cols = [f'dir_country_{c}' for c in top_countries] + ['dir_country_other']
    country_one_hot = pd.DataFrame(0, index=df.index, columns=country_one_hot_cols, dtype=int)
    for c in top_countries + ['other']:
        col = f'dir_country_{c}'
        # Important: for directors with no lang, all country one-hots stay 0 (NaN → False)
        country_one_hot[col] = (country_grouped == c).fillna(False).astype(int).values

    # ─── Assemble final block ────────────────────────────────────────────────
    out = pd.concat([
        bio_block[bio_cols].reset_index(drop=True),
        pd.Series(has_bio.values, name='has_director_bio'),
        lang_one_hot.reset_index(drop=True),
        country_one_hot.reset_index(drop=True),
        pd.Series(has_lang.values, name='has_director_lang'),
    ], axis=1)
    out.index = df.index
    return out
```

- [ ] **Step 6.4: Re-run function cell, then test cell. Confirm `✅ engineer_director_profile: shape=(4, 12), all 6 assertions passed`.**

If shape mismatch: usually means `top_n_lang`/`top_n_country` did not match the test's expected widths. Re-check column construction.

- [ ] **Step 6.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): add engineer_director_profile (broadcast + missing handling)"
```

---

## Task 7 — Extend `apply_modality_aware_scaling` (modify §2.5 cell)

**Files:** Modify `eda_v2.ipynb`, edit the existing `apply_modality_aware_scaling` cell.

The existing function handles five blocks (numerical, genre, language, decade, awards, plus `text_overview`). Add a branch for `director_profile`: L2-normalize the `dir_bio_pca_*` columns per row, leave the rest untouched.

- [ ] **Step 7.1: Open the §2.5 cell. Locate the `apply_modality_aware_scaling` body.**

- [ ] **Step 7.2: Inside the function body — typically near the end of the per-block branching — add this branch:**

```python
# Director profile: only the bio_pca columns get L2-normalized; the rest
# (one-hot lang, one-hot country, flags) are already 0/1 and pass through.
if 'director_profile' in blocks:
    block = blocks['director_profile'].copy()
    bio_cols = [c for c in block.columns if c.startswith('dir_bio_pca_')]
    if bio_cols:
        bio_arr = block[bio_cols].values.astype(np.float32)
        norms = np.linalg.norm(bio_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0   # zero-vector rows (no bio) stay zero
        block[bio_cols] = bio_arr / norms
    scaled_blocks['director_profile'] = block
    scalers['director_profile'] = None   # no scaler instance to persist; L2 is data-dependent
```

Make sure this code runs only if the block is present (the function must remain backward-compatible with the parent notebook's pre-extension runs).

- [ ] **Step 7.3: Add an inline test cell immediately below the function cell.**

```python
# Test: a director_profile mini-block with 2 rows (one with bio, one without)
mini = pd.DataFrame({
    'dir_bio_pca_0': [3.0, 0.0],
    'dir_bio_pca_1': [4.0, 0.0],
    'has_director_bio': [1, 0],
    'dir_lang_en': [1, 0],
    'has_director_lang': [1, 1],
})
out, sc = apply_modality_aware_scaling({'director_profile': mini})
result = out['director_profile']

# Row 0: (3, 4) → L2 norm 5 → (0.6, 0.8)
assert np.allclose(result.loc[0, ['dir_bio_pca_0', 'dir_bio_pca_1']].values, [0.6, 0.8])
# Row 1: zero-vector stays zero (no division by zero)
assert np.allclose(result.loc[1, ['dir_bio_pca_0', 'dir_bio_pca_1']].values, [0.0, 0.0])
# Other columns unchanged
assert result['has_director_bio'].tolist() == [1, 0]
assert result['dir_lang_en'].tolist() == [1, 0]

print("✅ apply_modality_aware_scaling: director_profile L2-norm correct, others bypass")
```

- [ ] **Step 7.4: Run the function cell, then the test cell. Confirm `✅`.**

- [ ] **Step 7.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): extend apply_modality_aware_scaling for director_profile"
```

---

## Task 8 — Update §3 execution flow (add steps 6–8)

**Files:** Modify `eda_v2.ipynb`, edit the existing `§3 Pipeline Execution` cell.

The existing cell already builds 5 blocks plus the overview text block. Add the three new steps from spec §4.4.

- [ ] **Step 8.1: Locate the existing line that builds `blocks['text_overview'] = compute_text_embeddings(...)`. After it, but BEFORE `blocks_scaled, scalers = apply_modality_aware_scaling(blocks)`, insert:**

```python
# ─── §3 step 6: Director-profile data load ──────────────────────────────
print("\n[§3.6] Loading director-profile CSVs...")
bios_df_raw, langs_df_raw, lang_to_country_df = load_director_profile_csvs(CONFIG['paths'])
bios_df, langs_df = normalize_director_in_profile_dfs(bios_df_raw, langs_df_raw)
print(f"   bios:  {len(bios_df):>5} directors with bio")
print(f"   langs: {len(langs_df):>5} director-language rows ({langs_df['director_name_norm'].nunique()} unique directors)")

# ─── §3 step 7: Director-bio embedding + PCA ────────────────────────────
print("\n[§3.7] Director-bio embedding + PCA...")
bio_emb_full = compute_director_bio_embeddings(
    bios_df,
    model_name=CONFIG['embedding_model'],
    cache_path=CONFIG['bio_embedding_cache'],
    meta_path=CONFIG['bio_embedding_meta'],
    batch_size=CONFIG['embedding_batch_size'],
)
bio_pca, pca_director = pca_reduce_director_bios(
    bio_emb_full, n_components=CONFIG['bio_pca_n_components'], random_state=CONFIG['seed'],
)
print(f"   bio embedding shape: {bio_emb_full.shape}")
print(f"   bio PCA shape:       {bio_pca.shape}")
print(f"   PCA explained var:   {pca_director.explained_variance_ratio_.sum():.3f}")

# ─── §3 step 8: Build director_profile block ────────────────────────────
print("\n[§3.8] Building director_profile block...")
blocks['director_profile'] = engineer_director_profile(
    df, bios_df, bio_pca, langs_df, lang_to_country_df,
    top_n_lang=CONFIG['top_n_languages'],
    top_n_country=CONFIG['top_n_director_country'],
    bio_pca_dim=CONFIG['bio_pca_n_components'],
)
print(f"   director_profile block shape: {blocks['director_profile'].shape}")
print(f"   has_director_bio  count: {int(blocks['director_profile']['has_director_bio'].sum()):,}")
print(f"   has_director_lang count: {int(blocks['director_profile']['has_director_lang'].sum()):,}")
```

- [ ] **Step 8.2: Run the entire §3 cell end-to-end.**

Expected stdout (last few lines, exact numbers may differ):
```
[§3.6] Loading director-profile CSVs...
   bios:    505 directors with bio       (any number ≥ 500 is fine)
   langs:  85881 director-language rows (85865 unique directors)

[§3.7] Director-bio embedding + PCA...
   [cache MISS] computing 505 director bio embeddings ...
   bio embedding shape: (505, 384)
   bio PCA shape:       (505, 64)
   PCA explained var:   0.78x        (must be ≥ 0.75 — see Task 9 assertion)

[§3.8] Building director_profile block...
   director_profile block shape: (329044, 113)
   has_director_bio  count: ~50,000 to ~100,000
   has_director_lang count: ~310,000 to ~325,000
```

If the bio embedding shape isn't `(~505, 384)`, re-check Task 2's empty-bio drop. If `has_director_bio count == 0`, the join key is wrong — re-check Task 3's normalization.

- [ ] **Step 8.3: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): wire director_profile into §3 execution flow"
```

---

## Task 9 — Update §3 final assertions (shape and modality balance)

**Files:** Modify `eda_v2.ipynb`, edit the §3 cell — the trailing assertion block.

- [ ] **Step 9.1: Locate the existing final assertion `assert_shape(X, (len(df), 451), 'X')`. Replace 451 with 564.**

```python
assert_shape(X, (len(df), 564), 'X')   # was 451 — director_profile adds 113 dims
```

- [ ] **Step 9.2: Locate the existing modality-balance assertion. The parent spec uses `var(text)/var(others) ∈ [0.5, 2.0]`. Update both bounds and the variance computation to include the new bio_pca dims as part of "text".**

Replace whatever the parent cell has with:

```python
# ─── Modality balance check (extended for director_profile) ──────────────
# "text" now includes both overview embedding + bio_pca cols
text_overview_var = blocks_scaled['text_overview'].values.var(axis=0).sum()
bio_pca_cols = [c for c in blocks_scaled['director_profile'].columns if c.startswith('dir_bio_pca_')]
bio_text_var = blocks_scaled['director_profile'][bio_pca_cols].values.var(axis=0).sum()
text_total_var = text_overview_var + bio_text_var

# "others" = everything except the two text components
others_var = 0.0
for name, blk in blocks_scaled.items():
    if name == 'text_overview':
        continue
    if name == 'director_profile':
        non_bio_cols = [c for c in blk.columns if not c.startswith('dir_bio_pca_')]
        others_var += blk[non_bio_cols].values.var(axis=0).sum()
    else:
        others_var += blk.values.var(axis=0).sum()

ratio = text_total_var / max(others_var, 1e-9)
print(f"\n[modality balance] var(text)/var(others) = {ratio:.3f}  (target: [0.5, 2.5])")
assert 0.5 <= ratio <= 2.5, f"modality balance ratio {ratio:.3f} outside [0.5, 2.5]"
```

- [ ] **Step 9.3: Locate the §3 sanity-print cell (the one that surfaces awards merge rate, has_vote count, etc.). Append three new headline lines:**

```python
n = len(df)
print(f"   has_director_bio          : {int(df_dir_block['has_director_bio'].sum()):>7,}  ({df_dir_block['has_director_bio'].mean()*100:.1f}%)")
print(f"   has_director_lang         : {int(df_dir_block['has_director_lang'].sum()):>7,}  ({df_dir_block['has_director_lang'].mean()*100:.1f}%)")
print(f"   bio PCA explained variance: {pca_director.explained_variance_ratio_.sum()*100:>6.1f}%")
```

(`df_dir_block` is just an alias — at the top of the sanity-print cell, do `df_dir_block = blocks['director_profile']`.)

- [ ] **Step 9.4: Re-run the §3 cell. Confirm:**
  - `assert_shape(X, (329044, 564), 'X')` passes
  - The modality balance ratio is printed and within [0.5, 2.5]
  - The sanity-print shows `has_director_bio ≥ 10%` and `has_director_lang ≥ 95%` and `bio PCA explained var ≥ 75%`. If any of these fall outside their target, the spec's §10 risk-and-mitigation table dictates the next action.

- [ ] **Step 9.5: Commit.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): assert (329044, 564) and updated modality balance"
```

---

## Task 10 — Add 5 new figures to §4

**Files:** Modify `eda_v2.ipynb`, append cells to the §4 visualizations section.

Each figure is its own cell to keep them re-runnable independently.

- [ ] **Step 10.1: F1 — Director-bio coverage by decade.**

```python
# §4.1.bis  Director-bio coverage by decade
fig, ax = plt.subplots(figsize=(11, 4))

dec_df = df.copy()
dec_df['has_dir_bio'] = blocks['director_profile']['has_director_bio'].values
dec_df = dec_df[dec_df['decade'].between(1900, 2020)]
cov = dec_df.groupby('decade')['has_dir_bio'].agg(['sum', 'count'])
cov['pct'] = cov['sum'] / cov['count'] * 100

ax.bar(cov.index.astype(int).astype(str), cov['pct'], color='#5DCAA5', alpha=0.85, edgecolor='white')
overall = blocks['director_profile']['has_director_bio'].mean() * 100
ax.axhline(overall, color='red', linestyle='--', alpha=0.7, label=f'Overall: {overall:.1f}%')
ax.set_ylabel('% of films with director bio')
ax.set_xlabel('Decade')
ax.set_title('Director Wikipedia-bio coverage at film level')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'director_bio_coverage.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 10.2: F2 — Bio PCA scree plot.**

```python
# §4.6.bis  Bio PCA scree
pca_full = PCA(n_components=min(150, bio_emb_full.shape[1]), random_state=42).fit(bio_emb_full)
cum = np.cumsum(pca_full.explained_variance_ratio_) * 100
n75 = int(np.argmax(cum >= 75)) + 1
n80 = int(np.argmax(cum >= 80)) + 1
n90 = int(np.argmax(cum >= 90)) + 1

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(cum) + 1), cum, marker='o', markersize=3, color='#4C72B0')
ax.axvline(64, color='black', linestyle=':', alpha=0.6, label='Selected: 64 PCs')
ax.axhline(75, color='red',    linestyle='--', alpha=0.6, label=f'75% → {n75} PCs')
ax.axhline(80, color='orange', linestyle='--', alpha=0.6, label=f'80% → {n80} PCs')
ax.axhline(90, color='green',  linestyle='--', alpha=0.6, label=f'90% → {n90} PCs')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Cumulative explained variance (%)')
ax.set_title('Director-bio embedding — PCA variance explained')
ax.legend()
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'director_bio_pca_scree.png', bbox_inches='tight')
plt.show()
print(f"At 64 PCs: {cum[63]:.1f}% explained variance")
```

- [ ] **Step 10.3: F3 — Bio PCA 2D scatter colored by country.**

```python
# §4.6.bis  Bio PCA 2D scatter — colored by country
pca_2d = PCA(n_components=2, random_state=42).fit_transform(bio_emb_full)

# Map each bio row → country via the dominant lang
l2c_map = dict(zip(lang_to_country_df['language'], lang_to_country_df['country']))
dom_lang = (
    langs_df.sort_values('nb', ascending=False)
    .drop_duplicates('director_name_norm', keep='first')
    .set_index('director_name_norm')['original_language']
)
bio_country = bios_df['director_name_norm'].map(dom_lang).map(l2c_map).fillna('Unknown')

top_countries = bio_country.value_counts().head(8).index.tolist()
bio_country_grouped = bio_country.where(bio_country.isin(top_countries), 'Other')

fig, ax = plt.subplots(figsize=(10, 7))
palette_8 = sns.color_palette('tab10', n_colors=len(top_countries) + 1)
for i, c in enumerate(top_countries + ['Other']):
    mask = (bio_country_grouped.values == c)
    ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], color=palette_8[i],
               label=f'{c} ({mask.sum()})', alpha=0.7, s=25)
ax.set_title('Director-bio PCA 2D — colored by country (top-8 + other)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(markerscale=1.2, fontsize=9, loc='best')
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'director_bio_pca_2d_scatter.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 10.4: F4 — Director-lang vs film-lang confusion heatmap.**

```python
# §4.4.bis  Director's dominant language vs film's original language
top_n = 12

# 1. Build per-director dominant lang, then broadcast to films
dir_lang_per_director = (
    langs_df.sort_values('nb', ascending=False)
    .drop_duplicates('director_name_norm', keep='first')
    .set_index('director_name_norm')['original_language']
)
film_dir_lang = df['director_name_norm'].map(dir_lang_per_director)

# 2. Top-N vocabularies (independent for each axis)
top_dir_langs  = dir_lang_per_director.value_counts().head(top_n).index
top_film_langs = df['original_language'].value_counts().head(top_n).index

dir_lang_grouped  = film_dir_lang.where(film_dir_lang.isin(top_dir_langs), 'other').fillna('—')
film_lang_grouped = df['original_language'].where(df['original_language'].isin(top_film_langs), 'other').fillna('—')

# 3. Crosstab — row-normalized (each row sums to 1)
confusion = pd.crosstab(dir_lang_grouped, film_lang_grouped, normalize='index')

# 4. Plot
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(confusion, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Row-normalized fraction'}, ax=ax)
ax.set_xlabel('Film original_language')
ax.set_ylabel("Director's dominant language")
ax.set_title('Director language × Film language — row-normalized')
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'director_lang_vs_film_lang.png', bbox_inches='tight')
plt.show()

# 5. Off-diagonal mass — restrict to languages present on BOTH axes for a fair diagonal
common_langs = sorted(set(confusion.index) & set(confusion.columns))
common_block = confusion.loc[common_langs, common_langs]
diag_mass    = float(np.diag(common_block.values).sum())
total_mass   = float(common_block.values.sum())
off_diag_pct = (1.0 - diag_mass / total_mass) * 100 if total_mass > 0 else 0.0
print(f"Off-diagonal mass over shared lang grid: {off_diag_pct:.1f}%   (target: ≥ 5% — feature is non-redundant)")
```

- [ ] **Step 10.5: F5 — Modality balance v2 (six modalities including director_profile).**

```python
# §4.5.bis  Modality balance — 6 blocks
modality_groups = {
    'Numerical\n(6)':         blocks_scaled['numerical'].values,
    'Genre\n(22)':            blocks_scaled['genre'].values,
    'Language\n(31)':         blocks_scaled['language'].values,
    'Decade\n(2)':            blocks_scaled['decade'].values,
    'Awards\n(6)':            blocks_scaled['awards'].values,
    'Text overview\n(384)':   blocks_scaled['text_overview'].values,
    'Director profile\n(113)': blocks_scaled['director_profile'].values,
}

variances = {n: float(v.var(axis=0).sum()) for n, v in modality_groups.items()}

fig, ax = plt.subplots(figsize=(11, 5))
colors_mod = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#5DCAA5']
ax.bar(list(variances.keys()), list(variances.values()), color=colors_mod, alpha=0.85, edgecolor='white')
for i, (k, v) in enumerate(variances.items()):
    ax.text(i, v + max(variances.values()) * 0.01, f'{v:.1f}', ha='center', fontsize=9)
ax.set_ylabel('Total variance (post-scaling)')
ax.set_title('Modality balance — variance contribution (6 modalities)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'modality_balance_v2.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 10.6: Run all 5 new figure cells in order. Verify all 5 PNG files exist.**

```bash
ls artifacts/figures/director_bio_coverage.png artifacts/figures/director_bio_pca_scree.png artifacts/figures/director_bio_pca_2d_scatter.png artifacts/figures/director_lang_vs_film_lang.png artifacts/figures/modality_balance_v2.png
```

- [ ] **Step 10.7: Commit.**

```bash
git add eda_v2.ipynb artifacts/figures/director_bio_coverage.png artifacts/figures/director_bio_pca_scree.png artifacts/figures/director_bio_pca_2d_scatter.png artifacts/figures/director_lang_vs_film_lang.png artifacts/figures/modality_balance_v2.png
git commit -m "feat(eda_v2): add 5 director_profile figures (F1-F5)"
```

---

## Task 11 — Update §5 persistence (save new artifacts)

**Files:** Modify `eda_v2.ipynb`, edit the existing `§5 Persistence` cell.

The bio embeddings are already cached by Task 4 (auto-saved on cold path). Two new artifacts need explicit saves: the fitted PCA and the metadata JSON.

- [ ] **Step 11.1: At the end of the §5 cell, append:**

```python
# ─── Director-profile artifacts ──────────────────────────────────────────
import pickle

with open(CONFIG['bio_pca_cache'], 'wb') as f:
    pickle.dump(pca_director, f)

director_profile_meta = {
    'n_directors_with_bio': int(len(bios_df)),
    'film_coverage_bio': float(blocks['director_profile']['has_director_bio'].mean()),
    'film_coverage_lang': float(blocks['director_profile']['has_director_lang'].mean()),
    'bio_pca_n_components': int(pca_director.n_components_),
    'bio_pca_explained_variance': float(pca_director.explained_variance_ratio_.sum()),
    'top_languages': [c.replace('dir_lang_', '') for c in blocks['director_profile'].columns if c.startswith('dir_lang_')],
    'top_countries': [c.replace('dir_country_', '') for c in blocks['director_profile'].columns if c.startswith('dir_country_')],
}
CONFIG['director_profile_meta'].write_text(
    json.dumps(director_profile_meta, indent=2)
)

print(f"\n✅ Director-profile artifacts saved:")
print(f"   {CONFIG['bio_embedding_cache']}")
print(f"   {CONFIG['bio_embedding_meta']}")
print(f"   {CONFIG['bio_pca_cache']}")
print(f"   {CONFIG['director_profile_meta']}")
```

- [ ] **Step 11.2: Locate the existing `feature_metadata.json` save block and verify it picks up the new column names automatically (it should iterate over `feature_names`, which now includes the 113 director_profile cols). If the parent's persistence step hard-codes block names, append the director_profile names explicitly.**

- [ ] **Step 11.3: Run the §5 cell. Verify all four files in stdout exist on disk.**

- [ ] **Step 11.4: Commit.**

```bash
git add eda_v2.ipynb artifacts/director_bio_pca.pkl artifacts/director_profile_metadata.json
git commit -m "feat(eda_v2): persist director_profile artifacts (PCA + metadata)"
```

---

## Task 12 — Reproducibility verification

**Files:** Run-only — no code changes.

- [ ] **Step 12.1: Run `Restart kernel & run all` from the notebook menu.**

Expected end-to-end: cold (no caches) → 30–60 minutes if both embeddings are cold; warm (caches present) → under 2 minutes.

- [ ] **Step 12.2: Capture the MD5 of `feature_matrix.npz`:**

```bash
md5 artifacts/feature_matrix.npz   # macOS
# or: md5sum artifacts/feature_matrix.npz   # Linux
```

Note the hash.

- [ ] **Step 12.3: Run `Restart kernel & run all` again. Capture the MD5 again. Compare.**

The two hashes **must be identical**. If they differ, the deterministic pipeline has been broken — most likely culprit is a missing `random_state` somewhere in Task 5 or 6. Bisect to find the offending step.

- [ ] **Step 12.4: Final sanity check — print run summary.**

Add a final cell at the bottom of the notebook:

```python
# Run summary
print("=" * 60)
print("CineEmbed eda_v2 — Director-profile Extension Run Summary")
print("=" * 60)
print(f"Feature matrix shape : {X.shape}")
print(f"Expected             : (329044, 564)")
print(f"NaN count            : {int(np.isnan(X).sum())}")
print(f"Director bio coverage: {blocks['director_profile']['has_director_bio'].mean()*100:.1f}%")
print(f"Director lang coverage: {blocks['director_profile']['has_director_lang'].mean()*100:.1f}%")
print(f"PCA explained var    : {pca_director.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Modality balance     : {ratio:.3f}  (target: [0.5, 2.5])")
print(f"Artifacts written    : {len(list(CONFIG['artifacts_dir'].rglob('*'))):>3} files")
print("=" * 60)
```

- [ ] **Step 12.5: Commit and tag.**

```bash
git add eda_v2.ipynb
git commit -m "feat(eda_v2): final run summary + reproducibility verified"
git tag eda_v2-director-profile-v1
```

---

## Verification Against Spec

After completing all 12 tasks, walk through the spec's success criteria one by one:

| Spec criterion | Verification |
|---|---|
| (Parent's six criteria) | Already verified in parent plan |
| Director bio film-level coverage ≥ 10% | Task 8.2 stdout + Task 12.4 summary |
| Director bio PCA explained variance ≥ 75% | Task 8.2 stdout + Task 9.4 sanity print |
| Director-lang ≠ film-lang in ≥ 5% of rows | Task 10.4 stdout (off-diagonal mass) |
| Modality balance ratio in [0.5, 2.5] | Task 9 assertion + Task 12.4 summary |
| Reproducibility (identical MD5 across two runs) | Task 12.3 |

If any criterion fails, see spec §10 (Risk and Mitigation) for the planned fallback action.
