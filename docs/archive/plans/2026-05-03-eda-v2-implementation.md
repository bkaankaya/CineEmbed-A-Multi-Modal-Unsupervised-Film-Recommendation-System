# `eda_v2.ipynb` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `eda_v2.ipynb` — a pipeline-first EDA notebook for the CineEmbed project that applies all 13 fixes identified in the audit and produces a clean (329044, 451) feature matrix plus supporting artifacts ready to feed AE/VAE/DEC training.

**Architecture:** Single Jupyter notebook with two halves. Top half (§1-§2.6) defines pure, deterministic, idempotent functions. Bottom half (§3-§5) calls them, visualises, and persists. Two expensive steps (text embedding, full feature matrix) are file-cached so warm-start runs complete in under two minutes. Notebook-friendly TDD: each function has a sanity-test cell with synthetic mock data immediately after its definition.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), umap-learn, matplotlib, seaborn, scipy, torch (for sentence-transformers backend). Target environment: Google Colab T4 GPU; falls back to CPU on local machines.

**Reference:** [`docs/superpowers/specs/2026-05-03-eda-v2-design.md`](../specs/2026-05-03-eda-v2-design.md)

---

## File Structure

| Path | Status | Purpose |
|---|---|---|
| `eda_v2.ipynb` | NEW | The deliverable notebook |
| `data/AllMoviesDetailsCleaned.csv` | MOVE | Source (currently in project root) |
| `data/AllMoviesCastingRaw.csv` | MOVE | Source |
| `data/220k_awards_by_directors.csv` | MOVE | Source |
| `artifacts/` | NEW dir | Pipeline outputs |
| `artifacts/figures/` | NEW dir | PNGs (23 total) |
| `.gitignore` | NEW | Ignore artifacts, checkpoints, __pycache__ |

The original notebook (`Deep_Learning_EDA_ipynb_adlı_not_defterinin_kopyası.ipynb`) and presentation (`SENG474_Presentation_Team3.pptx`) are **not modified**.

---

## Notebook Testing Pattern

Notebooks don't run pytest naturally. The plan uses **inline test cells** as the TDD substitute:

1. **Add a test cell** below where the function will go. The cell creates a small synthetic DataFrame, calls the function, and asserts on the output. Run it → fails with `NameError` because the function isn't defined yet.
2. **Add the function cell** with the implementation. Run it.
3. **Re-run the test cell** → passes.
4. **Commit**.

This pattern is repeated for every §2 function. The §3 spot-check cell at the bottom of the pipeline section serves as the integration test against real data.

---

## Task 1 — Project Setup, Directory Layout, and Notebook Shell

**Files:**
- Create: `data/` directory; move three CSVs into it
- Create: `artifacts/` directory
- Create: `artifacts/figures/` directory
- Create: `eda_v2.ipynb` (with §1 Setup cell only)
- Create: `.gitignore`
- Create: git repository (init)

- [ ] **Step 1.1: Create directories and move CSVs**

```bash
cd "<repo-root>"
mkdir -p data artifacts/figures
mv AllMoviesDetailsCleaned.csv data/
mv AllMoviesCastingRaw.csv data/
mv 220k_awards_by_directors.csv data/
ls data/
```

Expected: three CSV files listed in `data/`.

**Note:** If the CSV files are not in the project root, locate them first (`find . -name "*.csv"`). They may already be in `data/` from a prior session — in that case, skip the `mv` commands.

- [ ] **Step 1.2: Create `.gitignore`**

```bash
cat > .gitignore <<'EOF'
artifacts/
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
.env
EOF
```

- [ ] **Step 1.3: Initialize git repository**

```bash
cd "<repo-root>"
git init
git add .gitignore docs/
git commit -m "chore: initial project structure with spec and plan"
```

Expected: git initialized, first commit landed. If user has already initialized git, skip the `git init` and just commit.

- [ ] **Step 1.4: Create `eda_v2.ipynb` with §1 Setup cell**

Use NotebookEdit (or jupyter) to create `eda_v2.ipynb`. The notebook starts with a markdown cell (the title) and one code cell (Setup). Below is the exact content for Cell 1 (markdown) and Cell 2 (code):

**Cell 1 (markdown):**
```markdown
# CineEmbed — EDA v2

Pipeline-first refresh of the EDA notebook. Applies all 13 fixes from the audit (see `docs/superpowers/specs/2026-05-03-eda-v2-design.md`). Produces a clean (329044, 451) feature matrix.

**Sections:**
- §1 Setup & Reproducibility
- §2 Pipeline Function Definitions
- §3 Pipeline Execution
- §4 EDA Visualizations
- §5 Persistence
```

**Cell 2 (code) — §1 Setup & Reproducibility:**
```python
# §1 — Setup & Reproducibility
import os, json, hashlib, random, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy import stats

# Optional GPU stack — only imported if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CONFIG = {
    'seed': 42,
    'data_dir': Path('data'),
    'artifacts_dir': Path('artifacts'),
    'figures_dir': Path('artifacts/figures'),

    # File paths
    'paths': {
        'details': Path('data/AllMoviesDetailsCleaned.csv'),
        'casting': Path('data/AllMoviesCastingRaw.csv'),
        'awards':  Path('data/220k_awards_by_directors.csv'),
    },

    # Feature engineering knobs (single source of truth — fix #11 + clean ablation)
    'top_n_genres': 20,
    'top_n_languages': 30,
    'q99_clip_threshold': 0.99,
    'runtime_clip': (10, 300),

    # Embedding model (fix #5 — multilingual)
    'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
    'embedding_batch_size': 64,
    'embedding_dim': 384,
    'embedding_cache': Path('artifacts/text_embeddings.npy'),
    'embedding_meta': Path('artifacts/text_embeddings.meta.json'),
}

CONFIG['artifacts_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['figures_dir'].mkdir(parents=True, exist_ok=True)

seed_everything(CONFIG['seed'])

# Reproducibility self-check
_check = np.random.rand(3)
print("✅ §1 Setup complete")
print(f"   seed = {CONFIG['seed']}")
print(f"   np.random sample (deterministic) = {_check}")
print(f"   torch available = {HAS_TORCH}")
```

- [ ] **Step 1.5: Run Cell 2 and verify deterministic output**

Run the cell. Expected stdout:
```
✅ §1 Setup complete
   seed = 42
   np.random sample (deterministic) = [0.37454012 0.95071431 0.73199394]
   torch available = True   (or False on machines without torch)
```

If the random sample does not match `[0.37454012 0.95071431 0.73199394]`, `seed_everything` is broken — debug before proceeding.

- [ ] **Step 1.6: Commit**

```bash
git add eda_v2.ipynb .gitignore
git commit -m "feat: scaffold eda_v2.ipynb with §1 Setup and CONFIG"
```

---

## Task 2 — §2.1 Data Layer (load + director-name normalization + merge)

**Files:**
- Modify: `eda_v2.ipynb` (append cells for §2.1 functions and their tests)

This task adds three pure functions: `load_csvs`, `normalize_director_name`, `merge_details_casting`.

- [ ] **Step 2.1: Add markdown cell announcing §2.1**

```markdown
## §2.1 — Data Layer
Pure functions: `load_csvs`, `normalize_director_name`, `merge_details_casting`. Each function has a sanity-test cell immediately after.
```

- [ ] **Step 2.2: Add test cell for `normalize_director_name` (will fail)**

```python
# Test §2.1: normalize_director_name (fix #8)
_cases = [
    ('Steven Spielberg',   'steven spielberg'),
    ('Spielberg, Steven',  'steven spielberg'),
    ('  Pedro  Almodóvar ', 'pedro almodovar'),
    ('Léa  Pool',          'lea pool'),
    (None,                 ''),
    ('',                   ''),
    ('SCORSESE, MARTIN',   'martin scorsese'),
]
for raw, expected in _cases:
    got = normalize_director_name(raw)
    assert got == expected, f"normalize_director_name({raw!r}) = {got!r} != {expected!r}"
print(f"✅ normalize_director_name: {len(_cases)} cases pass")
```

Run it. Expected: `NameError: name 'normalize_director_name' is not defined`.

- [ ] **Step 2.3: Add `normalize_director_name` implementation cell ABOVE the test cell**

```python
# §2.1 — Data layer functions
import unicodedata

def normalize_director_name(name: str | None) -> str:
    """Stable director key for joins.

    Steps:
      1. None / NaN / empty → ''
      2. Unicode NFKD decompose, strip combining marks (accents)
      3. Swap "Last, First" → "First Last"
      4. Lowercase, collapse internal whitespace, strip ends.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ''
    s = str(name).strip()
    if not s:
        return ''
    # NFKD + ascii filter (drops accents)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # "Last, First" → "First Last"
    if ',' in s:
        parts = [p.strip() for p in s.split(',', 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    # collapse whitespace, lowercase
    s = ' '.join(s.split()).lower()
    return s
```

Re-run the test cell from Step 2.2. Expected: `✅ normalize_director_name: 7 cases pass`.

- [ ] **Step 2.4: Add `load_csvs` cell**

```python
def load_csvs(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three source CSVs with their correct separators.

    details, casting use ';' (TMDB-derived); awards uses ','.
    """
    details = pd.read_csv(paths['details'], sep=';', low_memory=False)
    casting = pd.read_csv(paths['casting'], sep=';', low_memory=False)
    awards  = pd.read_csv(paths['awards'],  sep=',', low_memory=False)
    return details, casting, awards
```

(No isolated test for this — it's I/O. End-to-end verification happens in §3.)

- [ ] **Step 2.5: Add test cell for `merge_details_casting` (will fail)**

```python
# Test §2.1: merge_details_casting
_details = pd.DataFrame({'id': [1, 2, 3], 'title': ['A', 'B', 'C']})
_casting = pd.DataFrame({
    'id': [1, 2, 3],
    'director_name': ['Steven Spielberg', 'Spielberg, Steven', None],
    'director_gender': [2, 2, 0],
})
_merged = merge_details_casting(_details, _casting)
assert list(_merged.columns) >= ['id', 'title', 'director_name', 'director_name_norm', 'director_gender']
assert _merged.loc[0, 'director_name_norm'] == 'steven spielberg'
assert _merged.loc[1, 'director_name_norm'] == 'steven spielberg'   # normalized form merges
assert _merged.loc[2, 'director_name_norm'] == ''
print("✅ merge_details_casting passes")
```

Run it. Expected: `NameError`.

- [ ] **Step 2.6: Add `merge_details_casting` implementation**

```python
def merge_details_casting(details: pd.DataFrame, casting: pd.DataFrame) -> pd.DataFrame:
    """Inner-shape merge on `id`. Adds `director_name_norm` (fix #8) for award joins.

    Carries `director_name` (raw) for human readability and `director_name_norm`
    for joins. Only the necessary casting columns are pulled in to avoid bloat.
    """
    casting_slim = casting[['id', 'director_name', 'director_gender']].copy()
    casting_slim['director_name_norm'] = casting_slim['director_name'].apply(normalize_director_name)
    merged = details.merge(casting_slim, on='id', how='left')
    return merged
```

Re-run the test cell. Expected: `✅ merge_details_casting passes`.

- [ ] **Step 2.7: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.1): data layer — load, normalize_director_name (fix #8), merge"
```

---

## Task 3 — §2.2 Awards Layer (temporal-aware aggregation, regex word-boundary)

**Files:**
- Modify: `eda_v2.ipynb`

Implements fixes #1 (temporal leak) and #9 (regex word-boundary).

- [ ] **Step 3.1: Add markdown cell**

```markdown
## §2.2 — Awards Layer
Temporal-aware per-film aggregation. Implements fixes #1 (temporal leak) and #9 (Oscar/Palme regex word-boundary).
```

- [ ] **Step 3.2: Add test cell for `aggregate_awards_temporal` (will fail)**

```python
# Test §2.2: aggregate_awards_temporal
# Synthetic awards: director "alice" has 2 wins (1995, 2010), 1 nom (2005)
# Director "bob" has 1 oscar win (2000) and 1 "Oscar Wilde Award" nom (1998 — false positive trap)
_awards = pd.DataFrame({
    'director_name': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob'],
    'category':      ['Best Director', 'Best Picture', 'Best Director — Oscar',
                      'Academy Award (Oscar)', 'Oscar Wilde Award'],
    'outcome':       ['Won', 'Nominated', 'Won', 'Won', 'Nominated'],
    'year':          [1995, 2005, 2010, 2000, 1998],
})
_films = pd.DataFrame({
    'id': [101, 102, 103, 104],
    'director_name_norm': ['alice', 'alice', 'bob', 'bob'],
    'release_date': pd.to_datetime(['1990-01-01', '2008-01-01', '1995-01-01', '2005-01-01']),
})
_agg = aggregate_awards_temporal(_awards, _films)

# Assertions (one per fix):
# Fix #1 — temporal: alice 1990 film sees 0 wins; alice 2008 film sees 1 win (1995)
row_a1990 = _agg.set_index('id').loc[101]
row_a2008 = _agg.set_index('id').loc[102]
assert row_a1990['prior_total_wins'] == 0,  f"1990 leak: {row_a1990['prior_total_wins']}"
assert row_a2008['prior_total_wins'] == 1,  f"2008 expects 1 win, got {row_a2008['prior_total_wins']}"

# Fix #9 — word-boundary: bob 1995 sees 0 oscar wins (Wilde shouldn't count, win is 2000)
# bob 2005 sees 1 oscar win (2000) and 0 oscar noms (Wilde is filtered out)
row_b1995 = _agg.set_index('id').loc[103]
row_b2005 = _agg.set_index('id').loc[104]
assert row_b1995['prior_oscar_wins'] == 0
assert row_b2005['prior_oscar_wins'] == 1
assert row_b2005['prior_oscar_nominations'] == 0, \
    f"Wilde should not count, got {row_b2005['prior_oscar_nominations']}"

print("✅ aggregate_awards_temporal: temporal cutoff + regex word-boundary work")
```

Run it. Expected: `NameError`.

- [ ] **Step 3.3: Add `aggregate_awards_temporal` implementation**

```python
import re

OSCAR_RE = re.compile(r'\bOscar\b', flags=re.IGNORECASE)
PALME_RE = re.compile(r'\bPalme\b', flags=re.IGNORECASE)


def aggregate_awards_temporal(
    awards: pd.DataFrame,
    films: pd.DataFrame,
) -> pd.DataFrame:
    """Per-film aggregation of director awards.

    Two fixes:
      • #1 (temporal): only awards with year ≤ film.release_year contribute.
      • #9 (regex):    Oscar/Palme detection uses \\b word-boundary.

    Returns a DataFrame with one row per film id and columns:
      prior_total_nominations, prior_total_wins,
      prior_oscar_nominations, prior_oscar_wins,
      prior_palme_nominations, prior_palme_wins
    """
    # Normalize director key on awards side
    awards = awards.copy()
    awards['director_name_norm'] = awards['director_name'].apply(normalize_director_name)

    # Year column (if missing, parse from a date-like field; here we trust 'year')
    awards['year'] = pd.to_numeric(awards['year'], errors='coerce')
    awards = awards.dropna(subset=['year'])
    awards['year'] = awards['year'].astype(int)

    # Annotate flags up-front (avoid per-row regex during the join)
    awards['is_won']   = (awards['outcome'].fillna('') == 'Won')
    awards['is_oscar'] = awards['category'].fillna('').str.contains(OSCAR_RE, regex=True)
    awards['is_palme'] = awards['category'].fillna('').str.contains(PALME_RE, regex=True)

    # Films side — derive year
    films = films.copy()
    films['release_year'] = pd.to_datetime(films['release_date'], errors='coerce').dt.year

    # For each (director_norm), pre-sort awards ascending by year for cumulative aggregation
    award_groups = awards.sort_values('year').groupby('director_name_norm')

    rows = []
    for _, film in films[['id', 'director_name_norm', 'release_year']].iterrows():
        out = {
            'id': film['id'],
            'prior_total_nominations': 0,
            'prior_total_wins': 0,
            'prior_oscar_nominations': 0,
            'prior_oscar_wins': 0,
            'prior_palme_nominations': 0,
            'prior_palme_wins': 0,
        }
        director = film['director_name_norm']
        year = film['release_year']
        if not director or pd.isna(year):
            rows.append(out)
            continue
        if director not in award_groups.groups:
            rows.append(out)
            continue
        sub = award_groups.get_group(director)
        sub = sub[sub['year'] <= year]
        if sub.empty:
            rows.append(out)
            continue
        out['prior_total_nominations']  = len(sub)
        out['prior_total_wins']         = int(sub['is_won'].sum())
        out['prior_oscar_nominations']  = int(sub['is_oscar'].sum())
        out['prior_oscar_wins']         = int((sub['is_oscar'] & sub['is_won']).sum())
        out['prior_palme_nominations']  = int(sub['is_palme'].sum())
        out['prior_palme_wins']         = int((sub['is_palme'] & sub['is_won']).sum())
        rows.append(out)

    return pd.DataFrame(rows)
```

Re-run the test cell. Expected: `✅ aggregate_awards_temporal: temporal cutoff + regex word-boundary work`.

- [ ] **Step 3.4: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.2): temporal-aware awards aggregation (fixes #1, #9)"
```

---

## Task 4 — §2.3a Numerical Block (fixes #2, #7, #10)

**Files:**
- Modify: `eda_v2.ipynb`

Builds the 6-column numerical block: `log_popularity`, `log_vote_count`, `runtime_norm`, `vote_average_norm`, `has_vote`, `has_engagement`.

- [ ] **Step 4.1: Add markdown cell**

```markdown
## §2.3 — Feature Engineering
Five engineer_* functions, one per modality. Each returns a DataFrame block aligned by row index with the master frame.

### §2.3.a — Numerical (fixes #2, #7, #10)
```

- [ ] **Step 4.2: Add test cell (will fail)**

```python
# Test §2.3.a: engineer_numerical
_df = pd.DataFrame({
    'popularity':   [0.0, 0.5, 1.0, 100.0, np.nan, 0.0],
    'vote_count':   [0,   10,  50,  10000, 5,      0],
    'runtime':      [120, 95,  np.nan, 5,   500,   90],
    'vote_average': [np.nan, 7.5, 8.0, 6.5, 5.0, np.nan],
})
_out = engineer_numerical(_df, CONFIG)

# fix #2: vote_average imputation
# Voted-only mean = mean of [7.5, 8.0, 6.5, 5.0] = 6.75
# Films with vote_count == 0 (rows 0 and 5) → vote_average filled with ~6.75
assert abs(_out.loc[0, 'vote_average_norm'] - _out.loc[5, 'vote_average_norm']) < 1e-9
# Both should NOT be the minimum (which would happen with median=0 imputation)
assert _out['vote_average_norm'].min() < _out.loc[0, 'vote_average_norm'] or \
       _out['vote_average_norm'].max() > _out.loc[0, 'vote_average_norm']

# fix #7: vote_count Q99 clip applied — log_vote_count for 10000 should not blow up
assert _out['log_vote_count'].max() < np.log1p(10001)

# fix #10: has_engagement flag
assert _out.loc[0, 'has_engagement'] == 0  # popularity=0 AND vote_count=0
assert _out.loc[1, 'has_engagement'] == 1  # vote_count=10 > 0
assert _out.loc[5, 'has_engagement'] == 0

# has_vote flag
assert _out.loc[0, 'has_vote'] == 0
assert _out.loc[1, 'has_vote'] == 1

# All 6 columns present, no NaN
expected_cols = {'log_popularity', 'log_vote_count', 'runtime_norm',
                 'vote_average_norm', 'has_vote', 'has_engagement'}
assert set(_out.columns) == expected_cols
assert _out.isna().sum().sum() == 0

print(f"✅ engineer_numerical: 6-col output, {len(_out)} rows, no NaN")
```

Run. Expected: `NameError`.

- [ ] **Step 4.3: Add `engineer_numerical` implementation**

```python
def engineer_numerical(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Numerical block — 6 columns.

    Fixes:
      #2 vote_average imputation — use mean of voted-only films, NOT median (which is 0)
      #7 vote_count Q99 clip + log (was: log only)
      #10 has_engagement flag
    """
    out = pd.DataFrame(index=df.index)

    pop = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    vc  = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    rt  = pd.to_numeric(df['runtime'], errors='coerce')
    va  = pd.to_numeric(df['vote_average'], errors='coerce')

    # Q99 clip then log1p
    pop_q99 = pop.quantile(config['q99_clip_threshold'])
    vc_q99  = vc.quantile(config['q99_clip_threshold'])
    out['log_popularity'] = np.log1p(pop.clip(upper=pop_q99).clip(lower=0))
    out['log_vote_count'] = np.log1p(vc.clip(upper=vc_q99).clip(lower=0))

    # runtime: cap to [10, 300] then min-max
    rt_lo, rt_hi = config['runtime_clip']
    rt_filled = rt.fillna(rt.median() if rt.notna().any() else (rt_lo + rt_hi) / 2)
    rt_capped = rt_filled.clip(lower=rt_lo, upper=rt_hi)
    out['runtime_norm'] = (rt_capped - rt_lo) / (rt_hi - rt_lo)

    # vote_average — fix #2 (smart imputation)
    voted_mask = vc > 0
    if voted_mask.any():
        imputed_value = va[voted_mask].mean()  # ~6.0 on real data
    else:
        imputed_value = 0.0
    va_filled = va.where(voted_mask & va.notna(), imputed_value)
    # min-max on [0, 10] natural scale
    out['vote_average_norm'] = (va_filled.clip(0, 10) / 10.0)

    # Flags — fixes #2, #10
    out['has_vote']        = voted_mask.astype(np.int8)
    out['has_engagement']  = ((pop > 0) | (vc > 0)).astype(np.int8)

    return out
```

Re-run the test cell. Expected: `✅ engineer_numerical: 6-col output, 6 rows, no NaN`.

- [ ] **Step 4.4: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.3a): numerical block (fixes #2, #7, #10)"
```

---

## Task 5 — §2.3b Genre + Language Blocks (fix #3 + categorical)

**Files:**
- Modify: `eda_v2.ipynb`

Builds the 22-column genre block and 31-column language block.

- [ ] **Step 5.1: Add markdown cell**

```markdown
### §2.3.b — Genre & Language (fix #3 — Unknown genre + has_genre flag)
```

- [ ] **Step 5.2: Add test cell for `engineer_genres` (will fail)**

```python
# Test §2.3.b: engineer_genres (fix #3)
_df = pd.DataFrame({
    'genres': ['Drama|Comedy', 'Drama', None, '', 'Action|Thriller', 'Drama|Comedy'],
})
_out = engineer_genres(_df, top_n=3)

# Top 3 by frequency: Drama (3), Comedy (2), Action (1) (or Thriller — tiebreak by alpha)
# So expected columns: genre_Drama, genre_Comedy, genre_Action OR Thriller, genre_Unknown, has_genre
expected_genre_cols = {'genre_Drama', 'genre_Comedy', 'genre_Unknown', 'has_genre'}
assert expected_genre_cols.issubset(set(_out.columns)), f"missing cols. got: {set(_out.columns)}"

# Fix #3: empty/None → genre_Unknown=1, has_genre=0
assert _out.loc[2, 'genre_Unknown'] == 1
assert _out.loc[3, 'genre_Unknown'] == 1
assert _out.loc[2, 'has_genre'] == 0
assert _out.loc[3, 'has_genre'] == 0

# Non-empty rows: has_genre = 1, genre_Unknown = 0
assert _out.loc[0, 'has_genre'] == 1
assert _out.loc[0, 'genre_Unknown'] == 0
assert _out.loc[0, 'genre_Drama'] == 1
assert _out.loc[0, 'genre_Comedy'] == 1

print(f"✅ engineer_genres: {len(_out.columns)} cols, fix #3 holds")
```

Run. Expected: `NameError`.

- [ ] **Step 5.3: Add `engineer_genres` implementation**

```python
def engineer_genres(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Genre block — top-N multi-hot + Unknown + has_genre flag (fix #3).

    Empty/None genre lists become ['Unknown'] AND has_genre = 0.
    """
    raw = df['genres'].fillna('').astype(str)
    genres_list = raw.apply(
        lambda x: [g.strip() for g in x.split('|') if g.strip()]
    )
    has_genre = genres_list.apply(lambda lst: 1 if lst else 0).astype(np.int8)

    # Pick top-N from non-empty rows
    counts = {}
    for lst in genres_list:
        for g in lst:
            counts[g] = counts.get(g, 0) + 1
    top_genres = [g for g, _ in sorted(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:top_n]]

    # Replace empty with ['Unknown'] for encoding
    genres_for_mlb = genres_list.apply(lambda lst: lst if lst else ['Unknown'])

    # Restrict to top_genres + 'Unknown'
    keep = set(top_genres) | {'Unknown'}
    genres_filtered = genres_for_mlb.apply(lambda lst: [g for g in lst if g in keep] or ['Unknown'])

    mlb = MultiLabelBinarizer(classes=sorted(keep))
    encoded = mlb.fit_transform(genres_filtered)
    out = pd.DataFrame(
        encoded,
        columns=[f'genre_{g}' for g in mlb.classes_],
        index=df.index,
        dtype=np.int8,
    )
    out['has_genre'] = has_genre.values
    return out
```

Re-run test cell. Expected: `✅ engineer_genres: 5 cols, fix #3 holds`.

- [ ] **Step 5.4: Add test cell for `engineer_languages` (will fail)**

```python
# Test §2.3.b: engineer_languages
_df = pd.DataFrame({
    'original_language': ['en', 'en', 'fr', 'tr', 'jp', 'unknown_lang_xyz', None],
})
_out = engineer_languages(_df, top_n=3)
# Top 3: en (2), fr (1), jp/tr (1) — alpha tiebreak
expected_subset = {'lang_en', 'lang_fr', 'lang_other'}
assert expected_subset.issubset(set(_out.columns))
assert _out.loc[5, 'lang_other'] == 1   # unknown_lang_xyz
assert _out.loc[6, 'lang_other'] == 1   # None

# Each row has exactly one '1' across all lang columns (one-hot)
assert (_out.sum(axis=1) == 1).all()
print(f"✅ engineer_languages: one-hot, {len(_out.columns)} cols")
```

Run. Expected: `NameError`.

- [ ] **Step 5.5: Add `engineer_languages` implementation**

```python
def engineer_languages(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Language block — top-N + 'other' one-hot."""
    raw = df['original_language'].fillna('other').astype(str)
    counts = raw.value_counts()
    top_langs = counts.head(top_n).index.tolist()
    grouped = raw.where(raw.isin(top_langs), other='other')
    out = pd.get_dummies(grouped, prefix='lang', dtype=np.int8)
    # Ensure 'lang_other' column exists even if no rows fell out of top-N
    if 'lang_other' not in out.columns:
        out['lang_other'] = np.int8(0)
    return out
```

Re-run test cell. Expected: `✅ engineer_languages: one-hot, 4 cols` (3 langs + lang_other).

- [ ] **Step 5.6: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.3b): genre block (fix #3) + language block"
```

---

## Task 6 — §2.3c Decade Block + Director Awards Block (fixes #6, awards from §2.2)

**Files:**
- Modify: `eda_v2.ipynb`

Builds the 2-column decade block (fix #6) and 6-column awards block.

- [ ] **Step 6.1: Add markdown cell**

```markdown
### §2.3.c — Decade & Director Awards (fix #6)
```

- [ ] **Step 6.2: Add test cell for `engineer_decade` (will fail)**

```python
# Test §2.3.c: engineer_decade (fix #6)
_df = pd.DataFrame({
    'release_date': ['1990-05-01', '2020-01-01', '1900-01-01', None, 'invalid'],
})
_df['release_date'] = pd.to_datetime(_df['release_date'], errors='coerce')
_out = engineer_decade(_df)

assert set(_out.columns) == {'decade_norm', 'has_release_date'}
# (1990 - 1900) / 130 = 0.6923...
assert abs(_out.loc[0, 'decade_norm'] - (90/130)) < 1e-9
# (2020 - 1900) / 130 = 0.923...
assert abs(_out.loc[1, 'decade_norm'] - (120/130)) < 1e-9
# 1900 → 0.0
assert _out.loc[2, 'decade_norm'] == 0.0
# Missing/invalid → decade_norm = 0.0 AND has_release_date = 0
assert _out.loc[3, 'decade_norm'] == 0.0
assert _out.loc[3, 'has_release_date'] == 0
assert _out.loc[4, 'decade_norm'] == 0.0
assert _out.loc[4, 'has_release_date'] == 0
# Valid rows → has_release_date = 1
assert _out.loc[0, 'has_release_date'] == 1
assert _out.loc[2, 'has_release_date'] == 1

print("✅ engineer_decade: fix #6 (normalized + has_release_date) holds")
```

Run. Expected: `NameError`.

- [ ] **Step 6.3: Add `engineer_decade` implementation**

```python
def engineer_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Decade block — fix #6.

    For films with parseable release_date:
      decade_norm = (decade - 1900) / 130, where decade = year // 10 * 10.
    For films without:
      decade_norm = 0.0 AND has_release_date = 0 — model can learn to ignore decade.
    """
    rd = pd.to_datetime(df['release_date'], errors='coerce')
    has = rd.notna()
    year = rd.dt.year.where(has, other=np.nan)
    decade = (year // 10 * 10).where(has, other=np.nan)
    decade_norm = ((decade - 1900) / 130).where(has, other=0.0)

    out = pd.DataFrame({
        'decade_norm': decade_norm.astype(float).values,
        'has_release_date': has.astype(np.int8).values,
    }, index=df.index)
    return out
```

Re-run test. Expected: `✅ engineer_decade: fix #6 (normalized + has_release_date) holds`.

- [ ] **Step 6.4: Add test cell for `engineer_director_awards` (will fail)**

```python
# Test §2.3.c: engineer_director_awards
_df = pd.DataFrame({
    'prior_total_nominations': [0, 5, 100],
    'prior_total_wins': [0, 2, 50],
    'prior_oscar_nominations': [0, 1, 10],
    'prior_oscar_wins': [0, 0, 3],
    'prior_palme_nominations': [0, 0, 1],
    'prior_palme_wins': [0, 0, 1],
})
_out = engineer_director_awards(_df)
# All log1p applied
assert abs(_out.loc[0, 'prior_log_total_wins']) < 1e-9
assert abs(_out.loc[2, 'prior_log_total_wins'] - np.log1p(50)) < 1e-9
# 6 columns expected
assert len(_out.columns) == 6
# All non-negative
assert (_out.values >= 0).all()
print(f"✅ engineer_director_awards: 6 cols, log1p applied")
```

Run. Expected: `NameError`.

- [ ] **Step 6.5: Add `engineer_director_awards` implementation**

```python
AWARDS_RAW_COLS = [
    'prior_total_nominations', 'prior_total_wins',
    'prior_oscar_nominations', 'prior_oscar_wins',
    'prior_palme_nominations', 'prior_palme_wins',
]

def engineer_director_awards(df: pd.DataFrame) -> pd.DataFrame:
    """Director awards block — log1p of the 6 prior_ columns from §2.2."""
    out = pd.DataFrame(index=df.index)
    for col in AWARDS_RAW_COLS:
        clean = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        out[f'prior_log_{col[len("prior_"):]}'] = np.log1p(clean)
    return out
```

Re-run test. Expected: `✅ engineer_director_awards: 6 cols, log1p applied`.

- [ ] **Step 6.6: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.3c): decade block (fix #6) + director awards block"
```

---

## Task 7 — §2.4 Embedding Layer with Cache (fixes #5, #12)

**Files:**
- Modify: `eda_v2.ipynb`

Implements `compute_text_embeddings` with multilingual model and MD5-keyed file cache.

- [ ] **Step 7.1: Add markdown cell**

```markdown
## §2.4 — Text Embedding (fixes #5, #12)
Multilingual sentence-transformers model with MD5-keyed file cache.
```

- [ ] **Step 7.2: Add test cell for cache logic (will fail)**

```python
# Test §2.4: compute_text_embeddings cache
import tempfile

with tempfile.TemporaryDirectory() as _tmp:
    _cache = Path(_tmp) / 'emb.npy'
    _meta  = Path(_tmp) / 'emb.meta.json'
    _ovs = pd.Series(['A short film', '', 'Another plot'])

    # Cold path: compute, save
    _emb1 = compute_text_embeddings(
        _ovs,
        model_name=CONFIG['embedding_model'],
        cache_path=_cache,
        cache_meta_path=_meta,
        batch_size=4,
    )
    assert _emb1.shape == (3, 384), f"got {_emb1.shape}"
    assert _cache.exists()

    # Empty overview must produce zero vector
    assert np.allclose(_emb1[1], 0.0), "empty overview should be zero vector"
    assert not np.allclose(_emb1[0], 0.0), "non-empty overview should be non-zero"

    # Warm path: should load from cache (verify by mtime — no recompute)
    _t0 = _cache.stat().st_mtime
    _emb2 = compute_text_embeddings(
        _ovs,
        model_name=CONFIG['embedding_model'],
        cache_path=_cache,
        cache_meta_path=_meta,
        batch_size=4,
    )
    assert np.array_equal(_emb1, _emb2), "warm load should equal cold result"
    assert _cache.stat().st_mtime == _t0, "cache should not have been rewritten"

print("✅ compute_text_embeddings: cold path, warm path, zero-vector all work")
```

Run. Expected: `NameError`.

- [ ] **Step 7.3: Add `compute_text_embeddings` implementation**

```python
def compute_text_embeddings(
    overviews: pd.Series,
    *,
    model_name: str,
    cache_path: Path,
    cache_meta_path: Path | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Sentence embeddings with MD5-keyed file cache.

    Behaviour:
      • Cache hit if model_name + len(overviews) + sum(char_lengths) match.
      • Empty overviews produce a zero vector (preserves old behaviour, info captured by has_overview flag elsewhere).
      • On cache miss, encodes ALL overviews (zeros included) for simplicity, then overwrites empty rows with zeros.

    The first cold run on 329k films takes ~30-60 min on Colab T4 GPU; warm load is < 5 s.
    """
    cache_path = Path(cache_path)
    if cache_meta_path is None:
        cache_meta_path = cache_path.with_suffix('.meta.json')

    overviews = overviews.fillna('').astype(str)
    total_chars = int(overviews.str.len().sum())
    expected_hash = hashlib.md5(
        f"{model_name}|{len(overviews)}|{total_chars}".encode()
    ).hexdigest()

    # Cache hit?
    if cache_path.exists() and cache_meta_path.exists():
        try:
            meta = json.loads(cache_meta_path.read_text())
            if meta.get('hash') == expected_hash:
                emb = np.load(cache_path)
                if emb.shape == (len(overviews), 384):
                    return emb
        except Exception:
            pass  # fall through to cold path

    # Cold path
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = overviews.tolist()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    # Force zero vectors for empty overviews
    empty_mask = (overviews.str.len() == 0).values
    emb[empty_mask] = 0.0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    cache_meta_path.write_text(json.dumps({
        'hash': expected_hash,
        'model': model_name,
        'n': len(overviews),
        'total_chars': total_chars,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    return emb
```

Re-run test cell. The first run will take 5-30 seconds (small input, model download if first time). Expected: `✅ compute_text_embeddings: cold path, warm path, zero-vector all work`.

- [ ] **Step 7.4: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.4): multilingual embedding layer with MD5 cache (fixes #5, #12)"
```

---

## Task 8 — §2.5 Assembly Layer (fix #4 — modality-aware scaling)

**Files:**
- Modify: `eda_v2.ipynb`

Implements the modality-aware scaling and the final matrix builder. This task contains the most subtle fix in the whole pipeline.

- [ ] **Step 8.1: Add markdown cell**

```markdown
## §2.5 — Assembly (fix #4 — modality-aware scaling)
StandardScaler for numerical/decade/awards; no scaling for genre/language one-hot; row L2-normalize for text embeddings.
```

- [ ] **Step 8.2: Add test cell for `apply_modality_aware_scaling` (will fail)**

```python
# Test §2.5: apply_modality_aware_scaling (fix #4)
_blocks = {
    # numerical block: very different scales per column
    'numerical': pd.DataFrame({
        'log_popularity':    [0.0, 1.0, 2.0, 3.0, 4.0],
        'log_vote_count':    [0.0, 5.0, 10.0, 0.0, 5.0],
        'has_vote':          [0, 1, 1, 1, 1],
    }),
    # genre block: 0/1 — must not be rescaled
    'genre': pd.DataFrame({'genre_A': [1, 0, 1, 0, 1], 'genre_B': [0, 1, 1, 0, 0]}),
    # text block: random, will be L2-normalized per row
    'text': np.random.RandomState(42).randn(5, 384).astype(np.float32),
}
_scaled, _scalers = apply_modality_aware_scaling(_blocks)

# numerical: each column ~ μ=0, σ=1
_n = _scaled['numerical']
assert np.allclose(_n.mean(axis=0), 0, atol=1e-6), f"numerical mean ≠ 0: {_n.mean(axis=0)}"
assert np.allclose(_n.std(axis=0, ddof=0), 1, atol=1e-6) or \
       np.allclose(_n.std(axis=0, ddof=0)[_n.std(axis=0, ddof=0) > 0], 1, atol=1e-6)

# genre: untouched
assert np.array_equal(_scaled['genre'].values, _blocks['genre'].values)

# text: every row has L2 norm = 1
_t = _scaled['text']
_norms = np.linalg.norm(_t, axis=1)
assert np.allclose(_norms, 1.0, atol=1e-6), f"text L2 norms not 1: min={_norms.min()}, max={_norms.max()}"

# scalers dict has the right keys
assert 'numerical' in _scalers
assert 'genre' not in _scalers   # we didn't scale it
print("✅ apply_modality_aware_scaling: numerical Standard, genre untouched, text L2-normalized")
```

Run. Expected: `NameError`.

- [ ] **Step 8.3: Add `apply_modality_aware_scaling` implementation**

```python
def apply_modality_aware_scaling(
    blocks: dict[str, np.ndarray | pd.DataFrame],
) -> tuple[dict[str, np.ndarray], dict]:
    """Modality-aware scaling — fix #4.

    Strategy per block:
      • numerical, decade, awards → StandardScaler (μ=0, σ=1 per dim)
      • genre, language           → no scaling (already {0, 1})
      • text                      → L2-normalize per ROW (each embedding becomes unit-length)
                                    → each text dim has variance ~ 1/384, the BLOCK total
                                    variance is ~1, comparable to numerical block's ~6
                                    and awards' ~6. Modal balance achieved.

    Returns scaled-blocks dict (numpy arrays) + scalers dict (sklearn instances for persistence).
    """
    scaled = {}
    scalers = {}

    standard_blocks = {'numerical', 'decade', 'awards'}
    no_scale_blocks = {'genre', 'language'}

    for name, block in blocks.items():
        arr = block.values if isinstance(block, pd.DataFrame) else np.asarray(block)

        if name in standard_blocks:
            scaler = StandardScaler()
            scaled[name] = scaler.fit_transform(arr.astype(np.float32))
            scalers[name] = scaler
        elif name in no_scale_blocks:
            scaled[name] = arr.astype(np.float32)
        elif name == 'text':
            # Per-row L2 normalization
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)   # zero-vector safe-guard
            scaled[name] = (arr / norms).astype(np.float32)
        else:
            raise ValueError(f"Unknown block name: {name!r}")

    return scaled, scalers
```

Re-run test. Expected: `✅ apply_modality_aware_scaling: numerical Standard, genre untouched, text L2-normalized`.

- [ ] **Step 8.4: Add test cell for `assemble_feature_matrix` (will fail)**

```python
# Test §2.5: assemble_feature_matrix
_blocks_scaled = {
    'numerical': np.zeros((4, 6), dtype=np.float32),
    'genre':     np.zeros((4, 22), dtype=np.float32),
    'language':  np.zeros((4, 31), dtype=np.float32),
    'decade':    np.zeros((4, 2), dtype=np.float32),
    'awards':    np.zeros((4, 6), dtype=np.float32),
    'text':      np.zeros((4, 384), dtype=np.float32),
}
_block_names = {
    'numerical': [f'num_{i}' for i in range(6)],
    'genre':     [f'genre_{i}' for i in range(22)],
    'language':  [f'lang_{i}' for i in range(31)],
    'decade':    [f'dec_{i}' for i in range(2)],
    'awards':    [f'aw_{i}' for i in range(6)],
    'text':      [f'txt_{i}' for i in range(384)],
}
_X, _names = assemble_feature_matrix(_blocks_scaled, _block_names)
assert _X.shape == (4, 451), f"got {_X.shape}"
assert len(_names) == 451
# Block ordering: numerical (6) | genre (22) | language (31) | decade (2) | awards (6) | text (384)
assert _names[0].startswith('num_')
assert _names[5].startswith('num_')
assert _names[6].startswith('genre_')
assert _names[27].startswith('genre_')   # 6+22-1
assert _names[28].startswith('lang_')
assert _names[58].startswith('lang_')    # 6+22+31-1
assert _names[59].startswith('dec_')
assert _names[61].startswith('aw_')
assert _names[67].startswith('txt_')
print(f"✅ assemble_feature_matrix: shape={_X.shape}, ordering correct")
```

Run. Expected: `NameError`.

- [ ] **Step 8.5: Add `assemble_feature_matrix` implementation**

```python
BLOCK_ORDER = ['numerical', 'genre', 'language', 'decade', 'awards', 'text']


def assemble_feature_matrix(
    blocks_scaled: dict[str, np.ndarray],
    block_names: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Concatenate blocks horizontally in fixed BLOCK_ORDER. Returns (X, feature_names)."""
    arrays = []
    names = []
    for b in BLOCK_ORDER:
        if b not in blocks_scaled:
            raise KeyError(f"Missing block: {b}")
        arr = np.asarray(blocks_scaled[b], dtype=np.float32)
        arrays.append(arr)
        names.extend(block_names[b])
    X = np.hstack(arrays).astype(np.float32)
    return X, names
```

Re-run test. Expected: `✅ assemble_feature_matrix: shape=(4, 451), ordering correct`.

- [ ] **Step 8.6: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.5): modality-aware scaling (fix #4) + matrix assembly"
```

---

## Task 9 — §2.6 Sanity Helpers

**Files:**
- Modify: `eda_v2.ipynb`

Four lightweight assertion helpers used throughout §3.

- [ ] **Step 9.1: Add markdown + implementation cell**

```markdown
## §2.6 — Sanity Helpers
Fail-fast assertion helpers with informative messages.
```

```python
def assert_shape(arr, expected: tuple, name: str = 'array') -> None:
    if hasattr(arr, 'shape'):
        got = tuple(arr.shape)
    else:
        got = (len(arr),)
    assert got == expected, f"{name}: shape {got} != expected {expected}"


def assert_no_nan(arr, name: str = 'array') -> None:
    if isinstance(arr, pd.DataFrame):
        n_nan = int(arr.isna().sum().sum())
    else:
        n_nan = int(np.isnan(np.asarray(arr, dtype=float)).sum())
    assert n_nan == 0, f"{name}: contains {n_nan} NaN values"


def assert_value_range(arr, lo: float, hi: float, name: str = 'array') -> None:
    a = np.asarray(arr, dtype=float)
    a_min, a_max = float(np.nanmin(a)), float(np.nanmax(a))
    assert a_min >= lo, f"{name}: min {a_min} < {lo}"
    assert a_max <= hi, f"{name}: max {a_max} > {hi}"


def assert_unit_norm(arr, name: str = 'array', atol: float = 1e-3) -> None:
    a = np.asarray(arr, dtype=float)
    norms = np.linalg.norm(a, axis=1)
    # zero-rows allowed (empty overviews)
    nonzero = norms > 1e-12
    if nonzero.any():
        bad = np.abs(norms[nonzero] - 1.0) > atol
        assert not bad.any(), f"{name}: {bad.sum()} rows have non-unit L2 norm"
```

- [ ] **Step 9.2: Add test cell**

```python
# Test §2.6 helpers
assert_shape(np.zeros((3, 4)), (3, 4), 'zeros')
assert_no_nan(np.zeros((3, 4)), 'zeros')
assert_value_range(np.array([0.1, 0.5, 0.9]), 0, 1, 'unit_interval')

_unit = np.random.RandomState(0).randn(5, 384)
_unit = _unit / np.linalg.norm(_unit, axis=1, keepdims=True)
assert_unit_norm(_unit, 'unit')

# Negative cases
try:
    assert_no_nan(np.array([1.0, np.nan]), 'nan_arr')
    assert False, "should have raised"
except AssertionError:
    pass

try:
    assert_shape(np.zeros(5), (4,), 'wrong')
    assert False, "should have raised"
except AssertionError:
    pass

print("✅ §2.6 helpers: positive + negative cases pass")
```

Run. Expected: `✅ §2.6 helpers: positive + negative cases pass`.

- [ ] **Step 9.3: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§2.6): sanity assertion helpers"
```

---

## Task 10 — §3 Pipeline Execution + Spot-Check Cell

**Files:**
- Modify: `eda_v2.ipynb`

Wires §2 functions into the actual end-to-end run on real data.

- [ ] **Step 10.1: Add markdown cell**

```markdown
## §3 — Pipeline Execution
Calls the §2 functions on the real data. After each block, an inline assertion catches bugs early. The §3.4 spot-check cell at the end surfaces the headline numbers for visual review.
```

- [ ] **Step 10.2: Add execution cell §3.1 — Load and merge**

```python
# §3.1 — Load + merge (uses §2.1 functions)
details, casting, awards = load_csvs(CONFIG['paths'])
print(f"   details: {details.shape}")
print(f"   casting: {casting.shape}")
print(f"   awards : {awards.shape}")

df = merge_details_casting(details, casting)
assert_shape(df, (details.shape[0], df.shape[1]), 'df after merge')
assert df['director_name_norm'].notna().all(), "director_name_norm should never be NaN (empty string OK)"
print(f"✅ §3.1 merged df: {df.shape}")
```

Run cell. Expected: roughly `details: (329044, 22)`, `casting: (329044, 19)`, `awards: (225675, 6)` (numbers may vary slightly), then `✅ §3.1 merged df: (329044, ~25)`.

- [ ] **Step 10.3: Add execution cell §3.2 — Awards aggregation**

```python
# §3.2 — Temporal-aware awards aggregation (uses §2.2)
awards_per_film = aggregate_awards_temporal(awards, df)
df = df.merge(awards_per_film, on='id', how='left')
for col in AWARDS_RAW_COLS:
    df[col] = df[col].fillna(0)

# Quick comparison: baseline merge rate (raw) vs normalized merge rate (verifies fix #8)
baseline_rate = (
    awards['director_name'].apply(lambda n: n if n else '')
    .isin(casting['director_name'].fillna(''))
    .mean()
)
normalized_rate = (
    awards['director_name'].apply(normalize_director_name)
    .isin(df['director_name_norm'])
    .mean()
)
print(f"   baseline director-match rate (raw):        {baseline_rate*100:.2f}%")
print(f"   normalized director-match rate (fix #8):   {normalized_rate*100:.2f}%")
print(f"   Δ: {(normalized_rate - baseline_rate)*100:.2f} pp")
print(f"✅ §3.2 df after awards merge: {df.shape}")
```

Run. Expected: `Δ ≥ 5 pp` (fix #8 verification). If less, investigate `normalize_director_name`.

- [ ] **Step 10.4: Add execution cell §3.3 — Build feature blocks**

```python
# §3.3 — Build feature blocks (uses §2.3)
blocks_df = {}
blocks_df['numerical'] = engineer_numerical(df, CONFIG)
assert_no_nan(blocks_df['numerical'], 'numerical block')
assert_shape(blocks_df['numerical'], (len(df), 6), 'numerical block')

blocks_df['genre']    = engineer_genres(df, top_n=CONFIG['top_n_genres'])
assert_shape(blocks_df['genre'], (len(df), 22), 'genre block')   # 20 + Unknown + has_genre

blocks_df['language'] = engineer_languages(df, top_n=CONFIG['top_n_languages'])
# Language dim depends on data — assert sane upper bound, not exact
n_lang = blocks_df['language'].shape[1]
assert 5 <= n_lang <= 35, f"unexpected language dim: {n_lang}"

blocks_df['decade']   = engineer_decade(df)
assert_shape(blocks_df['decade'], (len(df), 2), 'decade block')

blocks_df['awards']   = engineer_director_awards(df)
assert_shape(blocks_df['awards'], (len(df), 6), 'awards block')

print(f"✅ §3.3 blocks: {[(k, v.shape) for k, v in blocks_df.items()]}")
```

Run. Expected: each block printed with its shape; numerical (n, 6), genre (n, 22), language (n, ~31), decade (n, 2), awards (n, 6).

- [ ] **Step 10.5: Add execution cell §3.4 — Text embedding**

```python
# §3.4 — Text embedding (uses §2.4) — cached after first run
text_emb = compute_text_embeddings(
    df['overview'].fillna('').astype(str),
    model_name=CONFIG['embedding_model'],
    cache_path=CONFIG['embedding_cache'],
    cache_meta_path=CONFIG['embedding_meta'],
    batch_size=CONFIG['embedding_batch_size'],
)
assert_shape(text_emb, (len(df), 384), 'text_emb')
print(f"✅ §3.4 text_emb: {text_emb.shape}, dtype={text_emb.dtype}")
```

Run. **First run: 30-60 minutes on Colab T4 GPU, longer on CPU.** Subsequent runs: < 5 seconds (cache hit).

- [ ] **Step 10.6: Add execution cell §3.5 — Modality scaling and assembly**

```python
# §3.5 — Modality-aware scaling + assemble (uses §2.5)
blocks_arr = {k: v.values if isinstance(v, pd.DataFrame) else v for k, v in blocks_df.items()}
blocks_arr['text'] = text_emb

block_names = {
    'numerical': list(blocks_df['numerical'].columns),
    'genre':     list(blocks_df['genre'].columns),
    'language':  list(blocks_df['language'].columns),
    'decade':    list(blocks_df['decade'].columns),
    'awards':    list(blocks_df['awards'].columns),
    'text':      [f'text_{i}' for i in range(text_emb.shape[1])],
}

blocks_scaled, scalers = apply_modality_aware_scaling(blocks_arr)
X, feature_names = assemble_feature_matrix(blocks_scaled, block_names)

# Compute total dim from actual blocks (resilient to language dim change)
expected_dim = sum(blocks_arr[b].shape[1] for b in BLOCK_ORDER)
assert_shape(X, (len(df), expected_dim), 'X')
assert_no_nan(X, 'X')

# Modality balance check (fix #4 verification)
text_var = float(blocks_scaled['text'].var(axis=0).sum())
other_var = sum(
    float(blocks_scaled[b].var(axis=0).sum())
    for b in BLOCK_ORDER if b != 'text'
)
ratio = text_var / max(other_var, 1e-12)
print(f"   modality variance ratio (text / other) = {ratio:.3f}")
assert 0.3 <= ratio <= 3.0, f"⚠️  modality balance off (ratio={ratio:.2f}); expected 0.5-2.0"

print(f"✅ §3.5 X: {X.shape}, modal ratio = {ratio:.3f}")
```

Run. Expected: `X: (329044, ~451)` (depends on actual language dim), `modal ratio` between 0.5 and 2.0.

- [ ] **Step 10.7: Add execution cell §3.6 — Sanity print spot-check**

```python
# §3.6 — Spot-check headline numbers (operator visual review)
print("─── §3.6 Pipeline Spot-Checks ─────────────────────────")
print(f"  Films total                           : {len(df):,}")
print(f"  Awards merge rate (any prior win > 0) : {(df['prior_total_wins']>0).mean()*100:.2f}%")
print(f"  has_vote = 1                          : {blocks_df['numerical']['has_vote'].sum():,}")
print(f"  has_genre = 1                         : {blocks_df['genre']['has_genre'].sum():,}")
print(f"  has_release_date = 1                  : {blocks_df['decade']['has_release_date'].sum():,}")
print(f"  has_engagement = 1                    : {blocks_df['numerical']['has_engagement'].sum():,}")
print(f"  log_vote_count skewness               : {blocks_df['numerical']['log_vote_count'].skew():.2f}  (fix #7: target < 2)")
print(f"  modality variance ratio (text/other)  : {ratio:.3f}  (fix #4: target 0.5-2.0)")
print("────────────────────────────────────────────────────────")
```

Run. Verify each line is in the expected range:
- Films total ≈ 329,044
- Awards merge rate > 25%
- log_vote_count skewness < 2 (was 28 in original)
- Modality ratio 0.5-2.0

- [ ] **Step 10.8: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§3): pipeline execution + spot-check cell"
```

---

## Task 11 — §4 EDA Visualizations (slide-aligned)

**Files:**
- Modify: `eda_v2.ipynb`
- Create: `artifacts/figures/*.png` (23 PNGs total)

This task is large but each visualization cell is short. Group into 8 sub-sections matching the spec.

**Strategy:** Each sub-section is one cell. Each cell saves PNG to `artifacts/figures/`. Captions include a "(fix #X)" tag where the figure verifies a specific fix.

- [ ] **Step 11.1: Add §4.1 — Data Quality cells**

```markdown
## §4 — EDA Visualizations
```

```python
# §4.1 — Data quality: missing, awards merge quality
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: missing-data heatmap
key_cols = ['popularity', 'vote_average', 'vote_count', 'runtime', 'overview',
            'genres', 'original_language', 'release_date', 'director_name']
miss = df[key_cols].isnull().sum().sort_values(ascending=False) / len(df) * 100
axes[0].barh(miss.index, miss.values, color='#E74C3C')
axes[0].set_xlabel('Missing %')
axes[0].set_title('Missing Value Rate')
for i, v in enumerate(miss.values):
    axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

# Right: awards merge quality (slide 6 equivalent)
matched = (df['prior_total_wins'] > 0).sum()
has_dir = (df['director_name_norm'] != '').sum()
labels = ['No director\nname', 'Director, no\naward match', 'Has prior\naward']
vals = [len(df) - has_dir, has_dir - matched, matched]
axes[1].barh(labels, vals, color=['#D3D1C7', '#FAC775', '#5DCAA5'])
axes[1].set_xlabel('Film count')
axes[1].set_title('Awards Merge Quality')
for i, v in enumerate(vals):
    axes[1].text(v + 5000, i, f'{v:,} ({v/len(df)*100:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'awards_merge_quality.png', bbox_inches='tight')
plt.savefig(CONFIG['figures_dir'] / 'missing_analysis.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 11.2: Add §4.2 — Distributions (raw + log + box) cells**

```python
# §4.2 — Distributions: raw, log-transformed, boxplots
num_features = ['popularity', 'vote_average', 'vote_count', 'runtime']

# Raw
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, col in enumerate(num_features):
    data = pd.to_numeric(df[col], errors='coerce').dropna()
    data = data[data > 0] if col != 'vote_average' else data
    axes[i].hist(data, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(f'{col}\nskew={data.skew():.2f}')
plt.suptitle('Raw Distributions', fontsize=13)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'hist_raw.png', bbox_inches='tight')
plt.show()

# Log-transformed
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, col in enumerate(num_features):
    data = pd.to_numeric(df[col], errors='coerce').dropna()
    data = data[data > 0] if col != 'vote_average' else data
    log_data = np.log1p(data)
    axes[i].hist(log_data, bins=60, color='teal', edgecolor='white', alpha=0.85)
    axes[i].set_title(f'log1p({col})\nskew={log_data.skew():.2f}')
plt.suptitle('Log-Transformed Distributions', fontsize=13)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'hist_log.png', bbox_inches='tight')
plt.show()

# Boxplots
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, col in enumerate(num_features):
    data = pd.to_numeric(df[col], errors='coerce').dropna()
    data = data[data > 0] if col != 'vote_average' else data
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    iqr = data.quantile(0.75) - data.quantile(0.25)
    n_out = ((data < data.quantile(0.25) - 1.5*iqr) |
             (data > data.quantile(0.75) + 1.5*iqr)).sum()
    axes[i].set_title(f'{col}\n{n_out:,} IQR outliers')
plt.suptitle('Boxplots — Outlier Detection', fontsize=13)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'boxplots.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 11.3: Add §4.3 — Correlation cells**

```python
# §4.3 — Correlation (Spearman + Pearson)
corr_cols = ['popularity', 'vote_average', 'vote_count', 'runtime',
             'prior_total_nominations', 'prior_total_wins', 'prior_oscar_nominations']
df_corr = df[corr_cols].apply(pd.to_numeric, errors='coerce')

for method, fname in [('spearman', 'correlation_heatmap.png'),
                      ('pearson',  'correlation_pearson.png')]:
    plt.figure(figsize=(9, 7))
    corr = df_corr.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0)
    plt.title(f'{method.title()} Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / fname, bbox_inches='tight')
    plt.show()
```

- [ ] **Step 11.4: Add §4.4 — Categorical (genre/lang/decade) cells**

```python
# §4.4 — Genre, language, decade distributions
# Genre top-20 + pie
genre_counts = blocks_df['genre'].drop(columns=['has_genre']).sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].barh(
    [c.replace('genre_', '') for c in genre_counts.index[:20][::-1]],
    genre_counts.values[:20][::-1],
    color='#4C72B0', alpha=0.85,
)
axes[0].set_title('Top 20 Genres')
axes[0].set_xlabel('Film count')

top8 = genre_counts.head(8)
other = genre_counts.iloc[8:].sum()
pie_data = pd.concat([top8, pd.Series({'genre_Other': other})])
axes[1].pie(pie_data.values,
            labels=[c.replace('genre_', '') for c in pie_data.index],
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.2))
axes[1].set_title('Genre Distribution')
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'genre_distribution.png', bbox_inches='tight')
plt.show()

# Language + decade
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
lang_counts = df['original_language'].value_counts().head(15)
axes[0].barh(lang_counts.index[::-1], lang_counts.values[::-1],
             color='#55A868', alpha=0.85)
axes[0].set_title('Top 15 Original Languages')

decade = (pd.to_datetime(df['release_date'], errors='coerce').dt.year // 10 * 10).dropna()
decade_counts = decade.value_counts().sort_index()
decade_counts = decade_counts[decade_counts.index > 0]
axes[1].bar(decade_counts.index.astype(int).astype(str),
            decade_counts.values, color='#8172B2', alpha=0.85)
axes[1].set_title('Movies by Decade')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'categorical_distributions.png', bbox_inches='tight')
plt.show()

# Genre imbalance figure
top10 = genre_counts.head(10)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar([c.replace('genre_', '') for c in top10.index],
            top10.values / top10.sum() * 100,
            color='#4C72B0', alpha=0.85)
axes[0].axhline(y=10, color='red', linestyle='--', label='Balanced baseline (10%)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_title('Genre Imbalance (top 10)')
axes[0].legend()

cum = np.cumsum(genre_counts.values) / genre_counts.sum() * 100
axes[1].plot(range(1, len(genre_counts) + 1), cum, color='#4C72B0', linewidth=2)
axes[1].axhline(80, color='red', linestyle='--', alpha=0.7,
                label=f'80% → top {int(np.argmax(cum>=80)+1)}')
axes[1].axhline(90, color='orange', linestyle='--', alpha=0.7,
                label=f'90% → top {int(np.argmax(cum>=90)+1)}')
axes[1].set_xlabel('Number of genres')
axes[1].set_ylabel('Cumulative % of films')
axes[1].set_title('Genre Cumulative Distribution')
axes[1].legend()
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'imbalance_analysis.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 11.5: Add §4.5 — Modality Balance (NEW figure for fix #4) cell**

```python
# §4.5 — Modality balance: BEFORE vs AFTER scaling (fix #4)
def modality_variances(blocks):
    return {k: float(np.var(v, axis=0).sum()) for k, v in blocks.items()}

# BEFORE: raw blocks_arr (text not scaled, others not scaled)
before = modality_variances(blocks_arr)
# AFTER: blocks_scaled
after = modality_variances(blocks_scaled)

import matplotlib.patches as mpatches
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(before.keys())
x = np.arange(len(names))
w = 0.4
axes[0].bar(x - w/2, [before[n] for n in names], w, label='Before scaling', color='#C44E52')
axes[0].bar(x + w/2, [after[n] for n in names],  w, label='After modality-aware', color='#5DCAA5')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names)
axes[0].set_ylabel('Total variance (sum across dims)')
axes[0].set_title('Modality variance — before vs after\n(fix #4)')
axes[0].legend()
axes[0].set_yscale('log')

# Right plot: ratio text/other for each
text_v_after = after['text']
other_v_after = sum(v for k, v in after.items() if k != 'text')
ratio_after = text_v_after / max(other_v_after, 1e-12)

text_v_before = before['text']
other_v_before = sum(v for k, v in before.items() if k != 'text')
ratio_before = text_v_before / max(other_v_before, 1e-12)

axes[1].bar(['Before', 'After'], [ratio_before, ratio_after],
            color=['#C44E52', '#5DCAA5'])
axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.5)
axes[1].axhspan(0.5, 2.0, alpha=0.15, color='green', label='Acceptable [0.5, 2.0]')
axes[1].set_ylabel('text variance / other variance')
axes[1].set_title('Modality balance ratio')
axes[1].legend()
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'modality_balance_before_after.png', bbox_inches='tight')
plt.show()
print(f"   Modality ratio: BEFORE={ratio_before:.2f}, AFTER={ratio_after:.2f}")
```

- [ ] **Step 11.6: Add §4.6 — Text embedding PCA cell**

```python
# §4.6 — Text embedding PCA (5k sample to keep it fast)
from sklearn.decomposition import PCA

n_sample = min(5000, text_emb.shape[0])
sample_idx = np.random.RandomState(42).choice(text_emb.shape[0], size=n_sample, replace=False)
emb_sample = text_emb[sample_idx]
genre_sample = blocks_df['genre'].iloc[sample_idx].drop(columns=['has_genre'])

pca_emb = PCA(n_components=2, random_state=42).fit(emb_sample)
emb_2d = pca_emb.transform(emb_sample)

# Variance explained
pca_var = PCA(n_components=50, random_state=42).fit(emb_sample)
cumvar_emb = np.cumsum(pca_var.explained_variance_ratio_) * 100
n80_emb = int(np.argmax(cumvar_emb >= 80)) + 1

# Top genre per row
def get_top_genre(g_row):
    nz = g_row[g_row > 0].index
    if len(nz) == 0:
        return 'Other'
    g = nz[0].replace('genre_', '')
    return g

top_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror',
              'Romance', 'Documentary', 'Animation']
palette = {'Drama': '#4C72B0', 'Comedy': '#DD8452', 'Action': '#55A868',
           'Thriller': '#C44E52', 'Horror': '#8172B2', 'Romance': '#937860',
           'Documentary': '#DA8BC3', 'Animation': '#8C8C8C', 'Other': '#CCB974',
           'Unknown': '#444444'}
labels = genre_sample.apply(get_top_genre, axis=1).values

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for g, c in palette.items():
    mask = labels == g
    if mask.sum() > 0:
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                        c=c, label=f'{g} ({mask.sum()})', alpha=0.5, s=10)
axes[0].set_title(f'Text embedding PCA 2D ({n_sample} sample)')
axes[0].set_xlabel(f'PC1 ({pca_emb.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_emb.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].legend(markerscale=2, fontsize=8)

axes[1].plot(range(1, 51), cumvar_emb, marker='o', markersize=3, color='#4C72B0')
axes[1].axhline(80, color='red', linestyle='--', alpha=0.7,
                label=f'80% → {n80_emb} PCs')
axes[1].set_xlabel('Number of PCs')
axes[1].set_ylabel('Cumulative variance (%)')
axes[1].set_title('Text embedding — variance explained')
axes[1].legend()
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'embedding_analysis.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 11.7: Add §4.7 — Clusterability (full PCA) cell**

```python
# §4.7 — Clusterability: PCA on full feature matrix (50k sample)
from sklearn.decomposition import PCA

n_clust = min(50_000, X.shape[0])
clust_idx = np.random.RandomState(42).choice(X.shape[0], size=n_clust, replace=False)

pca_full = PCA(n_components=50, random_state=42).fit(X[clust_idx])
cumvar_full = np.cumsum(pca_full.explained_variance_ratio_) * 100
n80_full = int(np.argmax(cumvar_full >= 80)) + 1
n90_full = int(np.argmax(cumvar_full >= 90)) + 1

X_2d = PCA(n_components=2, random_state=42).fit_transform(X[clust_idx])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, 51), cumvar_full, marker='o', markersize=3, color='#4C72B0')
axes[0].axhline(80, color='red',    linestyle='--', alpha=0.7,
                label=f'80% → {n80_full} PCs')
axes[0].axhline(90, color='orange', linestyle='--', alpha=0.7,
                label=f'90% → {n90_full} PCs')
axes[0].set_xlabel('Number of PCs')
axes[0].set_ylabel('Cumulative explained variance (%)')
axes[0].set_title('Full feature matrix PCA variance')
axes[0].legend()

genre_clust = blocks_df['genre'].iloc[clust_idx].drop(columns=['has_genre'])
labels_clust = genre_clust.apply(get_top_genre, axis=1).values
for g, c in palette.items():
    mask = labels_clust == g
    if mask.sum() > 0:
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=c, label=g, alpha=0.35, s=4)
axes[1].set_title(f'Full PCA 2D scatter — {n_clust} sample')
axes[1].legend(markerscale=3, fontsize=7)

plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'clusterability_pca.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 11.8: Add §4.8 — Variance thresholding + new fix-verification figures**

```python
# §4.8 — Variance thresholding + figures verifying fixes #1, #2, #5

# Variance thresholding
variances = X.var(axis=0)
sorted_v = np.sort(variances)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].bar(range(len(sorted_v)), sorted_v, color='steelblue', alpha=0.7)
axes[0].axhline(0.01, color='red', linestyle='--', label='Threshold 0.01')
axes[0].set_yscale('log')
axes[0].set_title('All features — log scale')
axes[0].legend()
axes[1].bar(range(20), sorted_v[-20:][::-1], color='steelblue', alpha=0.7)
axes[1].axhline(0.01, color='red', linestyle='--')
axes[1].set_title('Bottom 20 features')
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'variance_thresholding.png', bbox_inches='tight')
plt.show()
n_below = int((variances < 0.01).sum())
print(f"   Features below threshold 0.01: {n_below}")

# Fix #2 — vote_average imputation impact (NEW figure)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
old_imp = df['vote_average'].fillna(df['vote_average'].median())  # what the original did
new_imp = blocks_df['numerical']['vote_average_norm']
axes[0].hist(old_imp, bins=80, color='#C44E52', alpha=0.6, label='Original (median=0)')
axes[0].set_title('vote_average — ORIGINAL imputation\n(yapay 0 yığılması)')
axes[0].set_xlabel('vote_average')
axes[0].legend()
axes[1].hist(new_imp, bins=80, color='#5DCAA5', alpha=0.6, label='Fix #2 (voted-only mean)')
axes[1].set_title('vote_average_norm — FIX #2\n(no artificial 0-spike)')
axes[1].set_xlabel('vote_average_norm')
axes[1].legend()
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'vote_average_imputation_impact.png', bbox_inches='tight')
plt.show()

# Fix #1 — temporal awards visualization (NEW figure)
# For each decade, show mean prior_total_wins
df_temp = df.copy()
df_temp['rel_year'] = pd.to_datetime(df_temp['release_date'], errors='coerce').dt.year
df_temp['rel_decade'] = (df_temp['rel_year'] // 10 * 10).astype('Int64')
trend = df_temp.dropna(subset=['rel_decade']).groupby('rel_decade')['prior_total_wins'].mean()
trend = trend[(trend.index >= 1900) & (trend.index <= 2020)]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trend.index.astype(int), trend.values, marker='o', color='#4C72B0')
ax.set_xlabel('Release decade')
ax.set_ylabel('Mean prior_total_wins (temporal-aware)')
ax.set_title('Fix #1 — average director award count AT release time, by decade\n'
             'Should rise monotonically (no leakage from future awards)')
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'awards_temporal_cutoff.png', bbox_inches='tight')
plt.show()

# Fix #5 — multilingual coverage (NEW figure)
overview_lang_avail = df.groupby('original_language')['overview'].apply(
    lambda s: (s.fillna('').str.len() > 0).mean()
).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(overview_lang_avail.index, overview_lang_avail.values, color='#8172B2')
ax.set_ylabel('Fraction with non-empty overview')
ax.set_title('Fix #5 — overview availability by language (top 15)\n'
             '(multilingual model now embeds these meaningfully)')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(CONFIG['figures_dir'] / 'multilingual_coverage.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 11.9: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§4): EDA visualizations — 19 originals + 4 fix-verification figures"
```

---

## Task 12 — §5 Persistence (fix #13)

**Files:**
- Modify: `eda_v2.ipynb`
- Create (at run time): `artifacts/feature_matrix.npz`, `artifacts/feature_matrix_raw.npz`, `artifacts/movies_eda_final.csv`, `artifacts/scalers.pkl`, `artifacts/feature_metadata.json`, `artifacts/pipeline_version.json`

- [ ] **Step 12.1: Add markdown cell**

```markdown
## §5 — Persistence
Saves all artifacts that the modelling notebooks will consume.
```

- [ ] **Step 12.2: Add persistence cell**

```python
# §5 — Save artifacts
import pickle, importlib.metadata

artifacts = CONFIG['artifacts_dir']

# 1. feature_matrix.npz (model input — primary)
np.savez_compressed(
    artifacts / 'feature_matrix.npz',
    X=X,
    feature_names=np.array(feature_names, dtype=object),
)

# 2. feature_matrix_raw.npz — pre-scaling, block dict (for ablation)
np.savez_compressed(
    artifacts / 'feature_matrix_raw.npz',
    **{k: np.asarray(v, dtype=np.float32) for k, v in blocks_arr.items()},
)

# 3. movies_eda_final.csv — raw + non-text blocks (text excluded for size/readability — fix #13)
encoded_concat = pd.concat(
    [blocks_df['numerical'].reset_index(drop=True),
     blocks_df['genre'].reset_index(drop=True),
     blocks_df['language'].reset_index(drop=True),
     blocks_df['decade'].reset_index(drop=True),
     blocks_df['awards'].reset_index(drop=True)],
    axis=1,
)
final = pd.concat([df.reset_index(drop=True), encoded_concat], axis=1)
final.to_csv(artifacts / 'movies_eda_final.csv', index=False)
print(f"   movies_eda_final.csv: {final.shape}")

# 4. scalers.pkl
with open(artifacts / 'scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)

# 5. feature_metadata.json
metadata = {
    'feature_names': feature_names,
    'block_order': BLOCK_ORDER,
    'block_dims': {b: int(blocks_arr[b].shape[1]) for b in BLOCK_ORDER},
    'total_dim': int(X.shape[1]),
    'n_films': int(X.shape[0]),
}
(artifacts / 'feature_metadata.json').write_text(json.dumps(metadata, indent=2))

# 6. pipeline_version.json — reproducibility audit (fix #11)
md5 = hashlib.md5()
md5.update(X.tobytes())
matrix_hash = md5.hexdigest()

def _ver(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return 'unknown'

version_info = {
    'seed': CONFIG['seed'],
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'library_versions': {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sentence_transformers': _ver('sentence-transformers'),
        'torch': _ver('torch'),
        'umap_learn': _ver('umap-learn'),
        'scikit_learn': _ver('scikit-learn'),
    },
    'model_name': CONFIG['embedding_model'],
    'n_films': int(X.shape[0]),
    'feature_dim': int(X.shape[1]),
    'feature_matrix_md5': matrix_hash,
}
(artifacts / 'pipeline_version.json').write_text(json.dumps(version_info, indent=2))

print("✅ §5 Persistence complete:")
for f in sorted(artifacts.iterdir()):
    if f.is_file():
        print(f"   {f.name:35s} {f.stat().st_size/1e6:8.2f} MB")
print(f"\n   feature_matrix MD5 = {matrix_hash}")
```

- [ ] **Step 12.3: Run cell, verify outputs**

Run. Expected:
- 7 files in `artifacts/` (excluding figures/)
- `feature_matrix.npz` ~ 600 MB (compressed; X is ~600 MB float32 raw)
- `movies_eda_final.csv` ~ 100 MB
- MD5 hash printed

- [ ] **Step 12.4: Commit**

```bash
git add eda_v2.ipynb
git commit -m "feat(§5): persistence — 7 artifacts including version manifest (fixes #11, #12, #13)"
```

---

## Task 13 — End-to-End Reproducibility & Cache Verification

**Files:**
- Modify: `eda_v2.ipynb` (read MD5, no functional change unless mismatch found)

- [ ] **Step 13.1: Run the notebook end-to-end fresh (clear outputs first)**

In Jupyter: `Kernel → Restart & Run All`. Time it. First run on Colab T4 expected: 30-60 minutes (dominated by §3.4 embedding).

Note the printed `feature_matrix MD5` from the §5 cell.

- [ ] **Step 13.2: Run the notebook end-to-end a SECOND time without clearing artifacts**

Restart & Run All again. Note the second run's MD5 and total time.

**Expected:**
- Second-run MD5 == first-run MD5 (exact byte-for-byte)
- Total time < 2 minutes (cache hit on §3.4)

- [ ] **Step 13.3: If MD5 mismatch, investigate before proceeding**

Common causes:
- Sentence-transformers using different precision (check `torch.backends.cudnn.deterministic`)
- `MultiLabelBinarizer.classes_` order non-deterministic (we sort explicitly — verify)
- `top_genres` tiebreak unstable (we sort by `(-count, name)` — verify)

**Do not proceed to Task 14 if MD5 doesn't match.** Reproducibility is a hard success criterion.

- [ ] **Step 13.4: If cache cold-start path takes much longer than 60 min on T4, profile**

If the embedding step is dramatically slower than expected, check:
- `model.encode` is using GPU (`model.device` should be `cuda`)
- `batch_size` honoured (CONFIG default 64 is fine for T4)
- Network: model download (first run ever) adds 1-2 minutes

- [ ] **Step 13.5: Commit any reproducibility fixes**

If you needed to add explicit sort order or seed steps, commit those.

```bash
git add eda_v2.ipynb
git commit -m "fix: ensure deterministic ordering in feature engineering (reproducibility audit)"
```

---

## Task 14 — Success Criteria Verification + Documentation

**Files:**
- Modify: `eda_v2.ipynb` — append a final markdown cell with a checked-off success criteria table.

- [ ] **Step 14.1: Append success-criteria checklist cell**

```markdown
## ✅ Success Criteria — Verification Checklist

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | All 13 fixes applied | ✅ | §3 spot-check + figures |
| 2 | Reproducibility — identical MD5 across two consecutive runs | ✅ | feature_matrix_md5 in pipeline_version.json |
| 3 | Cache works — second run < 2 min | ✅ | timing in Task 13.2 |
| 4 | Artifacts complete — 7 files in `artifacts/` + figures/ | ✅ | §5 listing |
| 5 | Modality balance — text/other ratio ∈ [0.5, 2.0] | ✅ | §3.5 + §4.5 figure |
| 6 | Sanity print silent — no warnings, values in expected ranges | ✅ | §3.6 |

**Final feature matrix:** (329044, 451)
**Reproducibility hash (seed 42):** `<paste from pipeline_version.json>`
```

- [ ] **Step 14.2: Verify each criterion against actual output**

For each row, replace the "✅" with "❌" if the criterion does not pass and document what failed. If anything fails, return to the relevant earlier task to fix.

- [ ] **Step 14.3: Replace `<paste from pipeline_version.json>` with the real MD5 hash**

- [ ] **Step 14.4: Commit final notebook**

```bash
git add eda_v2.ipynb
git commit -m "docs: success criteria checklist — eda_v2 complete"
```

- [ ] **Step 14.5: Final tag**

```bash
git tag -a eda_v2_complete -m "EDA v2 — 13 fixes applied, deterministic, ready for modelling phase"
```

---

## Summary

After completing all 14 tasks:
- `eda_v2.ipynb` is a comprehensive, reproducible notebook
- All 13 audit fixes verified
- Artifacts ready for AE/VAE/DEC modelling phase (next spec)
- Total commits: ~14 (one per task) + bug fixes
- Reproducibility hash recorded in `artifacts/pipeline_version.json`

The next spec will cover the modelling phase: AE, VAE, DEC training notebooks that consume `artifacts/feature_matrix.npz`.
