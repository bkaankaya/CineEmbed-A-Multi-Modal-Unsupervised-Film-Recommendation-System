# CineEmbed Modeling (AE / VAE / DEC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package `src/cineembed/` containing a multi-modal AutoEncoder backbone with three model heads (AE / VAE / DEC), then train 21-22 models on the EDA `(329044, 564)` feature matrix to produce a comparative study suitable for the SENG 474 final report.

**Architecture:** Pure, testable Python modules under `src/cineembed/` with pytest unit tests under `tests/`. Five Jupyter notebooks under `notebooks/` orchestrate training (heavy compute on Colab T4) and produce final figures. The package follows a `src/` layout — `pyproject.toml` at repo root auto-discovers `src/cineembed/`. Local development tests use synthetic small data (Mac CPU); full 329k training runs in Colab.

**Tech Stack:** Python 3.10+, PyTorch 2.10+ (CUDA on Colab, CPU on Mac), scikit-learn, numpy, pandas, umap-learn, matplotlib, seaborn, pytest. The sentence-transformers stack from the EDA phase is NOT a dependency here — we consume the pre-computed `feature_matrix.npz`.

**Reference:** [`docs/superpowers/specs/2026-05-04-modeling-design.md`](../specs/2026-05-04-modeling-design.md) and decision log [`docs/adr/0001-modeling-hybrid-architecture.md`](../../adr/0001-modeling-hybrid-architecture.md).

---

## Prerequisites

The EDA phase must be **complete and verified**:

1. `eda_v2.ipynb` runs end-to-end (last verified by user 2026-05-04, MD5 `e99cee84b6891ea352a7b44d5d7d0ee4`).
2. Required artifacts uploaded to a known location accessible by the implementer (Mac local OR Drive-mounted on Colab):
   - `artifacts/feature_matrix.npz` — `X: (329044, 564)`, `feature_names: (564,)`
   - `artifacts/feature_metadata.json`
   - `artifacts/movies_eda_final.csv` — `(329044, ~146)`
   - `artifacts/pipeline_version.json` — for MD5 audit
   - `artifacts/director_profile_metadata.json`

If the user has only the small subset on Mac (the metadata JSONs and 2 figures from `artifacts/figures/`), Tasks 1-7 (the package + tests) can still proceed using synthetic feature matrices in pytest fixtures. The notebooks (Tasks 8-12) require the full data on Colab.

---

## File Structure

| Path | Status | Purpose |
|---|---|---|
| `pyproject.toml` | NEW | Declares `cineembed` package, lists dependencies, points to `src/` layout |
| `src/cineembed/__init__.py` | NEW | Package marker; re-exports public API |
| `src/cineembed/data.py` | NEW | `load_feature_matrix`, `get_block_indices`, `get_labels`, `make_dataloader`, `train_val_split` |
| `src/cineembed/backbone.py` | NEW | `MultiModalBackbone` — modality-specific projections + FC stack |
| `src/cineembed/losses.py` | NEW | `compute_block_weights`, `director_block_loss`, `weighted_recon_loss`, `weighted_recon_loss_uniform`, `vae_elbo`, `dec_loss`, `LearnedWeightedLoss` (W4 stretch) |
| `src/cineembed/heads.py` | NEW | `AEHead`, `VAEHead`, `DECHead` — three head modules |
| `src/cineembed/train.py` | NEW | `train_model` — generic training loop with early stopping + checkpoint resume |
| `src/cineembed/eval.py` | NEW | `cluster_assignments_kmeans`, `cluster_assignments_dec`, `evaluate_run`, `umap_plot`, `linear_probe` |
| `tests/conftest.py` | NEW | Pytest fixtures: synthetic feature matrix, synthetic labels, dummy DataLoader |
| `tests/test_data.py` | NEW | Tests for `data.py` |
| `tests/test_losses.py` | NEW | Tests for `losses.py` |
| `tests/test_backbone.py` | NEW | Tests for `backbone.py` |
| `tests/test_heads.py` | NEW | Tests for `heads.py` |
| `tests/test_train.py` | NEW | Tests for `train.py` |
| `tests/test_eval.py` | NEW | Tests for `eval.py` |
| `notebooks/01_smoke_test.ipynb` | NEW | Backbone forward pass, head smoke tests, all in <1 min |
| `notebooks/02_train_ae.ipynb` | NEW | Runs 3-6 (vanilla AE z=64 + multi-modal AE z={32,64,128}) + ablations 19-22 |
| `notebooks/03_train_vae.ipynb` | NEW | Runs 7-9 (multi-modal VAE z={32,64,128}) |
| `notebooks/04_train_dec.ipynb` | NEW | Runs 10-18 (DEC z={32,64,128} × k={10,21,30}) |
| `notebooks/05_results.ipynb` | NEW | Loads `artifacts/eval/results.json`, produces comparison tables and final figures |
| `artifacts/models/` | NEW dir | Per-run `.pt` checkpoints (gitignored) |
| `artifacts/eval/` | NEW dir | `results.json`, `linear_probing.json`, `ablation_deltas.json` (gitignored) |
| `.gitignore` | MODIFY | Append `artifacts/models/`, `artifacts/eval/`, `*.pt`, `__pycache__/`, `.pytest_cache/` |

---

## Testing Pattern

The package is tested locally on Mac CPU with **pytest** + synthetic feature matrices. Each module has a focused test file that imports from `cineembed.*`. The conftest provides shared fixtures (mini feature matrix, label dict, etc.) so tests don't duplicate setup.

**Workflow per task:**

1. **Add the test cell** in `tests/test_<module>.py`. Run `pytest` → fails because the function doesn't exist.
2. **Implement** the function in `src/cineembed/<module>.py`.
3. **Re-run pytest** → passes.
4. **Commit** with a descriptive message.

Notebooks are tested by manual execution; the smoke-test notebook (`01_smoke_test.ipynb`) serves as the integration test against the actual package.

**Local pytest setup (one-time):**
```bash
cd "<repo-root>"
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]   # installs cineembed + pytest, torch, etc.
pytest tests/ -v
```

For Colab the same package installs via `!pip install -e /content/cineembed-repo` — see spec §7.4.

---

## Task 1 — Project Skeleton + pyproject.toml + Pytest Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `src/cineembed/__init__.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Modify: `.gitignore`

This task creates the Python package skeleton and verifies `pip install -e .` + `pytest` work end-to-end.

- [ ] **Step 1.1: Create `pyproject.toml` at repo root.**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cineembed"
version = "0.1.0"
description = "Multi-modal AE/VAE/DEC for movie metadata clustering (SENG 474 final project)"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "torch>=2.0",
    "umap-learn>=0.5",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 1.2: Create `src/cineembed/__init__.py`.**

```python
"""CineEmbed — Multi-modal AE/VAE/DEC for movie metadata clustering.

See docs/superpowers/specs/2026-05-04-modeling-design.md for the full design.
"""

__version__ = "0.1.0"

# Public API re-exports (filled in as modules land)
__all__ = []
```

- [ ] **Step 1.3: Create `tests/__init__.py` (empty file) and `tests/conftest.py`.**

`tests/__init__.py`:
```python
```

`tests/conftest.py`:
```python
"""Shared pytest fixtures for the cineembed test suite.

Synthetic feature matrices match the (329044, 564) production layout but are
small enough to run instantly on Mac CPU.
"""
import numpy as np
import pandas as pd
import pytest
import torch


# Block dimensions per spec §3 (numerical=6, genre=22, language=31, decade=2,
# awards=6, text=384, director=113 → total 564).
BLOCK_DIMS = {
    'numerical': 6,
    'genre':     22,
    'language':  31,
    'decade':    2,
    'awards':    6,
    'text':      384,
    'director':  113,
}
TOTAL_DIM = sum(BLOCK_DIMS.values())  # 564
BLOCK_ORDER = list(BLOCK_DIMS.keys())


def _block_slices(block_dims=BLOCK_DIMS):
    out, start = {}, 0
    for b in BLOCK_ORDER:
        out[b] = slice(start, start + block_dims[b])
        start += block_dims[b]
    return out


BLOCK_SLICES = _block_slices()


@pytest.fixture
def block_dims():
    return dict(BLOCK_DIMS)


@pytest.fixture
def block_slices():
    return dict(BLOCK_SLICES)


@pytest.fixture
def synthetic_feature_matrix():
    """Mini feature matrix matching production layout: (n=200, 564)."""
    rng = np.random.default_rng(42)
    n = 200
    X = np.zeros((n, TOTAL_DIM), dtype=np.float32)
    # numerical: random N(0,1) — StandardScaler-like
    X[:, BLOCK_SLICES['numerical']] = rng.standard_normal((n, BLOCK_DIMS['numerical'])).astype(np.float32)
    # genre: sparse one-hot, ~3 per row
    gi = BLOCK_SLICES['genre']
    for i in range(n):
        cols = rng.choice(BLOCK_DIMS['genre'], size=3, replace=False)
        X[i, gi.start + cols] = 1.0
    # language: sparse one-hot, exactly 1 per row
    li = BLOCK_SLICES['language']
    cols = rng.integers(0, BLOCK_DIMS['language'], size=n)
    for i, c in enumerate(cols):
        X[i, li.start + c] = 1.0
    # decade: float in [0,1] + binary flag
    di = BLOCK_SLICES['decade']
    X[:, di.start] = rng.uniform(0, 1, size=n).astype(np.float32)
    X[:, di.start + 1] = (rng.uniform(0, 1, size=n) > 0.1).astype(np.float32)
    # awards: lognormal-ish positive
    X[:, BLOCK_SLICES['awards']] = np.abs(rng.standard_normal((n, BLOCK_DIMS['awards']))).astype(np.float32)
    # text: random unit-norm vector per row (mimics L2-normalized embedding)
    text_raw = rng.standard_normal((n, BLOCK_DIMS['text'])).astype(np.float32)
    text_norm = np.linalg.norm(text_raw, axis=1, keepdims=True)
    X[:, BLOCK_SLICES['text']] = text_raw / np.maximum(text_norm, 1e-8)
    # director: bio_pca (64) sparse + has_bio + lang one-hot (31) + country one-hot (16) + has_lang
    di = BLOCK_SLICES['director']
    has_bio = (rng.uniform(0, 1, size=n) < 0.05).astype(np.float32)  # 5% bio coverage
    bio_raw = rng.standard_normal((n, 64)).astype(np.float32)
    bio_norm = np.linalg.norm(bio_raw, axis=1, keepdims=True)
    bio_unit = bio_raw / np.maximum(bio_norm, 1e-8) * has_bio[:, None]
    X[:, di.start:di.start + 64] = bio_unit
    X[:, di.start + 64] = has_bio
    # dir_lang: one-hot in 31 slots
    dl_cols = rng.integers(0, 31, size=n)
    for i, c in enumerate(dl_cols):
        X[i, di.start + 65 + c] = 1.0
    # dir_country: one-hot in 16 slots
    dc_cols = rng.integers(0, 16, size=n)
    for i, c in enumerate(dc_cols):
        X[i, di.start + 96 + c] = 1.0
    # has_dir_lang: mostly 1
    X[:, di.start + 112] = (rng.uniform(0, 1, size=n) > 0.001).astype(np.float32)
    return X


@pytest.fixture
def synthetic_feature_names():
    """Feature names matching production order — must align with BLOCK_DIMS above."""
    names = []
    names += ['log_popularity', 'log_vote_count', 'runtime_norm', 'vote_average_norm',
              'has_vote', 'has_engagement']
    names += [f'genre_g{i}' for i in range(20)] + ['genre_Unknown', 'has_genre']
    names += [f'lang_l{i}' for i in range(30)] + ['lang_other']
    names += ['decade_norm', 'has_release_date']
    names += ['prior_log_total_nominations', 'prior_log_total_wins',
              'prior_log_oscar_nominations', 'prior_log_oscar_wins',
              'prior_log_palme_nominations', 'prior_log_palme_wins']
    names += [f'text_{i}' for i in range(384)]
    names += [f'dir_bio_pca_{i}' for i in range(64)] + ['has_director_bio']
    names += [f'dir_lang_l{i}' for i in range(30)] + ['dir_lang_other']
    names += [f'dir_country_c{i}' for i in range(15)] + ['dir_country_other']
    names += ['has_director_lang']
    assert len(names) == TOTAL_DIM, f"got {len(names)}, expected {TOTAL_DIM}"
    return names


@pytest.fixture
def synthetic_labels():
    """Synthetic ground-truth labels for the 200-row fixture."""
    rng = np.random.default_rng(42)
    n = 200
    return {
        'primary_genre': rng.integers(0, 21, size=n),
        'decade_bin':    rng.integers(0, 13, size=n),
        'lang_top10':    rng.integers(0, 11, size=n),
    }


@pytest.fixture
def synthetic_blocks_dict(synthetic_feature_matrix, block_slices):
    """The same matrix split into a dict keyed by block name (torch tensors)."""
    X = synthetic_feature_matrix
    return {
        b: torch.from_numpy(X[:, slc]).float()
        for b, slc in block_slices.items()
    }


@pytest.fixture
def synthetic_has_bio(synthetic_blocks_dict):
    """has_director_bio flag extracted from director block (column 64)."""
    return synthetic_blocks_dict['director'][:, 64]
```

- [ ] **Step 1.4: Append entries to `.gitignore`.**

```bash
cd "<repo-root>"
cat >> .gitignore <<'EOF'

# Modeling phase
artifacts/models/
artifacts/eval/
*.pt
__pycache__/
.pytest_cache/
.venv/
*.egg-info/
src/cineembed.egg-info/
EOF
```

- [ ] **Step 1.5: Add a green-from-start import test.**

Create `tests/test_import.py`:
```python
"""Sanity test — verifies the package installs and exposes its version."""
import cineembed


def test_import_and_version():
    assert hasattr(cineembed, '__version__')
    assert cineembed.__version__ == "0.1.0"
```

- [ ] **Step 1.6: Install package in editable mode and verify pytest passes.**

```bash
cd "<repo-root>"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q
pip install -e ".[dev]" -q

# verify cineembed importable
python3 -c "import cineembed; print(f'cineembed v{cineembed.__version__}')"
# expected: cineembed v0.1.0

# pytest must be GREEN (1 passed) — exit code 0
pytest tests/ -v
# expected: tests/test_import.py::test_import_and_version PASSED  [100%]
#           ===== 1 passed in 0.0Xs =====
```

If `pip install -e .` errors, ensure `pyproject.toml` is at repo root (not nested) and `src/cineembed/__init__.py` exists.

- [ ] **Step 1.7: Commit.**

```bash
git add pyproject.toml src/ tests/ .gitignore
git commit -m "feat(cineembed): package skeleton + pytest scaffolding (1 import test)"
```

---

## Task 2 — `data.py` (load + block indices + labels + dataloader)

**Files:**
- Create: `src/cineembed/data.py`
- Create: `tests/test_data.py`

This module loads `feature_matrix.npz`, derives block boundaries from `feature_names`, extracts ground-truth labels from `movies_eda_final.csv`, and provides a PyTorch DataLoader for training.

- [ ] **Step 2.1: Write the failing tests in `tests/test_data.py`.**

```python
import json
import numpy as np
import pandas as pd
import pytest
import torch

from cineembed import data


def test_get_block_indices_matches_expected_dims(synthetic_feature_names):
    indices = data.get_block_indices(synthetic_feature_names)
    expected = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
                'awards': 6, 'text': 384, 'director': 113}
    for block, want_dim in expected.items():
        slc = indices[block]
        assert slc.stop - slc.start == want_dim, f"{block}: got {slc.stop-slc.start}, want {want_dim}"
    assert sum(slc.stop - slc.start for slc in indices.values()) == 564


def test_load_feature_matrix(tmp_path, synthetic_feature_matrix, synthetic_feature_names):
    path = tmp_path / "feature_matrix.npz"
    np.savez(path, X=synthetic_feature_matrix, feature_names=np.array(synthetic_feature_names, dtype=object))

    X, names = data.load_feature_matrix(path)
    assert isinstance(X, torch.Tensor)
    assert X.shape == synthetic_feature_matrix.shape
    assert X.dtype == torch.float32
    assert list(names) == synthetic_feature_names


def test_get_labels_from_csv(tmp_path):
    df = pd.DataFrame({
        'genres':            ['Drama|Crime', 'Action', '', 'Comedy|Romance|Drama'],
        'decade':            [1990, 2000, 0, 2010],
        'original_language': ['en', 'fr', 'en', 'es'],
    })
    csv_path = tmp_path / "movies_eda_final.csv"
    df.to_csv(csv_path, index=False)

    labels = data.get_labels(csv_path)
    assert labels['primary_genre'].tolist() == ['Drama', 'Action', 'Unknown', 'Comedy']
    assert labels['decade_bin'].tolist() == [1990, 2000, 0, 2010]
    assert set(labels['lang_top10'].unique()).issubset({'en', 'fr', 'es', 'other'})


def test_train_val_split_deterministic():
    n = 100
    train_idx, val_idx = data.train_val_split(n, val_frac=0.1, seed=42)
    assert len(train_idx) == 90
    assert len(val_idx) == 10
    assert set(train_idx).isdisjoint(set(val_idx))
    # determinism
    train_idx2, val_idx2 = data.train_val_split(n, val_frac=0.1, seed=42)
    np.testing.assert_array_equal(train_idx, train_idx2)
    np.testing.assert_array_equal(val_idx, val_idx2)


def test_make_dataloader_yields_blocks_and_has_bio(synthetic_feature_matrix, block_slices):
    X = torch.from_numpy(synthetic_feature_matrix)
    has_bio = X[:, block_slices['director'].start + 64]
    loader = data.make_dataloader(X, has_bio, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    assert 'blocks' in batch and 'has_bio' in batch
    assert batch['blocks']['numerical'].shape == (32, 6)
    assert batch['blocks']['text'].shape == (32, 384)
    assert batch['has_bio'].shape == (32,)
```

- [ ] **Step 2.2: Run the tests — confirm failures.**

```bash
pytest tests/test_data.py -v
```
Expected: all 5 tests fail with `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 2.3: Implement `src/cineembed/data.py`.**

```python
"""Data loading and label extraction for the cineembed modeling phase."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Mapping from block name to identifying prefixes in feature_names (per spec §3.1).
BLOCK_PREFIX_RULES = {
    'numerical': ('log_popularity', 'log_vote_count', 'runtime_norm',
                  'vote_average_norm', 'has_vote', 'has_engagement'),
    'genre':     ('genre_', 'has_genre'),
    'language':  ('lang_',),
    'decade':    ('decade_norm', 'has_release_date'),
    'awards':    ('prior_log_',),
    'text':      ('text_',),
    'director':  ('dir_bio_pca_', 'has_director_bio', 'dir_lang_',
                  'dir_country_', 'has_director_lang'),
}

BLOCK_ORDER = ['numerical', 'genre', 'language', 'decade', 'awards', 'text', 'director']


def _classify_feature(name: str) -> str:
    """Return the block name a feature column belongs to (deterministic priority)."""
    # 'director' takes priority over 'language' because dir_lang_* would also match lang_
    for block in ['numerical', 'genre', 'director', 'language', 'decade', 'awards', 'text']:
        for pfx in BLOCK_PREFIX_RULES[block]:
            if name == pfx or name.startswith(pfx):
                return block
    raise ValueError(f"Cannot classify feature {name!r}")


def get_block_indices(feature_names: list[str]) -> dict[str, slice]:
    """Derive per-block column slices by scanning feature_names left-to-right.

    Production feature_matrix has block-contiguous columns. Each block's slice
    is identified by classifying every column and grouping adjacent same-class runs.
    """
    classifications = [_classify_feature(n) for n in feature_names]
    out: dict[str, slice] = {}
    i = 0
    while i < len(classifications):
        block = classifications[i]
        if block in out:
            raise ValueError(f"Block {block!r} appears non-contiguously near col {i}")
        j = i
        while j < len(classifications) and classifications[j] == block:
            j += 1
        out[block] = slice(i, j)
        i = j
    missing = set(BLOCK_ORDER) - set(out.keys())
    if missing:
        raise ValueError(f"Missing blocks: {missing}")
    return out


def load_feature_matrix(path: str | Path) -> tuple[torch.Tensor, list[str]]:
    """Load (X, feature_names) from the EDA artifact.

    X is converted to torch.float32. feature_names is returned as a Python list.
    """
    archive = np.load(Path(path), allow_pickle=True)
    X_np = archive['X'].astype(np.float32)
    names = list(archive['feature_names'])
    return torch.from_numpy(X_np), names


def _bin_decade(value):
    try:
        v = int(value)
    except (ValueError, TypeError):
        return 0
    return v if v > 0 else 0


def get_labels(csv_path: str | Path, top_lang_n: int = 10) -> dict[str, np.ndarray]:
    """Derive three orthogonal label vectors from movies_eda_final.csv (spec §3.2)."""
    df = pd.read_csv(csv_path, low_memory=False)

    # primary_genre = first piece of 'genres' before '|'; empty → 'Unknown'
    genres_first = df.get('genres', pd.Series([''] * len(df))).fillna('').astype(str)
    primary_genre = genres_first.apply(lambda s: s.split('|')[0] if s else 'Unknown')
    primary_genre = primary_genre.replace('', 'Unknown').to_numpy()

    decade_bin = df.get('decade', pd.Series([0] * len(df))).apply(_bin_decade).to_numpy()

    lang = df.get('original_language', pd.Series([''] * len(df))).fillna('other').astype(str)
    top = lang.value_counts().head(top_lang_n).index
    lang_top10 = lang.where(lang.isin(top), 'other').to_numpy()

    return {
        'primary_genre': primary_genre,
        'decade_bin':    decade_bin,
        'lang_top10':    lang_top10,
    }


def train_val_split(n: int, val_frac: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random index split — deterministic given (n, val_frac, seed)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


class _BlocksDataset(Dataset):
    """Memory-efficient view of (X, has_bio) optionally restricted to indices.

    IMPORTANT: never copies X — both train and val datasets share the same underlying
    tensor and only differ in their `indices` array. With X at (329044, 564), copying
    would balloon Colab RAM to several GB unnecessarily.
    """
    def __init__(
        self,
        X: torch.Tensor,
        has_bio: torch.Tensor,
        indices: np.ndarray | None = None,
    ):
        self.X = X
        self.has_bio = has_bio
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0] if self.indices is None else len(self.indices)

    def __getitem__(self, i):
        real_i = int(self.indices[i]) if self.indices is not None else int(i)
        return {'X': self.X[real_i], 'has_bio': self.has_bio[real_i]}


def _split_into_blocks(X_batch: torch.Tensor, block_slices: dict[str, slice]) -> dict[str, torch.Tensor]:
    return {b: X_batch[:, slc] for b, slc in block_slices.items()}


def _collate(batch_list, block_slices):
    X = torch.stack([b['X'] for b in batch_list], dim=0)
    has_bio = torch.stack([b['has_bio'] for b in batch_list], dim=0)
    return {'blocks': _split_into_blocks(X, block_slices), 'has_bio': has_bio}


def make_dataloader(
    X: torch.Tensor,
    has_bio: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool = True,
    indices: np.ndarray | None = None,
    block_slices: dict[str, slice] | None = None,
    seed: int | None = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader yielding {'blocks': dict, 'has_bio': tensor} batches."""
    if block_slices is None:
        # Caller did not provide block_slices — assume production order via heuristic.
        # Production callers should pass explicit block_slices from get_block_indices().
        # Default fallback uses the canonical block dim layout.
        from cineembed.data import BLOCK_ORDER as _BO
        # Construct slices by cumulative sums based on canonical dims
        canonical_dims = [6, 22, 31, 2, 6, 384, 113]
        out, start = {}, 0
        for b, d in zip(_BO, canonical_dims):
            out[b] = slice(start, start + d)
            start += d
        block_slices = out

    dataset = _BlocksDataset(X, has_bio, indices=indices)
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        collate_fn=lambda batch: _collate(batch, block_slices),
    )
```

- [ ] **Step 2.4: Re-run tests — confirm passes.**

```bash
pytest tests/test_data.py -v
```
Expected: 5 passed.

- [ ] **Step 2.5: Commit.**

```bash
git add src/cineembed/data.py tests/test_data.py
git commit -m "feat(data): load_feature_matrix, get_block_indices, get_labels, dataloader"
```

---

## Task 3 — `losses.py` (W2 + G2 + canonical recon + ELBO + DEC KL)

**Files:**
- Create: `src/cineembed/losses.py`
- Create: `tests/test_losses.py`

Per spec §5. Includes the canonical `weighted_recon_loss` helper that consolidates W2 + G2 (avoiding double-count).

- [ ] **Step 3.1: Write failing tests.**

```python
import numpy as np
import pytest
import torch

from cineembed import losses


@pytest.fixture
def synthetic_decoded_input():
    """Decoded ≈ input + noise; per-block dicts matching block dims."""
    torch.manual_seed(42)
    n = 16
    block_dims = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
                  'awards': 6, 'text': 384, 'director': 113}
    inp = {b: torch.randn(n, d) for b, d in block_dims.items()}
    dec = {b: inp[b] + 0.1 * torch.randn_like(inp[b]) for b in inp}
    has_bio = torch.tensor([1.0] * 4 + [0.0] * 12)
    return dec, inp, has_bio


def test_compute_block_weights_clipping():
    """W2 weights should be clipped to [0.1, 10.0]."""
    block_indices = {'a': slice(0, 2), 'b': slice(2, 5)}
    # Block 'a' has near-zero variance → weight would be huge → clipped to 10.0
    # Block 'b' has high variance → weight tiny → clipped to 0.1
    X = np.zeros((100, 5), dtype=np.float32)
    X[:, 0] = 0.001  # near-zero variance
    X[:, 1] = 0.001
    X[:, 2:] = np.random.randn(100, 3) * 100  # high variance

    w = losses.compute_block_weights(X, block_indices, w_min=0.1, w_max=10.0)
    assert w['a'] == 10.0
    assert w['b'] == 0.1


def test_director_block_loss_masks_bio_when_no_bio():
    """When has_bio is all-zero, the bio_pca half contributes 0 (mask sums to 0
    → clamp_min(1) prevents NaN; numerator already 0)."""
    dec_dir = torch.randn(8, 113)
    inp_dir = torch.zeros(8, 113)
    has_bio = torch.zeros(8)
    loss = losses.director_block_loss(dec_dir, inp_dir, has_bio, w_block=1.0)
    # Only the non-bio cols (64..113) contribute. Bio cols (0..64) masked → no contribution.
    expected_other = ((dec_dir[:, 64:] - inp_dir[:, 64:]) ** 2).mean()
    expected_total = 0.5 * (0.0 + expected_other)
    assert torch.allclose(loss, torch.tensor(float(expected_total)), atol=1e-5)


def test_director_block_loss_uses_bio_when_all_present():
    """When has_bio is all-one, both halves contribute equally weighted."""
    torch.manual_seed(0)
    dec_dir = torch.randn(8, 113)
    inp_dir = torch.zeros(8, 113)
    has_bio = torch.ones(8)
    loss = losses.director_block_loss(dec_dir, inp_dir, has_bio, w_block=1.0)
    bio_diff = (dec_dir[:, :64] - inp_dir[:, :64]) ** 2
    bio_loss = bio_diff.mean()  # mean over (8, 64) when mask all-one
    other_loss = ((dec_dir[:, 64:] - inp_dir[:, 64:]) ** 2).mean()
    expected = 0.5 * (bio_loss + other_loss)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_weighted_recon_loss_no_double_count(synthetic_decoded_input):
    """Director must NOT be summed twice (regression test for the D10 bug)."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks = {b: 1.0 for b in inp}

    loss_full = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks)

    # Manual computation matching the spec §5.2.1 contract.
    other = sum(((dec[b] - inp[b]) ** 2).mean() for b in inp if b != 'director')
    dir_loss = losses.director_block_loss(dec['director'], inp['director'], has_bio, w_blocks['director'])
    expected = other + dir_loss
    assert torch.allclose(loss_full, expected, atol=1e-5)


def test_weighted_recon_loss_uniform_equals_uniform_weights(synthetic_decoded_input):
    """W1 baseline = canonical loss with all weights = 1."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks_one = {b: 1.0 for b in inp}
    expected = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks_one)
    actual = losses.weighted_recon_loss_uniform(dec, inp, has_bio)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_vae_elbo_returns_recon_kl_separately(synthetic_decoded_input):
    """ELBO returns (loss, recon_value, kl_value) — used for logging."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks = {b: 1.0 for b in inp}
    n = inp['numerical'].shape[0]
    mu = torch.randn(n, 16)
    log_var = torch.zeros(n, 16)  # σ=1 → KL has only mu^2 contribution
    loss, recon_val, kl_val = losses.vae_elbo(dec, inp, mu, log_var, has_bio, w_blocks, beta=0.5)
    expected_kl = 0.5 * (mu ** 2).sum(dim=1).mean()
    assert abs(kl_val - expected_kl.item()) < 1e-4
    # loss = recon + 0.5 * kl
    assert abs(loss.item() - (recon_val + 0.5 * kl_val)) < 1e-4


def test_weighted_recon_loss_exclude_blocks(synthetic_decoded_input):
    """F1/F2 ablation: exclude_blocks must skip the named block from the sum."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks = {b: 1.0 for b in inp}

    full = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks)
    no_text = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks, exclude_blocks={'text'})
    expected_diff = w_blocks['text'] * torch.nn.functional.mse_loss(dec['text'], inp['text'])
    assert torch.allclose(full - no_text, expected_diff, atol=1e-5)

    # Excluding 'director' must skip the G2 helper too
    no_dir = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks, exclude_blocks={'director'})
    expected_dir = losses.director_block_loss(dec['director'], inp['director'], has_bio, w_blocks['director'])
    assert torch.allclose(full - no_dir, expected_dir, atol=1e-5)


def test_dec_loss_runs_and_returns_components():
    """DEC loss returns (loss, kl_val, recon_val) and computes batch-wise P/Q."""
    torch.manual_seed(0)
    n, k, z_dim = 16, 5, 8
    z = torch.randn(n, z_dim)
    centers = torch.randn(k, z_dim)
    block_dims = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
                  'awards': 6, 'text': 384, 'director': 113}
    inp = {b: torch.randn(n, d) for b, d in block_dims.items()}
    dec = {b: inp[b] + 0.05 * torch.randn_like(inp[b]) for b in inp}
    has_bio = torch.zeros(n)
    w_blocks = {b: 1.0 for b in inp}

    loss, kl_val, recon_val = losses.dec_loss(z, dec, inp, centers, has_bio, w_blocks, lambda_recon=0.1)
    assert loss.requires_grad or loss.dim() == 0  # scalar
    assert kl_val >= 0
    assert recon_val >= 0
```

- [ ] **Step 3.2: Run tests — confirm failures.**

```bash
pytest tests/test_losses.py -v
```

- [ ] **Step 3.3: Implement `src/cineembed/losses.py`.**

```python
"""Loss functions for AE / VAE / DEC training (spec §5)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_block_weights(
    X: np.ndarray,
    block_indices: dict[str, slice],
    *,
    w_min: float = 0.1,
    w_max: float = 10.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Block-level inverse-variance weights with clipping (spec §5.1, D9 fix).

    w_b = clip(1 / total_variance(block), w_min, w_max)
    """
    weights = {}
    for block, slc in block_indices.items():
        block_var = X[:, slc].var(axis=0).sum()
        w_raw = 1.0 / max(float(block_var), eps)
        weights[block] = float(np.clip(w_raw, w_min, w_max))
    return weights


def director_block_loss(
    dec_dir: torch.Tensor,
    inp_dir: torch.Tensor,
    has_bio: torch.Tensor,
    w_block: float,
) -> torch.Tensor:
    """G2 masked director-block loss (spec §5.2).

    The 64 dir_bio_pca_* columns are MASKED to zero loss for rows where has_bio == 0.
    The remaining 49 columns (has_bio flag + lang/country one-hot + has_lang) are
    treated normally.

    Args:
        dec_dir, inp_dir: (B, 113)
        has_bio: (B,) in {0, 1}

    Returns:
        scalar tensor: w_block * 0.5 * (loss_bio + loss_other)
    """
    bio_pca_slice = slice(0, 64)
    other_slice = slice(64, 113)

    bio_diff = (dec_dir[:, bio_pca_slice] - inp_dir[:, bio_pca_slice]) ** 2
    bio_mask = has_bio.unsqueeze(1)  # (B, 1) broadcasts to (B, 64)
    masked_sum = (bio_diff * bio_mask).sum()
    n_masked = bio_mask.sum().clamp_min(1.0)
    loss_bio = masked_sum / n_masked / 64.0

    other_diff = (dec_dir[:, other_slice] - inp_dir[:, other_slice]) ** 2
    loss_other = other_diff.mean()

    return w_block * 0.5 * (loss_bio + loss_other)


def weighted_recon_loss(
    decoded: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    has_bio: torch.Tensor,
    w_blocks: dict[str, float],
    exclude_blocks: set | None = None,
) -> torch.Tensor:
    """Canonical W2 + G2 reconstruction loss (spec §5.2.1).

    'director' is excluded from the generic sum and added via director_block_loss
    to apply the G2 mask. Including 'director' in the sum would double-count.

    Args:
        exclude_blocks: optional set of block names to skip. Used by F1/F2
            modality ablation runs (spec §8.3.2) — skipping the masked block in
            both forward AND loss prevents the model from being penalized for
            failing to reconstruct the zero input. If 'director' is in
            exclude_blocks, its G2 helper is also skipped.
    """
    skip = exclude_blocks or set()
    other = sum(
        w_blocks[b] * F.mse_loss(decoded[b], target[b])
        for b in target
        if b != 'director' and b not in skip
    )
    if 'director' in skip:
        return other
    return other + director_block_loss(
        decoded['director'], target['director'], has_bio, w_blocks['director']
    )


def weighted_recon_loss_uniform(
    decoded: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    has_bio: torch.Tensor,
) -> torch.Tensor:
    """W1 ablation: uniform weights (spec §5.3)."""
    uniform = {b: 1.0 for b in target}
    return weighted_recon_loss(decoded, target, has_bio, uniform)


def vae_elbo(
    decoded: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    mu: torch.Tensor,
    log_var: torch.Tensor,
    has_bio: torch.Tensor,
    w_blocks: dict[str, float],
    beta: float,
) -> tuple[torch.Tensor, float, float]:
    """ELBO = recon + β * KL (spec §5.5).

    Returns (loss_tensor, recon_value_float, kl_value_float) for training-loop logging.
    """
    recon = weighted_recon_loss(decoded, target, has_bio, w_blocks)
    kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1).mean()
    loss = recon + beta * kl
    return loss, float(recon.item()), float(kl.item())


def dec_loss(
    z: torch.Tensor,
    decoded: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    cluster_centers: torch.Tensor,
    has_bio: torch.Tensor,
    w_blocks: dict[str, float],
    *,
    lambda_recon: float = 0.1,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """DEC KL on soft assignments + reconstruction grounding (spec §5.6, D10 batch-wise P).

    Args:
        z: (B, z_dim)
        cluster_centers: (k, z_dim) — learnable parameter

    Returns (loss, kl_value, recon_value).
    """
    # Soft assignments via Student-t kernel
    diff = z.unsqueeze(1) - cluster_centers.unsqueeze(0)  # (B, k, z_dim)
    q_unnorm = (1.0 + (diff ** 2).sum(dim=2) / alpha) ** -((alpha + 1.0) / 2.0)
    q = q_unnorm / q_unnorm.sum(dim=1, keepdim=True)  # (B, k)

    # Sharpened target P (batch-wise, detached)
    f = q.sum(dim=0)  # (k,)
    p_unnorm = q ** 2 / f
    p = p_unnorm / p_unnorm.sum(dim=1, keepdim=True)
    p = p.detach()

    kl = (p * (p / q.clamp_min(1e-12)).log()).sum(dim=1).mean()

    recon = weighted_recon_loss(decoded, target, has_bio, w_blocks)

    loss = kl + lambda_recon * recon
    return loss, float(kl.item()), float(recon.item())


class LearnedWeightedLoss(nn.Module):
    """W4 stretch: Kendall et al. 2018 learned uncertainty weighting (spec §5.4).

    Per-block scalar log_sigma is jointly trained with the network.
    """
    def __init__(self, block_names: list[str]):
        super().__init__()
        self.block_names = list(block_names)
        # init log_sigma=0 → exp(-log_sigma)=1 → uniform initial weighting
        self.log_sigma = nn.Parameter(torch.zeros(len(self.block_names)))

    def forward(
        self,
        decoded: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        has_bio: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0.0
        for i, b in enumerate(self.block_names):
            if b == 'director':
                # director uses G2 masked loss; the learned weight scales the whole director loss
                block_loss = director_block_loss(decoded[b], target[b], has_bio, w_block=1.0)
            else:
                block_loss = F.mse_loss(decoded[b], target[b])
            s = self.log_sigma[i]
            loss = loss + torch.exp(-s) * block_loss + 0.5 * s
        return loss
```

- [ ] **Step 3.4: Re-run tests — confirm passes.**

```bash
pytest tests/test_losses.py -v
```
Expected: 7 passed.

- [ ] **Step 3.5: Commit.**

```bash
git add src/cineembed/losses.py tests/test_losses.py
git commit -m "feat(losses): W2 + G2 canonical recon, ELBO, DEC KL, W4 learned weights"
```

---

## Task 4 — `backbone.py` (MultiModalBackbone)

**Files:**
- Create: `src/cineembed/backbone.py`
- Create: `tests/test_backbone.py`

The shared multi-modal encoder per spec §4.1: per-modality projection → concat → FC stack → latent.

- [ ] **Step 4.1: Write failing tests.**

```python
import torch

from cineembed import backbone


PROJ_DIMS = {
    'numerical': 16, 'genre': 16, 'language': 16, 'decade': 4,
    'awards': 16, 'text': 64, 'director': 32,
}
BLOCK_DIMS = {
    'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
    'awards': 6, 'text': 384, 'director': 113,
}


def test_backbone_output_shape(synthetic_blocks_dict):
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z = model(synthetic_blocks_dict)
    assert z.shape == (200, 64)


def test_backbone_deterministic_with_seed(synthetic_blocks_dict):
    torch.manual_seed(42)
    m1 = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z1 = m1(synthetic_blocks_dict)

    torch.manual_seed(42)
    m2 = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z2 = m2(synthetic_blocks_dict)
    assert torch.allclose(z1, z2)


def test_backbone_param_count_under_500k():
    """Backbone size sanity — too big = wrong design."""
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 10_000 < n_params < 500_000, f"got {n_params}"


def test_backbone_supports_different_latent_dims(synthetic_blocks_dict):
    for z_dim in [32, 64, 128]:
        model = backbone.MultiModalBackbone(
            block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=z_dim,
        )
        z = model(synthetic_blocks_dict)
        assert z.shape == (200, z_dim)


def test_backbone_block_mask_zeros_modality(synthetic_blocks_dict):
    """F1/F2 ablation: block_mask={'text': 0.0} → text projection contribution removed."""
    torch.manual_seed(42)
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z_full = model(synthetic_blocks_dict)
    mask_no_text = {b: 1.0 for b in BLOCK_DIMS}
    mask_no_text['text'] = 0.0
    z_no_text = model(synthetic_blocks_dict, block_mask=mask_no_text)
    # Ablating text MUST change the output
    assert not torch.allclose(z_full, z_no_text), \
        "block_mask={text: 0.0} should change z; backbone isn't honoring the mask"
```

- [ ] **Step 4.2: Run — confirm failures.**

```bash
pytest tests/test_backbone.py -v
```

- [ ] **Step 4.3: Implement `src/cineembed/backbone.py`.**

```python
"""Multi-modal projection backbone (spec §4.1)."""
from __future__ import annotations

import torch
import torch.nn as nn


# Default projection dimensions per spec §4.1 — modality-specific compression ratios.
DEFAULT_PROJ_DIMS = {
    'numerical': 16,
    'genre':     16,
    'language':  16,
    'decade':    4,
    'awards':    16,
    'text':      64,
    'director':  32,
}


class _BlockProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiModalBackbone(nn.Module):
    """Per-modality projection layers → concat → FC backbone → latent z.

    Same backbone is used by AEHead, VAEHead, DECHead — only the head differs.
    """
    def __init__(
        self,
        block_dims: dict[str, int],
        proj_dims: dict[str, int] | None = None,
        *,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        proj_dims = proj_dims if proj_dims is not None else dict(DEFAULT_PROJ_DIMS)
        self.block_order = list(block_dims.keys())
        self.proj_dims = proj_dims

        # Modality-specific projections
        self.projections = nn.ModuleDict({
            b: _BlockProjection(
                in_dim=block_dims[b],
                out_dim=proj_dims[b],
                dropout=0.2 if b in ('text', 'director') else 0.1,
            )
            for b in self.block_order
        })

        concat_dim = sum(proj_dims[b] for b in self.block_order)

        # Backbone FC stack
        self.backbone = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.concat_dim = concat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(
        self,
        blocks: dict[str, torch.Tensor],
        block_mask: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            blocks: per-block input tensors.
            block_mask: optional dict {block_name: 0.0 or 1.0}. A value of 0.0 zeros
                that block's projected output before concatenation, simulating
                "modality removed" for ablation studies (F1/F2 in spec §8.3.2).
                Missing keys default to 1.0 (kept).
        """
        projected = []
        for b in self.block_order:
            p = self.projections[b](blocks[b])
            if block_mask is not None and block_mask.get(b, 1.0) == 0.0:
                p = torch.zeros_like(p)
            projected.append(p)
        h = torch.cat(projected, dim=1)
        return self.backbone(h)
```

- [ ] **Step 4.4: Re-run — confirm passes.**

```bash
pytest tests/test_backbone.py -v
```
Expected: 4 passed.

- [ ] **Step 4.5: Commit.**

```bash
git add src/cineembed/backbone.py tests/test_backbone.py
git commit -m "feat(backbone): MultiModalBackbone with per-modality projections + FC stack"
```

---

## Task 5 — `heads.py` (AEHead, VAEHead, DECHead)

**Files:**
- Create: `src/cineembed/heads.py`
- Create: `tests/test_heads.py`

Three model heads per spec §4.2. AE: deterministic decoder. VAE: μ/σ posterior + reparameterization. DEC: cluster centroids + Student-t kernel.

- [ ] **Step 5.1: Write failing tests.**

```python
import numpy as np
import pytest
import torch

from cineembed import backbone, heads


PROJ_DIMS = {'numerical': 16, 'genre': 16, 'language': 16, 'decade': 4,
             'awards': 16, 'text': 64, 'director': 32}
BLOCK_DIMS = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
              'awards': 6, 'text': 384, 'director': 113}


@pytest.fixture
def fresh_backbone():
    return backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )


def test_ae_head_reconstructs_all_blocks(fresh_backbone, synthetic_blocks_dict):
    head = heads.AEHead(
        backbone=fresh_backbone, block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128,
    )
    decoded = head(synthetic_blocks_dict)
    assert set(decoded.keys()) == set(BLOCK_DIMS.keys())
    for b, d in BLOCK_DIMS.items():
        assert decoded[b].shape == (200, d), f"{b}: {decoded[b].shape}"


def test_vae_head_returns_mu_log_var_and_decoded(fresh_backbone, synthetic_blocks_dict):
    head = heads.VAEHead(
        backbone=fresh_backbone, block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128,
    )
    decoded, mu, log_var = head(synthetic_blocks_dict)
    assert mu.shape == (200, 64)
    assert log_var.shape == (200, 64)
    assert decoded['numerical'].shape == (200, 6)
    # Sampling means consecutive forward passes differ
    decoded_2, _, _ = head(synthetic_blocks_dict)
    assert not torch.allclose(decoded['numerical'], decoded_2['numerical'])


def test_dec_head_initialize_centers_from_kmeans(fresh_backbone, synthetic_blocks_dict):
    ae_head = heads.AEHead(
        backbone=fresh_backbone, block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128,
    )
    dec_head = heads.DECHead(
        backbone=fresh_backbone,
        ae_decoder=ae_head.decoder,
        n_clusters=10,
        latent_dim=64,
    )
    # Initialize from synthetic encoder outputs
    with torch.no_grad():
        z_init = fresh_backbone(synthetic_blocks_dict)
    dec_head.initialize_centers(z_init.numpy(), seed=42)
    assert dec_head.cluster_centers.shape == (10, 64)


def test_dec_head_forward_returns_z_decoded_q(fresh_backbone, synthetic_blocks_dict):
    ae_head = heads.AEHead(
        backbone=fresh_backbone, block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128,
    )
    dec_head = heads.DECHead(
        backbone=fresh_backbone, ae_decoder=ae_head.decoder, n_clusters=10, latent_dim=64,
    )
    with torch.no_grad():
        z_init = fresh_backbone(synthetic_blocks_dict)
    dec_head.initialize_centers(z_init.numpy(), seed=42)

    z, decoded, q = dec_head(synthetic_blocks_dict)
    assert z.shape == (200, 64)
    assert decoded['numerical'].shape == (200, 6)
    assert q.shape == (200, 10)
    # q rows sum to ~1 (probability distribution)
    row_sums = q.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(200), atol=1e-4)


def test_dec_head_reinit_collapsed_centers(fresh_backbone, synthetic_blocks_dict):
    """Re-init must replace centers whose cluster-count is below the floor."""
    ae_head = heads.AEHead(
        backbone=fresh_backbone, block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128,
    )
    dec_head = heads.DECHead(
        backbone=fresh_backbone, ae_decoder=ae_head.decoder, n_clusters=5, latent_dim=64,
    )
    # Initialize centers, then snapshot the current state
    with torch.no_grad():
        z_init = fresh_backbone(synthetic_blocks_dict)
    dec_head.initialize_centers(z_init.numpy(), seed=42)
    centers_before = dec_head.cluster_centers.clone()

    # Force two clusters into "collapsed" state (count < floor * n_total)
    counts = torch.tensor([100, 80, 0, 0, 100])  # clusters 2 and 3 are collapsed
    z_pool = z_init.numpy()
    n_reinit = dec_head.reinit_collapsed_centers(
        cluster_counts=counts, z_pool=z_pool, n_total=300, size_floor=0.001, seed=42,
    )
    assert n_reinit == 2

    # The two collapsed centers should have moved; the others should be unchanged
    diff = (dec_head.cluster_centers - centers_before).abs().sum(dim=1)
    assert diff[0] == 0 and diff[1] == 0 and diff[4] == 0, \
        f"non-collapsed centers were modified: {diff.tolist()}"
    assert diff[2] > 0 and diff[3] > 0, \
        f"collapsed centers were not re-initialized: {diff.tolist()}"
```

- [ ] **Step 5.2: Run — confirm failures.**

```bash
pytest tests/test_heads.py -v
```

- [ ] **Step 5.3: Implement `src/cineembed/heads.py`.**

```python
"""AE / VAE / DEC heads (spec §4.2)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from cineembed.backbone import MultiModalBackbone


class _BlockDecoder(nn.Module):
    """Per-block decoder: latent sub-vector → ReLU → block original dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(in_dim, max(in_dim, 32)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_dim, 32), out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(x)


class _MultiModalDecoder(nn.Module):
    """Mirror of MultiModalBackbone — latent → split into per-block sub-latents → reconstructions."""
    def __init__(
        self,
        block_dims: dict[str, int],
        proj_dims: dict[str, int],
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.block_order = list(block_dims.keys())
        self.proj_dims = proj_dims
        concat_dim = sum(proj_dims[b] for b in self.block_order)

        # Latent → hidden → concat (mirror of backbone FC)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, concat_dim),
        )
        # Per-block decoders
        self.decoders = nn.ModuleDict({
            b: _BlockDecoder(in_dim=proj_dims[b], out_dim=block_dims[b])
            for b in self.block_order
        })
        self._concat_dim = concat_dim

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fc(z)  # (B, concat_dim)
        # Split h into per-block sub-vectors (matching projection sizes)
        out = {}
        offset = 0
        for b in self.block_order:
            d = self.proj_dims[b]
            sub = h[:, offset:offset + d]
            out[b] = self.decoders[b](sub)
            offset += d
        return out


class AEHead(nn.Module):
    """Deterministic AutoEncoder head (spec §4.2.1)."""
    def __init__(
        self,
        backbone: MultiModalBackbone,
        block_dims: dict[str, int],
        proj_dims: dict[str, int],
        hidden_dim: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = _MultiModalDecoder(
            block_dims=block_dims,
            proj_dims=proj_dims,
            hidden_dim=hidden_dim,
            latent_dim=backbone.latent_dim,
        )

    def encode(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backbone(blocks)

    def forward(self, blocks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = self.encode(blocks)
        return self.decoder(z)


class VAEHead(nn.Module):
    """Variational AutoEncoder head with reparameterization (spec §4.2.2)."""
    def __init__(
        self,
        backbone: MultiModalBackbone,
        block_dims: dict[str, int],
        proj_dims: dict[str, int],
        hidden_dim: int,
    ):
        super().__init__()
        self.backbone = backbone
        z_dim = backbone.latent_dim
        # Two heads on top of backbone output for μ, log_var
        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_log_var = nn.Linear(z_dim, z_dim)
        self.decoder = _MultiModalDecoder(
            block_dims=block_dims,
            proj_dims=proj_dims,
            hidden_dim=hidden_dim,
            latent_dim=z_dim,
        )

    def encode(self, blocks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(blocks)
        return self.fc_mu(h), self.fc_log_var(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, blocks: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(blocks)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var


class DECHead(nn.Module):
    """Deep Embedded Clustering head (spec §4.2.3, D10 batch-wise P)."""
    def __init__(
        self,
        backbone: MultiModalBackbone,
        ae_decoder: _MultiModalDecoder,
        n_clusters: int,
        latent_dim: int,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = ae_decoder
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        # Cluster centers, learnable. Initialized via initialize_centers().
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    @torch.no_grad()
    def initialize_centers(self, z_array: np.ndarray, seed: int = 42) -> None:
        """Initialize cluster centers via KMeans on a precomputed latent array."""
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=seed)
        km.fit(z_array)
        centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        self.cluster_centers.data.copy_(centers)

    @torch.no_grad()
    def reinit_collapsed_centers(
        self,
        cluster_counts: torch.Tensor,
        z_pool: np.ndarray,
        n_total: int,
        size_floor: float = 0.001,
        seed: int = 42,
    ) -> int:
        """Re-initialize cluster centers that hold < size_floor * n_total samples.

        Mitigation for cluster collapse (spec §10). Called every epoch in the DEC
        training loop. New center is sampled from a random latent point in z_pool.

        Args:
            cluster_counts: (k,) tensor of per-cluster argmax counts over the dataset
            z_pool: (n_total, latent_dim) latent vectors to sample re-init points from
            n_total: total number of samples (denominator for floor check)

        Returns: number of clusters re-initialized.
        """
        floor_count = max(1, int(round(size_floor * n_total)))
        rng = np.random.default_rng(seed)
        n_reinit = 0
        for j in range(self.n_clusters):
            if int(cluster_counts[j].item()) < floor_count:
                # Sample a random latent point as the new center
                new_idx = int(rng.integers(0, z_pool.shape[0]))
                self.cluster_centers.data[j] = torch.from_numpy(z_pool[new_idx]).to(
                    self.cluster_centers.device
                )
                n_reinit += 1
        return n_reinit

    def soft_assignments(self, z: torch.Tensor) -> torch.Tensor:
        """Student-t kernel soft assignments q (B, k)."""
        diff = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)  # (B, k, z)
        q_unnorm = (1.0 + (diff ** 2).sum(dim=2) / self.alpha) ** -((self.alpha + 1.0) / 2.0)
        q = q_unnorm / q_unnorm.sum(dim=1, keepdim=True)
        return q

    def forward(
        self, blocks: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z = self.backbone(blocks)
        decoded = self.decoder(z)
        q = self.soft_assignments(z)
        return z, decoded, q
```

- [ ] **Step 5.4: Re-run — confirm passes.**

```bash
pytest tests/test_heads.py -v
```
Expected: 4 passed.

- [ ] **Step 5.5: Commit.**

```bash
git add src/cineembed/heads.py tests/test_heads.py
git commit -m "feat(heads): AEHead, VAEHead with reparameterization, DECHead with kmeans init"
```

---

## Task 6 — `train.py` (generic training loop with early stopping + checkpoint)

**Files:**
- Create: `src/cineembed/train.py`
- Create: `tests/test_train.py`

Per spec §4.3 (defaults). The loop is generic — works for AE, VAE, DEC by accepting a loss function callable.

- [ ] **Step 6.1: Write failing tests.**

```python
import json
import torch

from cineembed import data, heads, backbone, losses, train


PROJ_DIMS = {'numerical': 16, 'genre': 16, 'language': 16, 'decade': 4,
             'awards': 16, 'text': 64, 'director': 32}
BLOCK_DIMS = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
              'awards': 6, 'text': 384, 'director': 113}


def _make_loaders(synthetic_feature_matrix, block_slices):
    X = torch.from_numpy(synthetic_feature_matrix)
    has_bio = X[:, block_slices['director'].start + 64]
    train_idx, val_idx = data.train_val_split(X.shape[0], val_frac=0.1, seed=42)
    train_loader = data.make_dataloader(X, has_bio, batch_size=32, shuffle=True,
                                         indices=train_idx, block_slices=block_slices, seed=42)
    val_loader = data.make_dataloader(X, has_bio, batch_size=32, shuffle=False,
                                       indices=val_idx, block_slices=block_slices, seed=42)
    return train_loader, val_loader


def test_train_ae_loss_decreases(synthetic_feature_matrix, block_slices):
    bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=64, latent_dim=32)
    head = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=64)
    train_loader, val_loader = _make_loaders(synthetic_feature_matrix, block_slices)
    w_blocks = losses.compute_block_weights(synthetic_feature_matrix,
                                             {b: block_slices[b] for b in BLOCK_DIMS})

    def loss_fn(model, batch):
        decoded = model(batch['blocks'])
        return losses.weighted_recon_loss(decoded, batch['blocks'], batch['has_bio'], w_blocks)

    history = train.train_model(
        model=head, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=3, lr=1e-3, early_stop_patience=10, device='cpu',
    )
    assert 'train_loss' in history and 'val_loss' in history
    assert history['train_loss'][-1] < history['train_loss'][0], \
        f"train loss did not decrease: {history['train_loss']}"


def test_checkpoint_save_and_resume(tmp_path, synthetic_feature_matrix, block_slices):
    bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=64, latent_dim=32)
    head = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=64)
    train_loader, val_loader = _make_loaders(synthetic_feature_matrix, block_slices)
    w_blocks = losses.compute_block_weights(synthetic_feature_matrix,
                                             {b: block_slices[b] for b in BLOCK_DIMS})

    def loss_fn(model, batch):
        decoded = model(batch['blocks'])
        return losses.weighted_recon_loss(decoded, batch['blocks'], batch['has_bio'], w_blocks)

    ckpt = tmp_path / "ae.pt"
    train.train_model(
        model=head, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=2, lr=1e-3, device='cpu', checkpoint_path=ckpt,
    )
    assert ckpt.exists()
    state = torch.load(ckpt, weights_only=False)
    assert 'model_state' in state and 'epoch' in state and 'val_loss' in state
```

- [ ] **Step 6.2: Run — confirm failures.**

```bash
pytest tests/test_train.py -v
```

- [ ] **Step 6.3: Implement `src/cineembed/train.py`.**

```python
"""Generic training loop with early stopping + checkpoint save/resume (spec §4.3)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def train_model(
    *,
    model: nn.Module,
    loss_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    device: str = 'cuda',
    checkpoint_path: str | Path | None = None,
    extra_params: list[nn.Parameter] | None = None,
    seed: int = 42,
) -> dict:
    """Generic training loop.

    Args:
        loss_fn: callable (model, batch, epoch) → scalar tensor or tuple. Batch dict
                 has keys 'blocks' (per-block tensors) and 'has_bio'. The `epoch`
                 argument enables schedules like VAE β warmup. For backward
                 compatibility, the signature is auto-detected: if the function
                 accepts only (model, batch), epoch is omitted.
        extra_params: optional extra parameters to pass to the optimizer (e.g.,
                      learned-uncertainty log_sigmas in W4 stretch).

    Returns:
        history dict with 'train_loss' and 'val_loss' lists.
    """
    import inspect
    torch.manual_seed(seed)
    model = model.to(device)
    params = list(model.parameters()) + (list(extra_params) if extra_params else [])
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    # Auto-detect whether loss_fn expects an epoch argument
    sig = inspect.signature(loss_fn)
    accepts_epoch = len(sig.parameters) >= 3

    def _call_loss(model, batch, epoch):
        return loss_fn(model, batch, epoch) if accepts_epoch else loss_fn(model, batch)

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # ─── train ───
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = _call_loss(model, batch, epoch)
            if isinstance(loss, tuple):  # vae_elbo / dec_loss return (loss, ...)
                loss = loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_avg = sum(train_losses) / max(len(train_losses), 1)
        history['train_loss'].append(train_avg)

        # ─── validation ───
        val_avg = float('inf')
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = _move_batch_to_device(batch, device)
                    loss = _call_loss(model, batch, epoch)
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    val_losses.append(float(loss.item()))
            val_avg = sum(val_losses) / max(len(val_losses), 1)
        history['val_loss'].append(val_avg)

        # ─── early stopping + checkpoint ───
        improved = (best_val - val_avg) > early_stop_min_delta
        if improved:
            best_val = val_avg
            epochs_no_improve = 0
            if checkpoint_path is not None:
                _save_checkpoint(model, epoch, val_avg, train_avg, history, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                break

    history['final_val_loss'] = best_val
    history['n_epochs_completed'] = epoch + 1
    return history


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {
        'blocks': {b: t.to(device) for b, t in batch['blocks'].items()},
        'has_bio': batch['has_bio'].to(device),
    }


def _save_checkpoint(
    model: nn.Module, epoch: int, val_loss: float, train_loss: float,
    history: dict, path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state':  model.state_dict(),
        'epoch':        epoch,
        'val_loss':     val_loss,
        'train_loss':   train_loss,
        'history':      history,
    }, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: str = 'cpu') -> dict:
    """Load model state and metadata. Returns the checkpoint dict (sans model_state)."""
    state = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    return {k: v for k, v in state.items() if k != 'model_state'}
```

- [ ] **Step 6.4: Re-run — confirm passes.**

```bash
pytest tests/test_train.py -v
```
Expected: 2 passed.

- [ ] **Step 6.5: Commit.**

```bash
git add src/cineembed/train.py tests/test_train.py
git commit -m "feat(train): generic training loop with early stopping + checkpoint save/load"
```

---

## Task 7 — `eval.py` (NMI/ARI/UMAP/linear probing)

**Files:**
- Create: `src/cineembed/eval.py`
- Create: `tests/test_eval.py`

Per spec §8. Cluster assignments + L4 three-axis evaluation + linear probing + UMAP.

- [ ] **Step 7.1: Write failing tests.**

```python
import numpy as np
import torch

from cineembed import eval as cev


def test_cluster_assignments_kmeans_returns_int_array():
    z = np.random.randn(200, 32).astype(np.float32)
    c = cev.cluster_assignments_kmeans(z, k=10, seed=42)
    assert c.shape == (200,)
    assert c.dtype.kind == 'i'
    assert set(np.unique(c)).issubset(set(range(10)))


def test_evaluate_run_three_axes(synthetic_labels):
    n = 200
    np.random.seed(0)
    cluster_ids = np.random.randint(0, 21, size=n)
    metrics = cev.evaluate_run(cluster_ids, synthetic_labels)
    for axis in ['genre', 'decade', 'lang']:
        assert f'{axis}_nmi' in metrics
        assert f'{axis}_ari' in metrics
        assert 0.0 <= metrics[f'{axis}_nmi'] <= 1.0
        assert -1.0 <= metrics[f'{axis}_ari'] <= 1.0


def test_linear_probe_returns_accuracy():
    z = np.random.randn(200, 32).astype(np.float32)
    labels = np.random.randint(0, 5, size=200)
    train_idx = np.arange(160)
    val_idx = np.arange(160, 200)
    result = cev.linear_probe(z, labels, train_idx=train_idx, val_idx=val_idx,
                               n_classes=5, n_epochs=10, lr=1e-2, seed=42)
    assert 'val_accuracy' in result
    assert 0.0 <= result['val_accuracy'] <= 1.0


def test_umap_plot_creates_file(tmp_path):
    z = np.random.randn(200, 32).astype(np.float32)
    labels = np.random.choice(['A', 'B', 'C'], size=200)
    out = tmp_path / "test_umap.png"
    cev.umap_plot(z, labels, title='test', savepath=out, seed=42)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial file
```

- [ ] **Step 7.2: Run — confirm failures.**

```bash
pytest tests/test_eval.py -v
```

- [ ] **Step 7.3: Implement `src/cineembed/eval.py`.**

```python
"""Evaluation helpers for AE/VAE/DEC runs (spec §8)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless / Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_assignments_kmeans(z: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Run KMeans(k) on latent vectors → cluster ids."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return km.fit_predict(z).astype(np.int64)


def cluster_assignments_dec(model, batches: list[dict], device: str = 'cpu') -> np.ndarray:
    """Argmax over DEC soft assignments (q) across all batches."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in batches:
            blocks = {b: t.to(device) for b, t in batch['blocks'].items()}
            _, _, q = model(blocks)
            chunks.append(q.argmax(dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.int64)


def evaluate_run(
    cluster_ids: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute NMI and ARI vs each label axis (spec §8.1)."""
    out = {}
    axis_aliases = {'genre': 'primary_genre', 'decade': 'decade_bin', 'lang': 'lang_top10'}
    for short, full in axis_aliases.items():
        labs = labels[full]
        out[f'{short}_nmi'] = float(normalized_mutual_info_score(labs, cluster_ids))
        out[f'{short}_ari'] = float(adjusted_rand_score(labs, cluster_ids))
    return out


def linear_probe(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_classes: int,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cpu',
) -> dict[str, float]:
    """Train a linear classifier on frozen z and report val accuracy (spec §8.3.1)."""
    torch.manual_seed(seed)
    z_t = torch.from_numpy(z).float().to(device)
    y_t = torch.from_numpy(np.asarray(labels)).long().to(device)
    model = nn.Linear(z.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx_t = torch.from_numpy(train_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()

    for _ in range(n_epochs):
        model.train()
        logits = model(z_t[train_idx_t])
        loss = nn.functional.cross_entropy(logits, y_t[train_idx_t])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(z_t[val_idx_t])
        val_acc = (val_logits.argmax(dim=1) == y_t[val_idx_t]).float().mean().item()
    return {'val_accuracy': val_acc}


def umap_plot(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    savepath: str | Path,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """2D UMAP scatter colored by labels, saved to disk (spec §8.2)."""
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=2, random_state=seed)
    z2d = reducer.fit_transform(z)

    fig, ax = plt.subplots(figsize=(10, 7))
    unique = np.unique(labels)
    cmap = plt.get_cmap('tab20', len(unique))
    for i, cls in enumerate(unique):
        mask = labels == cls
        ax.scatter(z2d[mask, 0], z2d[mask, 1], s=8, alpha=0.55,
                   color=cmap(i), label=f'{cls} ({mask.sum()})')
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=7, loc='best', ncol=2)
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.close(fig)
```

- [ ] **Step 7.4: Re-run — confirm passes.**

```bash
pytest tests/test_eval.py -v
```
Expected: 4 passed (umap_plot may take a few seconds the first time as umap-learn lazily compiles).

- [ ] **Step 7.5: Commit.**

```bash
git add src/cineembed/eval.py tests/test_eval.py
git commit -m "feat(eval): KMeans/DEC cluster assignments, NMI/ARI L4, linear probing, UMAP"
```

---

## Task 8 — Notebook 01: smoke test

**Files:**
- Create: `notebooks/01_smoke_test.ipynb`

Manual end-to-end validation that the package works on a small synthetic batch.

- [ ] **Step 8.1: Create the notebook with these cells (in order).**

**Cell 1 (markdown):**
```markdown
# CineEmbed — 01 Smoke Test

Validates that the `cineembed` package's backbone, three heads, losses, and eval
helpers all run end-to-end on a 200-row synthetic batch. Should complete in <1 minute
on Mac CPU.
```

**Cell 2 (code) — environment setup:**
```python
import sys, os
from pathlib import Path
# Local Mac path; for Colab replace with /content/cineembed-repo
REPO_ROOT = Path('..').resolve() if Path('../pyproject.toml').exists() else Path('.').resolve()
sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np
import torch

from cineembed import data, backbone, heads, losses, eval as cev, train

print(f"cineembed loaded from {REPO_ROOT / 'src'}")
print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()}")
```

**Cell 3 (code) — synthetic feature matrix (mirror of conftest fixture):**
```python
BLOCK_DIMS = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
              'awards': 6, 'text': 384, 'director': 113}
PROJ_DIMS = {'numerical': 16, 'genre': 16, 'language': 16, 'decade': 4,
             'awards': 16, 'text': 64, 'director': 32}

# Reuse the conftest synthetic-matrix logic inline
rng = np.random.default_rng(42)
n = 200
total = sum(BLOCK_DIMS.values())
X = np.zeros((n, total), dtype=np.float32)
slices, start = {}, 0
for b, d in BLOCK_DIMS.items():
    slices[b] = slice(start, start + d)
    start += d

# Fill blocks (simplified — production uses richer synthetic data)
X[:, slices['numerical']] = rng.standard_normal((n, 6)).astype(np.float32)
for i in range(n):
    cols = rng.choice(BLOCK_DIMS['genre'], size=3, replace=False)
    X[i, slices['genre'].start + cols] = 1.0
text_raw = rng.standard_normal((n, 384)).astype(np.float32)
X[:, slices['text']] = text_raw / np.linalg.norm(text_raw, axis=1, keepdims=True).clip(1e-8)
has_bio = (rng.uniform(0, 1, size=n) < 0.05).astype(np.float32)
X[:, slices['director'].start + 64] = has_bio  # has_director_bio col

X_t = torch.from_numpy(X)
has_bio_t = torch.from_numpy(has_bio)
print(f"X shape: {X.shape}; has_bio sum: {int(has_bio.sum())}")
```

**Cell 4 (code) — instantiate backbone and three heads:**
```python
torch.manual_seed(42)
bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=64)
ae = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
vae = heads.VAEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
dec_head = heads.DECHead(bb, ae.decoder, n_clusters=10, latent_dim=64)

print(f"AE   params: {sum(p.numel() for p in ae.parameters()):>7,}")
print(f"VAE  params: {sum(p.numel() for p in vae.parameters()):>7,}")
print(f"DEC  params: {sum(p.numel() for p in dec_head.parameters()):>7,}")
```

**Cell 5 (code) — run forward passes:**
```python
blocks = {b: X_t[:, slc] for b, slc in slices.items()}

decoded = ae(blocks)
print(f"AE  decoded keys: {sorted(decoded.keys())}")
print(f"    decoded['numerical'].shape: {decoded['numerical'].shape}")

decoded_v, mu, log_var = vae(blocks)
print(f"VAE mu, log_var shapes: {mu.shape}, {log_var.shape}")

# Initialize DEC centers from AE encoder output
with torch.no_grad():
    z_init = bb(blocks).numpy()
dec_head.initialize_centers(z_init, seed=42)
z_dec, decoded_d, q = dec_head(blocks)
print(f"DEC z, q shapes: {z_dec.shape}, {q.shape}")
print(f"    q row sums (should be ≈1): min={q.sum(1).min():.4f}, max={q.sum(1).max():.4f}")
```

**Cell 6 (code) — compute all loss variants:**
```python
w_blocks = losses.compute_block_weights(X, slices, w_min=0.1, w_max=10.0)
print(f"Block weights: {w_blocks}")

# AE / W2
loss_ae = losses.weighted_recon_loss(decoded, blocks, has_bio_t, w_blocks)
print(f"AE  W2 loss: {loss_ae.item():.4f}")

# AE / W1 baseline
loss_w1 = losses.weighted_recon_loss_uniform(decoded, blocks, has_bio_t)
print(f"AE  W1 loss: {loss_w1.item():.4f}")

# VAE ELBO with β=0.5
loss_vae, recon_v, kl_v = losses.vae_elbo(decoded_v, blocks, mu, log_var, has_bio_t, w_blocks, beta=0.5)
print(f"VAE elbo: {loss_vae.item():.4f}  (recon={recon_v:.4f}  kl={kl_v:.4f})")

# DEC loss
loss_dec, kl_d, recon_d = losses.dec_loss(z_dec, decoded_d, blocks, dec_head.cluster_centers,
                                            has_bio_t, w_blocks, lambda_recon=0.1)
print(f"DEC loss: {loss_dec.item():.4f}  (kl={kl_d:.4f}  recon={recon_d:.4f})")
```

**Cell 7 (code) — eval helpers smoke:**
```python
synthetic_labels = {
    'primary_genre': rng.integers(0, 21, size=n),
    'decade_bin':    rng.integers(0, 13, size=n),
    'lang_top10':    rng.integers(0, 11, size=n),
}

# KMeans on AE latent
with torch.no_grad():
    z_ae = ae.encode(blocks).numpy()
c_ids = cev.cluster_assignments_kmeans(z_ae, k=10, seed=42)
metrics = cev.evaluate_run(c_ids, synthetic_labels)
print("Synthetic NMI/ARI (random labels → near-zero):")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
```

**Cell 8 (markdown):**
```markdown
✅ Smoke test passed if all cells ran without error and printed non-NaN losses + per-axis NMI/ARI values.

For full data training: run `02_train_ae.ipynb`, `03_train_vae.ipynb`, `04_train_dec.ipynb` on Colab.
```

- [ ] **Step 8.2: Run all cells in order in JupyterLab / VSCode notebook UI.** Verify no errors; verify all printed values are sensible (non-NaN, non-Inf).

- [ ] **Step 8.3: Commit.**

```bash
git add notebooks/01_smoke_test.ipynb
git commit -m "feat(notebooks): 01_smoke_test — end-to-end package validation"
```

---

## Task 9 — Notebook 02: train AE (3 main + 3 ablations + optional W4)

**Files:**
- Create: `notebooks/02_train_ae.ipynb`

Trains AE main runs (z={32, 64, 128}), W1 ablation, F1/F2 modality ablations, optional W4 stretch. **Heavy compute → run in Colab.**

- [ ] **Step 9.1: Create the notebook structure.**

**Cell 1 (markdown):**
```markdown
# CineEmbed — 02 Train AE

Trains the AutoEncoder family per spec §6 run matrix:
- Run 3: vanilla concat-AE z=64 (baseline for architecture validation)
- Runs 4-6: multi-modal AE × {32, 64, 128}
- Run 19: W1 (uniform) ablation
- Run 20: F1 (no text) ablation
- Run 21: F2 (no director) ablation
- Run 22 (optional): W4 (Kendall) stretch

**Compute estimate**: ~1.5-2 hours on Colab T4. Each run saves a checkpoint in
`artifacts/models/` and appends metrics to `artifacts/eval/results.json`.
```

**Cell 2 (code) — Colab setup (with fallback for local):**
```python
import os, sys, json
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = Path('/content/cineembed-repo')  # adjust if you placed repo elsewhere
    ARTIFACTS = Path('/content/drive/MyDrive/cineembed_artifacts')
    if not REPO_ROOT.exists():
        # Upload .zip of repo to Drive; this expands it into Colab session storage.
        import shutil
        shutil.unpack_archive(str(ARTIFACTS / 'cineembed_repo.zip'), str(REPO_ROOT.parent))
    !pip install -e {REPO_ROOT} -q
else:
    REPO_ROOT = Path('..').resolve()
    ARTIFACTS = REPO_ROOT / 'artifacts'

sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np
import torch

from cineembed import data, backbone, heads, losses, train, eval as cev

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Repo: {REPO_ROOT}\nArtifacts: {ARTIFACTS}\nDevice: {DEVICE}")
```

**Cell 3 (code) — load feature matrix and labels:**
```python
X, feature_names = data.load_feature_matrix(ARTIFACTS / 'feature_matrix.npz')
labels = data.get_labels(ARTIFACTS / 'movies_eda_final.csv')
block_slices = data.get_block_indices(feature_names)
has_bio = X[:, block_slices['director'].start + 64].clone()

print(f"X: {X.shape}; has_bio sum: {int(has_bio.sum())}")
print(f"Labels: {[(k, len(set(v))) for k, v in labels.items()]}")
print(f"Block slices: {[(b, slc.stop - slc.start) for b, slc in block_slices.items()]}")
```

**Cell 4 (code) — compute block weights and prepare loaders:**
```python
BLOCK_DIMS = {b: (slc.stop - slc.start) for b, slc in block_slices.items()}
PROJ_DIMS = backbone.DEFAULT_PROJ_DIMS

w_blocks = losses.compute_block_weights(X.numpy(), block_slices, w_min=0.1, w_max=10.0)
print(f"Block weights (W2): {w_blocks}")

train_idx, val_idx = data.train_val_split(X.shape[0], val_frac=0.1, seed=42)
train_loader = data.make_dataloader(X, has_bio, batch_size=512, shuffle=True,
                                     indices=train_idx, block_slices=block_slices, seed=42)
val_loader = data.make_dataloader(X, has_bio, batch_size=512, shuffle=False,
                                    indices=val_idx, block_slices=block_slices, seed=42)
```

**Cell 5 (code) — generic AE trainer helper:**
```python
def train_ae_run(run_name, z_dim, w_blocks_to_use, *,
                 use_proj_dims=PROJ_DIMS, mask_text=False, mask_director=False,
                 n_epochs=100, vanilla=False):
    """Train one AE run; save checkpoint and metrics. Returns metrics dict.

    Modality ablation (mask_text / mask_director) uses the backbone's `block_mask`
    parameter (zeros that block's projected encoder output) AND `weighted_recon_loss`
    `exclude_blocks` (skips that block in the reconstruction loss).
    """
    torch.manual_seed(42)
    if vanilla:
        # Single FC encoder, no modality projection — vanilla concat-AE baseline.
        # Uses the same per-block decoder as multi-modal AE for fair comparison.
        bb_fc = nn.Sequential(
            nn.Linear(564, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, z_dim),
        )
        class _VanillaWrap(nn.Module):
            def __init__(self, fc, z_dim_):
                super().__init__()
                self.fc = fc
                self.latent_dim = z_dim_
                self.block_order = list(BLOCK_DIMS.keys())
            def forward(self, blocks, block_mask=None):
                # block_mask intentionally ignored — vanilla baseline doesn't support
                # ablation. F1/F2 ablations only apply to multi-modal AE.
                X_cat = torch.cat([blocks[b] for b in self.block_order], dim=1)
                return self.fc(X_cat)
        bb = _VanillaWrap(bb_fc, z_dim)
    else:
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, use_proj_dims, hidden_dim=128, latent_dim=z_dim)
    head = heads.AEHead(bb, BLOCK_DIMS, use_proj_dims, hidden_dim=128)

    # Determine which blocks to ablate (gated at both forward AND loss).
    masked_blocks: set = set()
    if mask_text:
        masked_blocks.add('text')
    if mask_director:
        masked_blocks.add('director')
    block_mask = ({b: 0.0 for b in masked_blocks} | {b: 1.0 for b in BLOCK_DIMS if b not in masked_blocks}
                  if masked_blocks else None)

    def loss_fn(model, batch):
        if block_mask is not None and not vanilla:
            z = model.backbone(batch['blocks'], block_mask=block_mask)
            decoded = model.decoder(z)
        else:
            decoded = model(batch['blocks'])
        return losses.weighted_recon_loss(
            decoded, batch['blocks'], batch['has_bio'], w_blocks_to_use,
            exclude_blocks=masked_blocks if masked_blocks else None,
        )

    ckpt_path = ARTIFACTS / 'models' / f'{run_name}.pt'
    history = train.train_model(
        model=head, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_epochs, lr=1e-3, weight_decay=1e-5,
        early_stop_patience=10, early_stop_min_delta=1e-4,
        device=DEVICE, checkpoint_path=ckpt_path, seed=42,
    )

    # Embed all 329k films with best-checkpoint weights
    train.load_checkpoint(head, ckpt_path, device=DEVICE)
    head.eval()
    with torch.no_grad():
        z_full = []
        full_loader = data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                             block_slices=block_slices)
        for batch in full_loader:
            batch_blocks = {k: v.to(DEVICE) for k, v in batch['blocks'].items()}
            z_full.append(head.encode(batch_blocks).cpu().numpy())
    z_all = np.concatenate(z_full, axis=0)

    # KMeans on full-data latent and L4 evaluation
    c_ids = cev.cluster_assignments_kmeans(z_all, k=21, seed=42)
    metrics = cev.evaluate_run(c_ids, labels)
    metrics['run_name'] = run_name
    metrics['n_epochs'] = history['n_epochs_completed']
    metrics['final_val_loss'] = history['final_val_loss']
    metrics['z_dim'] = z_dim
    print(f"[{run_name}] z={z_dim}  epochs={history['n_epochs_completed']}  "
          f"val_loss={history['final_val_loss']:.4f}  genre_NMI={metrics['genre_nmi']:.3f}")
    return metrics, z_all

import torch.nn as nn  # for vanilla wrap
```

**Cell 6 (code) — run baseline + main + ablations:**
```python
all_metrics = []

# Run 3: vanilla concat-AE z=64
m, _ = train_ae_run('vanilla_ae_z64', z_dim=64, w_blocks_to_use=w_blocks, vanilla=True)
all_metrics.append(m)

# Runs 4-6: multi-modal AE × {32, 64, 128}
for z in [32, 64, 128]:
    m, _ = train_ae_run(f'ae_z{z}', z_dim=z, w_blocks_to_use=w_blocks)
    all_metrics.append(m)

# Run 19: W1 ablation
w_uniform = {b: 1.0 for b in w_blocks}
m, _ = train_ae_run('ae_z64_w1', z_dim=64, w_blocks_to_use=w_uniform)
all_metrics.append(m)

# Run 20: F1 (no text)
m, _ = train_ae_run('ae_z64_no_text', z_dim=64, w_blocks_to_use=w_blocks, mask_text=True)
all_metrics.append(m)

# Run 21: F2 (no director)
m, _ = train_ae_run('ae_z64_no_director', z_dim=64, w_blocks_to_use=w_blocks, mask_director=True)
all_metrics.append(m)
```

**Cell 7 (code) — append metrics to results.json:**
```python
results_path = ARTIFACTS / 'eval' / 'results.json'
results_path.parent.mkdir(parents=True, exist_ok=True)
existing = json.loads(results_path.read_text()) if results_path.exists() else {}
for m in all_metrics:
    existing[m['run_name']] = m
results_path.write_text(json.dumps(existing, indent=2))
print(f"Saved {len(all_metrics)} runs to {results_path}")
```

**Cell 8 (markdown):**
```markdown
**Optional: W4 Kendall stretch.** Uncomment the cell below to train run 22 (learned uncertainty weighting). Adds ~20 minutes.
```

**Cell 9 (code) — W4 stretch (commented out by default):**
```python
# from cineembed.losses import LearnedWeightedLoss
# torch.manual_seed(42)
# bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=64)
# head = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
# learned_loss = LearnedWeightedLoss(list(BLOCK_DIMS.keys()))
# def loss_fn(model, batch):
#     decoded = model(batch['blocks'])
#     return learned_loss(decoded, batch['blocks'], batch['has_bio'])
# history = train.train_model(model=head, loss_fn=loss_fn,
#     train_loader=train_loader, val_loader=val_loader, n_epochs=100, lr=1e-3,
#     device=DEVICE, checkpoint_path=ARTIFACTS/'models'/'ae_z64_w4.pt',
#     extra_params=list(learned_loss.parameters()), seed=42)
```

- [ ] **Step 9.2: Run notebook in Colab end-to-end.** Verify all 6 main runs (vanilla + 3 main + 3 ablations) complete and `results.json` is populated.

- [ ] **Step 9.3: Commit.**

```bash
git add notebooks/02_train_ae.ipynb
git commit -m "feat(notebooks): 02_train_ae — vanilla + multi-modal AE × 3 dims + W1/F1/F2 ablations"
```

---

## Task 10 — Notebook 03: train VAE (3 dims with β warmup)

**Files:**
- Create: `notebooks/03_train_vae.ipynb`

- [ ] **Step 10.1: Create notebook with setup cells (1-4) and the VAE-specific trainer.**

**Cell 1 (markdown):**
```markdown
# CineEmbed — 03 Train VAE

Trains the multi-modal VAE family per spec §6 run matrix:
- Runs 7-9: VAE × {32, 64, 128} latent dims
- β warmup schedule: 0 → 1 over first 10 epochs (spec §4.3, D9)

**Compute estimate**: ~1 hour on Colab T4. Saves checkpoints to `artifacts/models/`
and appends metrics to `artifacts/eval/results.json`.
```

**Cell 2 (code) — Colab setup (identical to 02_train_ae.ipynb):**
```python
import os, sys, json
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = Path('/content/cineembed-repo')
    ARTIFACTS = Path('/content/drive/MyDrive/cineembed_artifacts')
    if not REPO_ROOT.exists():
        import shutil
        shutil.unpack_archive(str(ARTIFACTS / 'cineembed_repo.zip'), str(REPO_ROOT.parent))
    !pip install -e {REPO_ROOT} -q
else:
    REPO_ROOT = Path('..').resolve()
    ARTIFACTS = REPO_ROOT / 'artifacts'

sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np
import torch

from cineembed import data, backbone, heads, losses, train, eval as cev

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Repo: {REPO_ROOT}\nArtifacts: {ARTIFACTS}\nDevice: {DEVICE}")
```

**Cell 3 (code) — load feature matrix and labels:**
```python
X, feature_names = data.load_feature_matrix(ARTIFACTS / 'feature_matrix.npz')
labels = data.get_labels(ARTIFACTS / 'movies_eda_final.csv')
block_slices = data.get_block_indices(feature_names)
has_bio = X[:, block_slices['director'].start + 64].clone()

print(f"X: {X.shape}; has_bio sum: {int(has_bio.sum())}")
```

**Cell 4 (code) — block weights + dataloaders:**
```python
BLOCK_DIMS = {b: (slc.stop - slc.start) for b, slc in block_slices.items()}
PROJ_DIMS = backbone.DEFAULT_PROJ_DIMS

w_blocks = losses.compute_block_weights(X.numpy(), block_slices, w_min=0.1, w_max=10.0)
print(f"Block weights (W2): {w_blocks}")

train_idx, val_idx = data.train_val_split(X.shape[0], val_frac=0.1, seed=42)
train_loader = data.make_dataloader(X, has_bio, batch_size=512, shuffle=True,
                                     indices=train_idx, block_slices=block_slices, seed=42)
val_loader = data.make_dataloader(X, has_bio, batch_size=512, shuffle=False,
                                    indices=val_idx, block_slices=block_slices, seed=42)
```

**Cell 5 (code) — VAE trainer using `train.train_model` with β warmup via the epoch-aware loss_fn:**
```python
BETA_WARMUP_EPOCHS = 10
BETA_TARGET = 1.0

def train_vae_run(run_name, z_dim, w_blocks_to_use, n_epochs=100):
    """Train one VAE run with β warmup wired through train_model's epoch arg."""
    torch.manual_seed(42)
    bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=z_dim)
    head = heads.VAEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)

    # IMPORTANT: 3-arg signature (model, batch, epoch) — train_model auto-detects this
    # and forwards the current epoch each step, enabling β warmup without manual loops.
    def loss_fn(model, batch, epoch):
        decoded, mu, log_var = model(batch['blocks'])
        beta = min(epoch / BETA_WARMUP_EPOCHS, 1.0) * BETA_TARGET
        loss, recon_v, kl_v = losses.vae_elbo(
            decoded, batch['blocks'], mu, log_var, batch['has_bio'], w_blocks_to_use, beta)
        return loss

    ckpt_path = ARTIFACTS / 'models' / f'{run_name}.pt'
    history = train.train_model(
        model=head, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_epochs, lr=1e-3, weight_decay=1e-5,
        early_stop_patience=10, early_stop_min_delta=1e-4,
        device=DEVICE, checkpoint_path=ckpt_path, seed=42,
    )

    # Embed all 329k films using μ (deterministic) for downstream clustering
    train.load_checkpoint(head, ckpt_path, device=DEVICE)
    head.eval()
    z_full = []
    full_loader = data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                         block_slices=block_slices)
    with torch.no_grad():
        for batch in full_loader:
            blk = {k: v.to(DEVICE) for k, v in batch['blocks'].items()}
            mu, _ = head.encode(blk)
            z_full.append(mu.cpu().numpy())
    z_all = np.concatenate(z_full, axis=0)

    c_ids = cev.cluster_assignments_kmeans(z_all, k=21, seed=42)
    metrics = cev.evaluate_run(c_ids, labels)
    metrics['run_name'] = run_name
    metrics['z_dim'] = z_dim
    metrics['final_val_loss'] = history['final_val_loss']
    metrics['n_epochs'] = history['n_epochs_completed']
    return metrics, z_all
```

**Cell 6 (code) — run VAE × 3 dims:**
```python
all_metrics = []
for z in [32, 64, 128]:
    m, _ = train_vae_run(f'vae_z{z}', z_dim=z, w_blocks_to_use=w_blocks)
    all_metrics.append(m)
    print(f"[vae_z{z}] genre_NMI={m['genre_nmi']:.3f}  decade_NMI={m['decade_nmi']:.3f}  lang_NMI={m['lang_nmi']:.3f}")

# Append to results.json
results_path = ARTIFACTS / 'eval' / 'results.json'
existing = json.loads(results_path.read_text()) if results_path.exists() else {}
for m in all_metrics:
    existing[m['run_name']] = m
results_path.write_text(json.dumps(existing, indent=2))
```

- [ ] **Step 10.2: Run on Colab.** Verify VAE runs converge (β warmup means initial loss is just reconstruction; KL kicks in around epoch 10).

- [ ] **Step 10.3: Commit.**

```bash
git add notebooks/03_train_vae.ipynb
git commit -m "feat(notebooks): 03_train_vae — multi-modal VAE × 3 dims with β warmup"
```

---

## Task 11 — Notebook 04: train DEC (3 dims × 3 k values = 9 runs)

**Files:**
- Create: `notebooks/04_train_dec.ipynb`

Loads pretrained AE checkpoints from `02_train_ae.ipynb` outputs. Each (z, k) combination is a separate fine-tune.

- [ ] **Step 11.1: Create notebook with setup cells (1-4) and the DEC-specific trainer.**

**Cell 1 (markdown):**
```markdown
# CineEmbed — 04 Train DEC

Trains DEC × {32, 64, 128} × {10, 21, 30} = 9 runs (spec §6, D8).
Each run loads the matching pretrained AE checkpoint produced by `02_train_ae.ipynb`.

**Prerequisite**: `02_train_ae.ipynb` must have completed; `ae_z32.pt`, `ae_z64.pt`, `ae_z128.pt`
must exist in `artifacts/models/`.

**Compute estimate**: ~1.5 hours on Colab T4 (9 runs × ~10 min each).
```

**Cell 2 (code) — Colab setup (same as Task 9 / Task 10):**
```python
import os, sys, json
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = Path('/content/cineembed-repo')
    ARTIFACTS = Path('/content/drive/MyDrive/cineembed_artifacts')
    if not REPO_ROOT.exists():
        import shutil
        shutil.unpack_archive(str(ARTIFACTS / 'cineembed_repo.zip'), str(REPO_ROOT.parent))
    !pip install -e {REPO_ROOT} -q
else:
    REPO_ROOT = Path('..').resolve()
    ARTIFACTS = REPO_ROOT / 'artifacts'

sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np
import torch

from cineembed import data, backbone, heads, losses, train, eval as cev

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Repo: {REPO_ROOT}\nArtifacts: {ARTIFACTS}\nDevice: {DEVICE}")
```

**Cell 3 (code) — load feature matrix and verify AE checkpoints exist:**
```python
X, feature_names = data.load_feature_matrix(ARTIFACTS / 'feature_matrix.npz')
labels = data.get_labels(ARTIFACTS / 'movies_eda_final.csv')
block_slices = data.get_block_indices(feature_names)
has_bio = X[:, block_slices['director'].start + 64].clone()

# Verify AE prerequisites
for z in [32, 64, 128]:
    p = ARTIFACTS / 'models' / f'ae_z{z}.pt'
    assert p.exists(), f"Missing prerequisite: {p} — run 02_train_ae.ipynb first"
print("AE prerequisites OK; X:", X.shape)
```

**Cell 4 (code) — block weights + dataloaders:**
```python
BLOCK_DIMS = {b: (slc.stop - slc.start) for b, slc in block_slices.items()}
PROJ_DIMS = backbone.DEFAULT_PROJ_DIMS
w_blocks = losses.compute_block_weights(X.numpy(), block_slices, w_min=0.1, w_max=10.0)
train_idx, val_idx = data.train_val_split(X.shape[0], val_frac=0.1, seed=42)
train_loader = data.make_dataloader(X, has_bio, batch_size=512, shuffle=True,
                                     indices=train_idx, block_slices=block_slices, seed=42)
val_loader = data.make_dataloader(X, has_bio, batch_size=512, shuffle=False,
                                    indices=val_idx, block_slices=block_slices, seed=42)
```

**Cell 5 (code) — DEC trainer with cluster-collapse re-init hook:**
```python
def _compute_full_latents_and_counts(model, X, has_bio, block_slices, device):
    """Forward pass over the entire dataset; return (z_array, per-cluster counts)."""
    model.eval()
    full_loader = data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                         block_slices=block_slices)
    z_chunks, q_argmax_chunks = [], []
    with torch.no_grad():
        for batch in full_loader:
            blk = {k_: v.to(device) for k_, v in batch['blocks'].items()}
            z, _, q = model(blk)
            z_chunks.append(z.cpu().numpy())
            q_argmax_chunks.append(q.argmax(dim=1).cpu().numpy())
    z_all = np.concatenate(z_chunks, axis=0)
    assignments = np.concatenate(q_argmax_chunks, axis=0)
    counts = np.bincount(assignments, minlength=model.n_clusters)
    return z_all, torch.from_numpy(counts)


def train_dec_run(run_name, z_dim, k, ae_checkpoint_path, n_epochs=50,
                   cluster_size_floor=0.001):
    """DEC training with explicit cluster-collapse re-init between epochs."""
    torch.manual_seed(42)
    bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=z_dim)
    ae_head = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
    train.load_checkpoint(ae_head, ae_checkpoint_path, device=DEVICE)
    ae_head = ae_head.to(DEVICE)

    # Compute initial latents on full data for k-means init
    ae_head.eval()
    z_init = []
    full_loader = data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                         block_slices=block_slices)
    with torch.no_grad():
        for batch in full_loader:
            blk = {k_: v.to(DEVICE) for k_, v in batch['blocks'].items()}
            z_init.append(ae_head.encode(blk).cpu().numpy())
    z_init = np.concatenate(z_init, axis=0)

    dec_head = heads.DECHead(bb, ae_head.decoder, n_clusters=k, latent_dim=z_dim)
    dec_head.initialize_centers(z_init, seed=42)
    dec_head = dec_head.to(DEVICE)

    # Manual training loop because we need a per-epoch cluster-count check.
    opt = torch.optim.Adam(dec_head.parameters(), lr=1e-3, weight_decay=1e-5)
    history = {'train_loss': [], 'val_loss': [], 'n_reinit': []}
    best_val = float('inf'); patience = 0
    ckpt_path = ARTIFACTS / 'models' / f'{run_name}.pt'

    n_total = X.shape[0]

    for epoch in range(n_epochs):
        dec_head.train()
        train_losses = []
        for batch in train_loader:
            blk = {k_: v.to(DEVICE) for k_, v in batch['blocks'].items()}
            hb = batch['has_bio'].to(DEVICE)
            opt.zero_grad()
            z, decoded, _ = dec_head(blk)
            loss, _, _ = losses.dec_loss(z, decoded, blk, dec_head.cluster_centers,
                                           hb, w_blocks, lambda_recon=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_head.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))
        history['train_loss'].append(sum(train_losses) / max(len(train_losses), 1))

        # ─── Cluster-collapse re-init (spec §10 mitigation) ───
        z_full, cluster_counts = _compute_full_latents_and_counts(
            dec_head, X, has_bio, block_slices, DEVICE,
        )
        n_reinit = dec_head.reinit_collapsed_centers(
            cluster_counts=cluster_counts, z_pool=z_full,
            n_total=n_total, size_floor=cluster_size_floor, seed=42 + epoch,
        )
        history['n_reinit'].append(n_reinit)
        if n_reinit > 0:
            print(f"  [epoch {epoch}] re-initialized {n_reinit} collapsed clusters")

        # Validation pass
        dec_head.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                blk = {k_: v.to(DEVICE) for k_, v in batch['blocks'].items()}
                hb = batch['has_bio'].to(DEVICE)
                z, decoded, _ = dec_head(blk)
                v_loss, _, _ = losses.dec_loss(z, decoded, blk, dec_head.cluster_centers,
                                                 hb, w_blocks, lambda_recon=0.1)
                val_losses.append(float(v_loss.item()))
        val_avg = sum(val_losses) / max(len(val_losses), 1)
        history['val_loss'].append(val_avg)

        if best_val - val_avg > 1e-4:
            best_val = val_avg; patience = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state': dec_head.state_dict(), 'epoch': epoch,
                        'val_loss': val_avg, 'history': history}, ckpt_path)
        else:
            patience += 1
            if patience >= 8:
                break

    # Final cluster assignments via DEC argmax
    train.load_checkpoint(dec_head, ckpt_path, device=DEVICE)
    dec_head.eval()
    full_batches = list(data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                                block_slices=block_slices))
    c_ids = cev.cluster_assignments_dec(dec_head, full_batches, device=DEVICE)
    metrics = cev.evaluate_run(c_ids, labels)
    metrics['run_name'] = run_name
    metrics['z_dim'] = z_dim
    metrics['k'] = k
    metrics['total_reinit'] = int(sum(history['n_reinit']))
    metrics['final_val_loss'] = best_val
    return metrics
```

**Cell 6 (code) — run DEC × 9 (3 z × 3 k):**
```python
all_metrics = []
for z in [32, 64, 128]:
    for k in [10, 21, 30]:
        ae_ckpt = ARTIFACTS / 'models' / f'ae_z{z}.pt'
        m = train_dec_run(f'dec_z{z}_k{k}', z_dim=z, k=k, ae_checkpoint_path=ae_ckpt)
        all_metrics.append(m)
        print(f"[dec_z{z}_k{k}] genre_NMI={m['genre_nmi']:.3f}  decade_NMI={m['decade_nmi']:.3f}")

# Append to results.json
results_path = ARTIFACTS / 'eval' / 'results.json'
existing = json.loads(results_path.read_text()) if results_path.exists() else {}
for m in all_metrics:
    existing[m['run_name']] = m
results_path.write_text(json.dumps(existing, indent=2))
```

- [ ] **Step 11.2: Run on Colab end-to-end.** Verify all 9 DEC runs complete; flag any cluster collapse (cluster_size_floor mitigation).

- [ ] **Step 11.3: Commit.**

```bash
git add notebooks/04_train_dec.ipynb
git commit -m "feat(notebooks): 04_train_dec — DEC × 3 z × 3 k = 9 runs from AE checkpoints"
```

---

## Task 12 — Notebook 05: results aggregation + final figures

**Files:**
- Create: `notebooks/05_results.ipynb`

Reads `results.json`, builds main comparison table, generates final figures, computes ablation deltas, runs linear probing on z=64 models.

- [ ] **Step 12.1: Create notebook.**

**Cell 1 (markdown):**
```markdown
# CineEmbed — 05 Results

Reads `artifacts/eval/results.json`, generates:
- Main results table (3 axes × NMI/ARI per run)
- Baseline-vs-deep comparison
- Ablation deltas (W1, F1, F2, optional W4)
- Linear probing accuracy at z=64
- 12 main report figures + 18+ supplementary UMAPs (per spec §8.2)
- Final results summary in JSON
```

**Cell 2 (code) — Colab setup (identical pattern to Tasks 9-11):**
```python
import os, sys, json
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = Path('/content/cineembed-repo')
    ARTIFACTS = Path('/content/drive/MyDrive/cineembed_artifacts')
    if not REPO_ROOT.exists():
        import shutil
        shutil.unpack_archive(str(ARTIFACTS / 'cineembed_repo.zip'), str(REPO_ROOT.parent))
    !pip install -e {REPO_ROOT} -q
else:
    REPO_ROOT = Path('..').resolve()
    ARTIFACTS = REPO_ROOT / 'artifacts'

sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np
import torch

from cineembed import data, backbone, heads, losses, train, eval as cev

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

X, feature_names = data.load_feature_matrix(ARTIFACTS / 'feature_matrix.npz')
labels = data.get_labels(ARTIFACTS / 'movies_eda_final.csv')
block_slices = data.get_block_indices(feature_names)
has_bio = X[:, block_slices['director'].start + 64].clone()
BLOCK_DIMS = {b: (slc.stop - slc.start) for b, slc in block_slices.items()}
PROJ_DIMS = backbone.DEFAULT_PROJ_DIMS
print(f"Loaded {X.shape}; results dir: {ARTIFACTS / 'eval'}")
```

**Cell 3 (code) — sklearn baselines (Run 1: KMeans on raw, Run 2: PCA + KMeans):**
```python
from sklearn.cluster import KMeans as SK_KMeans
from sklearn.decomposition import PCA

# Run 1: KMeans on raw 564-dim
c_raw = SK_KMeans(n_clusters=21, n_init=20, random_state=42).fit_predict(X.numpy())
m_raw = cev.evaluate_run(c_raw, labels)
m_raw['run_name'] = 'kmeans_raw_k21'
m_raw['z_dim'] = 564

# Run 2: PCA + KMeans
pca = PCA(n_components=64, random_state=42).fit(X.numpy())
X_pca = pca.transform(X.numpy())
c_pca = SK_KMeans(n_clusters=21, n_init=20, random_state=42).fit_predict(X_pca)
m_pca = cev.evaluate_run(c_pca, labels)
m_pca['run_name'] = 'pca_kmeans_k21'
m_pca['z_dim'] = 64

results_path = ARTIFACTS / 'eval' / 'results.json'
existing = json.loads(results_path.read_text()) if results_path.exists() else {}
existing[m_raw['run_name']] = m_raw
existing[m_pca['run_name']] = m_pca
results_path.write_text(json.dumps(existing, indent=2))

print(f"Baselines: kmeans_raw NMI(genre)={m_raw['genre_nmi']:.3f}, "
      f"pca_kmeans NMI(genre)={m_pca['genre_nmi']:.3f}")
```

**Cell 4 (code) — main results table:**
```python
import pandas as pd
results = json.loads(results_path.read_text())
rows = []
for name, m in sorted(results.items()):
    rows.append({
        'run': name,
        'z': m.get('z_dim', '-'),
        'k': m.get('k', '-'),
        'genre_NMI': m.get('genre_nmi', np.nan),
        'genre_ARI': m.get('genre_ari', np.nan),
        'decade_NMI': m.get('decade_nmi', np.nan),
        'decade_ARI': m.get('decade_ari', np.nan),
        'lang_NMI': m.get('lang_nmi', np.nan),
        'lang_ARI': m.get('lang_ari', np.nan),
    })
df_results = pd.DataFrame(rows).round(3)
print(df_results.to_string(index=False))
df_results.to_csv(ARTIFACTS / 'eval' / 'results_table.csv', index=False)
```

**Cell 5 (code) — ablation deltas:**
```python
ae_baseline = results.get('ae_z64', {})
ablations = ['ae_z64_w1', 'ae_z64_no_text', 'ae_z64_no_director', 'ae_z64_w4']
deltas = {}
for ab in ablations:
    if ab not in results:
        continue
    deltas[ab] = {
        'genre_nmi_delta':  results[ab]['genre_nmi'] - ae_baseline.get('genre_nmi', 0),
        'decade_nmi_delta': results[ab]['decade_nmi'] - ae_baseline.get('decade_nmi', 0),
        'lang_nmi_delta':   results[ab]['lang_nmi'] - ae_baseline.get('lang_nmi', 0),
    }
print("Ablation deltas vs ae_z64 baseline:")
for ab, d in deltas.items():
    print(f"  {ab}: {d}")

(ARTIFACTS / 'eval' / 'ablation_deltas.json').write_text(json.dumps(deltas, indent=2))
```

**Cell 6 (code) — linear probing for z=64 main runs:**
```python
from sklearn.preprocessing import LabelEncoder

probe_results = {}
le_genre = LabelEncoder().fit(labels['primary_genre'])
le_decade = LabelEncoder().fit(labels['decade_bin'])
y_genre = le_genre.transform(labels['primary_genre'])
y_decade = le_decade.transform(labels['decade_bin'])
train_idx_p, val_idx_p = data.train_val_split(X.shape[0], val_frac=0.2, seed=42)

for run in ['ae_z64', 'vae_z64', 'dec_z64_k21']:
    ckpt = ARTIFACTS / 'models' / f'{run}.pt'
    if not ckpt.exists():
        continue
    # Reload the model and recompute embeddings
    if run.startswith('ae_'):
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=64)
        h = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
    elif run.startswith('vae_'):
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=64)
        h = heads.VAEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
    else:  # dec_
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=64)
        ae_h = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
        h = heads.DECHead(bb, ae_h.decoder, n_clusters=21, latent_dim=64)
    train.load_checkpoint(h, ckpt, device='cpu')
    h.eval()
    # Embed
    z_full = []
    with torch.no_grad():
        full_loader = data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                             block_slices=block_slices)
        for batch in full_loader:
            blk = batch['blocks']
            if run.startswith('vae_'):
                mu, _ = h.encode(blk)
                z_full.append(mu.numpy())
            else:
                z_full.append(h.encode(blk).numpy() if hasattr(h, 'encode') else
                              h.backbone(blk).numpy())
    z_all = np.concatenate(z_full, axis=0)

    probe_results[run] = {
        'genre_acc':  cev.linear_probe(z_all, y_genre, train_idx_p, val_idx_p,
                                         n_classes=len(le_genre.classes_), seed=42)['val_accuracy'],
        'decade_acc': cev.linear_probe(z_all, y_decade, train_idx_p, val_idx_p,
                                         n_classes=len(le_decade.classes_), seed=42)['val_accuracy'],
    }

print("Linear probing accuracy:")
for r, p in probe_results.items():
    print(f"  {r}: genre={p['genre_acc']:.3f}, decade={p['decade_acc']:.3f}")
(ARTIFACTS / 'eval' / 'linear_probing.json').write_text(json.dumps(probe_results, indent=2))
```

**Cell 7 (code) — generate UMAP plots for best AE / VAE / DEC + 3 baselines:**
```python
# Choose "best" by genre_nmi within each family
def best_run_in(prefix):
    candidates = [(name, m) for name, m in results.items()
                   if name.startswith(prefix) and 'genre_nmi' in m]
    return max(candidates, key=lambda kv: kv[1]['genre_nmi']) if candidates else None

best_ae = best_run_in('ae_z')
best_vae = best_run_in('vae_z')
best_dec = best_run_in('dec_z')

print(f"Best AE: {best_ae[0] if best_ae else '—'}")
print(f"Best VAE: {best_vae[0] if best_vae else '—'}")
print(f"Best DEC: {best_dec[0] if best_dec else '—'}")

# Generate UMAPs for the 3 best runs × 3 axes = 9 main figures
# (Reuse embedding code pattern from cell 6 if needed)

fig_dir = ARTIFACTS / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

# Baseline UMAPs at genre axis only (3 figures)
cev.umap_plot(X.numpy(), labels['primary_genre'],
              title='Raw 564-dim — genre',
              savepath=fig_dir / 'umap_raw_genre.png', seed=42)
cev.umap_plot(X_pca, labels['primary_genre'],
              title='PCA(64) — genre',
              savepath=fig_dir / 'umap_pca_genre.png', seed=42)

print(f"Generated baseline UMAP figures in {fig_dir}")

# Best-of-each UMAPs: reload each best run's checkpoint, embed, and plot 3 axes per run.
def _embed_run(run_name, z_dim):
    """Reload checkpoint, embed full 329k films. Returns numpy array (n, z_dim)."""
    if run_name.startswith('vae_'):
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=z_dim)
        h = heads.VAEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
    elif run_name.startswith('dec_'):
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=z_dim)
        ae_h = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
        k = int(run_name.split('_k')[-1])
        h = heads.DECHead(bb, ae_h.decoder, n_clusters=k, latent_dim=z_dim)
    else:  # ae_
        bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=128, latent_dim=z_dim)
        h = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=128)
    train.load_checkpoint(h, ARTIFACTS / 'models' / f'{run_name}.pt', device='cpu')
    h.eval()
    z_chunks = []
    with torch.no_grad():
        full_loader = data.make_dataloader(X, has_bio, batch_size=2048, shuffle=False,
                                             block_slices=block_slices)
        for batch in full_loader:
            blk = batch['blocks']
            if run_name.startswith('vae_'):
                mu, _ = h.encode(blk)
                z_chunks.append(mu.numpy())
            else:
                z_chunks.append(h.backbone(blk).numpy() if not hasattr(h, 'encode')
                                  else h.encode(blk).numpy())
    return np.concatenate(z_chunks, axis=0)

axis_label_map = {'genre': 'primary_genre', 'decade': 'decade_bin', 'lang': 'lang_top10'}
for best in [best_ae, best_vae, best_dec]:
    if best is None:
        continue
    name, m = best
    z_emb = _embed_run(name, m['z_dim'])
    for axis_short, label_key in axis_label_map.items():
        cev.umap_plot(z_emb, labels[label_key],
                      title=f'{name} — {axis_short}',
                      savepath=fig_dir / f'umap_{name}_{axis_short}.png', seed=42)
print(f"Generated 9 best-of-each main report figures.")
```

- [ ] **Step 12.2: Run notebook end-to-end.** Verify `results_table.csv`, `ablation_deltas.json`, `linear_probing.json` are produced and figures saved.

- [ ] **Step 12.3: Commit.**

```bash
git add notebooks/05_results.ipynb
git commit -m "feat(notebooks): 05_results — main table, ablation deltas, linear probing, UMAPs"
```

---

## Task 13 — Final Integration: pytest, results audit, reproducibility

**Files:**
- Run-only — no new files.

- [ ] **Step 13.1: Run full local test suite.**

```bash
cd "<repo-root>"
pytest tests/ -v --tb=short
```
Expected: all 26 tests pass (5 data + 7 losses + 4 backbone + 4 heads + 2 train + 4 eval).

- [ ] **Step 13.2: Verify all checkpoints exist (after Colab runs complete).**

```bash
ls -1 artifacts/models/ | sort
```
Expected output (subset, depending on optional W4):
```
ae_z128.pt
ae_z32.pt
ae_z64.pt
ae_z64_no_director.pt
ae_z64_no_text.pt
ae_z64_w1.pt
dec_z128_k10.pt  dec_z128_k21.pt  dec_z128_k30.pt
dec_z32_k10.pt   dec_z32_k21.pt   dec_z32_k30.pt
dec_z64_k10.pt   dec_z64_k21.pt   dec_z64_k30.pt
vanilla_ae_z64.pt
vae_z128.pt   vae_z32.pt   vae_z64.pt
```
21 baseline checkpoints (without optional W4 = 22).

- [ ] **Step 13.3: Verify success criteria (spec §9).**

Run a small Python check:
```python
import json
results = json.loads(open('artifacts/eval/results.json').read())

# Best deep model NMI(genre)
deep_runs = {k: v for k, v in results.items()
              if not k.startswith(('kmeans_raw', 'pca_kmeans')) and not k.startswith('vanilla_ae')}
best_deep_nmi = max(v.get('genre_nmi', 0) for v in deep_runs.values())

# Best baseline NMI(genre)
baseline_nmi = max(
    results.get('kmeans_raw_k21', {}).get('genre_nmi', 0),
    results.get('pca_kmeans_k21', {}).get('genre_nmi', 0),
    results.get('vanilla_ae_z64', {}).get('genre_nmi', 0),
)

print(f"Best deep NMI(genre):     {best_deep_nmi:.3f}")
print(f"Best baseline NMI(genre): {baseline_nmi:.3f}")
print(f"Relative improvement:     {(best_deep_nmi / max(baseline_nmi, 1e-6) - 1) * 100:.1f}%")
print(f"Absolute floor (>=0.15):  {'PASS' if best_deep_nmi >= 0.15 else 'FAIL'}")
print(f"Relative gain (>=10%):    {'PASS' if best_deep_nmi >= baseline_nmi * 1.10 else 'FAIL'}")
```

If criteria fail, that is a **finding**, not a project failure — see spec §9.

- [ ] **Step 13.4: Reproducibility check.**

Re-run a single training notebook cell (e.g., `ae_z64`) with the same seed. Compare final NMI:
```python
# Old NMI from results.json
old_nmi = results['ae_z64']['genre_nmi']
# Re-run by calling train_ae_run('ae_z64_repro', z_dim=64, w_blocks_to_use=w_blocks)
# new_nmi from the re-run
assert abs(old_nmi - new_nmi) < 0.02, f"reproducibility broken: {old_nmi} vs {new_nmi}"
```

- [ ] **Step 13.5: Final commit.**

```bash
git add docs/superpowers/plans/2026-05-04-modeling-implementation.md
git commit -m "feat(modeling): full AE/VAE/DEC comparative study complete"
git tag modeling-v1
```

---

## Verification Against Spec

Each spec requirement → task implementing it:

| Spec section | Implementing task(s) |
|---|---|
| §3 Data inputs (load + labels + split) | Task 2 |
| §4.1 MultiModalBackbone | Task 4 |
| §4.2 Three heads (AE/VAE/DEC) | Task 5 |
| §4.3 Hyperparameters | Tasks 6, 9, 10, 11 |
| §5.1-5.6 All loss functions | Task 3 |
| §6 Run matrix (21-22 runs) | Tasks 9, 10, 11 |
| §7 Code organization | Tasks 1-7 (package), 8-12 (notebooks) |
| §8.1 Always-on metrics | Task 7 (`evaluate_run`) |
| §8.2 UMAP visualization | Task 7 (`umap_plot`), Task 12 |
| §8.3.1 Linear probing | Task 7 (`linear_probe`), Task 12 |
| §8.3.2 Modality ablations | Task 9 (F1, F2 runs) |
| §8.4 Results aggregation | Task 12 |
| §9 Success criteria | Task 13 (verification) |
| §10 Risk mitigation | Tasks 6, 11 (early stop, cluster floor) |
| §13 Bridge to plan | This document |

If any spec requirement is missing a task, add the task before execution.
