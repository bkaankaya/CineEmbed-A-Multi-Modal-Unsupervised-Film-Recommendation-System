"""Data loading and label extraction for the cineembed modeling phase."""
from __future__ import annotations

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
    """Return the block name a feature column belongs to (deterministic priority).

    'director' is checked before 'language' so that dir_lang_* columns are not
    mis-classified into the language block.
    """
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


def _column_or_default(df: pd.DataFrame, col: str, default_value) -> pd.Series:
    """Return df[col] if present, else a Series of `default_value` matching df length.

    Direct column access is preferred over df.get(...) because pandas type stubs
    flag DataFrame.get's return type as Optional, even though pandas itself never
    returns None when a default is provided. Cast to Series satisfies Pyright,
    which can't narrow `df[col]`'s return type away from DataFrame.
    """
    if col in df.columns:
        result = df[col]
        # Defensive: in single-column edge cases df[col] could return DataFrame.
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        return result
    return pd.Series([default_value] * len(df))


def get_labels(csv_path: str | Path, top_lang_n: int = 10) -> dict[str, np.ndarray]:
    """Derive three orthogonal label vectors from movies_eda_final.csv (spec §3.2)."""
    df = pd.read_csv(csv_path, low_memory=False)

    # primary_genre = first piece of 'genres' before '|'; empty → 'Unknown'
    genres_first = _column_or_default(df, 'genres', '').fillna('').astype(str)
    primary_genre = genres_first.apply(lambda s: s.split('|')[0] if s else 'Unknown')
    primary_genre = primary_genre.replace('', 'Unknown').to_numpy()

    # decade_bin: prefer raw 'decade' column; fall back to reconstructing from 'decade_norm'
    # (EDA's movies_eda_final.csv stores only normalized form: (decade - 1900) / 130).
    if 'decade' in df.columns:
        decade_bin = df['decade'].apply(_bin_decade).to_numpy()
    elif 'decade_norm' in df.columns:
        norm_arr = np.asarray(df['decade_norm'].fillna(0.0), dtype=np.float64)
        decade_raw_arr = (norm_arr * 130.0 + 1900.0).round().astype(np.int64)
        # Mask missing rows (has_release_date == 0) → 0 ('Unknown' decade bin)
        if 'has_release_date' in df.columns:
            has_date_arr = np.asarray(df['has_release_date'].fillna(0), dtype=np.float64) > 0
            decade_raw_arr = np.where(has_date_arr, decade_raw_arr, 0)
        # Round down to nearest decade
        decade_bin = (decade_raw_arr // 10) * 10
    else:
        decade_bin = np.zeros(len(df), dtype=int)

    lang = _column_or_default(df, 'original_language', '').fillna('other').astype(str)
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
        self.X = X                 # NEVER copies
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


class _ContrastivePairDataset(Dataset):
    """Yields a single row twice — once per call, the dataset itself returns
    the unmodified blocks dict; the augmentation (modality dropout) is applied
    at COLLATE time so each batch gets two stochastic views with their own
    independent random masks per row.

    Spec: docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md §2.1.

    Note on design: we apply the dropout in the collate function rather than
    in __getitem__ because the augmentation needs to be applied as a per-block
    multiplicative mask on the tensor, and broadcasting that against the (B, d)
    block tensor is cheaper than per-row masking N times.
    """
    def __init__(
        self,
        X: torch.Tensor,
        has_bio: torch.Tensor,
        indices: np.ndarray | None = None,
    ):
        self.X = X                 # NEVER copies (same convention as _BlocksDataset)
        self.has_bio = has_bio
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0] if self.indices is None else len(self.indices)

    def __getitem__(self, i):
        real_i = int(self.indices[i]) if self.indices is not None else int(i)
        return {'X': self.X[real_i], 'has_bio': self.has_bio[real_i]}


def _sample_block_mask_batch(
    n: int,
    block_order: list[str],
    drop_prob: float,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Sample per-row 0/1 masks of shape (n, 1) for each block.

    Guarantees that every row has at least one block kept (otherwise the encoder
    receives all zeros and the InfoNCE loss becomes degenerate).
    """
    # (n, n_blocks)
    probs = rng.random((n, len(block_order)))
    mask_np = (probs >= drop_prob).astype(np.float32)

    # Re-sample rows where all blocks were dropped
    all_dropped = (mask_np.sum(axis=1) == 0)
    if all_dropped.any():
        for idx in np.where(all_dropped)[0]:
            # Keep at least one random block for this row
            mask_np[idx, rng.integers(0, len(block_order))] = 1.0

    return {
        b: torch.from_numpy(mask_np[:, i : i + 1])
        for i, b in enumerate(block_order)
    }


def _contrastive_collate(
    batch_list,
    block_slices: dict[str, slice],
    block_order: list[str],
    drop_prob: float,
    rng: np.random.Generator,
):
    """Collate two augmented views per row, each with its own per-row random block_mask.

    Returns:
        {
            'view_a': {'blocks': dict, 'has_bio': tensor, 'block_mask': dict},
            'view_b': {'blocks': dict, 'has_bio': tensor, 'block_mask': dict},
        }
    Each `block_mask` contains (B, 1) tensors. This provides per-row stochastic
    augmentation, preventing the model from co-adapting to a batch-wide mask.
    """
    X = torch.stack([b['X'] for b in batch_list], dim=0)
    has_bio = torch.stack([b['has_bio'] for b in batch_list], dim=0)
    blocks = _split_into_blocks(X, block_slices)
    n = X.shape[0]

    mask_a = _sample_block_mask_batch(n, block_order, drop_prob, rng)
    mask_b = _sample_block_mask_batch(n, block_order, drop_prob, rng)
    return {
        'view_a': {'blocks': blocks, 'has_bio': has_bio, 'block_mask': mask_a},
        'view_b': {'blocks': blocks, 'has_bio': has_bio, 'block_mask': mask_b},
    }


def make_contrastive_dataloader(
    X: torch.Tensor,
    has_bio: torch.Tensor,
    batch_size: int,
    *,
    block_slices: dict[str, slice],
    block_order: list[str] | None = None,
    drop_prob: float = 0.3,
    indices: np.ndarray | None = None,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader that yields paired-view contrastive batches.

    Spec: docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md §2.1.

    Each batch is a dict with keys 'view_a' and 'view_b'. Each view contains
    {'blocks': per-block tensors, 'has_bio': bio flag, 'block_mask': per-block
    keep/drop dict to be passed to backbone(blocks, block_mask=...)}.

    Args:
        block_slices: per-block column slices (from `get_block_indices`).
        block_order: explicit block ordering; defaults to BLOCK_ORDER.
        drop_prob: per-block dropout probability for each view (independent of
                   the other view). 0.3 means each view drops ~2 of 7 modalities
                   on expectation. Tune via the contrastive validation curve.
        seed: rng seed. Use different seeds for train/val loaders.
    """
    block_order = list(block_order) if block_order is not None else list(BLOCK_ORDER)
    rng = np.random.default_rng(seed)
    dataset = _ContrastivePairDataset(X, has_bio, indices=indices)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        collate_fn=lambda batch: _contrastive_collate(
            batch, block_slices, block_order, drop_prob, rng,
        ),
    )


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
        canonical_dims = [6, 22, 31, 2, 6, 384, 113]
        out, start = {}, 0
        for b, d in zip(BLOCK_ORDER, canonical_dims):
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
