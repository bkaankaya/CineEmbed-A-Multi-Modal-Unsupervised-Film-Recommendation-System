"""Shared pytest fixtures for the cineembed test suite.

Synthetic feature matrices match the (329044, 564) production layout but are
small enough to run instantly on Mac CPU.
"""
import numpy as np
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
