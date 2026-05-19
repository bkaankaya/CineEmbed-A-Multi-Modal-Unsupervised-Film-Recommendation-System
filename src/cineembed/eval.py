"""Evaluation helpers for AE/VAE/DEC runs (spec §8).

Clustering-improvement extensions (spec
docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md):
- AMI alongside NMI in `evaluate_run`               (§2.4)
- `evaluate_run_per_axis_k` — per-axis k-sweep      (§2.2)
- GMM / spectral / HDBSCAN cluster assignments      (§2.3)
- `multilabel_macro_nmi` — multi-label genre eval   (§2.5)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless / Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# Axis aliases used by `evaluate_run` and `evaluate_run_per_axis_k`.
_AXIS_ALIASES: dict[str, str] = {
    'genre': 'primary_genre',
    'decade': 'decade_bin',
    'lang': 'lang_top10',
}

# Default k per axis matching ground-truth label cardinality (spec §2.2).
DEFAULT_AXIS_K: dict[str, int] = {'genre': 21, 'decade': 12, 'lang': 11}


def cluster_assignments_kmeans(z: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Run KMeans(k) on latent vectors → cluster ids."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)  # type: ignore[arg-type]
    return km.fit_predict(z).astype(np.int64)


def cluster_assignments_gmm(
    z: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    n_init: int = 5,
    covariance_type: str = 'diag',
) -> np.ndarray:
    """Soft-clustering via Gaussian Mixture (spec §2.3).

    Uses `predict()` for hard argmax over the posterior. For soft probabilities,
    callers can fit `GaussianMixture` directly and use `predict_proba`.

    Args:
        covariance_type: 'diag' is a good default for high-dim latents; 'full'
            is more flexible but slower and prone to singular cov on sparse data.
    """
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=seed,
        max_iter=200,
        reg_covar=1e-4,
    )
    return gm.fit_predict(z).astype(np.int64)


def cluster_assignments_spectral(
    z: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Spectral clustering on a k-NN cosine-affinity graph (spec §2.3).

    Useful when the latent has non-convex cluster shapes that KMeans/GMM can't
    capture. Computationally heavier than KMeans (eigendecomposition of an N×N
    affinity matrix); recommended for N ≤ ~10k.
    """
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=k,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        assign_labels='kmeans',
        random_state=seed,
        n_jobs=-1,
    )
    return sc.fit_predict(z).astype(np.int64)


def cluster_assignments_hdbscan(
    z: np.ndarray,
    *,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Density-based clustering — discovers cluster count automatically (spec §2.3).

    Returns int labels; -1 means "noise / not assigned to any cluster" (HDBSCAN's
    standard convention). Callers that want a fully-partitioned output should
    relabel -1 to a sentinel cluster id before computing NMI/ARI.
    """
    try:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "HDBSCAN requires scikit-learn >= 1.3 or the standalone `hdbscan` "
            "package. The project pins sklearn>=1.4 so this should work."
        ) from e
    h = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
    )
    return h.fit_predict(z).astype(np.int64)


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
    """Compute NMI, ARI, and AMI vs each label axis (spec §8.1, §2.4).

    AMI (Adjusted Mutual Information, Vinh et al. JMLR 2016) is added alongside
    NMI to give a chance-adjusted complement — relevant for our imbalanced
    `primary_genre` labels where NMI is biased upward.

    Output keys (per axis short ∈ {'genre', 'decade', 'lang'}):
        '{short}_nmi', '{short}_ari', '{short}_ami'
    """
    out: dict[str, float] = {}
    for short, full in _AXIS_ALIASES.items():
        labs = labels[full]
        out[f'{short}_nmi'] = float(normalized_mutual_info_score(labs, cluster_ids))
        out[f'{short}_ari'] = float(adjusted_rand_score(labs, cluster_ids))
        out[f'{short}_ami'] = float(adjusted_mutual_info_score(labs, cluster_ids))
    return out


def evaluate_run_per_axis_k(
    z: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    axis_k: dict[str, int] | None = None,
    seed: int = 42,
    cluster_fn=cluster_assignments_kmeans,
) -> dict[str, float]:
    """Per-axis k-sweep evaluation (spec §2.2).

    For each axis, run clustering with k matching that axis's ground-truth
    cardinality, then score only that axis's metric from the corresponding
    partition. Removes the cardinality-mismatch penalty that fixed-k=21 suffers
    on `decade_bin` (~12 classes) and `lang_top10` (11 classes).

    Args:
        z: (N, d) latent matrix.
        labels: ground-truth dict matching `_AXIS_ALIASES` keys.
        axis_k: per-axis cluster count. Defaults to DEFAULT_AXIS_K.
        cluster_fn: clustering function (z, k, seed) → cluster_ids. Defaults to
            KMeans; callers can pass `cluster_assignments_gmm` etc.

    Returns:
        Dict with keys '{short}_nmi_kN', '{short}_ari_kN', '{short}_ami_kN'
        where N is the axis-specific k actually used.
    """
    axis_k = dict(axis_k) if axis_k is not None else dict(DEFAULT_AXIS_K)
    out: dict[str, float] = {}
    for short, full in _AXIS_ALIASES.items():
        if short not in axis_k:
            continue
        k = axis_k[short]
        cluster_ids = cluster_fn(z, k=k, seed=seed)
        labs = labels[full]
        out[f'{short}_nmi_k{k}'] = float(normalized_mutual_info_score(labs, cluster_ids))
        out[f'{short}_ari_k{k}'] = float(adjusted_rand_score(labs, cluster_ids))
        out[f'{short}_ami_k{k}'] = float(adjusted_mutual_info_score(labs, cluster_ids))
    return out


def multilabel_macro_nmi(
    cluster_ids: np.ndarray,
    genre_onehot: np.ndarray,
    *,
    metric: str = 'nmi',
    min_genre_count: int = 1,
) -> dict:
    """Per-genre binary clustering quality, macro-averaged (spec §2.5).

    For each of the 21 genres, treat membership as a binary label
    (1 = movie has this genre, 0 = doesn't) and compute NMI/AMI between the
    cluster assignment and that binary label. Then macro-average across genres.
    This gives a fair multi-label number that doesn't penalize the model for
    legitimately encoding multi-genre membership (e.g., action-comedy films
    that single-label `primary_genre` collapses arbitrarily).

    Args:
        cluster_ids: (N,) cluster assignments.
        genre_onehot: (N, G) multi-label genre indicators in {0, 1}. Pass the
            21-column genre block from the feature matrix (excluding the
            `has_genre` flag).
        metric: 'nmi' or 'ami'.
        min_genre_count: skip genres with fewer than this many positive labels
            (degenerate binary partitions of all-zero / all-one).

    Returns:
        {
            'macro': float,                # macro-averaged across genres
            'per_genre': dict[int, float], # per-genre score by column index
            'n_genres_evaluated': int,     # number of non-degenerate genres
        }
    """
    if metric not in ('nmi', 'ami'):
        raise ValueError(f"metric must be 'nmi' or 'ami', got {metric!r}")
    score_fn = (normalized_mutual_info_score
                if metric == 'nmi' else adjusted_mutual_info_score)

    if genre_onehot.ndim != 2:
        raise ValueError(f"genre_onehot must be 2D, got shape {genre_onehot.shape}")
    if genre_onehot.shape[0] != cluster_ids.shape[0]:
        raise ValueError(
            f"row mismatch: cluster_ids has {cluster_ids.shape[0]} rows, "
            f"genre_onehot has {genre_onehot.shape[0]}"
        )

    G = genre_onehot.shape[1]
    per_genre: dict[int, float] = {}
    for g in range(G):
        col = genre_onehot[:, g].astype(np.int64)
        positive = int(col.sum())
        if positive < min_genre_count or positive >= len(col):
            continue  # degenerate (all zeros or all ones) → skip
        per_genre[g] = float(score_fn(col, cluster_ids))

    macro = float(np.mean(list(per_genre.values()))) if per_genre else 0.0
    return {
        'macro': macro,
        'per_genre': per_genre,
        'n_genres_evaluated': len(per_genre),
    }


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
    import umap  # type: ignore[import]
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    # Cast to ndarray — umap-learn stubs annotate fit_transform's return as
    # `float` on some platforms; runtime is always (N, 2) ndarray.
    z2d = np.asarray(reducer.fit_transform(z))

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
