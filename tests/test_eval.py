import numpy as np

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


# ---- New: clustering-improvement-techniques tests ------------------------------
# Spec: docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md


def test_evaluate_run_includes_ami(synthetic_labels):
    """AMI must be present alongside NMI/ARI for each axis (spec §2.4)."""
    n = 200
    np.random.seed(0)
    cluster_ids = np.random.randint(0, 21, size=n)
    metrics = cev.evaluate_run(cluster_ids, synthetic_labels)
    for axis in ('genre', 'decade', 'lang'):
        assert f'{axis}_ami' in metrics
        # AMI lower-bound is technically slightly negative under chance — bound at -0.1.
        assert -0.1 <= metrics[f'{axis}_ami'] <= 1.0


def test_evaluate_run_per_axis_k_uses_axis_specific_k(synthetic_labels):
    """Output keys must encode the per-axis k actually used (spec §2.2)."""
    z = np.random.randn(200, 16).astype(np.float32)
    metrics = cev.evaluate_run_per_axis_k(z, synthetic_labels)
    # Default k matches DEFAULT_AXIS_K → genre=21, decade=12, lang=11
    assert 'genre_nmi_k21' in metrics
    assert 'decade_nmi_k12' in metrics
    assert 'lang_nmi_k11' in metrics
    for k in metrics.values():
        assert -0.1 <= k <= 1.0


def test_evaluate_run_per_axis_k_custom(synthetic_labels):
    """Caller may override per-axis k."""
    z = np.random.randn(200, 16).astype(np.float32)
    custom_k = {'genre': 5, 'decade': 5, 'lang': 5}
    metrics = cev.evaluate_run_per_axis_k(z, synthetic_labels, axis_k=custom_k)
    assert 'genre_nmi_k5' in metrics
    assert 'decade_ari_k5' in metrics
    assert 'lang_ami_k5' in metrics


def test_cluster_assignments_gmm_returns_int_array():
    z = np.random.randn(200, 16).astype(np.float32)
    c = cev.cluster_assignments_gmm(z, k=8, seed=42)
    assert c.shape == (200,)
    assert c.dtype.kind == 'i'
    assert set(np.unique(c)).issubset(set(range(8)))


def test_cluster_assignments_spectral_returns_int_array():
    np.random.seed(0)
    z = np.random.randn(80, 8).astype(np.float32)  # smaller N for spectral speed
    c = cev.cluster_assignments_spectral(z, k=4, seed=42, n_neighbors=10)
    assert c.shape == (80,)
    assert c.dtype.kind == 'i'
    assert set(np.unique(c)).issubset(set(range(4)))


def test_cluster_assignments_hdbscan_runs():
    """HDBSCAN auto-discovers cluster count; -1 = noise is allowed."""
    np.random.seed(0)
    z = np.random.randn(150, 8).astype(np.float32)
    c = cev.cluster_assignments_hdbscan(z, min_cluster_size=10, min_samples=5)
    assert c.shape == (150,)
    assert c.dtype.kind == 'i'
    # All values must be ≥ -1 (sentinel for noise)
    assert (c >= -1).all()


def test_multilabel_macro_nmi_basic_contract():
    """Output dict has macro / per_genre / n_genres_evaluated keys."""
    np.random.seed(0)
    n, G = 200, 5
    cluster_ids = np.random.randint(0, 4, size=n)
    onehot = (np.random.rand(n, G) > 0.7).astype(np.int64)
    out = cev.multilabel_macro_nmi(cluster_ids, onehot, metric='nmi')
    assert 'macro' in out
    assert 'per_genre' in out
    assert 'n_genres_evaluated' in out
    assert out['n_genres_evaluated'] == len(out['per_genre'])
    assert 0.0 <= out['macro'] <= 1.0


def test_multilabel_macro_nmi_skips_degenerate_genres():
    """Genres with all-zero or all-one columns are skipped."""
    n = 100
    cluster_ids = np.random.randint(0, 5, size=n)
    onehot = np.zeros((n, 4), dtype=np.int64)
    onehot[:, 0] = 1                   # all-one  → degenerate, skip
    onehot[:, 1] = 0                   # all-zero → degenerate, skip
    onehot[:50, 2] = 1                 # half-and-half → valid
    onehot[::2, 3] = 1                 # alternating → valid
    out = cev.multilabel_macro_nmi(cluster_ids, onehot, metric='nmi')
    assert out['n_genres_evaluated'] == 2
    assert set(out['per_genre'].keys()) == {2, 3}


def test_multilabel_macro_nmi_metric_choice():
    """metric='ami' and metric='nmi' produce different but valid scores."""
    np.random.seed(0)
    n = 200
    cluster_ids = np.random.randint(0, 10, size=n)
    onehot = (np.random.rand(n, 6) > 0.6).astype(np.int64)
    out_nmi = cev.multilabel_macro_nmi(cluster_ids, onehot, metric='nmi')
    out_ami = cev.multilabel_macro_nmi(cluster_ids, onehot, metric='ami')
    assert out_nmi['n_genres_evaluated'] == out_ami['n_genres_evaluated']
    # AMI is chance-corrected; for random clusterings, AMI ≈ 0 < NMI.
    assert -0.1 <= out_ami['macro'] <= out_nmi['macro'] + 0.1


def test_multilabel_macro_nmi_rejects_unknown_metric():
    cluster_ids = np.random.randint(0, 5, size=10)
    onehot = np.zeros((10, 3), dtype=np.int64)
    onehot[::2, 0] = 1
    import pytest
    with pytest.raises(ValueError):
        cev.multilabel_macro_nmi(cluster_ids, onehot, metric='something_else')
