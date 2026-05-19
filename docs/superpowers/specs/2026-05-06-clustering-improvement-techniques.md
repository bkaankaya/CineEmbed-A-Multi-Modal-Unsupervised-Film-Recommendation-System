# Clustering-Improvement Techniques — Implementation Spec

**Date:** 2026-05-06
**Authors:** Internal dev sprint
**Status:** APPROVED for implementation (developer-facing only — no report/PPTX edits)

---

## 1. Motivation

The MVP's best clustering scores (`gNMI = 0.332`, `gARI = 0.244`) are task-appropriate for heterogeneous tabular+text+sparse-categorical data with multi-label long-tailed genres. They are *not* "weak" — they sit comfortably in the 0.20–0.45 NMI band that 2024–2025 deep-clustering literature reports for similar problems. However, several established techniques can plausibly push the numbers higher with bounded engineering effort. This spec captures the five that survived a codebase-feasibility audit.

### Out-of-scope (acknowledged but not implemented this sprint)

- **Larger text embedding** (e.g., BGE-large, e5-mistral, text-embedding-3-small) — requires re-running EDA on raw text. The pipeline supports it but the re-embedding pass is several hours and the raw text CSV is gitignored.
- **MMCMAE** (Multi-Modal Contrastive Masked AE, CVPR 2025) — two-stage progressive pre-training; ~2-week reimplementation.
- **TabClusterNet** — replaces encoder with TabNet, invalidates all existing checkpoints.

---

## 2. Techniques in scope

### 2.1 Contrastive pre-training stage (SimCLR-style, modality-dropout augmentation)

**Goal:** add a self-supervised pretext stage before AE/DEC training that pulls together two augmented views of the same row in latent space and pushes apart different rows.

**Insight specific to our project:** the existing `MultiModalBackbone.forward(blocks, block_mask)` already supports zeroing out arbitrary modality blocks (originally for F1/F2 ablation). This is *exactly* the right augmentation primitive for contrastive learning on heterogeneous data — drop a random subset of modalities to create two stochastic views of the same row, treat them as a positive pair.

**API:**

```python
# losses.py
def info_nce_loss(z_a: Tensor, z_b: Tensor, temperature: float = 0.5) -> Tensor:
    """Symmetric InfoNCE over two views (B, d). Standard SimCLR formulation."""

# heads.py
class ContrastiveHead(nn.Module):
    """Backbone + 2-layer projection MLP (d_z → d_z → d_proj). SimCLR-paper style.
    
    The projection is used ONLY during contrastive training; downstream
    AE/DEC heads operate directly on the backbone latent z (the projection
    is discarded after pretext, per Chen et al. 2020).
    """
    def __init__(self, backbone: MultiModalBackbone, projection_dim: int = 128): ...
    def forward(self, blocks, block_mask=None) -> Tensor: ...

# data.py
class ContrastivePairDataset(Dataset):
    """Yields a single row twice with two independent random modality-dropout masks."""
    def __init__(self, X: Tensor, has_bio: Tensor,
                 block_order: list[str], drop_prob: float = 0.3,
                 indices: np.ndarray | None = None): ...

def make_contrastive_dataloader(X, has_bio, batch_size, *,
                                block_slices, drop_prob=0.3,
                                indices=None, seed=42, num_workers=0) -> DataLoader: ...
```

**Training entry-point:** uses the existing `train.train_model` with a closure that calls `model(view_a) → z_a`, `model(view_b) → z_b`, `info_nce_loss(z_a, z_b)`. No changes to `train.py` itself.

**Expected payoff:** 5–12% NMI lift after 30–60 epochs of pretext, before any AE/DEC fine-tune. Confirmed in 2024–2025 deep-clustering literature (TCSS, SCAN family, sgSDC).

**Hyperparameters:**
- `temperature ∈ {0.1, 0.5}` — sweep both. Default 0.1 (heterogeneous tabular
  signal is denser than natural-image embeddings, so lower temperature sharpens
  the contrastive objective more effectively); SimCLR default 0.5 included as
  baseline. **Amendment 2026-05-16:** original spec specified 0.5 only; default
  changed to 0.1 after follow-on research, sweep retains 0.5 as comparison.
- `projection_dim = 128` (2× latent_dim is the SimCLR rule of thumb)
- `drop_prob = 0.3` per modality (each view drops ~2 of 7 modalities on expectation)
- `batch_size = 1024` to keep enough negatives per batch
- **Masking granularity: per-row.** Each row in a batch gets independent block
  masks (shape `(B, 1)` per block). Original implementation used per-batch
  scalar masks ("shatters negatives" concern); per-row prevents batch-level
  co-adaptation and gives stronger negatives. Backbone forward accepts either
  scalar `float` (legacy F1/F2 ablation) or `Tensor (B,1)` (contrastive views).
  **Amendment 2026-05-16:** added; original spec was silent on granularity.

### 2.2 Per-axis k-sweep evaluation

**Goal:** stop penalizing decade and language NMI mathematically. Currently we use `k = 21` for all three axes (matching genre cardinality). When evaluating against `decade_bin` (~12) or `lang_top10` (11), the partition has too many clusters to align cleanly — that's a known NMI/ARI hit.

**API:**

```python
# eval.py
DEFAULT_AXIS_K = {'genre': 21, 'decade': 12, 'lang': 11}

def evaluate_run_per_axis_k(z: np.ndarray,
                            labels: dict[str, np.ndarray],
                            *,
                            axis_k: dict[str, int] | None = None,
                            seed: int = 42) -> dict[str, float]:
    """Run KMeans 3 times (one per axis with axis-matched k). Report per-axis
    NMI/ARI/AMI from the corresponding k. Output keys: '{short}_nmi_kN', etc.
    """
```

**Expected payoff:** lNMI typically jumps +5–10% absolute when `k` matches axis cardinality. dNMI similar. Pure measurement-honesty fix; doesn't change models.

### 2.3 Soft / non-KMeans clustering algorithms

**Goal:** KMeans hard-assigns each row to one cluster, which is a poor inductive bias when films are multi-genre. GMM gives soft probabilities; spectral clustering captures non-convex structure; HDBSCAN handles variable density.

**API:**

```python
# eval.py
def cluster_assignments_gmm(z, k, *, seed=42, n_init=5) -> np.ndarray: ...
def cluster_assignments_spectral(z, k, *, seed=42, affinity='cosine',
                                  n_neighbors=15) -> np.ndarray: ...
def cluster_assignments_hdbscan(z, *, min_cluster_size=50,
                                 min_samples=5) -> np.ndarray: ...
```

GMM is the most defensible substitute for KMeans on our data (continuous Gaussian-ish latent, multi-membership semantics via the soft posterior). HDBSCAN is for exploration — it doesn't take `k`, it discovers the cluster count. Spectral helps if the latent has non-convex manifolds.

**Expected payoff:** GMM typically +2–5% gARI over KMeans on Gaussian latents. HDBSCAN gives a *different* signal (number of natural clusters in the latent) that's useful for the final report's interpretability section.

### 2.4 AMI (Adjusted Mutual Information)

**Goal:** report a chance-adjusted complement to NMI. Per JMLR 2016 (Vinh et al.), NMI is biased upward when reference clusterings are imbalanced (which `primary_genre` absolutely is). AMI is the chance-adjusted variant.

**API:** purely additive to existing `evaluate_run`:

```python
def evaluate_run(cluster_ids, labels) -> dict[str, float]:
    out = {}
    for short, full in axis_aliases.items():
        labs = labels[full]
        out[f'{short}_nmi'] = float(normalized_mutual_info_score(labs, cluster_ids))
        out[f'{short}_ari'] = float(adjusted_rand_score(labs, cluster_ids))
        out[f'{short}_ami'] = float(adjusted_mutual_info_score(labs, cluster_ids))  # NEW
    return out
```

**Expected payoff:** "honesty fix" — AMI < NMI for our imbalanced labels, but it's the fair number to report alongside the JMLR-cited NMI.

### 2.5 Multi-label genre evaluation

**Goal:** stop collapsing multi-genre films to a single `primary_genre` for evaluation. The full multi-label genre vector is already on disk inside the feature matrix's `genre` block (22 dims = 21 one-hot indicators + `has_genre` flag) — we just don't use it for eval.

**Method:** for each of the 21 genres, treat cluster membership as a binary classification problem ("does this cluster correlate with genre G?") and compute binary NMI. Then macro-average across genres.

**API:**

```python
# eval.py
def multilabel_macro_nmi(cluster_ids: np.ndarray,
                         genre_onehot: np.ndarray,
                         *,
                         metric: str = 'nmi') -> dict[str, float]:
    """Per-genre binary NMI (or AMI), macro-averaged across genres.
    
    Args:
        cluster_ids: (N,) cluster assignments
        genre_onehot: (N, 21) multi-label genre indicators (0/1)
        metric: 'nmi' | 'ami'
    
    Returns:
        {'macro_nmi': float, 'per_genre': dict[int, float]}
    """
```

**Expected payoff:** the macro number will be *similar* in absolute value to single-label NMI but tells a more honest story — it accounts for the multi-genre encoding the model legitimately learns. The per-genre breakdown is also a great diagnostic for which genres the model captures vs not (long-tail genres typically score lower).

---

## 3. File-by-file changes

| File | Change | Lines |
|---|---|---:|
| `src/cineembed/losses.py` | + `info_nce_loss(z_a, z_b, temperature)` | +25 |
| `src/cineembed/heads.py` | + `ContrastiveHead(backbone, projection_dim)` | +35 |
| `src/cineembed/data.py` | + `ContrastivePairDataset` + `make_contrastive_dataloader` | +75 |
| `src/cineembed/eval.py` | + GMM/spectral/HDBSCAN, AMI in `evaluate_run`, `evaluate_run_per_axis_k`, `multilabel_macro_nmi` | +120 |
| `tests/test_losses.py` | + InfoNCE tests | +30 |
| `tests/test_heads.py` | + ContrastiveHead tests | +25 |
| `tests/test_data.py` | + ContrastivePairDataset tests | +30 |
| `tests/test_eval.py` | + GMM/spectral/HDBSCAN/AMI/per-axis-k/multilabel-NMI tests | +90 |

**Total: ~430 LOC added across 8 files.** No existing line removed; all additive. No public API of existing functions changes (additive `_ami` keys in `evaluate_run` output won't break callers that read `_nmi`/`_ari` only).

## 4. Acceptance

- [x] All new functions have docstrings referencing this spec.
- [x] `pytest -q` passes with 100% pass rate including new tests.
- [x] No deprecation warnings from sklearn/torch.
- [x] No existing test breaks (additive-only contract).
- [x] Type-checks clean under the project's existing Pyright config (modulo the known python-pptx/sklearn stub limitations the team has already accepted).
- [x] Imports follow existing style (`from __future__ import annotations`, type hints, docstring spec-cross-refs).

**Status 2026-05-16:** All acceptance items met as of commit 8097685. Phase 1
contrastive sweep launched 2026-05-16 (commit 5fa95ff) on Colab; results land
in wandb group `phase-1-sweep`.

## 5. Out-of-scope (explicit non-goals for this sprint)

- Running any of these techniques on the production data. The existing checkpoints in `artifacts/models/` are unchanged. Anyone can run any of these on existing latents whenever.
- Updating `intermediate-progress-report.tex`, `slides.py`, or any other report/PPTX content.
- Modifying `artifacts/eval/results.json`. Stays as the canonical record of the MVP's six pre-registered runs.
- Modifying `train.py`. The new contrastive loop reuses `train_model` via the existing `loss_fn` callable interface.

**Amendment 2026-05-16:** Out-of-scope item 1 ("Not running these techniques on
production data") is superseded — the Phase 1 sweep is currently running 3
contrastive configs on the production feature matrix (Colab, wandb group
`phase-1-sweep`).
