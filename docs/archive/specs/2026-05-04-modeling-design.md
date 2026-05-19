# CineEmbed — Modeling Phase (AE / VAE / DEC) Design

**Status:** Draft (2026-05-04)
**Author:** Baran Dinçoğuz (with Claude)
**Course:** SENG 474 — Deep Learning · TED University · Spring 2026
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya
**Predecessor:** [`2026-05-03-eda-v2-design.md`](2026-05-03-eda-v2-design.md), [`2026-05-04-eda-v2-director-profile-extension-design.md`](2026-05-04-eda-v2-director-profile-extension-design.md)
**Decision log:** [`docs/adr/0001-modeling-hybrid-architecture.md`](../../adr/0001-modeling-hybrid-architecture.md) (D1–D9)

---

## 1. Overview

The EDA phase produced a `(329044, 564)` feature matrix from seven heterogeneous modality blocks (numerical, genre, language, decade, awards, text-overview, director-profile). The matrix exhibits a severe modality variance imbalance — text/bio blocks (L2-normalized) contribute ≈2 to total variance versus ≈18-22 from the StandardScaler/one-hot blocks. Naïve unsupervised methods will therefore learn from non-text features and ignore the text/bio signals.

This document specifies a **comparative study** of three representation-learning approaches for movie metadata clustering: a deterministic **AutoEncoder (AE)**, a **Variational AutoEncoder (VAE)**, and **Deep Embedded Clustering (DEC)**. All three share a **modality-aware backbone** that projects each block through a learned linear layer before fusion, addressing the variance imbalance at the architectural level (rather than as a post-hoc loss-weighting hack).

**Explicit framing:** This study does NOT claim to identify "the best" deep clustering model for movie metadata. It asks how three increasingly complex representation-learning approaches handle a heterogeneous multi-modal feature matrix. The hypothesis is **architectural** — modality-aware backbone outperforms a flat concat baseline. Reported clustering metrics (NMI/ARI vs primary genre, decade, language) are probes of latent-space structure across three orthogonal axes, not leaderboard scores.

The deliverables are:
1. A shared Python package `src/cineembed/` containing the multi-modal backbone, three model heads (AE, VAE, DEC), weighted-MSE loss with bio masking, and evaluation helpers.
2. Five Jupyter notebooks: `01_smoke_test`, `02_train_ae`, `03_train_vae`, `04_train_dec`, `05_results`.
3. 21-22 trained model checkpoints (3 baselines + 3 AE + 3 VAE + 9 DEC + 3 ablations + optional W4 stretch).
4. A consolidated `results.json` with all per-run metrics.
5. A final `05_results.ipynb` notebook with comparison tables, latent-space visualizations, and ablation deltas suitable for the SENG 474 final report.

---

## 2. Goals and Non-Goals

### Goals
- Train three model families (AE, VAE, DEC) at three latent dimensions (z ∈ {32, 64, 128}) on the 564-dim feature matrix from `artifacts/feature_matrix.npz`.
- Train three reference baselines (raw KMeans, PCA+KMeans, vanilla concat-AE) to validate the multi-modal backbone hypothesis.
- Address modality variance imbalance (ratio ≈ 0.046) via two complementary mechanisms: architectural (modality-specific projection layers) and loss-level (W2 inverse-variance weighting with clipping).
- Address the sparse director-bio coverage (3.2% of films) via masked reconstruction loss (G2): `has_director_bio == 0` rows do not receive gradient on the 64 `dir_bio_pca_*` dimensions.
- Evaluate every run on three orthogonal axes — genre (21 classes), decade (~13 classes), language top-10+other (11 classes) — using NMI and ARI.
- Produce a results table with shape (21-22 runs × ≥4 metrics), suitable for academic comparison.
- Demonstrate via ablations that the design choices (multi-modal backbone, W2 weighting, text/director modalities) measurably affect clustering quality.

### Non-Goals
- Building yet more EDA features (the feature matrix is locked).
- Training a "best" production model — this is a comparative study.
- Hyperparameter sweeps beyond the three explicit grids (z, k for DEC, weighting strategy).
- Multi-label NMI variants (L3 in D7) — adds complexity without proportional insight.
- Deploying a clustering API or recommendation system.
- Re-running the EDA pipeline (artifacts `feature_matrix.npz`, `feature_metadata.json`, `movies_eda_final.csv` are locked inputs).

---

## 3. Data Inputs

All inputs come from `artifacts/` produced by the EDA phase. The modeling phase MUST NOT re-run the upstream pipeline.

| Path | Shape / size | Role |
|---|---|---|
| `artifacts/feature_matrix.npz` | `X: (329044, 564)`, `feature_names: (564,)` | Model input — all training |
| `artifacts/feature_metadata.json` | dict | Block boundary lookup, column→block mapping |
| `artifacts/movies_eda_final.csv` | `(329044, ~146)` | Raw + encoded columns; source for ground-truth labels |
| `artifacts/director_profile_metadata.json` | dict | Sanity reference; not used at training time |
| `artifacts/pipeline_version.json` | dict (incl. `feature_matrix_md5`) | Reproducibility audit |

### 3.1 Block boundary discovery

The package's `data.py` derives block boundaries from `feature_names`:

```python
BLOCK_PREFIXES = {
    'numerical': ['log_', 'runtime_', 'vote_', 'has_vote', 'has_engagement'],
    'genre':     ['genre_', 'has_genre'],
    'language':  ['lang_'],
    'decade':    ['decade_norm', 'has_release_date'],
    'awards':    ['prior_log_'],
    'text':      ['text_'],
    'director':  ['dir_bio_pca_', 'has_director_bio', 'dir_lang_', 'dir_country_', 'has_director_lang'],
}
```

Verified expected dims: numerical=6, genre=22, language=31, decade=2, awards=6, text=384, director=113. Total = 564.

### 3.2 Ground-truth labels

Three orthogonal label vectors derived from `movies_eda_final.csv` and `feature_matrix`:

| Axis | Computation | Class count |
|---|---|---:|
| `primary_genre` | `df['genres'].str.split('|').str[0].fillna('Unknown')` | 21 |
| `decade_bin` | `df['decade']` cast to int (0, 1900, 1910, ..., 2020) | 13-14 |
| `lang_top10` | top-10 of `df['original_language']` + 'other' bucket | 11 |

These vectors are fixed once at the start of evaluation and reused across all 21-22 runs.

### 3.3 Train / validation split

We adopt a 90/10 split with `random_state=42` for early-stopping validation, and use ALL 329,044 embeddings for downstream evaluation:

- **Encoder/decoder training**: 90% of 329,044 films for gradient updates; 10% held out as a validation set used solely to track weighted reconstruction loss for early stopping. Final embedding extraction uses the trained encoder over the full 329,044 dataset.
- **Why split despite "unsupervised"**: early stopping needs a held-out signal. Computing it on training data invites overfitting that we cannot detect.
- **Cluster evaluation (NMI / ARI / UMAP)**: computed on all 329,044 embeddings (the encoder is frozen during eval, so train/val membership is irrelevant downstream).
- **Linear probing (Tier 2 eval)**: separate 80/20 split with the same `random_state=42` for the LINEAR classifier itself; encoder remains frozen.
- **Reproducibility**: same seed (42) used everywhere via `seed_everything()` from the EDA spec.

---

## 4. Architecture

### 4.1 Shared multi-modal backbone

Per ADR D1. Each modality block is projected to a fixed sub-dim before fusion.

**Modality projection layers (preliminary sizes; tunable):**

| Block | Input dim | Projection | Output dim |
|---|---:|---|---:|
| numerical | 6 | `Linear(6, 16) → ReLU → Dropout(0.1)` | 16 |
| genre | 22 | `Linear(22, 16) → ReLU → Dropout(0.1)` | 16 |
| language | 31 | `Linear(31, 16) → ReLU → Dropout(0.1)` | 16 |
| decade | 2 | `Linear(2, 4) → ReLU` | 4 |
| awards | 6 | `Linear(6, 16) → ReLU → Dropout(0.1)` | 16 |
| text | 384 | `Linear(384, 64) → ReLU → Dropout(0.2)` | 64 |
| director | 113 | `Linear(113, 32) → ReLU → Dropout(0.2)` | 32 |
| **Concat** | | | **164** |

**Backbone FC stack (post-concat):**
```
concat (164) → Linear(164, 128) → ReLU → Dropout(0.2) → Linear(128, z)
```

`z ∈ {32, 64, 128}` per the ablation grid. The backbone is identical across AE, VAE, DEC; only the head differs.

### 4.2 Three model heads

#### 4.2.1 AE Head

Mirror of the encoder: `Linear(z, 128) → ReLU → Linear(128, 164) → split into per-block decoders → reconstruction`.

Per-block decoders mirror the projection layers in reverse:
```
sub-z (e.g., 16 for numerical) → ReLU → Linear(16, original_dim)
```

Output: dict mapping block name to reconstructed tensor.

Loss: W2 weighted MSE with G2 bio mask (see §5).

#### 4.2.2 VAE Head

Encoder backbone outputs split into `μ` and `log_var` via two parallel heads:
```
backbone_output (z) → split into μ (z), log_var (z)
sample: ε ~ N(0, I), z = μ + ε * exp(0.5 * log_var)
```

Decoder identical to AE head structure, fed sampled `z`.

Loss: ELBO with β warmup (see §5).

#### 4.2.3 DEC Head

Initialization protocol:
1. Load pretrained AE checkpoint at the matching latent dim.
2. Compute `Z = encoder(X)` for all 329,044 films.
3. Run `KMeans(n_clusters=k, random_state=42, n_init=20).fit(Z)` → `μ_j` cluster centroids.
4. Initialize DEC head's `μ_j` parameters from those centroids.

Forward pass:
```
z_i = encoder(x_i)
q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2) / Σ_j' (...)   # Student-t kernel, α=1
p_ij = q_ij² / Σ_i q_ij  →  p_ij /= Σ_j p_ij              # sharpened target
loss = KL(P || Q) + λ_recon * weighted_MSE(decoded, x)    # λ_recon = 0.1 default
```

Both encoder and `μ_j` are fine-tuned. **Target distribution `P` is computed batch-wise** as a practical approximation of the full-dataset target — i.e., for each mini-batch, `Q` is computed from the batch's `z` vectors and `P` is derived from `Q` via the standard sharpening formula. This deviates from the original DEC paper's full-dataset `P` (refreshed every T epochs) but is the conventional pragmatic choice for academic reproductions and is mathematically valid as an unbiased estimator under uniform mini-batch sampling. Documented as a deliberate simplification.

`k ∈ {10, 21, 30}` per the ablation grid; with z ∈ {32, 64, 128} this gives 9 DEC runs per main grid.

### 4.3 Hyperparameters (defaults)

```python
TRAINING = {
    'optimizer':    'Adam',
    'lr':           1e-3,
    'weight_decay': 1e-5,
    'batch_size':   512,
    'max_epochs':   100,
    'early_stop_patience': 10,    # on weighted recon loss
    'early_stop_min_delta': 1e-4,
    'gradient_clip_norm':  1.0,
    'seed':         42,
}

VAE_EXTRA = {
    'beta_warmup_epochs': 10,     # β: 0 → 1 linear over first 10 epochs
    'beta_target':        1.0,
}

DEC_EXTRA = {
    'kmeans_n_init':      20,
    'lambda_recon':       0.1,    # weight of reconstruction term
    'finetune_epochs':    50,
    'cluster_size_floor': 0.001,  # if any cluster < 0.1% of data → re-init that center
    # P target distribution is computed batch-wise (see §4.2.3)
}
```

These are starting points; tunable inside each notebook if convergence requires.

---

## 5. Loss Functions

### 5.1 W2 weighted MSE (with clipping)

Per ADR D3 + D9. Computed once on the full training matrix:

```python
def compute_block_weights(X, block_indices, eps=1e-6, w_min=0.1, w_max=10.0):
    weights = {}
    for block, slc in block_indices.items():
        block_var = X[:, slc].var(axis=0).sum()
        w_raw = 1.0 / max(block_var, eps)
        weights[block] = float(np.clip(w_raw, w_min, w_max))
    return weights
```

Clipping prevents low-variance blocks from receiving extreme weights (e.g., a one-hot block in a heavily skewed distribution).

### 5.2 G2 bio masking

Per ADR D4. Applied within the `director` block reconstruction loss.

```python
def director_block_loss(decoded_dir, input_dir, has_bio, w_block):
    # input_dir, decoded_dir: (B, 113); has_bio: (B,) in {0, 1}
    bio_pca_slice = slice(0, 64)
    other_slice = slice(64, 113)
    
    bio_diff = (decoded_dir[:, bio_pca_slice] - input_dir[:, bio_pca_slice]) ** 2
    bio_mask = has_bio.unsqueeze(1)  # (B, 1) broadcasts to (B, 64)
    loss_bio = (bio_diff * bio_mask).sum() / bio_mask.sum().clamp_min(1.0) / 64
    
    other_diff = (decoded_dir[:, other_slice] - input_dir[:, other_slice]) ** 2
    loss_other = other_diff.mean()
    
    return w_block * 0.5 * (loss_bio + loss_other)
```

The 0.5 split balances the two sub-components within the director block.

### 5.2.1 Canonical reconstruction loss (used by AE main, VAE recon term, DEC recon term)

This consolidates the "W2 weighted MSE + G2 bio mask" pattern into a single helper. Director is excluded from the generic sum and its loss is computed via `director_block_loss` to apply the G2 mask. Including director in the sum would double-count.

```python
def weighted_recon_loss(decoded, input, has_bio, w_blocks):
    """W2 weighted MSE on all blocks except director, plus G2 masked director loss.
    
    decoded, input: dict[block_name → Tensor]
    has_bio:         (B,) tensor, 1 if has_director_bio else 0
    w_blocks:        dict[block_name → float] from compute_block_weights(...)
    """
    other = sum(
        w_blocks[b] * F.mse_loss(decoded[b], input[b])
        for b in BLOCKS
        if b != 'director'
    )
    return other + director_block_loss(
        decoded['director'], input['director'], has_bio, w_blocks['director']
    )
```

This is the canonical AE main training loss. VAE adds a KL term to it (§5.5). DEC scales it by `lambda_recon` and adds the cluster KL (§5.6).

### 5.3 W1 ablation (uniform weighting)

Same pattern as the canonical loss but with all block weights set to 1 (the G2 mask still applies — we are isolating the *weighting* effect, not the missing-handling behavior).

```python
def weighted_recon_loss_uniform(decoded, input, has_bio):
    uniform_w = {b: 1.0 for b in BLOCKS}
    return weighted_recon_loss(decoded, input, has_bio, uniform_w)
```

### 5.4 W4 stretch (Kendall et al. 2018 learned uncertainty)

```python
class LearnedWeightedLoss(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_blocks))   # init s=0 → exp(s)=1
    def forward(self, mse_per_block):
        # mse_per_block: dict of scalar tensors, one per block
        loss = 0
        for i, b in enumerate(BLOCKS):
            loss += torch.exp(-self.log_sigma[i]) * mse_per_block[b] + 0.5 * self.log_sigma[i]
        return loss
```

`log_sigma` parameters jointly trained with the network. Adds 7 learned scalars (one per block).

### 5.5 VAE ELBO with β warmup

Per ADR D9. β scheduled linearly:

```python
def beta_schedule(epoch, warmup_epochs=10, beta_target=1.0):
    return min(epoch / warmup_epochs, 1.0) * beta_target

def vae_loss(decoded, input, mu, log_var, has_bio, w_blocks, beta):
    # IMPORTANT: 'director' is excluded from the generic sum — its loss is computed
    # via director_block_loss(...) which applies the G2 bio mask. Including 'director'
    # in BLOCKS would double-count its contribution.
    recon = sum(
        w_blocks[b] * mse_per_block(decoded[b], input[b])
        for b in BLOCKS
        if b != 'director'
    )
    recon = recon + director_block_loss(decoded['director'], input['director'], has_bio, w_blocks['director'])
    kl = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1).mean()
    return recon + beta * kl, recon.item(), kl.item()   # returned tuple for logging
```

### 5.6 DEC KL loss

```python
def dec_loss(z, decoded, input, cluster_centers, has_bio, w_blocks, lambda_recon=0.1):
    # Soft assignments via Student-t kernel
    diff = z.unsqueeze(1) - cluster_centers.unsqueeze(0)   # (B, k, z_dim)
    q = (1 + (diff ** 2).sum(dim=2)) ** -1                   # (B, k); α=1 hard-coded
    q = q / q.sum(dim=1, keepdim=True)
    
    # Sharpened target P (computed without grad, refreshed every T batches)
    f = q.sum(dim=0)
    p = (q ** 2 / f) / ((q ** 2 / f).sum(dim=1, keepdim=True))
    p = p.detach()
    
    kl = (p * (p / q.clamp_min(1e-12)).log()).sum(dim=1).mean()
    
    # Reconstruction term keeps encoder grounded in input space.
    # 'director' is excluded from the generic sum — handled via director_block_loss
    # with G2 bio masking. Including in BLOCKS would double-count.
    recon = sum(
        w_blocks[b] * mse_per_block(decoded[b], input[b])
        for b in BLOCKS
        if b != 'director'
    )
    recon = recon + director_block_loss(decoded['director'], input['director'], has_bio, w_blocks['director'])
    
    return kl + lambda_recon * recon, kl.item(), recon.item()
```

---

## 6. Run Matrix

Per ADR D9. Total 21-22 runs.

| # | Run name | Model | z | k | Loss | Source / init | Compute |
|---:|---|---|---|---|---|---|---|
| 1 | `kmeans_raw_k21` | KMeans | — | 21 | — | sklearn on raw 564 | <1 min |
| 2 | `pca_kmeans_k21` | PCA + KMeans | 64 | 21 | — | sklearn pipeline | <1 min |
| 3 | `vanilla_ae_z64` | concat-AE | 64 | — | W2 (with clipping) | from scratch | ~15 min |
| 4 | `ae_z32` | multi-modal AE | 32 | — | W2 + G2 mask | from scratch | ~15 min |
| 5 | `ae_z64` | multi-modal AE | 64 | — | W2 + G2 mask | from scratch | ~15 min |
| 6 | `ae_z128` | multi-modal AE | 128 | — | W2 + G2 mask | from scratch | ~15 min |
| 7 | `vae_z32` | multi-modal VAE | 32 | — | W2 + G2 + ELBO (β warmup) | from scratch | ~20 min |
| 8 | `vae_z64` | multi-modal VAE | 64 | — | W2 + G2 + ELBO (β warmup) | from scratch | ~20 min |
| 9 | `vae_z128` | multi-modal VAE | 128 | — | W2 + G2 + ELBO (β warmup) | from scratch | ~20 min |
| 10 | `dec_z32_k10` | DEC | 32 | 10 | DEC KL + W2 recon | from `ae_z32.pt` | ~10 min |
| 11 | `dec_z32_k21` | DEC | 32 | 21 | DEC KL + W2 recon | from `ae_z32.pt` | ~10 min |
| 12 | `dec_z32_k30` | DEC | 32 | 30 | DEC KL + W2 recon | from `ae_z32.pt` | ~10 min |
| 13 | `dec_z64_k10` | DEC | 64 | 10 | DEC KL + W2 recon | from `ae_z64.pt` | ~10 min |
| 14 | `dec_z64_k21` | DEC | 64 | 21 | DEC KL + W2 recon | from `ae_z64.pt` | ~10 min |
| 15 | `dec_z64_k30` | DEC | 64 | 30 | DEC KL + W2 recon | from `ae_z64.pt` | ~10 min |
| 16 | `dec_z128_k10` | DEC | 128 | 10 | DEC KL + W2 recon | from `ae_z128.pt` | ~10 min |
| 17 | `dec_z128_k21` | DEC | 128 | 21 | DEC KL + W2 recon | from `ae_z128.pt` | ~10 min |
| 18 | `dec_z128_k30` | DEC | 128 | 30 | DEC KL + W2 recon | from `ae_z128.pt` | ~10 min |
| 19 | `ae_z64_w1` | multi-modal AE | 64 | — | W1 (uniform) + G2 | from scratch | ~15 min |
| 20 | `ae_z64_no_text` | multi-modal AE | 64 | — | W2 + G2, text projected to zero | from scratch | ~15 min |
| 21 | `ae_z64_no_director` | multi-modal AE | 64 | — | W2 + G2, director projected to zero | from scratch | ~15 min |
| 22 | `ae_z64_w4` (optional) | multi-modal AE | 64 | — | W4 (Kendall) + G2 | from scratch | ~20 min |

**Total compute estimate:** ~5-6 hours wall-clock on free Colab T4. Splittable across 2-3 sessions.

**Dependency graph for execution ordering:**
```
[1][2]                       (instant, sklearn)
[3]                          (vanilla AE baseline)
[4][5][6]    AE main         (must precede DEC)
[7][8][9]    VAE main        (independent of AE/DEC, parallelizable)
[10..18]     DEC main        (each loads matching ae_z*.pt)
[19][20][21][22]  ablations  (independent, last)
```

---

## 7. Code Organization

Per ADR D5.

### 7.1 Repository layout

```
deep learning movie project/
├── eda_v2.ipynb                          # untouched
├── data/                                 # untouched (gitignored)
├── artifacts/
│   ├── feature_matrix.npz                # produced by EDA, gitignored
│   ├── feature_metadata.json
│   ├── movies_eda_final.csv
│   ├── pipeline_version.json
│   ├── director_profile_metadata.json
│   ├── models/                           # NEW, gitignored
│   │   ├── ae_z32.pt, ae_z64.pt, ae_z128.pt
│   │   ├── vae_z32.pt, vae_z64.pt, vae_z128.pt
│   │   ├── dec_z{32,64,128}_k{10,21,30}.pt   # 9 files
│   │   ├── vanilla_ae_z64.pt
│   │   ├── ae_z64_w1.pt, ae_z64_no_text.pt, ae_z64_no_director.pt
│   │   └── ae_z64_w4.pt   # optional
│   ├── eval/                             # NEW
│   │   ├── results.json                  # all per-run metrics
│   │   ├── linear_probing.json
│   │   └── ablation_deltas.json
│   └── figures/                          # extends EDA's figures dir
│       ├── modeling_results_table.png
│       ├── latent_umap_<run>_<axis>.png  # 9 main runs × 3 axes = 27 PNGs
│       └── final_comparison.png
├── src/
│   └── cineembed/                        # NEW Python package
│       ├── __init__.py
│       ├── backbone.py                   # MultiModalBackbone
│       ├── heads.py                      # AEHead, VAEHead, DECHead
│       ├── losses.py                     # W2, W1, W4, G2 mask, ELBO, DEC KL
│       ├── data.py                       # load_feature_matrix, get_labels, train_val_split
│       ├── eval.py                       # NMI/ARI/MSE/UMAP/linear_probe helpers
│       └── train.py                      # generic training loop
├── notebooks/                            # NEW directory
│   ├── 01_smoke_test.ipynb
│   ├── 02_train_ae.ipynb
│   ├── 03_train_vae.ipynb
│   ├── 04_train_dec.ipynb
│   └── 05_results.ipynb
├── docs/
│   ├── superpowers/specs/...             # this file lives here
│   └── adr/0001-modeling-hybrid-architecture.md
└── pyproject.toml                        # NEW — declares cineembed package
```

### 7.2 Package public API

```python
# src/cineembed/backbone.py
class MultiModalBackbone(nn.Module):
    def __init__(self, block_dims: dict[str, int], proj_dims: dict[str, int],
                 hidden_dim: int = 128, latent_dim: int = 64): ...
    def forward(self, blocks: dict[str, Tensor]) -> Tensor: ...   # returns (B, z)

# src/cineembed/heads.py
class AEHead(nn.Module):
    def __init__(self, backbone, block_dims, proj_dims, hidden_dim): ...
    def forward(self, blocks): ...   # returns dict of reconstructions

class VAEHead(nn.Module):
    def __init__(self, backbone, block_dims, proj_dims, hidden_dim): ...
    def forward(self, blocks): ...   # returns (decoded_dict, mu, log_var)

class DECHead(nn.Module):
    def __init__(self, backbone, ae_decoder, n_clusters, latent_dim, alpha=1.0): ...
    def initialize_centers(self, X): ...
    def forward(self, blocks): ...   # returns (z, decoded_dict, q)

# src/cineembed/losses.py
def compute_block_weights(X, block_indices, w_min=0.1, w_max=10.0) -> dict: ...
def weighted_recon_loss(decoded, target, has_bio, w_blocks) -> Tensor: ...
def vae_elbo(decoded, target, mu, log_var, has_bio, w_blocks, beta) -> tuple: ...
def dec_loss(z, decoded, target, centers, has_bio, w_blocks, lambda_recon=0.1) -> tuple: ...

# src/cineembed/data.py
def load_feature_matrix(path='artifacts/feature_matrix.npz') -> tuple[Tensor, list[str]]: ...
def get_block_indices(feature_names) -> dict[str, slice]: ...
def get_labels(csv_path='artifacts/movies_eda_final.csv') -> dict[str, ndarray]: ...
def make_dataloader(X, has_bio, batch_size, shuffle=True) -> DataLoader: ...

# src/cineembed/eval.py
def cluster_assignments_kmeans(z, k, seed=42) -> ndarray: ...
def cluster_assignments_dec(model, X) -> ndarray: ...
def evaluate_run(z, cluster_ids, labels_dict) -> dict: ...   # NMI/ARI per axis
def umap_plot(z, labels, title, savepath) -> None: ...
def linear_probe(z, labels, train_idx, val_idx) -> dict: ...

# src/cineembed/train.py
def train_model(model, dataloader, optimizer, loss_fn, n_epochs, ...) -> dict: ...
```

### 7.3 Notebook responsibilities

| Notebook | Owner (suggested) | Outputs | Compute |
|---|---|---|---|
| `01_smoke_test` | Joint | Verify backbone forward pass on dummy data; confirm imports | <1 min |
| `02_train_ae` | Baran | Runs 3-6 + ablations 19-22 (and optional W4) | ~1.5-2 hr |
| `03_train_vae` | Arda | Runs 7-9 | ~1 hr |
| `04_train_dec` | Kaan | Runs 10-18 (loads ae_z*.pt from #02) | ~1.5 hr |
| `05_results` | Joint | Reads `artifacts/eval/results.json`, produces tables and figures | <30 min |

Each training notebook follows the same skeleton: load data → train → save checkpoint → evaluate → append to `results.json`.

### 7.4 Colab setup snippet (top of every training notebook)

The package uses a `src/` layout — `pyproject.toml` lives at the repository root, and modern setuptools auto-discovers `src/cineembed/`. The install path is the **repository root**, NOT the package subdirectory.

```python
!git clone <repo-url> /content/cineembed-repo  # or upload manually
!pip install -e /content/cineembed-repo -q     # install from repo root, not src/cineembed/

from cineembed import data, backbone, heads, losses, eval, train

# Load feature matrix from Drive (mounted) or session uploads
X, feature_names = data.load_feature_matrix('/content/drive/MyDrive/cineembed_artifacts/feature_matrix.npz')
labels = data.get_labels('/content/drive/MyDrive/cineembed_artifacts/movies_eda_final.csv')
```

If `pip install -e` fails due to Colab caching or pyproject.toml issues, fallback:
```python
import sys
sys.path.insert(0, '/content/cineembed-repo/src')
from cineembed import ...   # works because src/ is on path
```

---

## 8. Evaluation Methodology

Per ADR D6 + D7. Three orthogonal axes × NMI + ARI for every run; Tier 2 add-ons for select runs.

### 8.1 Always-on metrics (every run)

For run with cluster assignments `c_i`:
```python
for axis in ['genre', 'decade', 'lang']:
    nmi = normalized_mutual_info_score(labels[axis], c_i)
    ari = adjusted_rand_score(labels[axis], c_i)
    metrics[f'{axis}_nmi'] = nmi
    metrics[f'{axis}_ari'] = ari
```

Also computed per run:
- Per-block reconstruction MSE (numerical, genre, language, decade, awards, text, director)
- Total weighted MSE (W2-weighted)
- Final epoch loss curve (for plotting)
- VAE-only: ELBO components (recon, KL) per epoch
- DEC-only: per-cluster purity vs primary genre, soft-assignment confidence histogram

### 8.2 Latent-space visualization

For all 9 main runs (AE/VAE/DEC × {32, 64, 128}), produce 3 UMAP scatter plots colored by `genre` / `decade` / `lang_top10`. Total: 27 figures saved to `artifacts/figures/latent_umap_<run>_<axis>.png`.

UMAP parameters fixed for reproducibility: `n_neighbors=15`, `min_dist=0.1`, `random_state=42`.

**Reporting strategy:** All 27 figures are produced and committed for completeness (supplementary material). The **final report figure set** uses only the BEST run per model family across the three axes — i.e., 3 runs (best AE / best VAE / best DEC, selected by genre NMI) × 3 axes = **9 main figures** + the 3 baseline UMAPs (KMeans-raw, PCA+KMeans, vanilla concat-AE) at the genre axis only = **12 figures in main report**. The remaining 18+ go to appendix / supplementary section.

### 8.3 Tier 2 add-ons (z=64 main runs only)

#### 8.3.1 Linear probing
Train an `nn.Linear(64 → n_classes)` classifier on frozen `z`:
- Predict `primary_genre` (21 classes) → val accuracy
- Predict `decade_bin` (~13 classes) → val accuracy
- 80/20 train/val split, seed=42, Adam lr=1e-3, 20 epochs

This decouples encoder quality from clustering algorithm quality. A high probing accuracy with low NMI suggests the latent space encodes useful info but KMeans/DEC can't exploit it.

#### 8.3.2 Modality leave-one-out ablation
Two runs at z=64, AE only:
- **F1 (no text)**: zero out the text projection layer's input AT TRAINING TIME (decoder still tries to reconstruct text but has no signal). Measure delta in `genre_nmi`.
- **F2 (no director)**: same for director_profile block.

Implementation: pass a `block_mask` to the model that zeros specific block inputs. The reconstruction loss for the masked block is excluded (so the model isn't penalized for not reconstructing zero).

### 8.4 Results aggregation

After all 21-22 runs complete, `05_results.ipynb` reads `artifacts/eval/results.json` and produces:

1. **Main results table** (~30 cells):
   ```
                z=32 (g/d/l NMI)   z=64 (g/d/l NMI)   z=128 (g/d/l NMI)
   AE           x.xx/x.xx/x.xx     x.xx/x.xx/x.xx     x.xx/x.xx/x.xx
   VAE          x.xx/x.xx/x.xx     x.xx/x.xx/x.xx     x.xx/x.xx/x.xx
   DEC k=10     x.xx/x.xx/x.xx     x.xx/x.xx/x.xx     x.xx/x.xx/x.xx
   DEC k=21     ...
   DEC k=30     ...
   
   Baselines:
   KMeans-raw                       x.xx/x.xx/x.xx
   PCA+KMeans                       x.xx/x.xx/x.xx
   Vanilla concat-AE z=64           x.xx/x.xx/x.xx
   ```

2. **Ablation deltas table:**
   ```
   AE z=64 baseline: NMI(genre)=x.xx
   ├── W1 (uniform)               : Δ=±x.xx
   ├── F1 (no text)               : Δ=±x.xx
   ├── F2 (no director)           : Δ=±x.xx
   └── W4 (Kendall) [stretch]     : Δ=±x.xx     ← highlight if Δ > 0
   ```

3. **Linear probing table:**
   ```
                z=64 genre acc   z=64 decade acc
   AE           x.xx             x.xx
   VAE          x.xx             x.xx
   DEC          x.xx             x.xx
   ```

4. **Final figure** (composite): one panel per axis, each panel a bar chart of best-NMI per model family, with baselines as dashed reference lines.

---

## 9. Success Criteria

Per ADR D9 (peer-review revisions). Relative-improvement framing with absolute floor.

| Criterion | Verification |
|---|---|
| All 21-22 runs complete; checkpoints + results.json exist | `ls artifacts/models/` + `results.json` populated |
| All training losses converge (no divergence/NaN) | Per-run loss curve plateaus, val MSE < 0.5× initial |
| Best deep model NMI(genre) ≥ best baseline NMI(genre) × 1.10 | 10% relative improvement over PCA+KMeans |
| Best deep model NMI(genre) ≥ 0.15 absolute floor | Guards against vacuous "improvement over noise" |
| Multi-modal AE z=64 NMI(genre) > vanilla concat-AE z=64 NMI(genre) × 1.05 | Validates D1 architectural choice |
| W2 main NMI > W1 ablation NMI × 1.05 (at z=64 AE) | Validates D3 weighting |
| F1 ablation Δ measurable (text block contributes) | Confirms text modality is not noise |
| F2 ablation Δ measurable (director block contributes or is shown not to) | Confirms or refutes director modality value |
| Two re-runs of any model produce same final NMI ± 0.02 | Reproducibility (seed=42 throughout) |
| Final report figures generated, results table populated | `05_results.ipynb` runs end-to-end |

If a "Δ measurable" criterion fails (Δ ≈ 0), this is a **finding worth reporting**, not a failure — it means the modality didn't contribute to clustering structure.

---

## 10. Risk and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Multi-modal backbone fails to outperform vanilla concat-AE | Medium | High | Vanilla concat-AE is the explicit ablation; if fails, reframe report as "modality projection adds capacity but no clustering gain on this data" — still publishable |
| VAE KL collapse (μ→0, σ→1, latent uninformative) | Medium | High | β warmup (D9) is standard mitigation; if still fails, add free bits or KL annealing tweak |
| DEC cluster-center degeneration (one cluster swallows >95%) | Medium | Medium | `cluster_size_floor=0.001` re-init protocol in DEC training loop |
| W2 weight clipping bounds (0.1, 10.0) too aggressive or too loose | Low | Medium | Inspect actual `block_weights` values at training start; widen bounds if needed (single CONFIG change) |
| Bio mask gradient issue (`bio_mask.sum()==0` in batch) | Low | High | `clamp_min(1.0)` already in formula; protected |
| Overfitting on 329k samples (model memorizes) | Low | Medium | Dropout + weight_decay default; early stopping on weighted recon |
| Free Colab T4 disconnects mid-run | Medium | Low | Save checkpoint every 5 epochs; resume training capability in `train_model()` |
| Compute budget overflows free T4 daily limit | Medium | Low | Run matrix splittable across 2-3 sessions; results.json appendable |
| All deep models perform worse than baseline | Low | High | If happens, report it honestly — "deep methods do not outperform classical clustering on this feature space"; validates academic honesty |
| UMAP visualizations lack visible cluster structure | Medium | Low | Cosmetic; use t-SNE as fallback (also supported in eval.py); structure shown in NMI numbers regardless |
| F2 ablation Δ ≈ 0 (director block irrelevant) | Medium | Low | Honest finding; explains why bio coverage 3.2% wasn't worth the modality |

---

## 11. Out of Scope

- New EDA features or feature-matrix changes (locked at 564 dims).
- Multi-label NMI implementations (`scikit-multilearn` or hand-rolled).
- Hyperparameter sweeps beyond the explicit grids (z, k, weighting strategy).
- Cross-validation folds (single train pass on full data is methodologically standard for reconstruction-based unsupervised methods).
- Model deployment, REST API, or recommendation system.
- Comparison to non-AE methods beyond the three baselines (e.g., spectral clustering, DBSCAN, HDBSCAN).
- Training on multiple seeds for variance estimation (single seed=42 throughout per the EDA spec convention).
- VAE β-VAE hyperparameter sweep (β=1 fixed, only the warmup schedule varies).

---

## 12. Bridge from EDA, Bridge to Final Report

### 12.1 From EDA

The model loads:
```python
data = np.load('artifacts/feature_matrix.npz')
X = data['X']                     # (329044, 564)
feature_names = data['feature_names']
```

Block boundaries auto-derived from `feature_names` via `data.get_block_indices()`.

The EDA's MD5 (`e99cee84b6891ea352a7b44d5d7d0ee4` per `pipeline_version.json`) is recorded in each run's metadata for reproducibility audit:
```python
results[run_name]['eda_md5'] = pipeline_version['feature_matrix_md5']
```

### 12.2 To final report

The SENG 474 final report draws from:
- `artifacts/eval/results.json` — main results table
- `artifacts/eval/linear_probing.json` — supplementary table
- `artifacts/eval/ablation_deltas.json` — ablation discussion
- `artifacts/figures/modeling_results_table.png` — main figure
- `artifacts/figures/latent_umap_*.png` — supplementary figures
- `artifacts/figures/final_comparison.png` — final comparison
- `docs/adr/0001-modeling-hybrid-architecture.md` — methodology / decision log

Report sections map to spec sections:
- Methodology section: §4 (architecture), §5 (losses), §6 (run matrix)
- Evaluation section: §8 (eval methodology)
- Discussion section: §9 (success criteria results), §10 (which risks materialized)
- Limitations: §11 (out of scope) + ablation findings

---

## 13. Bridge to Implementation Plan

The next document is `docs/superpowers/plans/2026-05-04-modeling-implementation.md`, generated via the writing-plans skill. It will decompose this spec into 15-20 concrete tasks with TDD test cells, exact code, expected outputs, and commit messages — following the same pattern as the EDA implementation plans.

The implementation plan will likely partition into:
1. Package skeleton + `pyproject.toml` setup (Task 1)
2. `data.py` with load + label extraction + block indices (Task 2)
3. `backbone.py` MultiModalBackbone with synthetic test (Task 3)
4. `losses.py` W2 + G2 + ELBO + DEC KL with synthetic tests (Task 4)
5. `heads.py` AEHead + VAEHead + DECHead with smoke tests (Task 5)
6. `train.py` generic training loop with checkpoint/resume (Task 6)
7. `eval.py` NMI/ARI/UMAP/linear_probe helpers (Task 7)
8. `01_smoke_test.ipynb` (Task 8)
9. `02_train_ae.ipynb` (Tasks 9 — runs 3-6, 19-22)
10. `03_train_vae.ipynb` (Task 10 — runs 7-9)
11. `04_train_dec.ipynb` (Task 11 — runs 10-18)
12. `05_results.ipynb` (Task 12)
13. Final integration + reproducibility audit (Task 13)

Each task is self-contained, locally testable (with synthetic small data) where possible, with the heavy training deferred to Colab.
