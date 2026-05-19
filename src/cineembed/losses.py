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
    # Tensor-typed accumulator (avoids `sum(...)` returning Literal[0] on empty generator)
    device = next(iter(decoded.values())).device
    other = torch.zeros((), device=device)
    for b in target:
        if b == 'director' or b in skip:
            continue
        other = other + w_blocks[b] * F.mse_loss(decoded[b], target[b])
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


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE over two views (spec §2.1, Chen et al. 2020 SimCLR).

    Treats each row as a positive pair (z_a[i], z_b[i]); all other pairs in the
    batch are negatives. Symmetric in the sense that the loss is computed
    a→b and b→a and averaged.

    Args:
        z_a, z_b: (B, d) — projected representations of two augmented views
                  of the same B input rows. Inputs are L2-normalized internally.
        temperature: NT-Xent temperature. Default 0.1 (lower than SimCLR's 0.5
                     for natural images). Lower temperatures sharpen the
                     contrastive signal — appropriate for heterogeneous tabular
                     data where the latent geometry is denser and modality
                     dropout produces less radical view differences than image
                     augmentations. Sweep {0.1, 0.5} in Phase 1 (spec §2.1).

    Returns:
        scalar loss.
    """
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    # Concatenate views: rows 0..B-1 are view-a, B..2B-1 are view-b.
    # For row i in view-a, the positive is row i in view-b (logical index B+i).
    z = torch.cat([z_a, z_b], dim=0)               # (2B, d)
    sim = z @ z.t() / temperature                  # (2B, 2B)

    B = z_a.shape[0]
    # Mask out self-similarity along the diagonal.
    diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, float('-inf'))

    # For row i in [0, B), the positive index is B + i (its view-b counterpart).
    # For row B + i in [B, 2B), the positive index is i (its view-a counterpart).
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)

    return F.cross_entropy(sim, targets)


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
        # Tensor-typed accumulator (avoids `loss = 0.0` causing Tensor|float union)
        device = next(iter(decoded.values())).device
        loss = torch.zeros((), device=device)
        for i, b in enumerate(self.block_names):
            if b == 'director':
                # director uses G2 masked loss; learned weight scales the whole director loss
                block_loss = director_block_loss(decoded[b], target[b], has_bio, w_block=1.0)
            else:
                block_loss = F.mse_loss(decoded[b], target[b])
            s = self.log_sigma[i]
            loss = loss + torch.exp(-s) * block_loss + 0.5 * s
        return loss
