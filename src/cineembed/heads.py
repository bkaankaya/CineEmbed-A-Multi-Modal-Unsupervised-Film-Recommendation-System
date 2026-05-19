"""AE / VAE / DEC heads (spec §4.2)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .backbone import MultiModalBackbone


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


class ContrastiveHead(nn.Module):
    """SimCLR-style projection head for self-supervised pretext training.

    See `docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`
    §2.1. Wraps the shared `MultiModalBackbone` with a 2-layer MLP that maps
    the latent z into a projection space used ONLY for the InfoNCE loss. The
    projection is discarded after pretext (Chen et al. 2020); downstream
    AE/DEC heads operate directly on backbone latent z.

    The forward signature accepts an optional `block_mask` so callers can pass
    different stochastic masks for each augmented view (the augmentation
    primitive for contrastive learning on heterogeneous data).
    """
    def __init__(
        self,
        backbone: MultiModalBackbone,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        z_dim = backbone.latent_dim
        # 2-layer MLP: z → z → projection_dim, BN+ReLU between (SimCLR paper).
        self.projection = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, projection_dim),
        )
        self.projection_dim = projection_dim

    def encode(self, blocks: dict[str, torch.Tensor],
               block_mask: dict[str, float] | None = None) -> torch.Tensor:
        return self.backbone(blocks, block_mask=block_mask)

    def forward(self, blocks: dict[str, torch.Tensor],
                block_mask: dict[str, float] | None = None) -> torch.Tensor:
        z = self.encode(blocks, block_mask=block_mask)
        return self.projection(z)


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
        # sklearn 1.4+ stub annotates n_init as `'auto' | int`; some sklearn-stubs
        # versions have a stale `str`-only annotation. Runtime accepts int 20 fine.
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=seed)  # type: ignore[arg-type]
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
