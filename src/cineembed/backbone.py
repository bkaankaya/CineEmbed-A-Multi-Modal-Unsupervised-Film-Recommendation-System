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
        block_mask: dict[str, torch.Tensor | float] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            blocks: per-block input tensors.
            block_mask: optional dict {block_name: mask}. Mask can be a float (0.0 or 1.0)
                for per-batch masking, or a Tensor of shape (B, 1) for per-row masking.
                A value of 0.0 zeros that block's projected output before concatenation.
                Missing keys default to 1.0 (kept).
        """
        projected = []
        for b in self.block_order:
            p = self.projections[b](blocks[b])
            if block_mask is not None:
                m = block_mask.get(b, 1.0)
                if isinstance(m, torch.Tensor):
                    # Per-row masking: m has shape (B, 1) or (B, projection_dim)
                    # Broadcast multiplication zero out rows where m=0
                    p = p * m
                elif m == 0.0:
                    # Per-batch masking: zero out entire block
                    p = torch.zeros_like(p)
            projected.append(p)
        h = torch.cat(projected, dim=1)
        return self.backbone(h)
