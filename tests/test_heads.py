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


def test_contrastive_head_output_shape_and_projection_dim(fresh_backbone, synthetic_blocks_dict):
    """ContrastiveHead.forward returns (B, projection_dim) — not the latent dim."""
    head = heads.ContrastiveHead(backbone=fresh_backbone, projection_dim=128)
    out = head(synthetic_blocks_dict)
    assert out.shape == (200, 128)


def test_contrastive_head_block_mask_zeroes_modality(fresh_backbone, synthetic_blocks_dict):
    """When block_mask sets a modality to 0, the encoded view must differ from
    the unmasked one — confirms mask is propagated through to the backbone."""
    head = heads.ContrastiveHead(backbone=fresh_backbone, projection_dim=64)
    head.eval()
    with torch.no_grad():
        out_full = head(synthetic_blocks_dict)
        mask = {'text': 0.0}  # drop text block
        out_masked = head(synthetic_blocks_dict, block_mask=mask)
    # Outputs must differ — text is the largest modality, masking it changes the latent.
    assert not torch.allclose(out_full, out_masked, atol=1e-3)


def test_contrastive_head_encode_returns_latent_not_projection(fresh_backbone, synthetic_blocks_dict):
    """The `encode` method returns the backbone latent (z_dim), not the projection."""
    head = heads.ContrastiveHead(backbone=fresh_backbone, projection_dim=128)
    head.eval()
    with torch.no_grad():
        z = head.encode(synthetic_blocks_dict)
    assert z.shape == (200, 64)  # backbone latent_dim, not projection_dim


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
