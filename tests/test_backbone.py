import torch

from cineembed import backbone


PROJ_DIMS = {
    'numerical': 16, 'genre': 16, 'language': 16, 'decade': 4,
    'awards': 16, 'text': 64, 'director': 32,
}
BLOCK_DIMS = {
    'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
    'awards': 6, 'text': 384, 'director': 113,
}


def test_backbone_output_shape(synthetic_blocks_dict):
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z = model(synthetic_blocks_dict)
    assert z.shape == (200, 64)


def test_backbone_deterministic_with_seed(synthetic_blocks_dict):
    torch.manual_seed(42)
    m1 = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z1 = m1(synthetic_blocks_dict)

    torch.manual_seed(42)
    m2 = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    z2 = m2(synthetic_blocks_dict)
    assert torch.allclose(z1, z2)


def test_backbone_param_count_under_500k():
    """Backbone size sanity — too big = wrong design."""
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 10_000 < n_params < 500_000, f"got {n_params}"


def test_backbone_supports_different_latent_dims(synthetic_blocks_dict):
    for z_dim in [32, 64, 128]:
        model = backbone.MultiModalBackbone(
            block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=z_dim,
        )
        z = model(synthetic_blocks_dict)
        assert z.shape == (200, z_dim)


def test_backbone_per_batch_block_mask(synthetic_blocks_dict):
    """F1/F2 ablation: block_mask={'text': 0.0} → entire text block zeroed for batch."""
    torch.manual_seed(42)
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    model.eval()
    z_full = model(synthetic_blocks_dict)
    
    mask_no_text = {b: 1.0 for b in BLOCK_DIMS}
    mask_no_text['text'] = 0.0
    z_no_text = model(synthetic_blocks_dict, block_mask=mask_no_text)
    
    assert not torch.allclose(z_full, z_no_text), "Entire batch should change"


def test_backbone_per_row_block_mask(synthetic_blocks_dict):
    """Ensure that (B, 1) tensor masks correctly zero out individual rows per block."""
    torch.manual_seed(42)
    model = backbone.MultiModalBackbone(
        block_dims=BLOCK_DIMS, proj_dims=PROJ_DIMS, hidden_dim=128, latent_dim=64,
    )
    model.eval()  # Disable Dropout for deterministic comparison
    B = 200
    # Create a mask that zeros out the 'text' block only for the first row
    text_mask = torch.ones((B, 1))
    text_mask[0, 0] = 0.0
    
    mask = {b: 1.0 for b in BLOCK_DIMS}
    mask['text'] = text_mask
    
    z_full = model(synthetic_blocks_dict)
    z_masked = model(synthetic_blocks_dict, block_mask=mask)
    
    assert not torch.allclose(z_full[0], z_masked[0]), "Row 0 should be different"
    assert torch.allclose(z_full[1:], z_masked[1:]), "Other rows should be identical"
