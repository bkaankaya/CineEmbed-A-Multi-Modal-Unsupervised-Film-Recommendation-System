import numpy as np
import pytest
import torch

from cineembed import losses


@pytest.fixture
def synthetic_decoded_input():
    """Decoded ≈ input + noise; per-block dicts matching block dims."""
    torch.manual_seed(42)
    n = 16
    block_dims = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
                  'awards': 6, 'text': 384, 'director': 113}
    inp = {b: torch.randn(n, d) for b, d in block_dims.items()}
    dec = {b: inp[b] + 0.1 * torch.randn_like(inp[b]) for b in inp}
    has_bio = torch.tensor([1.0] * 4 + [0.0] * 12)
    return dec, inp, has_bio


def test_compute_block_weights_clipping():
    """W2 weights should be clipped to [0.1, 10.0]."""
    block_indices = {'a': slice(0, 2), 'b': slice(2, 5)}
    # Block 'a' has near-zero variance → weight would be huge → clipped to 10.0
    # Block 'b' has high variance → weight tiny → clipped to 0.1
    X = np.zeros((100, 5), dtype=np.float32)
    X[:, 0] = 0.001  # near-zero variance
    X[:, 1] = 0.001
    X[:, 2:] = np.random.randn(100, 3) * 100  # high variance

    w = losses.compute_block_weights(X, block_indices, w_min=0.1, w_max=10.0)
    assert w['a'] == 10.0
    assert w['b'] == 0.1


def test_director_block_loss_masks_bio_when_no_bio():
    """When has_bio is all-zero, the bio_pca half contributes 0 (mask sums to 0
    → clamp_min(1) prevents NaN; numerator already 0)."""
    dec_dir = torch.randn(8, 113)
    inp_dir = torch.zeros(8, 113)
    has_bio = torch.zeros(8)
    loss = losses.director_block_loss(dec_dir, inp_dir, has_bio, w_block=1.0)
    # Only the non-bio cols (64..113) contribute. Bio cols (0..64) masked → no contribution.
    expected_other = ((dec_dir[:, 64:] - inp_dir[:, 64:]) ** 2).mean()
    expected_total = 0.5 * (0.0 + expected_other)
    assert torch.allclose(loss, torch.tensor(float(expected_total)), atol=1e-5)


def test_director_block_loss_uses_bio_when_all_present():
    """When has_bio is all-one, both halves contribute equally weighted."""
    torch.manual_seed(0)
    dec_dir = torch.randn(8, 113)
    inp_dir = torch.zeros(8, 113)
    has_bio = torch.ones(8)
    loss = losses.director_block_loss(dec_dir, inp_dir, has_bio, w_block=1.0)
    bio_diff = (dec_dir[:, :64] - inp_dir[:, :64]) ** 2
    bio_loss = bio_diff.mean()  # mean over (8, 64) when mask all-one
    other_loss = ((dec_dir[:, 64:] - inp_dir[:, 64:]) ** 2).mean()
    expected = 0.5 * (bio_loss + other_loss)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_weighted_recon_loss_no_double_count(synthetic_decoded_input):
    """Director must NOT be summed twice (regression test for the D10 bug)."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks = {b: 1.0 for b in inp}

    loss_full = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks)

    # Manual computation matching the spec §5.2.1 contract.
    other = sum(((dec[b] - inp[b]) ** 2).mean() for b in inp if b != 'director')
    dir_loss = losses.director_block_loss(dec['director'], inp['director'], has_bio, w_blocks['director'])
    expected = other + dir_loss
    assert torch.allclose(loss_full, expected, atol=1e-5)


def test_weighted_recon_loss_uniform_equals_uniform_weights(synthetic_decoded_input):
    """W1 baseline = canonical loss with all weights = 1."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks_one = {b: 1.0 for b in inp}
    expected = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks_one)
    actual = losses.weighted_recon_loss_uniform(dec, inp, has_bio)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_vae_elbo_returns_recon_kl_separately(synthetic_decoded_input):
    """ELBO returns (loss, recon_value, kl_value) — used for logging."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks = {b: 1.0 for b in inp}
    n = inp['numerical'].shape[0]
    mu = torch.randn(n, 16)
    log_var = torch.zeros(n, 16)  # σ=1 → KL has only mu^2 contribution
    loss, recon_val, kl_val = losses.vae_elbo(dec, inp, mu, log_var, has_bio, w_blocks, beta=0.5)
    expected_kl = 0.5 * (mu ** 2).sum(dim=1).mean()
    assert abs(kl_val - expected_kl.item()) < 1e-4
    # loss = recon + 0.5 * kl
    assert abs(loss.item() - (recon_val + 0.5 * kl_val)) < 1e-4


def test_weighted_recon_loss_exclude_blocks(synthetic_decoded_input):
    """F1/F2 ablation: exclude_blocks must skip the named block from the sum."""
    dec, inp, has_bio = synthetic_decoded_input
    w_blocks = {b: 1.0 for b in inp}

    full = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks)
    no_text = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks, exclude_blocks={'text'})
    expected_diff = w_blocks['text'] * torch.nn.functional.mse_loss(dec['text'], inp['text'])
    assert torch.allclose(full - no_text, expected_diff, atol=1e-5)

    # Excluding 'director' must skip the G2 helper too
    no_dir = losses.weighted_recon_loss(dec, inp, has_bio, w_blocks, exclude_blocks={'director'})
    expected_dir = losses.director_block_loss(dec['director'], inp['director'], has_bio, w_blocks['director'])
    assert torch.allclose(full - no_dir, expected_dir, atol=1e-5)


def test_info_nce_loss_identical_views_lower_than_random():
    """Identical-views loss must be strictly lower than independent-views loss.

    Note: identical views does NOT drive the loss to 0 — within each view, rows
    are still independent random Gaussians, so the *negative* pairs (other rows
    within the batch) contribute non-zero similarity terms. The correct
    correctness criterion is *relative*: aligning the positives must lower the
    loss compared to random (z_a, z_b) pairs.
    """
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    loss_aligned = losses.info_nce_loss(z, z.clone(), temperature=0.5)

    z_a = torch.randn(32, 16)
    z_b = torch.randn(32, 16)
    loss_random = losses.info_nce_loss(z_a, z_b, temperature=0.5)

    assert loss_aligned.item() < loss_random.item(), \
        f"aligned loss {loss_aligned.item():.3f} should be < random loss {loss_random.item():.3f}"


def test_info_nce_loss_low_temperature_drives_aligned_to_zero():
    """At very small temperature, perfectly aligned positives dominate
    exponentially and the loss collapses toward 0."""
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    loss_lowT = losses.info_nce_loss(z, z.clone(), temperature=0.01)
    # exp(100) so dominant that loss ≈ 0
    assert loss_lowT.item() < 0.05


def test_info_nce_loss_random_views_in_expected_range():
    """Independent random views → loss in the [1, log(2B-1)+1] band.

    Theoretical baseline for random partitions is ~log(2B-1); for B=32 that's ~4.14.
    """
    torch.manual_seed(0)
    B = 32
    z_a = torch.randn(B, 16)
    z_b = torch.randn(B, 16)
    loss = losses.info_nce_loss(z_a, z_b, temperature=0.5)
    assert 1.0 < loss.item() < 6.0  # generous bracket around log(2B-1) ≈ 4.14


def test_info_nce_loss_is_symmetric():
    """Swapping (z_a, z_b) must produce the same loss (symmetric formulation)."""
    torch.manual_seed(0)
    z_a = torch.randn(16, 8)
    z_b = torch.randn(16, 8)
    loss_ab = losses.info_nce_loss(z_a, z_b, temperature=0.5)
    loss_ba = losses.info_nce_loss(z_b, z_a, temperature=0.5)
    assert torch.allclose(loss_ab, loss_ba, atol=1e-5)


def test_info_nce_loss_backward_grads_flow():
    """Gradients must propagate through both views."""
    torch.manual_seed(0)
    z_a = torch.randn(8, 4, requires_grad=True)
    z_b = torch.randn(8, 4, requires_grad=True)
    loss = losses.info_nce_loss(z_a, z_b, temperature=0.5)
    loss.backward()
    assert z_a.grad is not None and z_a.grad.abs().sum() > 0
    assert z_b.grad is not None and z_b.grad.abs().sum() > 0


def test_dec_loss_runs_and_returns_components():
    """DEC loss returns (loss, kl_val, recon_val) and computes batch-wise P/Q."""
    torch.manual_seed(0)
    n, k, z_dim = 16, 5, 8
    z = torch.randn(n, z_dim)
    centers = torch.randn(k, z_dim)
    block_dims = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
                  'awards': 6, 'text': 384, 'director': 113}
    inp = {b: torch.randn(n, d) for b, d in block_dims.items()}
    dec = {b: inp[b] + 0.05 * torch.randn_like(inp[b]) for b in inp}
    has_bio = torch.zeros(n)
    w_blocks = {b: 1.0 for b in inp}

    loss, kl_val, recon_val = losses.dec_loss(z, dec, inp, centers, has_bio, w_blocks, lambda_recon=0.1)
    assert loss.requires_grad or loss.dim() == 0  # scalar
    assert kl_val >= 0
    assert recon_val >= 0
