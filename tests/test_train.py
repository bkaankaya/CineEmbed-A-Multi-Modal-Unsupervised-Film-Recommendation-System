import torch

from cineembed import data, heads, backbone, losses, train


PROJ_DIMS = {'numerical': 16, 'genre': 16, 'language': 16, 'decade': 4,
             'awards': 16, 'text': 64, 'director': 32}
BLOCK_DIMS = {'numerical': 6, 'genre': 22, 'language': 31, 'decade': 2,
              'awards': 6, 'text': 384, 'director': 113}


def _make_loaders(synthetic_feature_matrix, block_slices):
    X = torch.from_numpy(synthetic_feature_matrix)
    has_bio = X[:, block_slices['director'].start + 64]
    train_idx, val_idx = data.train_val_split(X.shape[0], val_frac=0.1, seed=42)
    train_loader = data.make_dataloader(X, has_bio, batch_size=32, shuffle=True,
                                         indices=train_idx, block_slices=block_slices, seed=42)
    val_loader = data.make_dataloader(X, has_bio, batch_size=32, shuffle=False,
                                       indices=val_idx, block_slices=block_slices, seed=42)
    return train_loader, val_loader


def test_train_ae_loss_decreases(synthetic_feature_matrix, block_slices):
    bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=64, latent_dim=32)
    head = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=64)
    train_loader, val_loader = _make_loaders(synthetic_feature_matrix, block_slices)
    w_blocks = losses.compute_block_weights(synthetic_feature_matrix,
                                             {b: block_slices[b] for b in BLOCK_DIMS})

    def loss_fn(model, batch):
        decoded = model(batch['blocks'])
        return losses.weighted_recon_loss(decoded, batch['blocks'], batch['has_bio'], w_blocks)

    history = train.train_model(
        model=head, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=3, lr=1e-3, early_stop_patience=10, device='cpu',
    )
    assert 'train_loss' in history and 'val_loss' in history
    assert history['train_loss'][-1] < history['train_loss'][0], \
        f"train loss did not decrease: {history['train_loss']}"


def test_checkpoint_save_and_resume(tmp_path, synthetic_feature_matrix, block_slices):
    bb = backbone.MultiModalBackbone(BLOCK_DIMS, PROJ_DIMS, hidden_dim=64, latent_dim=32)
    head = heads.AEHead(bb, BLOCK_DIMS, PROJ_DIMS, hidden_dim=64)
    train_loader, val_loader = _make_loaders(synthetic_feature_matrix, block_slices)
    w_blocks = losses.compute_block_weights(synthetic_feature_matrix,
                                             {b: block_slices[b] for b in BLOCK_DIMS})

    def loss_fn(model, batch):
        decoded = model(batch['blocks'])
        return losses.weighted_recon_loss(decoded, batch['blocks'], batch['has_bio'], w_blocks)

    ckpt = tmp_path / "ae.pt"
    train.train_model(
        model=head, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=2, lr=1e-3, device='cpu', checkpoint_path=ckpt,
    )
    assert ckpt.exists()
    state = torch.load(ckpt, weights_only=False)
    assert 'model_state' in state and 'epoch' in state and 'val_loss' in state
