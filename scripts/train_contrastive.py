"""Phase 1 contrastive pretext training (spec 2026-05-06 §2.1).

Pre-trains MultiModalBackbone via SimCLR-style InfoNCE on stochastic
modality-dropout views. After pretext the backbone is saved (projection
head discarded per Chen et al. 2020), then encodes the full dataset and
reports per-axis NMI / ARI / AMI versus the MVP baseline. Everything is
logged into a SINGLE W&B run (training curves + final eval + backbone
artifact + UMAP image).

Usage (local):
    python scripts/train_contrastive.py \
        --artifacts artifacts \
        --tau 0.1 --drop-prob 0.3 --proj-dim 128 --batch-size 1024 \
        --epochs 30 --run-name contrastive_pretext_tau0p1

Usage (Colab):
    !python /content/cineembed-repo/scripts/train_contrastive.py \
        --artifacts /content/drive/MyDrive/CineEmbed/artifacts \
        --device cuda --epochs 60 \
        --wandb-project cineembed --wandb-group phase-1-sweep

Outputs (under <artifacts>/models/<run-name>/):
    pretext_backbone.pt    # backbone state_dict, ready for AE/DEC fine-tune
    pretext_full.pt        # backbone + projection state_dict (for resume)
    history.json           # train/val loss curves
    eval.json              # post-pretext NMI/ARI/AMI on KMeans + GMM
    umap.png               # (optional, --umap) UMAP plot colored by genre
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from cineembed import data as cdata
from cineembed.backbone import MultiModalBackbone
from cineembed.heads import ContrastiveHead
from cineembed.losses import info_nce_loss
from cineembed.train import train_model
from cineembed.eval import (
    cluster_assignments_kmeans,
    cluster_assignments_gmm,
    evaluate_run,
    evaluate_run_per_axis_k,
    multilabel_macro_nmi,
    umap_plot,
)


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Contrastive pretext training.')
    p.add_argument('--artifacts', type=Path, default=Path('artifacts'),
                   help='Directory with feature_matrix.npz + movies_eda_final.csv.')
    p.add_argument('--run-name', type=str, default='contrastive_pretext',
                   help='Output subdir + wandb run name.')

    # InfoNCE / augmentation hyperparams (spec §2.1)
    p.add_argument('--tau', type=float, default=0.1, help='InfoNCE temperature.')
    p.add_argument('--drop-prob', type=float, default=0.3,
                   help='Per-block dropout probability for each view.')
    p.add_argument('--proj-dim', type=int, default=128, help='Projection MLP output dim.')

    # Optim
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=8, help='Early-stop patience.')
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--hidden-dim', type=int, default=128)

    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-frac', type=float, default=0.1)

    # wandb
    p.add_argument('--wandb-project', type=str, default=None,
                   help='If set, log to this W&B project. Else offline-safe no-op.')
    p.add_argument('--wandb-entity', type=str, default=None,
                   help='W&B team/user. None = default from netrc.')
    p.add_argument('--wandb-group', type=str, default=None,
                   help='W&B run group (e.g. "phase-1-sweep") to compare runs.')
    p.add_argument('--wandb-tags', type=str, nargs='*', default=['contrastive', 'phase-1'])
    p.add_argument('--wandb-mode', type=str, default=None,
                   choices=['online', 'offline', 'disabled', 'shared'],
                   help='Override W&B mode. None = read WANDB_MODE env or default online.')

    # Diagnostics
    p.add_argument('--umap', action='store_true',
                   help='After eval, render a UMAP plot and log it to W&B.')

    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def _build_contrastive_loss(tau: float):
    """Closure with train_model's (model, batch) signature."""
    def _loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
        view_a = batch['view_a']
        view_b = batch['view_b']
        z_a = model(view_a['blocks'], block_mask=view_a['block_mask'])
        z_b = model(view_b['blocks'], block_mask=view_b['block_mask'])
        return info_nce_loss(z_a, z_b, temperature=tau)
    return _loss


@torch.no_grad()
def _encode_all(model: ContrastiveHead, X: torch.Tensor, has_bio: torch.Tensor,
                block_slices: dict, batch_size: int, device: str) -> np.ndarray:
    """Encode every row through the BACKBONE only (projection discarded)."""
    model.eval()
    loader = cdata.make_dataloader(
        X, has_bio, batch_size=batch_size, shuffle=False,
        block_slices=block_slices, seed=None,
    )
    chunks = []
    for batch in loader:
        blocks_dev = {b: t.to(device) for b, t in batch['blocks'].items()}
        z = model.backbone(blocks_dev)
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _do_eval(model, X, has_bio, block_slices, batch_size, device, artifacts, seed):
    """Encode + run all clustering metrics. Returns a nested dict and the
    latent matrix z (kept for optional UMAP)."""
    print('[eval] encoding all rows with backbone...')
    z = _encode_all(model, X, has_bio, block_slices, batch_size, device)
    print(f'[eval] z={z.shape}  computing clustering metrics...')

    labels = cdata.get_labels(artifacts / 'movies_eda_final.csv')
    kmeans_assign = cluster_assignments_kmeans(z, k=21, seed=seed)
    gmm_assign = cluster_assignments_gmm(z, k=21, seed=seed)

    eval_out = {
        'kmeans_k21':        evaluate_run(kmeans_assign, labels),
        'gmm_k21':           evaluate_run(gmm_assign, labels),
        'per_axis_k_kmeans': evaluate_run_per_axis_k(z, labels, seed=seed),
    }

    # Multi-label macro-NMI on the genre block columns (excluding has_genre flag).
    g = block_slices['genre']
    genre_onehot = X[:, g.start:g.start + 21].numpy().astype(np.float32)
    eval_out['multilabel_genre_macro_nmi_kmeans'] = multilabel_macro_nmi(
        kmeans_assign, genre_onehot, metric='nmi',
    )
    return eval_out, z, labels


def _print_eval_summary(eval_out: dict) -> None:
    km = eval_out['kmeans_k21']
    gm = eval_out['gmm_k21']
    pa = eval_out.get('per_axis_k_kmeans', {})
    print('=' * 60)
    print(f"[eval] KMeans  k=21    g={km['genre_nmi']:.3f}  d={km['decade_nmi']:.3f}  l={km['lang_nmi']:.3f}")
    print(f"[eval] GMM     k=21    g={gm['genre_nmi']:.3f}  d={gm['decade_nmi']:.3f}  l={gm['lang_nmi']:.3f}")
    if pa:
        print(f"[eval] per-axis-k       g={pa.get('genre_nmi_k21',0):.3f}  "
              f"d={pa.get('decade_nmi_k12',0):.3f}  l={pa.get('lang_nmi_k11',0):.3f}")
    print(f"[eval] MVP baseline     g=0.332 (dec_z64_k21 genre_nmi)")
    print('=' * 60)


def _push_eval_to_wandb(run, eval_out, out_dir, run_name):
    """Push every metric block to the same run with stable key prefixes, plus
    the backbone checkpoint as a versioned artifact."""
    from cineembed.wandb_integration import log_eval, log_artifact

    # log_eval auto-computes geo_nmi when (genre, decade, lang) NMI keys present.
    log_eval(run, eval_out['kmeans_k21'],         prefix='km_')
    log_eval(run, eval_out['gmm_k21'],            prefix='gmm_', add_geo_nmi=False)

    pa = eval_out.get('per_axis_k_kmeans', {})
    if pa:
        log_eval(run, pa, prefix='axis_', add_geo_nmi=False)

    ml = eval_out.get('multilabel_genre_macro_nmi_kmeans', {})
    if isinstance(ml, dict) and 'macro_nmi' in ml:
        run.log({'km_multilabel_macro_nmi': float(ml['macro_nmi'])})

    # Headline number (geo NMI) is logged by log_eval above with prefix km_;
    # also write to run.summary so it surfaces on the project's run-list.
    km = eval_out['kmeans_k21']
    run.summary['headline_km_genre_nmi'] = float(km['genre_nmi'])
    if all(k in km for k in ('genre_nmi', 'decade_nmi', 'lang_nmi')):
        prod = max(km['genre_nmi'] * km['decade_nmi'] * km['lang_nmi'], 0.0)
        run.summary['headline_km_geo_nmi'] = prod ** (1/3) if prod > 0 else 0.0

    # Version the backbone checkpoint as an artifact so downstream
    # AE/DEC fine-tune notebooks can `wandb.use_artifact(...)`.
    backbone_path = out_dir / 'pretext_backbone.pt'
    if backbone_path.exists():
        log_artifact(run, backbone_path,
                     name=f'pretext_backbone__{run_name}',
                     type='model',
                     description='Backbone-only state_dict after contrastive pretext.')


def main():
    args = _build_args()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)

    artifacts = args.artifacts.resolve()
    out_dir = artifacts / 'models' / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] artifacts={artifacts}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device}")
    print(f"[setup] tau={args.tau}  drop_prob={args.drop_prob}  proj_dim={args.proj_dim}")
    if args.wandb_project:
        print(f"[setup] wandb project={args.wandb_project}  group={args.wandb_group}")

    # ── data ────────────────────────────────────────────────────────────────
    X, feature_names = cdata.load_feature_matrix(artifacts / 'feature_matrix.npz')
    block_slices = cdata.get_block_indices(feature_names)
    has_bio = X[:, block_slices['director'].start + 64].clone()
    block_dims = {b: (slc.stop - slc.start) for b, slc in block_slices.items()}
    print(f"[data] X={tuple(X.shape)}  has_bio_sum={int(has_bio.sum())}")

    train_idx, val_idx = cdata.train_val_split(len(X), val_frac=args.val_frac, seed=args.seed)
    train_loader = cdata.make_contrastive_dataloader(
        X, has_bio, batch_size=args.batch_size,
        block_slices=block_slices, drop_prob=args.drop_prob,
        indices=train_idx, seed=args.seed,
    )
    val_loader = cdata.make_contrastive_dataloader(
        X, has_bio, batch_size=args.batch_size,
        block_slices=block_slices, drop_prob=args.drop_prob,
        indices=val_idx, seed=args.seed + 1,
    )

    # ── model ───────────────────────────────────────────────────────────────
    backbone = MultiModalBackbone(
        block_dims=block_dims,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    model = ContrastiveHead(backbone=backbone, projection_dim=args.proj_dim)
    loss_fn = _build_contrastive_loss(args.tau)

    # ── W&B context (no-op if --wandb-project not set) ──────────────────────
    if args.wandb_project:
        from cineembed.wandb_integration import wandb_run as _wandb_run
        wandb_ctx = _wandb_run(
            config={
                'phase': 'contrastive-pretext',
                'tau': args.tau, 'drop_prob': args.drop_prob,
                'proj_dim': args.proj_dim, 'latent_dim': args.latent_dim,
                'hidden_dim': args.hidden_dim, 'batch_size': args.batch_size,
                'epochs': args.epochs, 'lr': args.lr, 'weight_decay': args.weight_decay,
                'val_frac': args.val_frac, 'seed': args.seed,
            },
            run_name=args.run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=list(args.wandb_tags),
            mode=args.wandb_mode,
            notes='Spec 2026-05-06 §2.1 contrastive pretext (per-row block-mask, SimCLR-style).',
        )
    else:
        from contextlib import nullcontext
        wandb_ctx = nullcontext(None)

    with wandb_ctx as run:
        # ── train (logs per-epoch loss into `run`) ──────────────────────────
        history = train_model(
            model=model, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            early_stop_patience=args.patience,
            checkpoint_path=out_dir / 'pretext_full.pt',
            device=device, seed=args.seed, wandb_run=run,
        )

        # Backbone-only checkpoint is what AE/DEC fine-tune loads.
        torch.save(model.backbone.state_dict(), out_dir / 'pretext_backbone.pt')
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[ckpt] saved backbone -> {out_dir / 'pretext_backbone.pt'}")

        # ── eval (still inside the wandb context so metrics share the run) ──
        eval_out, z, labels = _do_eval(
            model, X, has_bio, block_slices, args.batch_size, device,
            artifacts, args.seed,
        )
        with open(out_dir / 'eval.json', 'w') as f:
            json.dump(eval_out, f, indent=2)
        _print_eval_summary(eval_out)

        if run is not None:
            _push_eval_to_wandb(run, eval_out, out_dir, args.run_name)

        # ── optional UMAP visual ────────────────────────────────────────────
        if args.umap:
            umap_path = out_dir / 'umap.png'
            try:
                print('[eval] rendering UMAP (this is slow)...')
                umap_plot(z, labels['primary_genre'],
                          title=f'Contrastive pretext: {args.run_name}',
                          savepath=str(umap_path))
                if run is not None:
                    from cineembed.wandb_integration import log_image
                    log_image(run, str(umap_path), key='umap_genre')
                print(f'[eval] umap -> {umap_path}')
            except Exception as e:
                print(f'[eval] UMAP skipped: {e!r}')

        if run is not None:
            print(f"[wandb] run URL: {run.url}")


if __name__ == '__main__':
    main()
