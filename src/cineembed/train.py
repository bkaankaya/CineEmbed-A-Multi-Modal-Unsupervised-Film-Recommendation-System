"""Generic training loop with early stopping + checkpoint save/resume (spec §4.3)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def train_model(
    *,
    model: nn.Module,
    loss_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    device: str = 'cuda',
    checkpoint_path: str | Path | None = None,
    extra_params: list[nn.Parameter] | None = None,
    seed: int = 42,
    wandb_run: Any | None = None,
) -> dict:
    """Generic training loop.

    Args:
        loss_fn: callable (model, batch, epoch) → scalar tensor or tuple. Batch dict
                 has keys 'blocks' (per-block tensors) and 'has_bio'. The `epoch`
                 argument enables schedules like VAE β warmup. For backward
                 compatibility, the signature is auto-detected: if the function
                 accepts only (model, batch), epoch is omitted.
        extra_params: optional extra parameters to pass to the optimizer (e.g.,
                      learned-uncertainty log_sigmas in W4 stretch).
        wandb_run: optional W&B run object from `wandb_integration.wandb_run(...)`.
                   When provided, train/val loss + lr are logged each epoch.
                   When None, no logging — keeps notebooks/tests offline-safe.

    Returns:
        history dict with 'train_loss' and 'val_loss' lists.
    """
    import inspect

    # Optional wandb logger — keeps wandb dep optional and side-effects local
    try:
        from .wandb_integration import log_epoch as _log_epoch
    except ImportError:  # pragma: no cover — defensive only
        def _log_epoch(*args, **kwargs):  # type: ignore[no-redef]
            pass
    torch.manual_seed(seed)
    model = model.to(device)
    params = list(model.parameters()) + (list(extra_params) if extra_params else [])
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    # Auto-detect whether loss_fn expects an epoch argument
    try:
        sig = inspect.signature(loss_fn)
        accepts_epoch = len(sig.parameters) >= 3
    except (ValueError, TypeError):
        accepts_epoch = False

    def _call_loss(model, batch, epoch):
        return loss_fn(model, batch, epoch) if accepts_epoch else loss_fn(model, batch)

    history: dict = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    epochs_no_improve = 0
    epoch = 0

    _t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] training start: max {n_epochs} epochs, "
          f"lr={lr}, weight_decay={weight_decay}, "
          f"early_stop_patience={early_stop_patience}, device={device}")

    for epoch in range(n_epochs):
        _t_epoch = time.time()
        # ─── train ───
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = _call_loss(model, batch, epoch)
            if isinstance(loss, tuple):  # vae_elbo / dec_loss return (loss, ...)
                loss = loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_avg = sum(train_losses) / max(len(train_losses), 1)
        history['train_loss'].append(train_avg)

        # ─── validation ───
        val_avg = float('inf')
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = _move_batch_to_device(batch, device)
                    loss = _call_loss(model, batch, epoch)
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    val_losses.append(float(loss.item()))
            val_avg = sum(val_losses) / max(len(val_losses), 1)
        history['val_loss'].append(val_avg)

        # ─── early stopping + checkpoint ───
        improved = (best_val - val_avg) > early_stop_min_delta
        _marker = '↓' if improved else '·'
        _epoch_elapsed = time.time() - _t_epoch
        print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch+1:3d}/{n_epochs} | "
              f"train={train_avg:.4f} val={val_avg:.4f} {_marker} "
              f"(best={best_val:.4f}) | {_epoch_elapsed:.1f}s")

        # ─── W&B per-epoch logging (no-op if wandb_run is None) ───
        _log_epoch(
            wandb_run,
            epoch=epoch,
            train_loss=train_avg,
            val_loss=val_avg,
            lr=optimizer.param_groups[0]['lr'],
            extra={'epoch_seconds': _epoch_elapsed, 'best_val': best_val},
        )
        if improved:
            best_val = val_avg
            epochs_no_improve = 0
            if checkpoint_path is not None:
                _save_checkpoint(model, epoch, val_avg, train_avg, history, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"[{time.strftime('%H:%M:%S')}] early stop at epoch {epoch+1} "
                      f"(best val={best_val:.4f}, patience exhausted)")
                break

    _total = time.time() - _t_start
    print(f"[{time.strftime('%H:%M:%S')}] training done: {epoch+1} epochs in "
          f"{_total/60:.1f} min, final best_val={best_val:.4f}")
    history['final_val_loss'] = best_val
    history['n_epochs_completed'] = epoch + 1
    return history


def _move_batch_to_device(batch: dict, device: str) -> dict:
    """Move a batch dict to `device`.

    Two layouts are supported:
      - Flat:        {'blocks': {block: tensor}, 'has_bio': tensor}
      - Contrastive: {'view_a': <flat>, 'view_b': <flat>} where each view may
                     additionally carry 'block_mask': {block: tensor}. Used by
                     `make_contrastive_dataloader` for InfoNCE pretext training
                     (spec 2026-05-06 §2.1).
    """
    if 'view_a' in batch and 'view_b' in batch:
        return {
            'view_a': _move_view_to_device(batch['view_a'], device),
            'view_b': _move_view_to_device(batch['view_b'], device),
        }
    return _move_view_to_device(batch, device)


def _move_view_to_device(view: dict, device: str) -> dict:
    out = {
        'blocks': {b: t.to(device) for b, t in view['blocks'].items()},
        'has_bio': view['has_bio'].to(device),
    }
    if 'block_mask' in view:
        out['block_mask'] = {b: t.to(device) for b, t in view['block_mask'].items()}
    return out


def _save_checkpoint(
    model: nn.Module, epoch: int, val_loss: float, train_loss: float,
    history: dict, path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state':  model.state_dict(),
        'epoch':        epoch,
        'val_loss':     val_loss,
        'train_loss':   train_loss,
        'history':      history,
    }, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: str = 'cpu') -> dict:
    """Load model state and metadata. Returns the checkpoint dict (sans model_state)."""
    state = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    return {k: v for k, v in state.items() if k != 'model_state'}
