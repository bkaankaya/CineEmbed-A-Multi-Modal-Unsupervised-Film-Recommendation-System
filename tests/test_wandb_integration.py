"""Tests for wandb_integration module — runs offline without network access.

Strategy:
  - WANDB_MODE=disabled is set at module-level → all wandb_run() calls yield None.
  - All log_* helpers must tolerate None gracefully (no-op).
  - Composite metric (geo_nmi) tested with FakeRun stub that captures payload.
"""
from __future__ import annotations

import os

# CRITICAL: set BEFORE any wandb import is triggered downstream
os.environ.setdefault("WANDB_MODE", "disabled")

import pytest

from cineembed.wandb_integration import (
    log_artifact,
    log_epoch,
    log_eval,
    log_image,
    log_table,
    wandb_run,
)


class _FakeRun:
    """Minimal stub matching the wandb.Run.log interface for unit testing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log(self, payload: dict) -> None:
        self.calls.append(payload)


# ---------- context manager ----------

def test_wandb_run_disabled_yields_none() -> None:
    """With WANDB_MODE=disabled, the context manager yields None."""
    with wandb_run(config={"foo": 1}) as run:
        assert run is None


def test_wandb_run_explicit_disabled_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if env says 'online', explicit mode='disabled' wins."""
    monkeypatch.setenv("WANDB_MODE", "online")
    with wandb_run(config={"foo": 1}, mode="disabled") as run:
        assert run is None


def test_wandb_run_invalid_mode_raises() -> None:
    """Invalid mode strings should raise ValueError before init attempt."""
    with pytest.raises(ValueError, match="mode must be one of"):
        with wandb_run(config={}, mode="bogus"):
            pass


# ---------- log_epoch ----------

def test_log_epoch_safe_with_none() -> None:
    """No-op when run is None — must not raise."""
    log_epoch(None, epoch=1, train_loss=0.5, val_loss=0.6)
    log_epoch(None, epoch=1, train_loss=0.5, val_loss=0.6, lr=1e-3)


def test_log_epoch_payload_shape() -> None:
    """When given a fake run, payload contains expected keys."""
    run = _FakeRun()
    log_epoch(run, epoch=3, train_loss=0.42, val_loss=0.51, lr=2e-4,
              extra={"best_val": 0.40})
    assert len(run.calls) == 1
    payload = run.calls[0]
    assert payload["epoch"] == 3
    assert payload["train_loss"] == 0.42
    assert payload["val_loss"] == 0.51
    assert payload["lr"] == 2e-4
    assert payload["best_val"] == 0.40


# ---------- log_eval ----------

def test_log_eval_safe_with_none() -> None:
    log_eval(None, {"genre_nmi": 0.3, "lang_nmi": 0.3, "decade_nmi": 0.3})


def test_log_eval_geo_nmi_computation() -> None:
    """When all 3 NMI keys present, geo_nmi = cube root of product."""
    run = _FakeRun()
    log_eval(run, {"genre_nmi": 0.3, "lang_nmi": 0.3, "decade_nmi": 0.3})
    assert len(run.calls) == 1
    payload = run.calls[0]
    assert "geo_nmi" in payload
    assert payload["geo_nmi"] == pytest.approx(0.3, abs=1e-6)


def test_log_eval_geo_nmi_dec_run_baseline() -> None:
    """Reproduces the dec_z64_k21 baseline geo_nmi from FINDINGS.md."""
    run = _FakeRun()
    log_eval(run, {
        "genre_nmi": 0.332,
        "lang_nmi": 0.294,
        "decade_nmi": 0.342,
    })
    payload = run.calls[0]
    expected = (0.332 * 0.294 * 0.342) ** (1 / 3)
    assert payload["geo_nmi"] == pytest.approx(expected, abs=1e-6)


def test_log_eval_partial_metrics_no_geo_nmi() -> None:
    """If any axis is missing, geo_nmi must NOT be auto-computed."""
    run = _FakeRun()
    log_eval(run, {"genre_nmi": 0.3, "lang_nmi": 0.3})  # missing decade
    assert "geo_nmi" not in run.calls[0]


def test_log_eval_geo_nmi_zero_when_axis_zero() -> None:
    """Geometric mean correctly drops to 0 when any axis is zero."""
    run = _FakeRun()
    log_eval(run, {"genre_nmi": 0.5, "lang_nmi": 0.0, "decade_nmi": 0.5})
    assert run.calls[0]["geo_nmi"] == 0.0


def test_log_eval_prefix() -> None:
    """Prefix prepends to all keys including geo_nmi."""
    run = _FakeRun()
    log_eval(run, {"genre_nmi": 0.3, "lang_nmi": 0.3, "decade_nmi": 0.3},
             prefix="val/")
    payload = run.calls[0]
    assert "val/genre_nmi" in payload
    assert "val/geo_nmi" in payload


def test_log_eval_disable_geo_nmi() -> None:
    run = _FakeRun()
    log_eval(run, {"genre_nmi": 0.3, "lang_nmi": 0.3, "decade_nmi": 0.3},
             add_geo_nmi=False)
    assert "geo_nmi" not in run.calls[0]


# ---------- log_image / log_artifact / log_table ----------

def test_log_image_safe_with_none() -> None:
    log_image(None, "/tmp/nonexistent.png")  # no-op, doesn't even check file


def test_log_image_raises_on_missing_file(tmp_path) -> None:
    """When run is non-None, missing file should raise (catch typos early)."""
    run = _FakeRun()
    with pytest.raises(FileNotFoundError):
        log_image(run, tmp_path / "missing.png")


def test_log_artifact_safe_with_none() -> None:
    log_artifact(None, "/tmp/nonexistent.pt", name="m")


def test_log_artifact_raises_on_missing_file(tmp_path) -> None:
    run = _FakeRun()
    with pytest.raises(FileNotFoundError):
        log_artifact(run, tmp_path / "missing.pt", name="m")


def test_log_table_safe_with_none() -> None:
    log_table(None, [[1, 2]], columns=["a", "b"])
