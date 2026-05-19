#!/usr/bin/env python3
"""One-shot script: import 6 historical MVP runs into the W&B dashboard.

Reads `artifacts/eval/results.json` and creates one W&B run per entry, tagged
with 'backfill' and 'mvp' so they can be filtered out of post-instrumentation
sweeps.

After running once, the dashboard at wandb.ai/<entity>/cineembed shows the full
MVP history (vanilla_ae_z64, ae_z64, ae_z64_w1, dec_z64_k21, kmeans_raw_k21,
pca_kmeans_k21) alongside any new runs that come from instrumented training.

USAGE
    .venv/bin/python scripts/backfill_wandb.py
    .venv/bin/python scripts/backfill_wandb.py --dry-run        # preview
    .venv/bin/python scripts/backfill_wandb.py --entity team3-seng474

DESIGN NOTES
    - Hyperparameters (z_dim, k, epochs) are extracted from the run name and
      results.json schema. For the model family / loss-weights config we encode
      what we know from FINDINGS.md / PROGRESS.md.
    - geo_nmi (composite) is auto-computed and pushed alongside.
    - We use `tags=['mvp', 'backfill']` so dashboard filters can hide them when
      analyzing post-instrumentation experiments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cineembed.wandb_integration import load_local_wandb_env

# Hand-curated hyperparameters per run, extracted from PROGRESS.md / FINDINGS.md.
# Anything not in results.json (model family, loss-weight schema, modality
# treatment) is encoded here. Edit if memories disagree with reality.
HYPERPARAMS: dict[str, dict] = {
    "vanilla_ae_z64": {
        "model": "AE",
        "modal": False,
        "loss_weights": "equal",
        "description": "concat-AE baseline (no per-block multi-modal handling)",
    },
    "ae_z64": {
        "model": "AE",
        "modal": True,
        "loss_weights": "W2",
        "description": "multi-modal AE with inverse-variance W2 weights",
    },
    "ae_z64_w1": {
        "model": "AE",
        "modal": True,
        "loss_weights": "W1",
        "description": "multi-modal AE with uniform W1 weights (ablation)",
    },
    "dec_z64_k21": {
        "model": "DEC",
        "modal": True,
        "loss_weights": "DEC_KL",
        "description": "DEC head on multi-modal AE backbone, k=21 = primary_genre count",
    },
    "kmeans_raw_k21": {
        "model": "KMeans baseline",
        "modal": False,
        "description": "KMeans on raw 564-dim feature matrix, no embedding",
    },
    "pca_kmeans_k21": {
        "model": "PCA+KMeans baseline",
        "modal": False,
        "description": "PCA reduction to z=64 then KMeans, ablation against deep encoders",
    },
}

# Group assignment for dashboard organization
GROUPS: dict[str, str] = {
    "vanilla_ae_z64": "ae-family",
    "ae_z64": "ae-family",
    "ae_z64_w1": "ae-family",
    "dec_z64_k21": "dec-family",
    "kmeans_raw_k21": "baseline",
    "pca_kmeans_k21": "baseline",
}


def _geo_nmi(metrics: dict) -> float | None:
    """Compute geometric mean of 3 NMI axes. Returns None if any axis missing."""
    g = metrics.get("genre_nmi")
    l = metrics.get("lang_nmi")
    d = metrics.get("decade_nmi")
    if not isinstance(g, (int, float)):
        return None
    if not isinstance(l, (int, float)):
        return None
    if not isinstance(d, (int, float)):
        return None
    product = max(g * l * d, 0.0)
    return product ** (1 / 3) if product > 0 else 0.0


def main() -> None:
    load_local_wandb_env(REPO_ROOT / ".env")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT", "cineembed"),
                    help="W&B project name (default: cineembed)")
    ap.add_argument("--entity", default=os.getenv("WANDB_ENTITY"),
                    help="W&B entity (team/user). Default: WANDB_ENTITY or netrc.")
    ap.add_argument("--results", default="artifacts/eval/results.json",
                    help="Path to results.json (default: artifacts/eval/results.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview what would be pushed without contacting W&B.")
    args = ap.parse_args()

    runs = json.loads(Path(args.results).read_text())
    print(f"Backfilling {len(runs)} historical runs into W&B project "
          f"'{args.project}' (entity={args.entity or 'default'})…")
    if args.dry_run:
        print("[DRY RUN — nothing will be pushed]")
        wandb = None  # type: ignore[assignment]
    else:
        import wandb  # lazy import — keeps --dry-run usable without wandb installed

    for name, metrics in runs.items():
        # Build config dict from results.json hyperparams + curated metadata
        curated = HYPERPARAMS.get(name, {"model": "unknown"})
        config = {
            "run_name": name,
            "z_dim": metrics.get("z_dim"),
            "k": metrics.get("k"),
            "n_epochs": metrics.get("n_epochs"),
            "total_reinit": metrics.get("total_reinit"),
            "backfilled_from": "artifacts/eval/results.json (Mayıs 5 MVP)",
            **curated,
        }
        config = {k: v for k, v in config.items() if v is not None}

        # Build metric payload
        log_payload = {
            "genre_nmi":  metrics.get("genre_nmi"),
            "genre_ari":  metrics.get("genre_ari"),
            "lang_nmi":   metrics.get("lang_nmi"),
            "lang_ari":   metrics.get("lang_ari"),
            "decade_nmi": metrics.get("decade_nmi"),
            "decade_ari": metrics.get("decade_ari"),
            "final_val_loss": metrics.get("final_val_loss"),
        }
        log_payload = {k: v for k, v in log_payload.items() if v is not None}
        geo = _geo_nmi(metrics)
        if geo is not None:
            log_payload["geo_nmi"] = geo

        if args.dry_run:
            print(f"\n  • {name}")
            print(f"      group:  {GROUPS.get(name)}")
            print(f"      config: {config}")
            print(f"      logs:   {log_payload}")
            continue

        assert wandb is not None  # type narrowing — set above when not dry-run
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=name,
            config=config,
            tags=["mvp", "backfill", "intermediate-report"],
            group=GROUPS.get(name, "default"),
            notes=curated.get("description"),
            reinit="finish_previous",
        )
        run.log(log_payload)
        wandb.finish()
        print(f"  ✓ {name:18s}  geo_nmi={log_payload.get('geo_nmi', 0):.4f}  "
              f"genre={log_payload.get('genre_nmi', 0):.3f}  "
              f"lang={log_payload.get('lang_nmi', 0):.3f}  "
              f"decade={log_payload.get('decade_nmi', 0):.3f}")

    if not args.dry_run:
        url = f"https://wandb.ai/{args.entity or '<your-entity>'}/{args.project}"
        print(f"\nDashboard: {url}")


if __name__ == "__main__":
    main()
