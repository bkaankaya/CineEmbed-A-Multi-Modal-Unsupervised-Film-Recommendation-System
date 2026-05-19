"""Aggregate per-backbone metadata into artifacts/backbones.json.

Source: each ae_z*/manifest.json (already contains retrieval stats)
plus the gNMI + genre@5 numbers from journal/10 baked in here.

This file is read by /api/backbones at boot to avoid runtime markdown
parsing.
"""

from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "artifacts" / "backbones.json"

# Numbers from docs/journal/10-results-table.md Round 2 section
METRICS = {
    "ae_z32":  {"z": 32,  "gnmi": 0.334, "genreAtFive": 0.723},
    "ae_z64":  {"z": 64,  "gnmi": 0.328, "genreAtFive": 0.715},
    "ae_z128": {"z": 128, "gnmi": 0.273, "genreAtFive": 0.722},
}

LABELS = {
    "ae_z32":  "AE z=32 (demo backbone)",
    "ae_z64":  "AE z=64 (MVP carry-over)",
    "ae_z128": "AE z=128 (over-parameterised)",
}


def main() -> None:
    out: list[dict] = []
    for bid, m in METRICS.items():
        manifest_path = REPO_ROOT / "artifacts" / "inference" / bid / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        out.append({
            "id": bid,
            "z": m["z"],
            "label": LABELS[bid],
            "genreAtFive": m["genreAtFive"],
            "gnmi": m["gnmi"],
            "preferred": bid == "ae_z32",
            "checkpointSha256_32": manifest.get("checkpoint_sha256_32", ""),
            "nFilms": manifest.get("n_films", 0),
        })
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[backbones] wrote {OUT} with {len(out)} entries")


if __name__ == "__main__":
    main()
