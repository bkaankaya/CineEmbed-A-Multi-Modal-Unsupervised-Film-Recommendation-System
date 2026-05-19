"""Build artifacts/inference/films_master.parquet — a single shared
329k-row films table consumed by every backbone-scoped endpoint."""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cineembed.enrich import parse_country  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "artifacts" / "movies_eda_final.csv"
BASE_PARQUET = REPO_ROOT / "artifacts" / "inference" / "ae_z32" / "films.parquet"
OUTPUT = REPO_ROOT / "artifacts" / "inference" / "films_master.parquet"


def main() -> None:
    print(f"[enrich] reading base parquet: {BASE_PARQUET}")
    films = pd.read_parquet(BASE_PARQUET)
    print(f"[enrich] base shape: {films.shape}")

    print(f"[enrich] reading CSV production_countries: {CSV_PATH}")
    csv = pd.read_csv(CSV_PATH, usecols=["id", "production_countries"])
    print(f"[enrich] CSV rows: {len(csv)}")

    print("[enrich] parsing country values...")
    csv["country"] = csv["production_countries"].apply(parse_country)
    csv = csv.drop(columns=["production_countries"])

    print("[enrich] merging on id...")
    merged = films.merge(csv, on="id", how="left")
    null_pct = merged["country"].isna().mean()
    print(f"[enrich] country null rate: {null_pct:.1%}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT, engine="pyarrow", index=False)
    print(f"[enrich] wrote {OUTPUT}  shape={merged.shape}")


if __name__ == "__main__":
    main()
