#!/usr/bin/env bash
# scripts/package-artifacts.sh — bundle gitignored artifacts for sharing.
# Run on Baran's machine; upload output zip to Drive/WeTransfer; share link
# with teammates. They extract into the repo root and skip the slow regen path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="cineembed-artifacts-$(date +%Y%m%d).tar.gz"

echo "Bundling gitignored artifacts..."
echo ""

# Verify all source files exist
for f in \
  artifacts/inference/films_master.parquet \
  artifacts/inference/ae_z32/embeddings.npy \
  artifacts/inference/ae_z32/films.parquet \
  artifacts/inference/ae_z64/embeddings.npy \
  artifacts/inference/ae_z64/films.parquet \
  artifacts/inference/ae_z128/embeddings.npy \
  artifacts/inference/ae_z128/films.parquet
do
  if [ ! -f "$f" ]; then
    echo "✗ Missing $f — run dev-up.sh once locally to generate, then retry."
    exit 1
  fi
done

tar -czf "$OUT" \
  artifacts/inference/films_master.parquet \
  artifacts/inference/ae_z32/embeddings.npy \
  artifacts/inference/ae_z32/films.parquet \
  artifacts/inference/ae_z64/embeddings.npy \
  artifacts/inference/ae_z64/films.parquet \
  artifacts/inference/ae_z128/embeddings.npy \
  artifacts/inference/ae_z128/films.parquet

SIZE=$(du -h "$OUT" | cut -f1)
echo ""
echo "✅ Created $OUT ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Upload to Google Drive / WeTransfer / Dropbox"
echo "  2. Share the download link with teammates via Slack/iMessage"
echo "  3. Teammates extract in repo root: tar -xzf $OUT"
echo "  4. (Also share .env via secure channel — NEVER commit it)"
