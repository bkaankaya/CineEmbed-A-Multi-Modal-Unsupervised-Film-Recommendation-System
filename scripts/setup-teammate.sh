#!/usr/bin/env bash
# scripts/setup-teammate.sh — one-shot setup for teammate clones
# Verifies prerequisites, installs deps, regenerates artifacts where possible,
# launches dev-up.sh when ready.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "  ┌──────────────────────────────────────────────┐"
echo "  │       CineEmbed — teammate setup              │"
echo "  │       SENG 474 · TED University · 2026        │"
echo "  └──────────────────────────────────────────────┘"
echo ""

# ─── 1. .env ─────────────────────────────────────────
if [ ! -f .env ]; then
  echo -e "${YELLOW}⚠${NC}  .env missing."
  echo "    Ask Baran for the file (Slack/iMessage). It contains:"
  echo "      TMDB_API_KEY=<v3 key>"
  echo "      TMDB_ACCESS_TOKEN=<v4 JWT>"
  echo "      CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000"
  echo ""
  echo "    Without it, posters fall back to gradient cards (demo still works)."
  echo ""
  read -r -p "Continue without .env? [y/N] " ans
  if [[ "${ans,,}" != "y" ]]; then
    echo "Aborted. Get .env from Baran, then re-run."
    exit 1
  fi
else
  echo -e "${GREEN}✓${NC}  .env present"
fi

# ─── 2. Source CSV (raw EDA, 252 MB, gitignored) ─────
SOURCE_CSV="artifacts/movies_eda_final.csv"
if [ ! -f "$SOURCE_CSV" ]; then
  echo -e "${RED}✗${NC}  $SOURCE_CSV missing (252 MB)."
  echo "    Required for regenerating films_master.parquet."
  echo "    Ask Baran for the Google Drive / WeTransfer link."
  echo "    Place at: $SOURCE_CSV"
  echo ""
  echo "    Alternative: ask for the pre-built artifact bundle"
  echo "    (films_master.parquet + 3× embeddings.npy + 3× films.parquet, ~580 MB)"
  echo "    and extract directly into artifacts/inference/ instead."
  exit 1
fi
echo -e "${GREEN}✓${NC}  source CSV present"

# ─── 3. Python deps ──────────────────────────────────
echo ""
echo "[1/5] Installing Python deps (cineembed + [demo] extras)..."
pip install -e ".[demo]" --quiet
echo -e "${GREEN}      done${NC}"

# ─── 4. Frontend deps ────────────────────────────────
echo "[2/5] Installing frontend deps (pnpm install)..."
if ! command -v pnpm > /dev/null 2>&1; then
  echo -e "${RED}      pnpm not found. Install with: npm install -g pnpm${NC}"
  exit 1
fi
(cd frontend && pnpm install --silent)
echo -e "${GREEN}      done${NC}"

# ─── 5. films_master.parquet ─────────────────────────
MASTER="artifacts/inference/films_master.parquet"
if [ ! -f "$MASTER" ]; then
  echo "[3/5] Regenerating films_master.parquet from CSV (~60s)..."
  python scripts/enrich_films.py
  echo -e "${GREEN}      done${NC}"
else
  echo -e "[3/5] films_master.parquet exists ${GREEN}✓${NC}"
fi

# ─── 6. Per-backbone embeddings + films.parquet ──────
echo "[4/5] Building per-backbone artifacts (3 backbones × ~30s)..."

build_if_missing() {
  local bb="$1"
  local checkpoint="$2"
  local latent_dim="$3"
  local out_dir="artifacts/inference/$bb"

  if [ ! -f "$out_dir/embeddings.npy" ] || [ ! -f "$out_dir/films.parquet" ]; then
    if [ ! -f "$checkpoint" ]; then
      echo -e "${RED}      ✗ $checkpoint missing — model checkpoint should be tracked in git. git pull?${NC}"
      exit 1
    fi
    echo "      building $bb (latent_dim=$latent_dim)..."
    python scripts/build_index.py \
      --checkpoint "$checkpoint" \
      --model-type ae \
      --latent-dim "$latent_dim" \
      --out "$out_dir/" 2>&1 | tail -3
  else
    echo -e "      ${GREEN}✓${NC} $bb (cached)"
  fi
}

build_if_missing ae_z32  artifacts/models/ae_z32/ae.pt  32
build_if_missing ae_z64  artifacts/models/ae_z64.pt     64
build_if_missing ae_z128 artifacts/models/ae_z128/ae.pt 128

# ─── 7. Verify everything ────────────────────────────
echo "[5/5] Final verification..."
MISSING=0
for f in \
  artifacts/inference/films_master.parquet \
  artifacts/inference/ae_z32/embeddings.npy \
  artifacts/inference/ae_z32/films.parquet \
  artifacts/inference/ae_z64/embeddings.npy \
  artifacts/inference/ae_z64/films.parquet \
  artifacts/inference/ae_z128/embeddings.npy \
  artifacts/inference/ae_z128/films.parquet \
  artifacts/inference/gallery.json
do
  if [ ! -f "$f" ]; then
    echo -e "${RED}      ✗ $f missing${NC}"
    MISSING=1
  fi
done

if [ "$MISSING" -ne 0 ]; then
  echo -e "${RED}Setup incomplete — see missing files above.${NC}"
  exit 1
fi
echo -e "${GREEN}      all artifacts present${NC}"
echo ""

# ─── 8. Launch ──────────────────────────────────────
echo ""
echo -e "  ${GREEN}✅ Setup complete.${NC}"
echo ""
echo "  Launching dev-up.sh (Ctrl-C to stop both servers)..."
echo "  → http://localhost:3000   (frontend)"
echo "  → http://localhost:8000   (API)"
echo ""
sleep 2
bash scripts/dev-up.sh
