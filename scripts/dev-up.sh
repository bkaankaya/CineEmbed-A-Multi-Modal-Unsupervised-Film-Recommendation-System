#!/usr/bin/env bash
# scripts/dev-up.sh — launch FastAPI + Next.js dev servers in parallel
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Cap BLAS threads to keep numpy from starving FastAPI's threadpool.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Source .env if present (TMDB_API_KEY, CORS_ORIGINS)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# First-time build of films_master.parquet if missing.
if [ ! -f artifacts/inference/films_master.parquet ]; then
  echo "[dev-up] films_master.parquet missing — running scripts/enrich_films.py (one-time, ~60s)..."
  python scripts/enrich_films.py
fi

cleanup() {
  echo "[dev-up] cleaning up..."
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[dev-up] starting FastAPI on :8000"
python -m uvicorn cineembed.api:app --port 8000 --reload --reload-dir src &
API_PID=$!

echo "[dev-up] starting Next.js on :3000"
# Direct binary bypasses pnpm 11's interactive sharp build-script gate.
# node_modules are populated once via the README's `pnpm install` step.
(cd frontend && ./node_modules/.bin/next dev) &
WEB_PID=$!

echo ""
echo "[dev-up] both processes started:"
echo "  FastAPI:  http://localhost:8000  (pid $API_PID)"
echo "  Next.js:  http://localhost:3000  (pid $WEB_PID)"
echo ""
echo "[dev-up] ready when /api/health returns 200 and Next.js compiles."
echo "         First Next.js compile of ~30 components can take 20-40 seconds."
echo "         Press Ctrl-C to stop both."
wait
