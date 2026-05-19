# CineEmbed

Multi-modal representation learning for movie metadata — autoencoder /
deep-embedded-clustering pipeline over a 329k-row, 564-feature matrix, with a
final web-app demo for cosine-similarity recommendations.

**Course:** SENG 474 — Spring 2026 · TED University
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya

---

## Demo — 5-line setup

```bash
# 1. Install Python deps
pip install -e ".[demo]"

# 2. Install frontend deps
cd frontend && pnpm install && cd ..

# 3. Set TMDb API key (optional but recommended)
cp .env.example .env
# edit .env: TMDB_API_KEY=your_v3_key_from_themoviedb.org

# 4. Launch
bash scripts/dev-up.sh

# 5. Open browser
open http://localhost:3000
```

See `docs/demo-script.md` for the presentation flow and known-good
query list.

---

## Teammate Quick Start (clone + run)

Code is on GitHub. Some files are gitignored (secrets + large model artifacts).
Get them from Baran via secure channels, then run one setup script.

### What you need from Baran (out-of-band)
| File | Size | How to share | What's in it |
|---|---|---|---|
| `.env` | <1 KB | Slack/iMessage DM | `TMDB_API_KEY` + `TMDB_ACCESS_TOKEN` + CORS config |
| `cineembed-artifacts-YYYYMMDD.tar.gz` | ~580 MB | Google Drive / WeTransfer link | Pre-computed embeddings (fastest path, **recommended**) |
| **OR** `artifacts/movies_eda_final.csv` | 252 MB | Google Drive | Raw EDA output — slower path, regenerates everything locally |

### Path A — Pre-computed bundle (fastest, ~3 min)

```bash
# 1. Clone
git clone https://github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System.git
cd CineEmbed-
git checkout main

# 2. Drop the .env Baran sent you into the repo root
cp ~/Downloads/.env .env

# 3. Extract the artifact bundle Baran sent
tar -xzf ~/Downloads/cineembed-artifacts-*.tar.gz

# 4. Install deps + launch
pip install -e ".[demo]"
cd frontend && pnpm install && cd ..
bash scripts/dev-up.sh

# 5. Open
open http://localhost:3000
```

### Path B — Regenerate from source (slower, ~10–15 min)

```bash
# 1. Clone
git clone https://github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System.git
cd CineEmbed- && git checkout main

# 2. Drop the .env Baran sent you
cp ~/Downloads/.env .env

# 3. Drop the raw EDA CSV Baran sent
cp ~/Downloads/movies_eda_final.csv artifacts/movies_eda_final.csv

# 4. One-shot setup (deps + regen + launch)
bash scripts/setup-teammate.sh
```

The script installs Python + frontend deps, regenerates `films_master.parquet`
from the CSV, builds per-backbone `embeddings.npy` + `films.parquet` from the
tracked model checkpoints (`artifacts/models/ae_z*/`), then starts `dev-up.sh`.

### What you'll see
- Frontend: http://localhost:3000 (Next.js + Tailwind + shadcn)
- API: http://localhost:8000 (FastAPI; `/api/health` should return JSON)
- 5 pages: home, /cluster, /cluster/[k], /gallery, /about
- Without `.env`, posters fall back to gradient cards (demo still works)

### For Baran — packaging the bundle
```bash
bash scripts/package-artifacts.sh
# → outputs cineembed-artifacts-YYYYMMDD.tar.gz in repo root
# → upload to Drive, share link, also share .env separately via Slack
```

---

## Quick start

```bash
# 1. Install (dev + wandb extras)
pip install -e ".[dev,wandb]"

# 2. Run the test suite (68 tests)
pytest -q

# 3. (Optional) Launch the Colab notebooks
#    Open notebooks/00_colab_setup.ipynb in Colab — it does `git clone` + `pip install -e`.
#    Then run 01_smoke_test → 02_train_ae → 03_train_contrastive → 04_train_dec → 05_results → 06_umap.

# 4. (Forthcoming) Run the FastAPI demo
uvicorn cineembed.api:app --reload
# → http://localhost:8000  (static UI served from /static/)
```

The pre-compute step `python scripts/build_index.py --checkpoint
artifacts/models/<winner>.pt` produces
`artifacts/inference/embeddings.npy` + `artifacts/inference/films.parquet`
(329k × 64 ≈ 80 MB) which the API loads at startup.

---

## Where to look

| Document | Purpose |
|---|---|
| `docs/PROGRESS.md` | Current state, two-round modeling strategy, web-app pivot, session log |
| `docs/FINDINGS.md` | Empirical results (Phase 0 MVP), running results (Phase 1, Rounds 1/2) |
| `docs/adr/0001-modeling-hybrid-architecture.md` | Architecture + decision log (D1–D14) |
| `docs/report/intermediate-progress-report.pdf` | Delivered intermediate report (2026-05-06) |
| `docs/report/final-report.tex` | Final report — in progress |
| `docs/superpowers/specs/` | Active design specs (clustering improvements, two-round modeling, web-app demo) |
| `docs/archive/` | Completed-phase specs, plans, and intermediate-report artifacts |

Start with `docs/PROGRESS.md` — it is the single source of truth for "where are we".

---

## Repository layout

```
src/cineembed/
├── backbone.py            multi-modal encoder (D1)
├── heads.py               AE / VAE / DEC / Contrastive heads
├── losses.py              W2 / G2 / ELBO / DEC-KL / InfoNCE
├── data.py                feature loader + ContrastivePairDataset
├── eval.py                NMI/ARI/AMI/per-axis-k/GMM/spectral/HDBSCAN
├── train.py               generic training loop with W&B hooks
└── wandb_integration.py   single-run-per-training context manager

notebooks/                 Colab-runnable: 00 → 06
scripts/                   train_contrastive.py, backfill_wandb.py, research_clustering.py
artifacts/                 models, eval/results.json, figures/umap/
tests/                     68 tests, all passing
docs/                      PROGRESS, FINDINGS, ADR, report, presentation, specs, archive
```

---

## Reproducibility

- `seed = 42` everywhere (numpy, torch, sklearn).
- Feature matrix MD5: `e99cee84b6891ea352a7b44d5d7d0ee4`.
- Per-block weights and split (90/10 with `random_state=42`) are deterministic
  given the feature matrix.
- All MVP runs are backfilled to W&B; new runs use the
  `cineembed.wandb_integration` context manager and resolve offline-safe if
  `WANDB_API_KEY` is not set.

---

## Current status (2026-05-16)

- ✅ Phase 0 (MVP): 6 runs complete; winner `dec_z64_k21` at gNMI=0.332.
- ✅ Clustering-improvement techniques implemented (commit `8097685`).
- ✅ W&B integration + 6-run backfill on the dashboard.
- 🔄 Phase 1 contrastive sweep running on Colab (wandb group `phase-1-sweep`).
- ⏸ Round 1 architecture comparison @ z=64 (9 rows) — blocked on Phase 1.
- ⏸ Round 2 z-sweep on Round-1 winner (3 rows) — blocked on Round 1.
- ⏸ Web-app demo (FastAPI + static UI) — deadline 2026-05-20.

See `docs/PROGRESS.md` for the full state table and `docs/FINDINGS.md` for the
running results.
