# CineEmbed — Project Progress Log

> **Living document.** Updated after each task completes. Source of truth for "where are we".
> If this document and TaskList disagree, trust the git log.

**Last updated:** 2026-05-17
**Current phase:** All modeling rounds complete; demo backbone locked at `ae_z32`. Web-app build pending teammate brainstorm.
**Course:** SENG 474 — Spring 2026 · TED University
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya
**Deadline:** 2026-05-20 (Wednesday)

---

## High-level project state

| Phase | Status | Artifacts | Notes |
|---|---|---|---|
| EDA v1 (original notebook) | ✅ Complete | `Deep_Learning_EDA_*.ipynb` | Untouched, slides reference it |
| EDA v2 (production pipeline) | ✅ Complete | `eda_v2.ipynb` | Feature matrix MD5 `e99cee84b6891ea352a7b44d5d7d0ee4` |
| EDA v2 extension (director profile) | ✅ Complete | folded into `eda_v2.ipynb` | (329044, 564) feature matrix |
| Modeling MVP (Phase 0) | ✅ Complete | 6 runs in `artifacts/eval/results.json` | NMI champion `dec_z64_k21` (later disqualified, see Phase 0 → retrieval pivot below) |
| Intermediate report + pptx | ✅ Delivered | `docs/report/intermediate-progress-report.pdf`, `docs/presentation/*.pptx` | Submitted 2026-05-06 |
| Clustering-improvement techniques | ✅ Implemented | `src/cineembed/{losses,heads,data,eval}.py` | Spec `2026-05-06`; 68/68 tests pass |
| W&B integration + backfill | ✅ Complete | `src/cineembed/wandb_integration.py`, `scripts/backfill_wandb.py` | 6 MVP runs on dashboard; Round 2 offline runs unsynced (skipped) |
| Phase 1 contrastive sweep | ❌ Negative result | 3 configs, group `phase-1-sweep` | All 3 underperformed cold-start `ae_z64`. Documented in journal/04 + journal/06 (ND-1). |
| Round 1 — architecture comparison @ z=64 | ❌ Negative result | `vae_z64` + 2× `dec_*_from_contrastive_*` | VAE posterior-collapse; DEC fine-tunes inherited contrastive poison. Journal/05 + journal/06 (ND-2, ND-3). |
| Phase 0 → retrieval pivot (2026-05-17 AM) | ✅ Locked | journal/07 | NMI champion `dec_z64_k21` disqualified on retrieval (cos≈1.000 collapse); demo backbone selected as `ae_z64` on `genre@5`. First paper-worthy methodological finding. |
| Round 2 — AE z-sweep | ✅ Complete | `artifacts/models/ae_z{32,128}/`, `artifacts/inference/ae_z{32,64,128}/` | U-curve in z; winner `ae_z32` (gNMI=0.334, genre@5=0.723). Second paper-worthy finding. Journal/12, ADR D15. |
| Demo backbone lock (2026-05-17 PM) | ✅ Locked | ADR D15 | `ae_z32` swap from `ae_z64`. Spec amendment in `2026-05-16-web-app-demo-design.md`. |
| Web app demo (FastAPI + static UI) | ⏸ Pending | `cineembed.api` (teammates), `scripts/build_index.py` ✅, frontend (teammates) | Backend + frontend owned by Arda + Kaan; Claude scope ends at inference pipeline. Brainstorm scheduled separately. |
| Final report draft | ⏸ Pending | `docs/report/final-report.tex` | Half-academic tier; 13-file journal serves as source material |
| Final presentation | ⏸ Pending | `.pptx` | Last step |

---

## Two-round modeling strategy

The "Modeling Full" exhaustive ablation (21-22 runs originally scoped in
`2026-05-04-modeling-design.md`) is **superseded** by a two-round strategy
that fits the remaining four working days and the "half-academic" report tier
(between bare demo and full ablation paper). See ADR D13 and
`docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`.

### Selection metric

A single composite ranks Round-1 architectures:

```
geo_NMI = (gNMI · dNMI · lNMI)^(1/3)
```

Geometric mean across the three label axes. Penalizes models that win one axis
and tank another. The winner of Round 1 by `geo_NMI` becomes the only
architecture re-trained in Round 2.

### Round 1 — architecture comparison @ z=64 (~30 min Colab)

9-row architecture comparison:

| Source | Run | Notes |
|---|---|---|
| MVP carry-over | `kmeans_raw_k21` | Non-deep baseline |
| MVP carry-over | `pca_kmeans_k21` | Non-deep baseline |
| MVP carry-over | `vanilla_ae_z64` | Simple deep baseline |
| MVP carry-over | `ae_z64` | Multi-modal (W2) |
| MVP carry-over | `ae_z64_w1` | Uniform-weight ablation |
| MVP carry-over | `dec_z64_k21` | MVP winner |
| NEW | `vae_z64` | Methodological completeness |
| NEW | best of `phase-1-sweep` | Phase 1 winner |
| NEW | `contrastive_pretext + DEC` | HERO RUN — Phase 1 payoff |

### Round 2 — z-sweep on Round-1 winner only (~20 min Colab)

Pick the Round-1 winner by `geo_NMI`. Train that exact architecture at z=32
and z=128 → 3-row z-dim sensitivity sub-table.

### Explicit scope cuts (justified future work)

| Skipped | Why | Justification in report |
|---|---|---|
| `ae_z32`, `ae_z128` | Covered by winner z-sweep | "z-sensitivity studied on the winning architecture" |
| `ae_z64_no_text` (F1) | Modality ablation | "modality contributions deferred to future work" |
| `ae_z64_no_director` (F2) | Modality ablation | "modality contributions deferred to future work" |
| `ae_z64_w4` | Kendall learned weighting | "W2 already validated at +99–277% over W1" |
| `dec_z32_*`, `dec_z64_k10/30`, `dec_z128_*` | k-grid | "k=21 won MVP; full k-sweep future work" |
| `vae_z32`, `vae_z128` | Unless VAE wins Round 1 | "VAE z-sweep contingent on Round-1 outcome" |

Total new training compute: 2 (Round 1) + 2 (Round 2) = 4 runs (~50 min Colab),
plus the 3 Phase 1 contrastive runs already underway.

---

## Web app demo

Final SENG 474 deliverable is a **working web app**, not just a report.
See ADR D14, D15 and `docs/superpowers/specs/2026-05-16-web-app-demo-design.md`.

### Endpoint sketch

| Method | Path | Returns |
|---|---|---|
| GET | `/api/films/search?q=...` | List of matching films (id, title, year) |
| GET | `/api/films/{id}/similar?top=N` | Top-N nearest neighbours by cosine on z=32 latent |
| GET | `/api/films/random?n=...` | Random films (cold-start for the UI) |

### Pre-compute step — DONE

`scripts/build_index.py` ✅ — backbone-agnostic. Round-2 winner `ae_z32`'s
state_dict encoded over all 329k rows, L2-normalized. Live deployment
artifacts at:

- `artifacts/inference/ae_z32/embeddings.npy` — float32 (329 044, 32) ≈ **42 MB**
- `artifacts/inference/ae_z32/films.parquet` — id, title, year, genres, lang
- `artifacts/inference/ae_z32/manifest.json` — checkpoint SHA, retrieval stats, eyeball top-5

Comparable artifacts also exist for `ae_z64` (MVP carry-over, 80 MB) and
`ae_z128` (over-parameterised, 168 MB) for ablation purposes.

### Runtime cosine search

In-RAM cosine: `(embeddings @ q) → topk`. 329k × 32 numpy matmul completes
in <5 ms on a laptop (2× faster than z=64); no FAISS needed.

### Frontend tier

Minimal static HTML + JS. Search box, top-N selector (2/5/10), result cards.
Posters DEFERRED — TMDb API on-demand using the `id` column in
`movies_eda_final.csv`.

### Deployment

`uvicorn cineembed.api:app --reload` on localhost:8000; static UI served from
the `/static/` mount.

### Ownership split

- **Claude scope (complete):** models, inference pipeline, journal, specs.
- **Teammates scope (pending):** backend (`cineembed.api`), frontend, API
  contract refinement, deployment. Brainstorm scheduled separately
  (per 2026-05-17 user decision).

### Remaining work (deadline 2026-05-20)

1. ✅ Models (MVP → Phase 1 → Round 1 → Round 2 → demo backbone lock)
2. ✅ Inference pipeline (`scripts/build_index.py` + 3 inference indices)
3. ⏸ REST API (teammates)
4. ⏸ Frontend UI (teammates)
5. ⏸ Final report `final-report.tex` (Claude — journal as source material)
6. ⏸ Final presentation `.pptx` (combined)

---

## Key decisions and references

| Question | Answer | Source |
|---|---|---|
| Why this architecture? | Multi-modal backbone solves modality-imbalance (variance ratio 0.046) at architectural level | ADR D1 |
| Why these latent dims? | 22 genres × 3 dim/class capacity → z=64 sweet spot, z-sweep validates | ADR D2 |
| Why W2 weighting? | Without it, text/bio dims get no gradient signal | ADR D3 |
| Why bio masking? | 96.8% bio missing → without mask, loss dominated by zero-vector reconstruction | ADR D4 |
| Why per-model notebooks? | Team parallelism (Baran→AE, Arda→VAE, Kaan→DEC), DRY backbone | ADR D5 |
| Why three eval axes? | Single axis (genre) leaves ambiguity about decade/lang structure | ADR D7 |
| Why k-sweep? | Single k = lucky guess; sweep distinguishes encoder quality from k-sensitivity | ADR D8 |
| Why baselines? | Multi-modal claim is unfalsifiable without controls (peer review D9) | ADR D9 |
| Why batch-wise DEC P? | Standard pragmatic approximation (peer review D10) | ADR D10 |
| Why contrastive pretext? | Tabular SimCLR-style stage adds 5–12% NMI lift (D11) | ADR D11 |
| Why per-row block masks + τ=0.1? | Per-row prevents batch co-adaptation; τ=0.1 fits dense heterogeneous tabular signal | ADR D12 |
| Why two rounds (not 21 runs)? | Cost-of-ablation vs deadline; `geo_NMI`-driven winner selection | ADR D13 |
| Why a web app? | SENG 474 final deliverable = working demo, not just a report | ADR D14 |
| Why `ae_z32` over `ae_z64` for demo? | U-curve in z: z=32 wins `genre@5` and `gNMI`; z=128 over-parameterised (near-dead dim). Information-bottleneck sweet spot. | ADR D15 |

**Where to look for context:**
- Modeling spec: `docs/archive/specs/2026-05-04-modeling-design.md`
- Decision log: `docs/adr/0001-modeling-hybrid-architecture.md` (D1–D15)
- Experimental journal: `docs/journal/` (13 files, ~4400 lines)
- Empirical findings: `docs/FINDINGS.md`
- Two-round strategy: `docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`
- Web-app design: `docs/superpowers/specs/2026-05-16-web-app-demo-design.md` (with 2026-05-17 AM + PM amendments)
- Clustering-improvement spec: `docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`
- Intermediate report: `docs/report/intermediate-progress-report.pdf`
- Completed-phase artifacts: `docs/archive/`

---

## Session log (append-only)

### 2026-05-04 — MVP scope kickoff
- Spec finalized (676 lines, ADR D1-D10); implementation plan written (3171 lines after 5 peer-review patches).
- MVP scope decided after second peer review: 11 tasks, 5–6 model runs.
- T1–T8 package + smoke test complete; T9p/T11p/T12p notebooks written (write-only, awaiting Colab).

### 2026-05-05 — MVP runs executed
- 6 MVP trainings executed on Colab T4: `kmeans_raw_k21`, `pca_kmeans_k21`, `vanilla_ae_z64`, `ae_z64`, `ae_z64_w1`, `dec_z64_k21`.
- All 3 pre-registered hypotheses (H1/H2/H3) PASS.
- Winner: `dec_z64_k21` — genre_NMI=0.332, lang_NMI=0.294, decade_NMI=0.342.
- Results vendored to `artifacts/eval/results.json` + figures to `artifacts/figures/umap/`.

### 2026-05-06 — Clustering-improvement techniques landed
- Spec written (`eebb32d`); 5 techniques implemented in one commit (`8097685`):
  - §2.1 InfoNCE contrastive pretext (`ContrastiveHead`, `make_contrastive_dataloader`)
  - §2.2 Per-axis k-sweep eval (`evaluate_run_per_axis_k`; decade k=12, lang k=11)
  - §2.3 GMM / spectral / HDBSCAN cluster assignments
  - §2.4 AMI keys in `evaluate_run` output
  - §2.5 Multi-label macro-NMI on the genre block
- All 68 tests pass; no existing API breaks.

### 2026-05-06 — Intermediate report + slides delivered
- LaTeX report scaffolded then filled section-by-section (`cc9adce` → `bde5c93` → `ef9d894`).
- python-pptx slide builder (`bb773d3` … `695bd06` … `29a2d70`) — 12 slides + metrics glossary.
- Final PDF + PPTX vendored: `docs/report/intermediate-progress-report.pdf`,
  `docs/presentation/intermediate-progress-presentation.pptx`.

### 2026-05-09 — W&B integration + historical backfill
- Optional `wandb` dependency added (`0afb351`); `.gitignore` updated for local cache.
- `src/cineembed/wandb_integration.py` (219 LOC) — context manager + per-epoch log helpers
  integrated into `train_model` (`d2e364c`).
- `scripts/backfill_wandb.py` pushed all 6 MVP runs to the dashboard (`31936a5`).

### 2026-05-16 — Contrastive amendments + Phase 1 launch
- Spec amended inline (`20472d4`): InfoNCE default τ 0.5 → 0.1; masking switched from
  per-batch scalar to per-row `Tensor (B,1)`. Backbone forward accepts both signatures.
- Phase 1 contrastive sweep script `scripts/train_contrastive.py` +
  `notebooks/03_train_contrastive.ipynb` (`5fa95ff`). Three configs launched on Colab,
  wandb group `phase-1-sweep`:
  - `contrastive_tau0p1_drop0p3` — done (UMAP rendering)
  - `contrastive_tau0p5_drop0p3` — queued
  - `contrastive_tau0p1_drop0p4` — queued
- W&B fix (`d0f08ac`): single-run-per-training pattern (train + eval + artifact in one
  wandb run, group support, offline-safe fallback).

### 2026-05-16 — Strategy + scope lock-in
- **Two-round modeling strategy locked** (ADR D13): Round 1 (architecture comparison @ z=64,
  9 rows) + Round 2 (z-sweep on winner only, 3 rows). Selection metric `geo_NMI =
  (gNMI · dNMI · lNMI)^(1/3)`. Explicit-skip list justified as future work.
- **Project deliverable pivots to web-app demo** (ADR D14): FastAPI + static frontend over
  L2-normalized z=64 cosine search. Deadline 2026-05-20. Report tier "half-academic".
- Repo cleanup: 3 stray root files removed; completed-phase specs/plans moved to
  `docs/archive/specs/` and `docs/archive/plans/`; intermediate report brief +
  presentation prompts moved to `docs/archive/2026-05-intermediate-report/`.

### 2026-05-17 (AM) — Phase 0 → retrieval pivot; demo backbone first lock
- **Phase 1 contrastive sweep documented as negative.** All 3 configs underperformed
  cold-start `ae_z64` on `geo_NMI`. Modality-dropout views poison the
  genre-prediction signal. See journal/04, journal/06 (ND-1).
- **Round 1 architecture comparison documented as negative.** VAE posterior-collapse
  at z=64 (early-stop epoch 12, geo=0.127); both AE→DEC fine-tunes inherited
  the contrastive poison. See journal/05, journal/06 (ND-2, ND-3).
- **`scripts/build_index.py` shipped** (`1e06a41`): backbone-agnostic inference
  precompute with retrieval-eval mode (genre@k mean/median/std) and curated
  eyeball top-5 table. Discovered `dec_z64_k21` has intra-cluster cosine ≈ 1.000
  — top-5 retrieval degenerates to random tie-break.
- **First methodological finding locked: NMI ≠ retrieval quality** (journal/07).
  `dec_z64_k21` disqualified despite winning `geo_NMI`. Demo backbone first
  selected as `ae_z64` (genre@5=0.714 vs 0.557). Web-app spec amended.
- **Structured experimental journal** written: 13 files / ~4400 lines under
  `docs/journal/` covering MVP, Phase 1, Round 1, the retrieval pivot,
  negative results, scope cuts, operational incidents, full results table,
  metrics deep-dive.
- **Round 2 retargeted from DEC z-sweep to AE z-sweep** in response to the
  Round 1 outcome. Strategy spec amended.

### 2026-05-17 (PM) — Round 2 complete; final demo backbone locked
- **Round 2 AE z-sweep complete** (notebook `08_round2_ae_zsweep.ipynb`,
  W&B group `round-2`, offline-mode):
  - `ae_z32`: gNMI=0.334, genre@5=**0.723**, dim_std_min=0.117
  - `ae_z64` (carry-over rebuild): gNMI=0.328, genre@5=0.715, dim_std_min=0.062
  - `ae_z128`: gNMI=0.273, genre@5=0.722, dim_std_min=0.025 (near-dead dim)
- **U-curve in z** confirmed. `ae_z128` ties `ae_z32` on `genre@5` within noise
  but collapses 6 pts on `gNMI`, shows angular-collapse precursor.
- **Second methodological finding locked: information-bottleneck sweet spot
  at z=32** (journal/12). Smaller-than-baseline latent wins demo task by
  forcing the encoder to concentrate on high-entropy modalities (text,
  director) at the expense of redundant ones (decade, language).
- **ADR D15 logged**: demo backbone swapped `ae_z64` → `ae_z32`. Spec
  amendment in `2026-05-16-web-app-demo-design.md`. `scripts/build_index.py`
  examples updated.
- **Round 2 artifacts integrated** (`ddb6cbd`): `artifacts/models/ae_z{32,128}/`
  + `artifacts/eval/round2_results.json` committed; inference indices
  (~553 MB) live locally under gitignore.
- **W&B offline-sync skipped** — Colab session ended without finding `wandb/`
  dir; runs not recoverable. Demo not blocked; artifacts authoritative.
- Push to `main` (`ddb6cbd`).
- **Next step (per user, separate session):** demo app brainstorm with teammates.
