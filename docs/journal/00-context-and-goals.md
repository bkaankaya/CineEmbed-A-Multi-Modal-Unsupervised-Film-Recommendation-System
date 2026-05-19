# 00 — Context and Goals

> Master synthesis. Everything in `docs/journal/` builds on the timeline,
> definitions, and decisions captured here. Read this first.

## Project

**CineEmbed** is a multi-modal movie clustering and recommendation system.
The model learns a single 64-dim latent representation that encodes
heterogeneous film metadata (numerical features, multi-label genre, language,
release decade, awards, English overview text embedding, director biography
embedding), and produces clusters / nearest-neighbour rankings useful for
content recommendation.

**Course:** SENG 474 — Spring 2026, TED University (Hacettepe Üniversitesi).
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya.
**Final deliverable:** a working web-app demo + a half-academic report, due
**2026-05-20 (Wednesday)**.

## Data

The EDA phase (complete, May 3-4, 2026) produced two canonical artifacts:

- `artifacts/feature_matrix.npz` — **(329 044, 564) float32**, MD5
  `e99cee84b6891ea352a7b44d5d7d0ee4`. Seven modality blocks contiguous on the
  column axis in this order:

  | Block | Dims | Contents |
  |---|---:|---|
  | numerical | 6 | log_popularity, log_vote_count, runtime_norm, vote_average_norm, has_vote, has_engagement |
  | genre | 22 | 21 one-hot genre indicators + has_genre flag (multi-label) |
  | language | 31 | original-language one-hot |
  | decade | 2 | decade_norm + has_release_date |
  | awards | 6 | log-transformed prior-Oscar / prior-Palme / prior-total nominations and wins |
  | text | 384 | overview sentence embedding (all-MiniLM-L6-v2) |
  | director | 113 | dir_bio_pca (64) + has_director_bio + dir_lang/country one-hots + has_director_lang |

- `artifacts/movies_eda_final.csv` — 329 044 rows aligned positionally with
  the feature matrix. Columns include `id` (TMDb), `imdb_id`, `title`,
  `original_title`, `release_date` (dd/mm/YYYY), `director_name`, `genres`
  (pipe-delimited), `overview`, `popularity`, `vote_average`, `vote_count`,
  `original_language`.

## Goal hierarchy

```
Primary goal:
  Produce a working web-app film recommender by 2026-05-20.

Sub-goals (in priority order, locked 2026-05-16):
  (1) Trained encoder backbone capable of producing useful 64-dim
      embeddings for cosine-similarity retrieval.
  (2) Inference pipeline: precompute embeddings + film metadata,
      serve via REST API.
  (3) Frontend UI: search box + result cards.
  (4) Posters and polish.
```

Note: the **demo-relevant model-quality metric is retrieval performance, not
clustering NMI.** This realignment happened on 2026-05-17 and is the
critical methodological finding of the project — see
`07-retrieval-vs-nmi-discovery.md`.

## Timeline

```
2026-05-03 to 05-04   EDA v2 (production pipeline, 75 cells)
2026-05-04            Modeling design spec (D1-D10 ADR)
2026-05-04 to 05-08   Modeling MVP implementation (T1-T12p)
2026-05-05            6 MVP runs executed
2026-05-06            Clustering-improvements spec (5 techniques)
2026-05-06            Intermediate progress report + PPTX delivered
2026-05-09            W&B integration + backfill of 6 MVP runs
2026-05-10            Multi-agent panel work (incidental, see git log)
2026-05-16            Per-row masking + InfoNCE tau=0.1 amendments
2026-05-16            Phase 1 contrastive sweep launched (Colab)
2026-05-16            Web-app pivot decision (final = demo + report)
2026-05-16            Two-round modeling strategy locked
2026-05-17            Phase 1 sweep complete (3 runs)
2026-05-17            Round 1 architecture comparison complete (3 runs)
2026-05-17            Diagnostic + retrieval-eval pivot to ae_z64
2026-05-17            Journal initialized (this directory)
[pending]             Round 2 z-sweep (ae_z32 + ae_z128) — GPU quota gating
[pending]             Web app integration (teammates working on API + UI)
2026-05-20            Final demo due
```

## Definitions

The journal uses these terms consistently:

- **gNMI / dNMI / lNMI.** Normalised Mutual Information between cluster
  assignments and `primary_genre` (g), `decade_bin` (d), `lang_top10` (l).
  All in [0, 1]. Higher = clusters align better with that axis.
- **gARI / gAMI.** Adjusted Rand Index and Adjusted Mutual Information against
  the same axes. AMI is chance-adjusted (Vinh et al. JMLR 2010).
- **geo_NMI.** Composite metric `(gNMI · dNMI · lNMI)^(1/3)`. Drops to 0 if any
  single axis is 0 — penalises over-fitting one axis. Selection metric for the
  two-round strategy.
- **genre@k.** Mean fraction of a query's top-k nearest neighbours that share
  its `primary_genre`. Introduced 2026-05-17 as the retrieval-task metric and
  is what the demo experience actually depends on.
- **MVP runs.** The six pre-registered model runs delivered for the
  intermediate report: `kmeans_raw_k21`, `pca_kmeans_k21`, `vanilla_ae_z64`,
  `ae_z64`, `ae_z64_w1`, `dec_z64_k21`.
- **Phase 1.** The three-config contrastive-pretext sweep, wandb group
  `phase-1-sweep`. See `04-phase1-contrastive-sweep.md`.
- **Round 1.** The architecture comparison at z=64 in the two-round modeling
  strategy. Three new configs (`vae_z64`, two contrastive→DEC fine-tunes)
  measured against the 6 MVP carry-overs. wandb group `round-1`. See
  `05-round1-architecture-comparison.md`.
- **Round 2.** The z-sweep on the Round-1 winner family — pending GPU quota,
  re-targeted from DEC to AE family on 2026-05-17 after the retrieval-eval
  pivot.

## Architecture (one paragraph)

A shared `MultiModalBackbone` projects each modality block to its own
compressed sub-vector (numerical→16, genre→16, language→16, decade→4,
awards→16, text→64, director→32), concatenates them (154 dims), passes the
concatenation through a 2-layer FC backbone (154 → 128 → 64) producing the
latent **z ∈ R⁶⁴**. The backbone is the only encoder the project uses; the
training "heads" (`AEHead`, `VAEHead`, `DECHead`, `ContrastiveHead`) each wrap
the same backbone with a different decoder / projection / cluster-centre
component and therefore a different loss. The web-app demo uses the backbone
alone — the heads are discarded after training.

## Decisions snapshot (as of 2026-05-17)

| Decision | Status | Source |
|---|---|---|
| Multi-modal projection backbone (D1) | Locked | ADR 0001 |
| z = 64, hidden = 128 (D2) | Locked at z=64; Round 2 will sweep z={32,128} | ADR 0001, two-round spec |
| W2 inverse-variance block weights (D3) | Locked (W1 ablation worse) | ADR 0001 |
| G2 director-block masked loss (D4) | Locked | ADR 0001 |
| primary_genre KMeans @ k=21 + DEC argmax (D6, D7) | Locked; AMI + per-axis-k added later | ADR 0001 + clustering-improvements spec |
| Batch-wise DEC P (D10) | Locked | ADR 0001 |
| Clustering-improvements (D11): InfoNCE + GMM/spectral/HDBSCAN + AMI + multilabel | Landed 2026-05-06 | clustering-improvements spec |
| Per-row block masking + InfoNCE tau=0.1 (D12) | Landed 2026-05-16 | spec amendments |
| Two-round modeling strategy (D13) | Round 1 done; Round 2 retargeted to AE family | two-round spec amendment |
| Project deliverable pivot to web app demo (D14) | Locked | web-app-demo spec |
| Demo backbone = `ae_z64` (NOT NMI champion) | Locked 2026-05-17 | web-app-demo spec amendment + journal 07 |

## What this journal does NOT contain

- Hyperparameter values for every run (those live in
  `artifacts/eval/results.json`, each run's `eval.json`, and the wandb
  dashboard for `cineembed`).
- Training-curve plots (in wandb).
- The intermediate report or final report drafts (those live under
  `docs/report/`).
- Per-file API documentation (Python code is self-documenting; see the
  src/cineembed/*.py module docstrings).
- Step-by-step "how to run training" guides (see `notebooks/` + `README.md`).

The journal is the **interpretation layer**: what the numbers mean, why we
made the choices we did, and what to do next.

## Cross-references

The rest of the journal:

- Foundation: `01-mvp-modeling-phase.md`, `02-clustering-improvements-spec.md`,
  `03-tooling-wandb-integration.md`
- Execution: `04-phase1-contrastive-sweep.md`, `05-round1-architecture-comparison.md`
- Analysis: `06-negative-results.md`, `07-retrieval-vs-nmi-discovery.md`,
  `08-scope-cuts-future-work.md`
- Ops: `09-operational-incidents.md`
- Data: `10-results-table.md`
