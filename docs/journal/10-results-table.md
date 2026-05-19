# 10 — Results table

Every model run × every metric in one place. Updated 2026-05-17.

**Source-of-truth files for each row:**

- MVP runs: `artifacts/eval/results.json` + `artifacts/models/dec_z64_k21/eval.json` (re-eval of MVP DEC, 2026-05-17).
- Phase 1: `artifacts/models/contrastive_*/eval.json`.
- Round 1: `artifacts/models/{vae_z64,dec_z64_k21_from_contrastive_*}/eval.json`.
- Retrieval: `artifacts/inference/{ae_z32,ae_z64,ae_z128,dec_z64_k21}/manifest.json`.
- Round 2 sweep eval: per-variant `artifacts/models/ae_z{32,128}/eval.json` (Drive on Colab account B; downloaded to local `artifacts/` 2026-05-17 post-z=128).

**Conventions:**

- `gNMI`, `dNMI`, `lNMI` are NMI against `primary_genre`, `decade_bin`,
  `lang_top10` respectively, computed on the reported method (KMeans or
  DEC-argmax).
- `geo_NMI = (gNMI · dNMI · lNMI)^(1/3)`. Selection metric for the two-round
  strategy.
- `genre@5` is the retrieval-task metric introduced 2026-05-17 — mean
  fraction of top-5 nearest neighbours sharing the query's `primary_genre`,
  measured over 500 random query films via `scripts/build_index.py
  --retrieval-eval`.
- `n/a` means the metric was not computed for that run (e.g. retrieval not
  evaluated, or method irrelevant).
- `—` means structurally not applicable (e.g. AMI for a method that didn't
  log it).

## Master comparison

Sorted by `geo_NMI` descending within each phase. Demo-relevant numbers in
**bold**.

### Round 2 — AE z-sweep (2026-05-17, wandb group `round-2`)

Cold-start AE at z={32, 64 (carry-over), 128}; identical recipe except for
`latent_dim`. `hidden_dim=128` held constant across z. See
`12-z-sweep-ae-z32-discovery.md` for the full narrative — Round 2 produced
the second methodological surprise of the project: a **U-curve** in z,
with the demo-optimal point at **z=32**.

| Run | latent_dim | Method | gNMI | dNMI | lNMI | geo_NMI | **genre@5 mean** | genre@5 median | pair_cos_std | dim_std_min | best_val_loss | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ae_z32`** | 32 | KMeans k=21 | **0.334** | 0.295 | 0.216 | 0.277 | **0.723** | 1.000 | 0.301 | 0.117 | 0.0223 | **Round 2 winner**; best gNMI; cleanest dim utilisation |
| `ae_z64` (MVP carry-over) | 64 | KMeans k=21 | 0.328 | 0.341 | 0.264 | 0.309 | 0.715 | 0.800 | 0.299 | 0.062 | ~0.024 | balanced allocation; best `geo_NMI`/dNMI/lNMI |
| `ae_z128` | 128 | KMeans k=21 | 0.273 | 0.275 | 0.272 | 0.274 | 0.722 | 0.800 | 0.289 | **0.025** | 0.0237 | over-parameterised; near-dead dim (std=0.025); gNMI collapses 6pt |

**Demo-backbone decision (locked):** swap from `ae_z64` to `ae_z32`. The
z=128 result confirmed the U-curve — its `genre@5` (0.722) ties z=32
within noise, but its `gNMI` collapses (-6.1 pts), `dim_std_min` drops
to near-dead, and `pair_cos_std` narrows (the angular-collapse precursor).
ae_z32 wins by Occam tiebreaker on `genre@5` and clearly on every other
informative axis. See `12-z-sweep-ae-z32-discovery.md` §9 for the full
sweet-spot analysis.

### Phase 0 — MVP runs (May 5-9)

| Run | Method | gNMI | dNMI | lNMI | geo_NMI | gARI | dARI | lARI | genre@5 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`dec_z64_k21`** | DEC argmax | **0.332** | 0.342 | 0.294 | **0.323** | 0.244 | 0.165 | 0.183 | **0.557** | MVP champion by NMI. Re-eval 2026-05-17. Angular-collapse on cosine retrieval. |
| **`ae_z64`** | KMeans k=21 | 0.328 | 0.341 | 0.264 | 0.309 | 0.226 | 0.158 | 0.156 | **0.714** | Multi-modal AE, W2 weights. **Demo backbone winner** (genre@5). |
| `vanilla_ae_z64` | KMeans k=21 | 0.287 | 0.369 | 0.095 | 0.211 | 0.183 | 0.182 | 0.030 | n/a | Concat-AE architecture baseline (no modality split). |
| `ae_z64_w1` | KMeans k=21 | 0.165 | 0.367 | 0.070 | 0.162 | 0.101 | 0.181 | 0.022 | n/a | W1 uniform-weight ablation. Confirms W2 matters. |
| `pca_kmeans_k21` | KMeans k=21 | 0.084 | 0.224 | 0.094 | 0.119 | 0.061 | 0.085 | 0.042 | n/a | Non-DL baseline (PCA → KMeans, z=64). |
| `kmeans_raw_k21` | KMeans k=21 | 0.109 | 0.233 | 0.075 | 0.124 | 0.063 | 0.093 | 0.026 | n/a | Non-DL baseline (raw 564-dim KMeans). |

Phase 0 selection metric: gNMI (MVP-era convention). Winner: `dec_z64_k21`.

### Phase 1 — Contrastive pretext sweep (May 16-17, wandb group `phase-1-sweep`)

Three SimCLR-style contrastive-pretext configs. Modality-dropout views,
per-row block masks, InfoNCE loss with default temperature 0.1 (amended from
SimCLR's 0.5 — see `02-clustering-improvements-spec.md`).

| Run | tau | drop_prob | epochs | gNMI | dNMI | lNMI | geo_NMI | gmm_gNMI | per_axis g_k21 | per_axis d_k12 | per_axis l_k11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contrastive_tau0p5_drop0p3` | 0.5 | 0.3 | 45 (early-stop) | 0.216 | 0.218 | 0.374 | **0.260** | 0.202 | 0.216 | 0.235 | 0.408 |
| `contrastive_tau0p1_drop0p3` | 0.1 | 0.3 | ~59 | 0.216 | 0.286 | 0.174 | 0.221 | 0.222 | 0.216 | 0.271 | 0.162 |
| `contrastive_tau0p1_drop0p4` | 0.1 | 0.4 | ? | 0.150 | 0.243 | 0.189 | 0.190 | 0.150 | 0.150 | 0.259 | 0.132 |

Phase 1 metric: KMeans k=21 on the **backbone** latent (projection MLP
discarded per Chen et al. 2020). All three runs underperform the MVP
baselines on `geo_NMI`. Phase 1 winner-by-geo: `contrastive_tau0p5_drop0p3`,
but its high `lNMI` (0.374) is dominated by language-block memorisation, not a
generally-useful representation (see `06-negative-results.md`).

### Round 1 — Architecture comparison at z=64 (May 17, wandb group `round-1`)

Three new configs against the six MVP carry-overs. Selection metric:
`geo_NMI` (DEC argmax for DEC family, KMeans k=21 for VAE family).

| Run | Family | Pretext source | gNMI | dNMI | lNMI | geo_NMI | KMeans-on-AE-latent gNMI | DEC-vs-KMeans delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `dec_z64_k21_from_contrastive_t0p5` | AE→DEC fine-tune | contrastive_tau0p5_drop0p3 | 0.098 | 0.487 | 0.125 | 0.181 | 0.098 | 0.000 |
| `vae_z64` | VAE cold start | — | 0.103 | 0.358 | 0.056 | 0.127 | (n/a) | — |
| `dec_z64_k21_from_contrastive_t0p1` | AE→DEC fine-tune | contrastive_tau0p1_drop0p3 | 0.120 | 0.641 | 0.016 | 0.107 | 0.119 | +0.001 |

VAE z=64 early-stopped at epoch 12 (β warmup over 10 epochs, β_max = 0.1).
Likely posterior collapse — `06-negative-results.md` documents the hypothesis.

The "DEC-vs-KMeans delta" column is the critical diagnostic: it shows that
DEC argmax and KMeans-on-the-same-latent produce within-0.005 gNMI. The DEC
step did **not** poison the encoder; the contrastive pretext did.

### Retrieval-eval comparison (May 17, `scripts/build_index.py`)

Two backbones evaluated for **demo-task quality** (genre@5 + eyeball top-5).
Three additional sanity statistics on the cosine distribution.

| Backbone | gNMI (KMeans) | geo_NMI | **genre@5 mean** | **genre@5 median** | genre@5 std | random_pair_cos mean | random_pair_cos std | Top-5 cos range observed | Demo verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ae_z64` | 0.328 | 0.309 | **0.714** | **0.800** | 0.349 | 0.303 | 0.299 | 0.93 — 0.99 | **SELECTED** |
| `dec_z64_k21` | 0.333 | 0.323 | 0.557 | 0.600 | 0.374 | 0.096 | 0.421 | 1.000 (collapsed) | rejected |

Retrieval-eval was performed on 316 effective query films (films with a
`primary_genre` label) drawn from a 500-row random subset; lower n_queries
reflect rows without a labelled primary genre. The angular sanity statistics
come from 5 000 random film pairs.

Note the seeming paradox: `dec_z64_k21` has a **wider** `random_pair_cos`
distribution (mean ≈ 0, std 0.42 vs ae_z64's 0.30) yet **collapsed top-5
cosines**. This is the angular-collapse signature: between random pairs the
DEC latent looks healthy, but inside any single cluster (≈ 15-17k films) the
vectors are angularly indistinguishable, so the top-5 of any query is a
random tie-break. See `07-retrieval-vs-nmi-discovery.md` for the full
analysis and eyeball comparison.

## What is NOT in this table

- The intermediate-report PPTX-only metrics (now superseded by re-eval).
- Per-epoch training-curve data (in wandb).
- Multi-label macro-NMI breakdowns (per-genre numbers exist in each run's
  `eval.json` but are too verbose to summarise here; macro means are
  available in the wandb run summary as `km_multilabel_macro_nmi`).
- AMI columns (logged by `evaluate_run` but redundant with NMI for the
  purpose of this table — AMI is consistently within 0.01-0.02 of NMI on
  our label distributions).

## Provenance

- All MVP numbers traced to commit `8097685` and re-confirmed via
  `dec_z64_k21` re-eval on 2026-05-17.
- Phase 1 numbers from `contrastive_*/eval.json` written by
  `scripts/train_contrastive.py` (commit `d0f08ac`).
- Round 1 numbers from `dec_z64_k21_from_contrastive_*/eval.json` and
  `vae_z64/eval.json` written by `notebooks/07_round1_finetune.ipynb`
  (commit `276f47d`).
- Retrieval numbers from `artifacts/inference/{ae_z32,ae_z64,ae_z128,dec_z64_k21}/manifest.json`
  written by `scripts/build_index.py` (commit `1e06a41`).
- Round 2 z-sweep results from `notebooks/08_round2_ae_zsweep.ipynb`
  output cells (2026-05-17 run on Colab account B, group `round-2`,
  W&B offline-mode pending sync).
