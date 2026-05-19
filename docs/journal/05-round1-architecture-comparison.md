# 05 — Round 1: architecture comparison @ z=64

> Execution log for the three Round 1 configs (wandb group `round-1`):
> `vae_z64`, `dec_z64_k21_from_contrastive_t0p1`, `dec_z64_k21_from_contrastive_t0p5`.
> Outcome: every new config underperformed the cold-start MVP baselines.
> The diagnostic added 2026-05-17 isolates the failure to the contrastive
> encoder, not the DEC step. Decision pivot: Round 1 winner family is MVP,
> not contrastive; Round 2 retargeted to AE z-sweep.

## 1. Context

Round 1 implements the two-round modeling strategy locked on 2026-05-16
(`docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`).
The goal is to pick a winning architecture family at z=64 by the
composite `geo_NMI` metric, then in Round 2 z-sweep the winner at
z={32, 128}. The architectural rationale is that the original
2026-05-04 modeling spec called for 21-22 runs across the
`{AE, VAE, DEC} × {z=32, 64, 128} × {k=10, 21, 30}` grid plus F1/F2/W4
ablations — not feasible in the four working days to the 2026-05-20
deadline alongside the web-app demo pivot. The two-round narrowing
trades grid coverage for a clean
architecture-comparison-then-z-sensitivity story.

Round 1 is therefore a 9-row table at z=64:

| Tier | Run | Source |
|---|---|---|
| Non-DL baseline | `kmeans_raw_k21` | MVP carry-over |
| Non-DL baseline | `pca_kmeans_k21` | MVP carry-over |
| Simple deep | `vanilla_ae_z64` | MVP carry-over |
| Ablation (W1) | `ae_z64_w1` | MVP carry-over |
| Multi-modal AE | `ae_z64` | MVP carry-over |
| Deep + DEC | `dec_z64_k21` | MVP carry-over |
| VAE | `vae_z64` | NEW |
| Contrastive A → DEC | `dec_z64_k21_from_contrastive_t0p1` | NEW (HERO candidate A) |
| Contrastive B → DEC | `dec_z64_k21_from_contrastive_t0p5` | NEW (HERO candidate B) |

The six MVP rows carry over from `artifacts/eval/results.json` without
re-training. Only three new trainings are needed, totalling ~30 min
Colab T4 per the spec. The Round 1 selection rule is "highest `geo_NMI`
on the row's measurement method" — KMeans-k21 for AE and VAE families,
DEC argmax for DEC family.

The HERO row (originally singular in the spec, expanded here to two
candidates) is the empirical payoff of the clustering-improvements spec:
take the Phase 1 contrastive backbone, drop it into AE→DEC fine-tune,
and measure whether the spec's expected +5-12% NMI lift materialises
end-to-end.

## 2. The three new configs

### `vae_z64` — cold start

- Family: VAE
- Latent dim 64, hidden 128, dropout 0.2
- β warmup from 0 to `beta_max = 0.1` over the first 10 epochs, then
  held constant; max 100 epochs with early-stop patience 10
- Loss: `vae_elbo(decoded, targets, mu, log_var, has_bio, w_blocks, beta)`
  — the W2 weighted reconstruction term plus β·KL, with the
  director-block mask gated on `has_bio`
- Justification: "methodological completeness" — the spec notes the
  report needs to answer "did you try VAE?" honestly even if VAE is
  unlikely to win on this matrix.

### `dec_z64_k21_from_contrastive_t0p1` — HERO candidate A

- Family: AE→DEC fine-tune
- AE stage: build a fresh `MultiModalBackbone`, load the Phase 1
  backbone state_dict from
  `artifacts/models/contrastive_tau0p1_drop0p3/pretext_backbone.pt`
  with `strict=False`, attach a randomly-initialised decoder, train
  end-to-end against the W2 weighted recon loss; max 100 epochs,
  early-stop patience 10
- DEC stage: build a `DECHead` wrapping the AE-stage backbone and
  decoder, initialise cluster centres via KMeans-k=21 on the
  full-dataset AE latents, train against
  `dec_loss(z, decoded, targets, centers, has_bio, w_blocks, lambda_recon=0.1)`
  with `reinit_collapsed_centers` invoked once per epoch (spec §10
  cluster-collapse mitigation); 50 epochs cap, early-stop patience 8
- Outputs: `ae.pt`, `dec.pt`, `backbone.pt` (encoder-only state_dict for
  inference convenience), `eval.json`

### `dec_z64_k21_from_contrastive_t0p5` — HERO candidate B

- Same recipe as candidate A, but the pretext backbone loaded is
  `artifacts/models/contrastive_tau0p5_drop0p3/pretext_backbone.pt`.
- Tests whether the Phase 1 "winner by geo" (tau=0.5) carries its
  lNMI advantage through the AE/DEC fine-tune.

The third Phase 1 run (`contrastive_tau0p1_drop0p4`) is not used as a
pretext source — `drop_prob = 0.4` was uniformly worse than 0.3 on Phase
1's own eval and adding it to Round 1 would not have changed the story.

## 3. Notebook layout

`notebooks/07_round1_finetune.ipynb` (commit `276f47d`). Eight cells.

| Cell ID | Purpose |
|---|---|
| `r1_intro` | Markdown intro + run-grid + prerequisites |
| `r1_setup` | Drive mount, repo clone, install, prerequisite check (both Phase 1 backbones must be on Drive) |
| `r1_wandb_cell` | W&B detection ladder (same shape as Phase 1 notebook) |
| `r1_data_cell` | One-time data load, block slicing, W2 weight precomputation, train/val split |
| `r1_helpers_cell` | The four training/eval helpers (see below) |
| `r1_sweep_cell` | The sweep loop with skip-if-done guard |
| `r1_summary_cell` | Read every `eval.json` on Drive, sort by `geo_NMI`, identify winner |
| `a42c70f4` | The 2026-05-17 diagnostic (added in `19962b8`) |

### The four training helpers

All defined in cell `r1_helpers_cell`. Together they mirror the patterns
in `02_train_ae.ipynb` and `04_train_dec.ipynb` (the MVP notebooks)
exactly, so that Round 1 numbers are directly comparable to MVP numbers
on the same eval contract.

- **`train_ae_finetune(run_name, pretext_backbone_path=None, ...)`** —
  builds a fresh AE, optionally loads pretext backbone weights with
  `strict=False`, trains end-to-end. The decoder is always randomly
  initialised (the pretext stage has no decoder). Returns the trained
  AEHead and a history dict. Used as stage 1 of both DEC fine-tunes.

- **`train_dec_finetune(run_name, ae_head, k=21, ...)`** — builds a
  DECHead from a trained AE, initialises centres via KMeans on the
  full-dataset latents (over 329 044 rows), runs the manual training
  loop with per-epoch `reinit_collapsed_centers` (spec §10) on a
  `cluster_size_floor = 0.001` threshold (a cluster below 0.1 % of the
  population is re-seeded from the densest latent region). Mirrors the
  MVP DEC notebook's training loop exactly.

- **`train_vae(run_name, beta_max=0.1, beta_warmup=10, ...)`** — VAE
  cold-start. Uses the same multi-modal backbone, attaches a `VAEHead`
  with mu/log_var heads, loss is `vae_elbo` with β warmup.

- **`encode_all_latents(any_head, device)`** — full-dataset forward
  pass that handles AEHead, DECHead, and VAEHead uniformly. For VAE
  it uses the deterministic `mu` output (not a sample).

### The eval helper

`do_eval(z, dec_model=None)` produces a dict with the same keys as
`scripts/train_contrastive.py`'s eval output — `kmeans_k21`, `gmm_k21`,
`per_axis_k_kmeans`, `multilabel_genre_macro_nmi_kmeans` — plus an
optional `dec_argmax` block when a DEC model is supplied. The
`dec_argmax` block uses the DEC head's soft assignment argmax as the
clustering, instead of running KMeans on the latent.

This shape-parity with the Phase 1 eval is intentional: it means the
nine `eval.json` files (3 Phase 1 + 3 Round 1 + 3 MVP carry-overs from
`artifacts/eval/results.json`) can be loaded into a single comparison
table without per-source schema munging. The `r1_summary_cell`
exploits this directly.

### W&B integration shape

Per the single-W&B-run protocol (commit `d0f08ac`), each Round 1 run
opens one wandb run and pushes everything into it: per-epoch train/val
loss from both stages, the eval-time KMeans/GMM/per-axis-k blocks (with
prefixes `km_`, `gmm_`, `axis_`), the DEC argmax block (prefix `dec_`),
the multi-label macro-NMI scalar, the run-summary headline keys
(`headline_km_geo_nmi`, `headline_dec_geo_nmi`), and the final model
artifacts. The DEC-stage per-epoch logger uses
`epoch = 10000 + epoch` so the DEC curve doesn't collide with the AE
curve on the wandb x-axis.

## 4. Per-run observations

The reported numbers come from `artifacts/models/<run>/eval.json` on
Drive and are also tabulated in `10-results-table.md`. The Round 1
sub-table summary (KMeans-k21 for VAE, DEC argmax for the DEC rows):

| Run | Family | Pretext source | gNMI | dNMI | lNMI | geo_NMI |
|---|---|---|---:|---:|---:|---:|
| `dec_z64_k21_from_contrastive_t0p5` | AE→DEC | tau=0.5 | 0.098 | 0.487 | 0.125 | 0.181 |
| `vae_z64` | VAE | — | 0.103 | 0.358 | 0.056 | 0.127 |
| `dec_z64_k21_from_contrastive_t0p1` | AE→DEC | tau=0.1 | 0.120 | 0.641 | 0.016 | 0.107 |

For comparison, the MVP carry-overs in the same table:

| Run | gNMI | dNMI | lNMI | geo_NMI |
|---|---:|---:|---:|---:|
| `dec_z64_k21` | 0.332 | 0.342 | 0.294 | 0.323 |
| `ae_z64` | 0.328 | 0.341 | 0.264 | 0.309 |
| `vanilla_ae_z64` | 0.287 | 0.369 | 0.095 | 0.215 |
| `ae_z64_w1` | 0.165 | 0.367 | 0.070 | 0.162 |

### `vae_z64`

Early-stopped at **epoch 12** with `best_val_loss = 0.2727` [verify
exact loss value — quoted from notebook output]. Patience = 10 was
exhausted on a val curve that did not improve after roughly epoch 2.

The early stop is suspicious. The β schedule warms from 0 to 0.1 over
the first 10 epochs and then holds — so the model spent only ~2 epochs
at the final β before patience ran out. Two possibilities:

1. **Posterior collapse.** β_max = 0.1 may already be too aggressive
   for the W2-weighted recon term on this matrix. The KL term pulls the
   posterior toward the unit Gaussian prior; if it dominates the recon
   gradient, the latent collapses to noise and the val recon loss is
   indistinguishable from a constant.
2. **Schedule mismatch.** `beta_warmup = 10` is correct in shape but
   the actual β never gets the chance to be "small enough" to learn
   reconstruction before the KL term ramps. A longer warmup (30-50
   epochs) and/or `beta_max = 0.01` would test this.

The eval numbers are consistent with collapse: gNMI = 0.103, dNMI =
0.358, lNMI = 0.056. The decade axis is the only one with any signal —
the easy continuous decade-norm feature can punch through even a
near-collapsed posterior. Genre and language die.

**Without instrumenting recon vs KL separately, this remains a
hypothesis.** No tracer was added to log `recon_loss` and `kl_loss`
separately; the helper logs only the combined ELBO. Suggested follow-up
documented in `08-scope-cuts-future-work.md`: instrument KL term per
epoch, lower `beta_max` to 0.01, try free-bits regularisation
(Kingma et al. 2016) as the standard mitigation.

### `dec_z64_k21_from_contrastive_t0p1`

gNMI = 0.120 (DEC argmax), dNMI = 0.641, lNMI = 0.016, geo_NMI = 0.107.

The dNMI is **anomalously high** — higher than every MVP run including
the MVP DEC champion (0.342). The lNMI is **catastrophically low** —
0.016 means the cluster assignments are essentially random with respect
to top-10 language.

The contrastive pretext itself (Phase 1's `tau0p1_drop0p3`) had
`lNMI = 0.174` on its own. So the pretext encoder DID have some
language signal — it just got demolished by the AE→DEC fine-tune.
Equivalently: the AE/DEC stages of this run pushed the encoder
into a "decade-only" regime, abandoning the modest language signal
the pretext had retained.

What clusters does this run produce? Effectively decade clusters.
The genre/language clustering objective the project optimises for is
nearly absent.

### `dec_z64_k21_from_contrastive_t0p5`

gNMI = 0.098, dNMI = 0.487, lNMI = 0.125, geo_NMI = 0.181.

Same pattern as candidate A but less extreme:

- decade is the dominant axis (0.487, again higher than any MVP run)
- language is held a little (0.125, but well below MVP DEC's 0.294)
- genre is catastrophic (0.098, three times worse than MVP DEC's 0.332)

This is the Phase 1 "winner by geo_NMI" carried forward, and it lost
its key advantage — the lNMI = 0.374 of the pretext stage — almost
entirely after fine-tune. The encoder retained 0.125 lNMI (a 67 %
drop). Whatever made the tau=0.5 pretext look strong on language did
not survive the W2 weighted recon objective.

This is the row that the `r1_summary_cell` mechanically identifies as
the "Round 1 winner" because it has the highest geo_NMI of the three
new rows. But 0.181 is below every MVP carry-over (the worst MVP row,
`ae_z64_w1`, is 0.162; the next-worst, `vanilla_ae_z64`, is 0.215).
The "winner" rule applied without re-thinking would crown a row that
is in fact the second-worst row in the table.

## 5. The diagnostic that pinpointed the failure

Added 2026-05-17 to the same notebook (commit `19962b8`), cell
`a42c70f4`. Two complementary diagnostics whose combined output forces
a single conclusion.

### Diagnostic 1 — Is the MVP DEC in `results.json`?

The summary cell prints the table sorted by geo_NMI but the MVP DEC
champion was missing from the top. The diagnostic scans
`artifacts/eval/results.json` and reports which MVP runs are tracked:

```text
MVP results.json runs: ['ae_z64', 'ae_z64_w1', 'vanilla_ae_z64']
  dec_z64_k21          MISSING from results.json
  kmeans_raw_k21       MISSING from results.json
  pca_kmeans_k21       MISSING from results.json
```

The MVP DEC checkpoint was on Drive at
`artifacts/models/dec_z64_k21/dec.pt` (W&B-backfilled in `31936a5`) but
the eval had never been re-written to `results.json` in the canonical
shape that the new comparison table expects. A local CPU re-eval was
performed against the existing checkpoint and produced
`g = 0.332, d = 0.342, l = 0.294 → geo = 0.323`, matching the
intermediate-report numbers. This re-eval is the
"dec_z64_k21 re-eval 2026-05-17" annotation in
`10-results-table.md` and `00-context-and-goals.md`.

With the MVP DEC row correctly present, the geo_NMI sorting is the
canonical one: `dec_z64_k21` (0.323) and `ae_z64` (0.309) at the top,
the Round 1 rows at the bottom. The "winner by geo" rule applied to
the full table picks an MVP, not a Round 1 run. (The next decision —
which MVP — is settled in `07-retrieval-vs-nmi-discovery.md`.)

### Diagnostic 2 — DEC vs KMeans on the same encoder latent

For each Round 1 DEC run, compare KMeans-k21 on the AE-stage latent
(what the encoder produced before the DEC step) versus DEC argmax
(what the DEC step produced). The logic:

- If KMeans-on-AE-latent is **materially better** than DEC argmax,
  the DEC head is the failure point (it poisoned an OK encoder), and
  we could salvage the encoder by skipping the DEC step.
- If they're **close**, the encoder is the failure point and DEC has
  nothing to work with.

Result, copied verbatim from the diagnostic cell output:

```text
dec_z64_k21_from_contrastive_t0p1
  KMeans on AE latent  g=0.119  d=0.664  l=0.014
  GMM on AE latent     g=0.118  d=0.652  l=0.014
  per-axis-k KMeans    g=0.119  d=0.646  l=0.013
  DEC argmax           g=0.120  d=0.641  l=0.016
  geo_NMI [KMeans]     0.104
  geo_NMI [DEC   ]     0.107

dec_z64_k21_from_contrastive_t0p5
  KMeans on AE latent  g=0.098  d=0.495  l=0.125
  GMM on AE latent     g=0.096  d=0.500  l=0.122
  per-axis-k KMeans    g=0.098  d=0.495  l=0.125
  DEC argmax           g=0.098  d=0.487  l=0.125
  geo_NMI [KMeans]     0.182
  geo_NMI [DEC   ]     0.181
```

The DEC-vs-KMeans gap on geo_NMI is **+0.003** for candidate A and
**-0.001** for candidate B — both **within 0.005**, well inside the
seed noise of the eval. The DEC step is neither helping nor hurting;
the encoder is producing latents that all four clustering methods
(KMeans-k21, GMM-k21, per-axis-k KMeans, DEC argmax) describe
identically.

**Conclusion: the encoder is the failure point.** The DEC stage of the
two HERO candidates is doing exactly what it should; it has nothing
useful to amplify because the AE stage has already collapsed onto the
decade axis.

This is the clean version of the answer Phase 1 was foreshadowing in a
softer form — the contrastive pretext doesn't only fail on its own, it
fails after AE/DEC fine-tune. The pretext is upstream of everything
and its representational deficits propagate.

## 6. What this tells us

Two complementary readings, both consistent with the diagnostic:

**Encoder reading.** The contrastive pretext encoder learned
representations that are good for invariance to dropped blocks
(decade, language, awards, director) and bad for the genre clustering
objective — because genre is itself a block that gets dropped during
the SimCLR augmentation. AE→DEC fine-tune optimises a recon objective
against the W2 weighted reconstruction loss, which gives plenty of
weight to the genre block (`w_genre ≈ 0.71` per the W2 inverse-variance
weighting) but also gives heavy weight to text (`w_text ≈ 1.35`) and
language (`w_lang ≈ 1.80`). The fine-tune can recover the genre signal
*only if the pretext encoder has not actively destroyed it*. The
diagnostic numbers (gNMI 0.098-0.120) say the pretext encoder did
destroy it: even the per-axis-k KMeans, which is the most forgiving
genre-NMI measurement available, returns ~0.10.

**Information-flow reading.** The Phase 1 InfoNCE objective and the
Round 1 W2 recon objective are not pulling in the same direction.
InfoNCE rewards block-invariance; W2 recon rewards block-fidelity. The
encoder is forced to compromise. With a randomly-initialised encoder
the compromise is mediated by the AE training, which is what produces
the MVP `ae_z64` row at gNMI = 0.328. With a contrastive-pretrained
encoder the compromise is biased toward invariance, and AE fine-tune
cannot un-do that bias in a 100-epoch schedule.

A third possibility — that the AE fine-tune itself is mis-configured
— is ruled out by the MVP `ae_z64` baseline. The same `train_ae_finetune`
helper (modulo the `pretext_backbone_path = None` branch) produces a
cold-start AE that lands at gNMI = 0.328. The recipe works; what
breaks it is the contrastive initialisation.

The full root-cause discussion, with attention to which augmentations
would have been viable alternatives, lives in `06-negative-results.md`.

## 7. Decision

Round 1 fails to produce a winner that beats the MVP carry-overs. The
two-round strategy's mechanical "winner is highest geo_NMI of Round 1"
rule, applied to the new rows alone, would pick
`dec_z64_k21_from_contrastive_t0p5` (0.181) as the winner of the new
configs. This is worse than every MVP carry-over.

Applied to the full 9-row table, the same rule picks `dec_z64_k21`
(0.323). That is the actual Round 1 winner.

**The honest decision is: the Round 1 winner family is the MVP DEC.**
The Phase 1 + Round 1 sprint produced negative results on its primary
hypothesis (contrastive pretext + DEC > MVP DEC).

The next question becomes: which MVP backbone deploys to the demo? The
two NMI front-runners (`dec_z64_k21` at 0.323 and `ae_z64` at 0.309)
are within 0.014 of each other on `geo_NMI` — well inside seed noise
on this matrix. The composite NMI metric does not crown one over the
other meaningfully. That tie is resolved by retrieval-eval (`genre@5`
plus eyeball top-5 on well-known titles), which surfaces the
angular-collapse pathology of the MVP DEC and selects `ae_z64` as the
demo backbone. The full retrieval-eval analysis and the methodological
finding it surfaces (NMI is not a retrieval metric, full stop) are in
`07-retrieval-vs-nmi-discovery.md`.

## 8. Round 2 retargeting

With AE z=64 as the demo backbone, the two-round strategy's Round 2
plan needs to change. Originally Round 2 was supposed to z-sweep the
Round 1 winner family — which the mechanical rule would have made
either VAE z={32, 128} (cheap) or DEC z={32, 128} preceded by fresh
contrastive pretext at z={32, 128} (~50 min Colab plus
pretext-fragility risk).

The 2026-05-17 amendment block at the top of the two-round spec
records the new plan: **`ae_z32` and `ae_z128` cold-start** (~20 min
Colab on T4, no failed pretext to recover from). This is a strict
simplification — cheaper, lower-risk, and aligned with the demo
backbone choice. Round 2 is GPU-quota-gated and may slip past the
demo deadline; that is acceptable because `ae_z64` is already the
demo-deployed backbone (`scripts/build_index.py`, commit `1e06a41`)
and Round 2 is for the report's z-sensitivity sub-table only.

Decision table from the spec amendment, for reference:

| Phase | Original (2026-05-16) | Revised (2026-05-17) | Rationale |
|---|---|---|---|
| Round 2 winner family | "highest geo_NMI of Round 1" | AE (highest `genre@5`) | Selection metric realigned to demo task |
| Round 2 runs | up to 2 new contrastive + 2 fine-tune | `ae_z32` cold-start, `ae_z128` cold-start | Cheaper, no failed pretext to recover from |
| Compute | up to ~50 min Colab | ~20 min Colab | T4-pod-day budget halved |

## 9. Cross-references

- `04-phase1-contrastive-sweep.md` — the pretext stage whose backbones
  feed this round.
- `06-negative-results.md` — the encoder-side root-cause analysis
  (modality-dropout structurally conflicts with the clustering label
  axis when the label IS one of the dropped modalities).
- `07-retrieval-vs-nmi-discovery.md` — the `genre@5` retrieval metric
  and the `ae_z64` vs `dec_z64_k21` demo-backbone selection.
- `08-scope-cuts-future-work.md` — what Round 2 will (and will not)
  cover, and the VAE follow-up suggestions.
- `09-operational-incidents.md` — GPU quota exhaustion mid-Round-1
  (forced the offline-mode fallback to be the canonical run mode), GMM
  singular-cov warnings on the DEC run, and the `results.json`
  schema-drift bug that made the MVP DEC champion invisible to the
  comparison cell until diagnosed.
- `10-results-table.md` — Round 1 row block with all reported numbers,
  including the DEC-vs-KMeans delta column.
- Two-round strategy spec
  (`docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`) —
  the canonical contract this round implements, with the 2026-05-17
  amendment block at the top capturing the retargeting decided here.
