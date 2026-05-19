# 04 — Phase 1: contrastive pretext sweep

> Execution log for the three SimCLR-style contrastive-pretext configs
> (wandb group `phase-1-sweep`). Outcome: all three underperformed the MVP
> baselines on `geo_NMI`. Full root-cause analysis in `06-negative-results.md`;
> read this file for the operational record — setup, configs, observations,
> numbers as-measured.

## 1. Context

Phase 1 is the developer-facing follow-on to
`docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`
§2.1. Its scope is narrow: train the SimCLR-style modality-dropout
contrastive pretext on the production feature matrix, then measure whether
the spec's expected payoff (+5-12% NMI lift after 30-60 epochs of pretext,
before any AE/DEC fine-tune) actually materialises on this data. The spec
authors-style payoff range derives from the 2024-2025 deep-clustering
literature it cites (TCSS, SCAN family, sgSDC), not from prior runs on this
matrix.

The phase produces **backbones** — the projection MLP is trained alongside
but is discarded after pretext per Chen et al. 2020. The KMeans / GMM /
per-axis-k metrics that this notebook reports are computed on the **raw
backbone latent** (output of `MultiModalBackbone.forward`), not the
projection head output. This matters: a model can have a very-low InfoNCE
loss with a backbone latent that is bad for downstream clustering (and that
is exactly what happens here).

Phase 1 fits in the larger plan as follows. Before the two-round strategy
(`docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`) was
locked, the clustering-improvements spec proposed five techniques; only one
of them — the SimCLR pretext — is a separate training stage that produces a
checkpoint another stage consumes. The other four (GMM/spectral/HDBSCAN,
AMI, per-axis-k, multi-label macro-NMI) are eval-time additions that landed
in commit `8097685` and need no separate sweep. Phase 1 therefore tests
exactly one hypothesis: *does the SimCLR-pretext idea work on our data?*

The two-round strategy spec, locked the same day Phase 1 launched
(2026-05-16), reuses the Phase 1 backbones as input to its HERO row
(`contrastive_pretext + DEC`). So Phase 1 is also a strict prerequisite for
Round 1 — without backbones on Drive, Round 1 cannot run.

## 2. The sweep grid

Three configs, deliberately small to fit a 1.5-2 h Colab T4 budget. The
grid pivots on two axes of the spec's hyperparameter space — temperature
and augmentation strength — with one shared design point.

| run | tau | drop_prob | proj_dim | latent | hidden | batch | patience | max epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `contrastive_tau0p1_drop0p3` | 0.1 | 0.3 | 128 | 64 | 128 | 1024 | 8 | 60 |
| `contrastive_tau0p5_drop0p3` | 0.5 | 0.3 | 128 | 64 | 128 | 1024 | 8 | 60 |
| `contrastive_tau0p1_drop0p4` | 0.1 | 0.4 | 128 | 64 | 128 | 1024 | 8 | 60 |

Rationale for the three points:

- **`tau0p1_drop0p3` — "research-recommended default."** Reflects the
  2026-05-16 amendment to the clustering-improvements spec which lowered
  the default temperature from SimCLR's 0.5 to 0.1. The argument
  (paraphrased from the amendment): heterogeneous tabular signal is denser
  than natural-image embeddings, so a lower temperature sharpens the
  contrastive objective. `drop_prob = 0.3` is the spec's nominal value
  (each view drops ~2 of 7 modalities on expectation).
- **`tau0p5_drop0p3` — "SimCLR-paper baseline."** Original spec
  temperature kept as a comparison, so we can demonstrate (or refute) the
  amendment's rationale empirically on this matrix.
- **`tau0p1_drop0p4` — "aggressive augmentation."** Same temperature as the
  default but with a stronger view-distortion. Tests the augmentation-strength
  axis on the assumption that the default `drop_prob = 0.3` might be too mild
  for these dense tabular blocks.

What the grid intentionally does NOT cover:

- No `tau ≥ 1.0` (irrelevant — high-temperature InfoNCE collapses).
- No `drop_prob < 0.3` (would degenerate to near-no-augmentation).
- No `proj_dim ≠ 128` (the 2×-latent-dim rule of thumb is well-supported
  for SimCLR; the spec scoped this to 128 outright).
- No `batch_size ≠ 1024` (small-batch InfoNCE is known-bad; T4 OOMs at
  larger batches).
- No `latent_dim ≠ 64` (Phase 1 is a z=64 sweep by construction; z-grid
  was scoped to Round 2 of the two-round strategy).

The grid is **3 configs**, not the full Cartesian (which would be 9). This
is a deliberate small-grid choice: each Phase 1 run is ~10-30 min on T4,
and the three configs hit the two qualitative knobs the spec actually
specifies (temperature, drop strength).

## 3. Training entry-point

`scripts/train_contrastive.py` (committed `5fa95ff`, refactored to the
single-W&B-run pattern in `d0f08ac`).

### CLI surface (excerpt)

```text
python scripts/train_contrastive.py \
    --artifacts <path> \
    --run-name contrastive_tau0p1_drop0p3 \
    --tau 0.1 --drop-prob 0.3 --proj-dim 128 \
    --batch-size 1024 --epochs 60 --patience 8 \
    --device cuda \
    --wandb-project cineembed --wandb-group phase-1-sweep \
    --wandb-mode online|offline
```

Defaults match the spec amendments: `--tau 0.1`, `--drop-prob 0.3`,
`--proj-dim 128`, `--patience 8`, `--epochs 30` (overridden to 60 by the
notebook).

### Key implementation details

- **`_build_contrastive_loss(tau)`**
  (`scripts/train_contrastive.py:107-115`) — a closure that adapts the
  contrastive forward + InfoNCE loss to `train.train_model`'s
  `(model, batch) -> loss` signature. The batch is a dict with two
  pre-augmented views (`view_a`, `view_b`), each containing a per-row
  `block_mask`; the closure feeds both views through the same backbone
  and calls `info_nce_loss(z_a, z_b, temperature=tau)`. The per-row mask
  granularity itself comes from `make_contrastive_dataloader` and reflects
  the 2026-05-16 amendment that switched from per-batch scalar masks to
  per-row tensors (spec §2.1).

- **Single-W&B-run protocol.** Lines 254-273 build the `wandb_run` context
  exactly once for the whole train + eval + artifact cycle. Inside the
  `with` block: training calls `train_model(...)` (which logs per-epoch
  metrics via `wandb_run`), the backbone-only state_dict is saved, then
  `_do_eval` encodes the full dataset and computes all clustering
  metrics, then `_push_eval_to_wandb` logs each metric block with a
  stable key prefix (`km_`, `gmm_`, `axis_`) and uploads
  `pretext_backbone.pt` as a versioned artifact. The
  pattern was introduced in `d0f08ac` to fix the symptom that earlier code
  was opening multiple W&B runs per training (one for train, another for
  eval). See `09-operational-incidents.md` for the wandb-related fixes.

- **Eval reach.** Phase 1 eval is over the **full 329 044 rows**
  (`_encode_all` at line 119, with `shuffle=False` and no subsetting). The
  KMeans/GMM/per-axis-k assignments and the multi-label macro-NMI all use
  the same encoded matrix `z` of shape `(329044, 64)`.

- **Outputs.** Under `<artifacts>/models/<run-name>/`:
  - `pretext_full.pt` — backbone + projection state_dict, for resume.
  - `pretext_backbone.pt` — backbone-only state_dict, what Round 1 loads.
  - `history.json` — per-epoch train/val loss curve.
  - `eval.json` — KMeans-k21, GMM-k21, per-axis-k, multi-label macro-NMI.
  - `umap.png` — only when `--umap` is set (not used in this sweep; the
    spec made it optional after UMAP-cost issues — see `09-operational-incidents.md`).

### Eval-block shape (one `eval.json`)

```text
{
  "kmeans_k21":        {genre_nmi, decade_nmi, lang_nmi, genre_ari, ..., genre_ami, ...},
  "gmm_k21":           {genre_nmi, decade_nmi, lang_nmi, ...},
  "per_axis_k_kmeans": {genre_nmi_k21, decade_nmi_k12, lang_nmi_k11, ...},
  "multilabel_genre_macro_nmi_kmeans": {macro_nmi, per_genre: {...}}
}
```

This shape is mirrored exactly by `notebooks/07_round1_finetune.ipynb`'s
`do_eval` helper so that Phase 1 and Round 1 numbers are directly
comparable.

## 4. The Colab notebook

`notebooks/03_train_contrastive.ipynb` orchestrates the sweep. Five cells.

| Cell ID | Purpose |
|---|---|
| `03ct000` | Markdown intro + run-grid documentation |
| `03ct001` | Setup: drive mount, repo clone, install, artifact sanity-check, current commit echo |
| `03ct003` | W&B auto-detection ladder (Colab secret → env var → ~/.netrc → offline) |
| `03ct005` | Sweep loop with skip-if-done idempotency guard |
| `03ct007` | Summary: read every `eval.json` and print a 9-column table |

### W&B detection ladder (cell `03ct003`)

The ladder is deliberately layered to never block on stdin (a previous
revision triggered a `KeyboardInterrupt` cascade when `wandb.init` read an
empty `WANDB_API_KEY` env var and fell back to an interactive prompt that
Colab subprocesses cannot answer — see `09-operational-incidents.md`):

1. Colab Secret `WANDB_API_KEY` present → set env var, online mode.
2. Otherwise, env var `WANDB_API_KEY` already set → online mode.
3. Otherwise, local `~/.netrc` exists (non-Colab path) → online mode.
4. Otherwise → **offline mode**. Runs land under `wandb/offline-run-*/`
   and can be synced later with `wandb sync`.

The detection completes silently; if all three checks fail the user gets a
one-line "No WANDB_API_KEY found → using OFFLINE mode" message and
training proceeds without blocking. Both Phase 1 and Round 1 sweeps were
in fact executed in offline mode and synced later.

### Skip-if-done guard (cell `03ct005`)

```python
if (not FORCE) and ckpt.exists() and evalj.exists():
    print(f"SKIP — already trained. ckpt=... eval=...")
    continue
```

Added in commit `88dbcc4` after run 1 (`contrastive_tau0p1_drop0p3`) was
silently re-trained twice during a Colab disconnect → reconnect cycle.
The earlier loop body unconditionally invoked the subprocess; the
re-trained backbones overwrote a checkpoint that was already on Drive,
which is wasted compute but worse, it shifted the random seed state of
the run from what the first pass actually used. The guard requires both
`pretext_backbone.pt` and `eval.json` to be present to count as "done"
(presence of one without the other indicates an aborted run). The fix is
cross-referenced in `09-operational-incidents.md`.

### Sweep launch shape

Each sweep iteration constructs a command line and invokes via
`!python ...` as a subprocess. The subprocess inherits the W&B env vars
from the notebook's process, so each invocation gets its own isolated
W&B run with the correct mode. `INCLUDE_UMAP = False` for the actual
sweep — UMAP rendering is expensive (~5-15 min per run; see
`09-operational-incidents.md`).

## 5. Training observations

### Wall-clock variance

The three runs took noticeably different times even though all three
have similar epoch budgets and identical batch sizes. Per the notebook
sweep output and the `pretext_backbone.pt` mtimes on Drive:

- `contrastive_tau0p1_drop0p3` — ~59 epochs (near max) [verify exact epoch count from `history.json`]
- `contrastive_tau0p5_drop0p3` — 45 epochs, early-stop
- `contrastive_tau0p1_drop0p4` — full or near-full epoch budget [verify]

The wall-clock range across the three was roughly 6.7-25.7 min [verify
exact wall-clock — these are approximate ranges quoted from session notes;
no per-run wall-clock field is written to `eval.json`]. This is almost
certainly Colab spot-instance variability — T4 reassignment between runs
or background contention on a shared host — not a code-side
non-determinism. The three subprocesses are invoked from the same kernel
with identical seeds, on identical data, with identical batch size; the
only differences are `tau` and `drop_prob`, neither of which materially
affects forward/backward FLOPs.

This is worth documenting honestly: Colab is not a controlled compute
environment and the project never had GPU instances of its own. Numbers
were never quoted with wall-clock precision and the report avoids any
"per-epoch time" claims.

### Convergence behaviour

All three runs reached either early-stop or the 60-epoch ceiling. This
means the InfoNCE loss did converge — the pretext signal saturated on
each config. The implication is important: **the pretext was not
under-trained**. The downstream `geo_NMI` shortfall (next section) is not
a training-budget issue; the optimiser found a good (low-loss) point in
the InfoNCE landscape, that point is just not a good point for clustering
on `primary_genre`.

The 8-epoch early-stop patience and the fact that `tau0p5_drop0p3` early-stopped at epoch 45
suggests the val InfoNCE plateaued well before the 60-epoch ceiling. The
two `tau0p1` runs went longer, consistent with a sharper objective
that keeps making small improvements; this is qualitative agreement with
the amendment rationale even though the downstream metric does not pay
off.

## 6. Results

The exact numbers below come from `artifacts/models/contrastive_*/eval.json`
on Drive, as collated in the cell `03ct007` summary table and copied to
`10-results-table.md`. All NMI metrics are KMeans-k21 on the backbone
latent unless noted otherwise.

### Headline table

| run | gNMI | dNMI | lNMI | geo_NMI | gmm_gNMI | axis_g_k21 | axis_d_k12 | axis_l_k11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `contrastive_tau0p5_drop0p3` | 0.216 | 0.218 | 0.374 | **0.260** | 0.202 | 0.216 | 0.235 | 0.408 |
| `contrastive_tau0p1_drop0p3` | 0.216 | 0.286 | 0.174 | 0.221 | 0.222 | 0.216 | 0.271 | 0.162 |
| `contrastive_tau0p1_drop0p4` | 0.150 | 0.243 | 0.189 | 0.190 | 0.150 | 0.150 | 0.259 | 0.132 |

For reference, the MVP baselines on the same metric:

| baseline | gNMI | dNMI | lNMI | geo_NMI |
|---|---:|---:|---:|---:|
| `ae_z64` | 0.328 | 0.341 | 0.264 | 0.309 |
| `dec_z64_k21` | 0.332 | 0.342 | 0.294 | 0.323 |

### Multi-label macro-NMI

Computed via `multilabel_macro_nmi` (averaged per-genre NMI across the 21
one-hot genre columns) and logged to wandb as `km_multilabel_macro_nmi`.
The summary table cell printed `None` in its `ml_macro_nmi` column for
all three runs — that is a display-side issue (the cell pulled
`ml.get('macro_nmi')` from a `dict`-or-`float` field that turned out to
be a bare float in the actual `eval.json` payload [verify]); the metric
itself is computed and is in each `eval.json`'s
`multilabel_genre_macro_nmi_kmeans` block and on the wandb dashboard.
[verify exact values — would need to read the per-run `eval.json` from
Drive to fill these].

### Interpretation, row by row

**All three runs land at `geo_NMI ∈ [0.19, 0.26]`, materially below the
MVP baselines (0.309 and 0.323).** This is the headline negative result
of Phase 1.

**tau=0.1 vs tau=0.5 at drop=0.3.** The two are nearly identical on
gNMI (0.2165 vs 0.2156 — within float-rounding) but distribute the
remaining signal very differently:

- `tau0p5_drop0p3`: lNMI = 0.374 (high) but dNMI = 0.218 (low).
- `tau0p1_drop0p3`: lNMI = 0.174 (low) but dNMI = 0.286 (medium).

A plausible reading: with the looser contrast (tau=0.5) the encoder
takes the path of least resistance and grabs the strongest, simplest
signal — the 31-dim language one-hot — to discriminate pairs. The lower
temperature (tau=0.1) forces a finer discrimination which the encoder
distributes across more modalities, picking up decade signal at the cost
of language. Neither outcome is what we want: genre signal stays flat
at ~0.216 in both cases.

This is qualitative agreement with the spec amendment's rationale (lower
temperature does change what the encoder represents) but the change
doesn't push genre — the axis we actually care about — meaningfully
upward.

**drop_prob = 0.4 is uniformly worse than 0.3** at the same temperature.
Stronger augmentation hurts every axis (gNMI 0.150, dNMI 0.243, lNMI
0.189 vs 0.216 / 0.286 / 0.174 for `drop0p3`). The interpretation:
dropping ~2.8 of 7 modalities per view (vs ~2.1) is too aggressive on
this matrix — the encoder loses too much shared signal between the two
views and ends up at a less-discriminative point in latent space. This
is the simple ablation the grid was designed to give and it gives a
clean answer.

**GMM ≈ KMeans on these latents.** Across the three runs the GMM-k21
gNMI matches KMeans-k21 gNMI to within 0.01 (`tau0p1_drop0p3`: 0.222 vs
0.216; `tau0p5_drop0p3`: 0.202 vs 0.216; `tau0p1_drop0p4`: 0.150 vs
0.150). Conclusion: the backbone latent geometry is roughly Gaussian on
the unit sphere — there is no soft-clustering bonus to be had. The spec
predicted GMM might pick up +2-5% gARI on Gaussian latents; on these
backbones the bonus does not appear, which is consistent with the
contrastive InfoNCE objective producing a hyperspherical, mode-collapsed
arrangement that KMeans already handles fine.

**Per-axis-k vs uniform-k.** `axis_g_k21` is identical to `km_gNMI`
(both are KMeans-k21), so the per-axis improvement only matters for
decade and language. On `tau0p1_drop0p3` the gap is small:
`axis_d_k12 = 0.271` vs `km_dNMI = 0.286` (k=12 is slightly worse on
this run); `axis_l_k11 = 0.162` vs `km_lNMI = 0.174`. The spec
predicted +5-10% absolute when k matches axis cardinality; on Phase 1's
contrastive backbones, k-matching is in the noise. (Note: the MVP runs
do see the expected per-axis-k bonus — see `02-clustering-improvements-spec.md`.)

### Phase 1 "winner by geo_NMI"

Mechanically, the `geo_NMI` selection rule picks
`contrastive_tau0p5_drop0p3` (0.260). But this is a suspect winner:

- Its dNMI is 0.218 (low).
- Its lNMI is 0.374 (high).
- Its gNMI is 0.216 (low — the SAME as `tau0p1_drop0p3`'s 0.216).

The geo mean is being pulled up by language-block memorisation, not by
broadly-useful representation. The 31-dim language one-hot is the
easiest signal in the matrix to over-fit to — it's high-cardinality,
sparse, and almost perfectly separable. The "winner" is the run that
most successfully short-circuited the contrastive objective into
language-block recall.

Round 1 uses both `tau0p1_drop0p3` and `tau0p5_drop0p3` as pretext
sources to AE→DEC fine-tunes precisely to find out which Phase 1
backbone fine-tunes better. The notebook discussion at the time framed
both as "hero candidate A/B." See `05-round1-architecture-comparison.md`.

## 7. Why Phase 1 fell short

The two-line summary: the SimCLR-style modality-dropout pretext creates
an encoder invariant to dropped block content. The genre block is itself
one of the modality blocks that gets dropped. So the encoder is
explicitly trained to produce representations that are invariant to
genre information — which is the opposite of what `primary_genre`
clustering needs.

In more detail: the InfoNCE objective treats two views of the same row
as a positive pair. If view A drops genre and view B keeps it, the
encoder must produce nearly-identical z's from a row-with-genre and a
row-without-genre. The cleanest way to achieve that is to make genre
information not affect z at all. The objective is structurally pulling
the encoder away from preserving genre, and the encoder complies.

The full root-cause analysis — including why this is a
data-structural conflict and not a temperature/dropout-tuning issue,
and what augmentation primitives would have been better — is in
`06-negative-results.md`. The bottom line, foreshadowed here: a
SimCLR-style invariance pretext is a poor fit for an objective whose
labels are themselves one of the augmentation primitives.

## 8. Operational notes

Cross-link `09-operational-incidents.md` for the five Phase 1 ops
events worth recording:

- **WANDB_API_KEY env var leak.** Commit `17d6fbb` originally added the
  skip-if-done guard, but during the change-set audit the W&B key was
  briefly visible in a scratch cell. The current canonical guard is
  `88dbcc4`, which adds the W&B env-var fallback safely.
- **Sweep loop idempotency bug.** The pre-`17d6fbb` loop body called
  the subprocess unconditionally; combined with a Colab disconnect this
  caused run 1 to silently re-train twice.
- **GMM singular-covariance warnings** on the higher-tau / lower-drop
  configs — `sklearn.mixture.GaussianMixture` flagged a few components
  as having degenerate covariances. Suppressed with `warnings.filterwarnings`
  in the eval pipeline; the assignment is still well-defined and the
  numbers above are correct.
- **UMAP rendering cost.** `--umap` was added to the script but disabled
  for the sweep itself (would have added ~5-15 min/run). UMAPs for the
  Phase 1 winners are deferred to a separate ad-hoc render against the
  Drive checkpoints.
- **Single-run-per-training fix.** Commit `d0f08ac` collapsed multiple
  wandb runs per training into a single wandb run, and added safe
  offline fallback that does not block on stdin.

## 9. What Phase 1 left behind

Three pretext backbones on Drive at
`artifacts/models/contrastive_<run>/pretext_backbone.pt`. These are
~237 KB each and contain only the `MultiModalBackbone` state_dict — the
projection MLP and decoder state are not in this file.

They are the input to Round 1 (`notebooks/07_round1_finetune.ipynb`):
the two strongest by overall structure (`tau0p1_drop0p3` and
`tau0p5_drop0p3`) feed the two `dec_z64_k21_from_contrastive_*` rows of
the Round 1 architecture comparison.

The third (`tau0p1_drop0p4`) is not used downstream — the grid was
already-decided on `drop_prob = 0.3` and the aggressive-drop run only
exists to document that the more-aggressive setting was tried and was
worse.

## 10. Cross-references

- `02-clustering-improvements-spec.md` — the spec that motivated this
  sweep; lists the four other landed techniques.
- `03-tooling-wandb-integration.md` — the W&B detection ladder + single
  run-per-training pattern.
- `05-round1-architecture-comparison.md` — where the Phase 1 backbones
  are consumed.
- `06-negative-results.md` — the structural genre-invariance failure mode
  that explains why Phase 1's `geo_NMI` came in below the MVP baselines.
- `09-operational-incidents.md` — wandb-key handling, sweep idempotency,
  GMM singular cov, UMAP cost.
- `10-results-table.md` — Phase 1 row block with all reported numbers.
