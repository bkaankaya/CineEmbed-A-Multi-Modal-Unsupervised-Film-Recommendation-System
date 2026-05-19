# 02 — Clustering-Improvements Spec

> The day after the intermediate report shipped (2026-05-06), the team
> approved a focused spec adding five techniques to the codebase. This file
> covers: what was specified, what was implemented, what changed after, and
> what the test suite looks like.
>
> Read `01-mvp-modeling-phase.md` first.

## §1 Motivation

The MVP-era best gNMI = 0.332 (`dec_z64_k21`) is **not** a sign of a broken
model. The spec's introduction is explicit on this:

> The MVP's best clustering scores (`gNMI = 0.332`, `gARI = 0.244`) are
> task-appropriate for heterogeneous tabular+text+sparse-categorical data
> with multi-label long-tailed genres. They are *not* "weak" — they sit
> comfortably in the 0.20-0.45 NMI band that 2024-2025 deep-clustering
> literature reports for similar problems. However, several established
> techniques can plausibly push the numbers higher with bounded engineering
> effort. This spec captures the five that survived a codebase-feasibility
> audit.

Source: `docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md` §1.

The two implicit goals of the sprint:

1. Add techniques that have a 5-12% NMI lift in the recent (2024-2025) deep
   clustering literature, with at most ~430 LOC of additions and no breaking
   changes to existing APIs.
2. Land the changes in a single commit so the rest of the team can roll the
   new tooling forward into Phase 1 / Round 1 without waiting for a
   multi-stage merge.

## §2 The five techniques

Each technique below: API surface, what changed, the spec's expected payoff,
and on-disk status.

### §2.1 Contrastive pretext

**API surface added:**

- `info_nce_loss(z_a, z_b, temperature)` at `src/cineembed/losses.py:165` —
  symmetric SimCLR/Chen-2020 NT-Xent over two views, fully differentiable.
- `ContrastiveHead(backbone, projection_dim=128)` at
  `src/cineembed/heads.py:133` — wraps `MultiModalBackbone` with a 2-layer
  MLP projection (z → z → d_proj, BN+ReLU). Projection is used **only**
  during pretext; downstream AE/DEC heads operate on the backbone latent z
  per Chen et al. 2020 §3.
- `_ContrastivePairDataset` at `src/cineembed/data.py:182` and
  `make_contrastive_dataloader` at `src/cineembed/data.py:271` — yield two
  augmented views of each row via independent random modality dropout. The
  existing `MultiModalBackbone.forward(blocks, block_mask)` (originally
  added for F1/F2 modality ablation) is exactly the right augmentation
  primitive for heterogeneous tabular data.

**What it changed:** training is now a two-stage option (pretext → AE/DEC
fine-tune) instead of cold AE start. The existing `train.train_model` is
reused with a closure that calls `model(view_a) → z_a`,
`model(view_b) → z_b`, `info_nce_loss(z_a, z_b)`. No changes to `train.py`
itself.

**Expected payoff (spec):** 5-12% NMI lift after 30-60 epochs of pretext,
before AE/DEC fine-tune. Confirmed in 2024-2025 deep-clustering literature
(TCSS, SCAN family, sgSDC).

**Actual payoff:** see `06-negative-results.md`.

### §2.2 Per-axis-k evaluation

**API surface added:**

- `evaluate_run_per_axis_k(z, labels, *, axis_k, seed)` at
  `src/cineembed/eval.py:162` — runs KMeans three times, once per axis, with
  axis-matched k.
- `DEFAULT_AXIS_K: dict[str, int] = {'genre': 21, 'decade': 12, 'lang': 11}`
  at `src/cineembed/eval.py:36`.

**What it changed:** stops penalizing decade and language NMI for
partition-cardinality mismatch. The MVP-era evaluation used k=21 for all
three axes (matching genre); per-axis k restricts the partition to the
axis's natural cardinality.

**Expected payoff:** +5-10% absolute on lNMI / dNMI when k matches axis
cardinality. Pure measurement-honesty fix; the encoder is unchanged.

### §2.3 Soft / non-KMeans clustering

**API surface added (all in `src/cineembed/eval.py`):**

- `cluster_assignments_gmm(z, k, *, seed=42, n_init=5)` at line 45 —
  Gaussian Mixture Model, soft posteriors.
- `cluster_assignments_spectral(z, k, *, seed=42, affinity='cosine', n_neighbors=15)` at line 74 —
  spectral clustering for non-convex manifolds.
- `cluster_assignments_hdbscan(z, *, min_cluster_size=50, min_samples=5)` at line 99 —
  HDBSCAN, auto-discovers cluster count (no k).

**What it changed:** KMeans hard-assigns each row to one cluster, which is a
poor inductive bias when films are multi-genre. GMM gives soft probabilities,
spectral captures non-convex structure, HDBSCAN handles variable density.
GMM is the most defensible KMeans substitute on our Gaussian-ish latent.

**Expected payoff:** GMM typically +2-5% gARI over KMeans on Gaussian
latents. HDBSCAN gives a *different* signal (number of natural clusters in
the latent) that's useful for interpretability in the final report.

### §2.4 AMI

**API surface added:**

- `adjusted_mutual_info_score` import in `eval.py:22`; new `*_ami` key in
  `evaluate_run` output (`src/cineembed/eval.py:158`).

**What it changed:** purely additive third metric alongside NMI/ARI. Per
Vinh et al. JMLR 2016, NMI is biased upward when reference clusterings are
imbalanced (`primary_genre` absolutely is — 21 long-tailed classes). AMI is
the chance-adjusted variant.

**Expected payoff:** "honesty fix" — AMI < NMI for imbalanced labels, but
it's the fair number to report alongside the JMLR-cited NMI.

### §2.5 Multi-label genre eval

**API surface added:**

- `multilabel_macro_nmi(cluster_ids, genre_onehot, *, metric='nmi')` at
  `src/cineembed/eval.py:202` — per-genre binary NMI/AMI, macro-averaged
  across the 21 genres. Returns `{'macro_nmi': float, 'per_genre': dict[int, float]}`.

**What it changed:** stops collapsing multi-genre films to a single
`primary_genre` for evaluation. The 21-dim genre one-hot block already lives
in the feature matrix; we now feed it to the eval function directly instead
of squashing to `primary_genre`.

**Expected payoff:** the macro number is *similar* in absolute value to
single-label NMI but tells a more honest story — it accounts for the
multi-genre encoding the model legitimately learns. The per-genre
breakdown is also a great diagnostic for which genres the model captures vs
not (long-tail genres typically score lower).

## §3 Implementation status

All five techniques landed in a single commit:

- Commit `8097685` — `feat(clustering-improvements): InfoNCE + ContrastiveHead
  + GMM/spectral/HDBSCAN + AMI + multi-label NMI`, 2026-05-06.
- Files changed: `data.py` (+121), `eval.py` (+232 net), `heads.py` (+40),
  `losses.py` (+39); tests `test_data.py` (+43), `test_eval.py` (+116),
  `test_heads.py` (+29), `test_losses.py` (+65). Total: **+678 / -7 LOC**
  across 8 files.

The acceptance § of the spec is checked off as of 2026-05-16 (see the
"Status 2026-05-16" footer in the spec):

> Status 2026-05-16: All acceptance items met as of commit 8097685. Phase 1
> contrastive sweep launched 2026-05-16 (commit `5fa95ff`) on Colab; results
> land in wandb group `phase-1-sweep`.

The acceptance checkboxes (spec §4):

- [x] All new functions have docstrings referencing this spec.
- [x] `pytest -q` passes with 100% pass rate including new tests.
- [x] No deprecation warnings from sklearn/torch.
- [x] No existing test breaks (additive-only contract).
- [x] Type-checks clean under the project's existing Pyright config.
- [x] Imports follow existing style (`from __future__ import annotations`,
      type hints, docstring spec-cross-refs).

## §4 Two amendments (2026-05-16)

Ten days after the spec landed, two amendments were made before Phase 1
launch. **Both amendments are documented inline in the spec file itself**
(see `docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`
§2.1). The rationales are reproduced here.

### 4.1 Per-row block masking (amendment 1)

**Original implementation:** per-batch scalar masks. Every row in a batch
view shares the same modality-dropout pattern.

**Concern raised during follow-on review:** the contrastive-views path needs
**independent per-row masks** to avoid batch-level co-adaptation. If every
row in `view_a` drops the same modalities, the negatives within a batch all
share the same masked-modality bias and InfoNCE has weaker negatives to
contrast against.

**Amendment:** the backbone forward signature was widened to accept either
shape per block:

- legacy scalar `float` (F1/F2 ablation paths) — every row in the batch shares the mask;
- `Tensor (B, 1)` (contrastive views) — per-row independent dropout pattern.

Implementation: commit `20472d4`,
`feat(contrastive): per-row modality dropout + InfoNCE tau=0.1 default`. The
backbone forward path is backwards compatible — existing F1/F2 callers do
not need to change.

The "shatters negatives" concern that originally argued for per-batch is
addressed by rejection-sampling rows where all blocks would drop to zero
(`_ContrastivePairDataset.__getitem__`).

### 4.2 InfoNCE default temperature 0.5 → 0.1 (amendment 2)

**Original spec (2026-05-06):** `temperature ∈ {0.1, 0.5}, sweep both.
Default 0.5 (SimCLR convention).`

**Amendment (2026-05-16):** `default 0.1` for the spec's recommended
starting point. Phase 1 sweep retains 0.5 as a comparison config.

**Rationale:** SimCLR's τ=0.5 is calibrated for the geometry of natural-image
sentence embeddings. The heterogeneous tabular signal here is **denser** —
each row already encodes seven complementary modalities — so the contrastive
objective with high temperature smears the softmax too widely. A lower
temperature (τ=0.1) sharpens the contrastive objective and produces stronger
gradients per positive pair.

Implementation: same commit (`20472d4`).

## §5 Test coverage

The pytest suite added the following alongside the spec
implementation (commit `8097685`, plus per-row mask tests in commit
`20472d4`). Test names verified against `tests/` directory contents.

| File | New tests | Coverage |
|---|---|---|
| `tests/test_losses.py` | 5 InfoNCE tests | `test_info_nce_loss_identical_views_lower_than_random`, `test_info_nce_loss_low_temperature_drives_aligned_to_zero`, `test_info_nce_loss_random_views_in_expected_range`, `test_info_nce_loss_is_symmetric`, `test_info_nce_loss_backward_grads_flow` |
| `tests/test_heads.py` | 3 ContrastiveHead tests | `test_contrastive_head_output_shape_and_projection_dim`, `test_contrastive_head_block_mask_zeroes_modality`, `test_contrastive_head_encode_returns_latent_not_projection` |
| `tests/test_data.py` | 3 contrastive-dataloader tests | including pair-yields-different-views + per-row mask shape assertion (amendment 1) |
| `tests/test_eval.py` | 14 eval-extension tests | per-axis-k (2), AMI (1), GMM (1), spectral (1), HDBSCAN (1), multi-label macro NMI (4), plus pre-existing baseline tests (4) |
| `tests/test_backbone.py` | per-row + per-batch mask coverage | amended for the per-row tensor shape signature |

Total tests in the suite as of 2026-05-17:

```
test_eval.py:               14
test_wandb_integration.py:  17
test_losses.py:             13
test_heads.py:               8
test_data.py:                7
test_backbone.py:            6
test_train.py:               2
test_import.py:              1
                           ----
Total                       68
```

Verify locally: `cd <repo> && .venv/bin/pytest -q` should report
**68/68 expected**.

## §6 What the spec did NOT promise

The spec is explicit about what it does *not* implement, even though all
three would plausibly produce additional gains:

| Out of scope | Reason |
|---|---|
| **Larger text embedding** (BGE-large, e5-mistral, text-embedding-3-small) | Requires re-running EDA on raw text. The pipeline supports it but the re-embedding pass is several hours and the raw text CSV is gitignored. Out of bandwidth for the May 6-16 window. |
| **MMCMAE** (Multi-Modal Contrastive Masked AE, CVPR 2025) | Two-stage progressive pre-training; ~2-week reimplementation. Out of timeline scope. |
| **TabClusterNet** | Replaces encoder with TabNet, invalidates all existing checkpoints. Would require redoing the MVP. |

These remain plausible future-work items if the project continues past the
2026-05-20 deadline.

The spec also explicitly disclaims (§5):

- Running these techniques on the production data — superseded
  2026-05-16 when Phase 1 launched (spec amendment 3, inline).
- Updating `intermediate-progress-report.tex` or `slides.py` — held; that
  spec is its own deliverable, and the clustering improvements were
  developer-facing only.
- Modifying `artifacts/eval/results.json` — held; it stays as the canonical
  record of the MVP's six pre-registered runs. (As of 2026-05-17 `dec_z64_k21`
  has new keys via re-eval, but the original schema is preserved.)
- Modifying `train.py` — held; the contrastive loop reuses `train_model`
  via the existing `loss_fn` callable interface.

## §7 Cross-references

- `04-phase1-contrastive-sweep.md` — the first time §2.1 was actually
  trained on production data (the three-config sweep).
- `06-negative-results.md` — what happened when Phase 1 ran. The spec's
  expected payoff of "5-12% NMI lift" did not materialise on this data.
- `07-retrieval-vs-nmi-discovery.md` — a critique of §2.4 AMI's relevance
  to the actual demo task. AMI is a fairer **clustering** metric, but the
  demo evaluates **retrieval**, where genre@5 turns out to be the
  decision-relevant number.
- `10-results-table.md` Phase 1 section — every Phase-1 number in one place.
