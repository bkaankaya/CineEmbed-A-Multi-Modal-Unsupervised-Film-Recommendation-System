# Two-Round Modeling Strategy — Design Spec

**Date:** 2026-05-16
**Status:** APPROVED — Round 1 executed, results below; Round 2 plan revised
**Supersedes the run grid in:** `docs/archive/specs/2026-05-04-modeling-design.md` (D2/D6/D8)
**Cross-ref:** ADR `0001-modeling-hybrid-architecture.md` D13;
`docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`

## Amendment — 2026-05-17 — Round 1 outcomes + Round 2 retargeting

**Round 1 result.** The 9-row architecture comparison at z=64 produced an
**unexpected outcome**: every contrastive-pretext-initialized fine-tune
underperformed the cold-start MVP baselines by a wide margin
(`geo_NMI`: contrastive+DEC 0.107–0.181 vs MVP `ae_z64` 0.309 vs MVP `dec_z64_k21` 0.323).
VAE z=64 also underperformed (geo_NMI=0.127), likely posterior collapse.

**Root cause (Phase 1 + Round 1 failure mode).** The SimCLR-style modality-dropout
contrastive pretext creates encoder invariance to dropped block content. Because
the genre block is itself a modality block subject to dropout, the encoder
learned genre-invariant representations — directly opposite to the downstream
clustering objective on `primary_genre`. The augmentation primitive structurally
conflicts with the task. We retain this as a documented negative result.

**Demo-backbone selection criterion change.** A retrieval evaluation
(`genre@5` over 500 random queries, plus an eyeball top-5 on 10 well-known
titles, see `scripts/build_index.py --retrieval-eval --eyeball`) revealed that
the NMI champion `dec_z64_k21` suffers **angular collapse**: every in-cluster
pair has cosine ≈ 1.000, so top-N retrieval ranks within a cluster are
effectively random tie-breaks. The smoother AE manifold gives genuinely
graded cosine neighbours.

| Backbone | geo_NMI | `genre@5` | Eyeball verdict |
|---|---:|---:|---|
| `dec_z64_k21` | 0.323 | 0.557 | Tied at cos=1.000, random order |
| **`ae_z64`** | 0.309 | **0.714** | Coherent Nolan / Pixar / Studio Ghibli groupings |

**Decision:** the demo backbone is **`ae_z64`** (chosen by `genre@5`, not NMI).
See `2026-05-16-web-app-demo-design.md` amendment of the same date.

**Round 2 retargeting.** Because the demo-deployed family is now AE, the
Round 2 z-sweep is re-aimed at `ae_z32` and `ae_z128` (cold-start AE, no
contrastive prereq). This is a strict simplification — original Round 2 plan
required either fresh contrastive pretext at z=32/128 (would have been ~50 min
Colab plus pretext-fragility risk) or just VAE z-sweep (only if VAE had won
Round 1, which it did not). Cold-start AE z-sweep is ~20 min Colab.

| Phase | Original (2026-05-16) | Revised (2026-05-17) | Rationale |
|---|---|---|---|
| Round 2 winner family | "highest geo_NMI of Round 1" | AE (highest `genre@5`) | Selection metric realigned to demo task |
| Round 2 runs | depends on winner; up to 2 new contrastive + 2 fine-tune | `ae_z32` cold-start, `ae_z128` cold-start | Cheaper, no failed pretext to recover from |
| Compute | up to ~50 min Colab | ~20 min Colab | T4-pod-day budget halved |

---

## 1. Motivation

The original modeling spec (2026-05-04) scoped 21-22 model runs across the
`{AE, VAE, DEC}` × `{z=32, 64, 128}` × `{k=10, 21, 30}` grid plus
F1/F2/W4 ablations. That matrix is **not feasible** in the four working days
to deadline (2026-05-20) when combined with the web-app demo pivot (ADR D14).

The report tier is **"half-academic"** — between a bare demo and a full
ablation paper. The goal is to deliver:

- a clean, defensible architecture-comparison story at one latent dim, AND
- a focused z-sensitivity check on the winning architecture,

while leaving the rest of the original ablation grid as explicit,
report-acknowledged future work. This document captures that strategy.

---

## 2. Selection metric — `geo_NMI`

Round-1 architectures are ranked by the geometric mean of NMI across the
three label axes:

```
geo_NMI = (gNMI · dNMI · lNMI)^(1/3)
```

Geometric mean (not arithmetic) penalizes a model that wins one axis and
tanks another — a real risk on this data, since the MVP showed no single
architecture wins all axes (vanilla wins decade_NMI + genre_ARI; multi-modal
wins decade_ARI; DEC wins genre_NMI + lang_NMI + lang_ARI).

The Round-1 winner by `geo_NMI` is the only architecture re-trained at
z={32, 128} in Round 2.

---

## 3. Round 1 — Architecture comparison @ z=64

**Compute estimate:** ~30 min Colab T4 total (3 new trainings × ~10 min;
MVP rows carry over from `artifacts/eval/results.json`).

### Run list (9 rows)

| Tier | Run | Source |
|---|---|---|
| Non-deep baseline | `kmeans_raw_k21` | MVP carry-over |
| Non-deep baseline | `pca_kmeans_k21` | MVP carry-over |
| Simple deep | `vanilla_ae_z64` | MVP carry-over |
| Ablation (W1) | `ae_z64_w1` | MVP carry-over |
| Multi-modal deep | `ae_z64` | MVP carry-over |
| Deep + DEC | `dec_z64_k21` | MVP carry-over (current `geo_NMI` leader at ≈ 0.322) |
| VAE | `vae_z64` | NEW — methodological completeness |
| Best contrastive | best of `phase-1-sweep` | NEW — Phase 1 winner |
| HERO | `contrastive_pretext + DEC` | NEW — demonstrates Phase 1 payoff |

`vae_z64` is included even though it is unlikely to win — the final report
needs to answer "did you try VAE?" honestly. The HERO row combines the
Phase 1 contrastive pretext (ContrastiveHead backbone weights) with a DEC
fine-tune head, the natural payoff of the clustering-improvement spec.

### Eval contract

Every Round-1 row reports, on the full 329k feature matrix:

- KMeans-k21 → `{g,d,l}_{nmi,ari,ami}` (the MVP-standard 6 axes)
- Per-axis-k → `{g_k21, d_k12, l_k11}_{nmi,ami}` (D11 §2.2)
- GMM-k21 → `{g,d,l}_nmi` (D11 §2.3) — optional defensive number
- `geo_NMI` from the KMeans-k21 numbers (primary selection metric)

DEC-style rows (`dec_z64_k21`, HERO) use the DEC soft-assignment argmax in
place of KMeans-k21.

---

## 4. Round 2 — Z-sweep on winner

**Compute estimate:** ~20 min Colab T4 total (2 new trainings; z=64 row
carries over from Round 1).

**Logic:**

1. Identify the Round-1 winner by `geo_NMI`.
2. Re-train the same architecture at z=32 and z=128 with identical
   hyperparameters except latent dim.
3. Report a 3-row sub-table (z=32, 64, 128) with the same eval contract as
   Round 1.

**Decision rule for the report:** the winner's `geo_NMI` ranking across the
z-sweep is the sole headline — sufficient to demonstrate that we explored
latent-dim sensitivity without inflating the run count.

If the Round-1 winner happens to be `vae_z64`, that triggers `vae_z32` and
`vae_z128` (the only branch where the VAE z-grid is run).

---

## 5. Explicit scope cuts

The following rows from the original 21-22 run matrix are intentionally
**not** trained, and the final report acknowledges each as future work:

| Skipped | Rationale | Future-work phrasing |
|---|---|---|
| `ae_z32`, `ae_z128` | Covered by winner z-sweep | "Z-dim sensitivity is studied on the winning architecture in Round 2; AE-specific z-grid is left to future work." |
| `ae_z64_no_text` (F1) | Modality ablation | "Modality-contribution ablations (text, director) deferred — Phase 1 pretext and the multi-modal vs vanilla comparison together suffice for the architectural claim." |
| `ae_z64_no_director` (F2) | Modality ablation | (same as above) |
| `ae_z64_w4` (Kendall learned weighting) | Marginal expected gain over W2 | "W2 already validated at +99% (genre) and +277% (lang) over W1; W4 stretch deferred." |
| `dec_z32_*`, `dec_z64_k10/30`, `dec_z128_*` | k-grid | "k=21 won at z=64 in MVP; full k-grid is future work pending the Round-2 winner." |
| `vae_z32`, `vae_z128` | Contingent on VAE winning Round 1 | "VAE z-grid is contingent on Round-1 outcome; this branch was not taken." |
| Linear probing on all z=64 models | Bandwidth-bound | "Linear probing performed on the demo-deployed winner; full probing matrix is future work." |
| All 27 UMAP figures | Report uses ~12 | "Best-of-each-architecture UMAPs are in the main body; the full grid is in supplementary." |

---

## 6. Reporting — how this slots into `final-report.tex`

- **Results §1 — architecture comparison.** The 9-row Round-1 table is the
  centerpiece, sorted by `geo_NMI`. The `geo_NMI` formula is stated in the
  Methodology section adjacent to the per-axis NMI/ARI definitions.
- **Results §2 — z-sensitivity sub-table.** A 3-row table on the winner.
- **Hero figures.** Two UMAPs (best architecture × genre, best × decade) +
  the missing-data sub-manifold finding from MVP (carry-over from
  `umap_dec_z64_k21_decade.png`).
- **Discussion.** The non-uniformity of axis winners across architectures
  motivates the `geo_NMI` composite. The contrastive-pretext + DEC HERO
  row is the empirical payoff of the clustering-improvement sprint.
- **Future work §.** The skip list above, lifted verbatim from §5.

---

## 7. Acceptance

- [ ] Phase 1 sweep complete (wandb group `phase-1-sweep`) and winner selected.
- [ ] `vae_z64`, best-contrastive, and HERO trained; rows added to
      `artifacts/eval/results.json`.
- [ ] Round-1 winner picked by `geo_NMI` and named in `docs/FINDINGS.md`.
- [ ] Round-2 z-sweep (z=32, z=128) of the winner trained.
- [ ] Final-report results tables filled in from
      `artifacts/eval/results.json`.
