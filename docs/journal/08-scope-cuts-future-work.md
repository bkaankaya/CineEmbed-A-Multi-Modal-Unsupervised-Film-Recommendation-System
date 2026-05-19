# 08 — Scope cuts and future work

> Everything we explicitly chose NOT to do, organised by category, with the
> condition under which a future maintainer would revisit each cut. Each
> item is paper-quotable as "future work" without the report having to
> invent justifications on the fly.
>
> Read `00-context-and-goals.md` first. Pair with
> [`06-negative-results.md`](06-negative-results.md) (the empirical basis
> for several cuts) and
> [`07-retrieval-vs-nmi-discovery.md`](07-retrieval-vs-nmi-discovery.md)
> (the metric re-alignment that made several cuts safe).

## §1 Framing

The CineEmbed deadline is **2026-05-20** and the deliverable is a working
web-app demo plus a half-academic report. Both of those constrain scope.
Many of the 21-22 originally-pre-registered runs from
`docs/archive/specs/2026-05-04-modeling-design.md` were not trained; many
of the originally-scoped figures, ablations, and probes were not produced.
This file is a definitive list of what we cut, why, and under what
circumstances a future researcher would revisit each.

Three categories:

- **A. Compute-budget cuts** — would help, but the GPU quota / wall-clock
  doesn't fit in 4 days. Mostly post-deadline future work.
- **B. Evidence-based cuts** — won't pay off based on what we've seen.
  Cutting these is informed by the negative results in
  [`06-negative-results.md`](06-negative-results.md).
- **C. Original-spec exclusions** — already-justified scope exclusions from
  the 2026-05-06 clustering-improvements spec, still excluded.
- **D. Modality-ablation cuts** — defensible-but-deferred F1 / F2 ablations.

For each item: the cut name, the one-sentence "why we cut it", and the
one-sentence "when to revisit".

## §2 Cut list

### §2.A Compute-budget cuts

These are cuts driven by the four-day timeline + free Colab T4 budget.
None are evidence-based "we wouldn't try this" decisions — they're
"we didn't have time" decisions.

#### `ae_z32`, `ae_z128` — Round 2 z-sweep on the AE family

- **Why cut.** Not actually cut — deferred. The Round 2 z-sweep on the AE
  family is currently pending GPU quota. Estimated ~20 min Colab T4 total
  (2 cold-start trainings, no contrastive prereq).
- **When to revisit.** As soon as the Colab budget refreshes. This is the
  one cut that the team would prioritise re-opening before the deadline if
  the schedule allows it; it is "in flight" rather than "future work".
- **Cross-ref.** [`05-round1-architecture-comparison.md`](05-round1-architecture-comparison.md) §8.
  `[verify]` (not yet written).

#### DEC k-grid: `dec_z{32,64,128}_k{10,21,30}` minus the trained row

- **Why cut.** Six DEC runs originally on the books (k = {10, 30} × z = {32,
  64, 128}). After ND-4 (DEC angular collapse breaks cosine retrieval; see
  [`06-negative-results.md`](06-negative-results.md) §5), there is no demo
  motivation to expand the DEC grid — DEC is not the deployed family.
- **When to revisit.** Only if a future researcher wants the academic
  comparison of DEC at multiple (z, k) settings. The report cites this as
  future work but does not promise it. The angular-collapse mechanism (§5
  of `06-negative-results.md`) suggests the trend across the grid is
  likely to be monotone bad for retrieval, so even the academic value
  is bounded.

#### 27 UMAP figures across all main runs × all axes

- **Why cut.** Pure visualisation overhead. The MVP delivered 4 hero UMAPs
  for the intermediate report; the full 9 runs × 3 axes = 27 UMAP figure
  grid was originally scoped for the final report.
- **When to revisit.** Generate on demand. UMAP runtime per figure is
  ~2-3 min on the laptop, so the grid is achievable in an afternoon if
  the report's discussion section needs them. The report will likely
  include 2-3 additional UMAPs (best-of-each-architecture × genre,
  best × decade) and reference the missing-data sub-manifold finding
  carry-over from MVP.

#### Linear-probing matrix across all checkpoints

- **Why cut.** Only the deployed-winner probe is necessary for the demo
  validation. Linear probing of all 9+ checkpoints (logistic regression
  on (z → label) for genre / decade / language) would multiply by ~3
  per checkpoint and provide a "feature-quality" complement to the
  clustering metrics, but is bandwidth-bound for the report tier.
- **When to revisit.** When the report has a probes-vs-clustering
  discussion section.

#### `vae_z32`, `vae_z128` — VAE z-sweep

- **Why cut.** Originally contingent on VAE winning Round 1. VAE z = 64
  lost Round 1 (ND-3, geo_NMI 0.127 vs MVP ae_z64 0.309); no reason to
  spend compute on the z-sweep.
- **When to revisit.** Only if a future researcher addresses the
  posterior-collapse hypothesis (slower β warmup, free-bits, lower lr)
  and re-runs vae_z64 with the recon/KL diagnostic logging.

### §2.B Evidence-based cuts

These are cuts informed by what we already empirically know. A
"future researcher" should not revisit them without first addressing
the underlying empirical finding that motivated the cut.

#### AE z-sweep beyond Round 2 (e.g. `ae_z16`, `ae_z256`)

- **Why cut.** MVP and Phase 1 results suggest z = 64 is already the
  sweet spot — `ae_z64` produced the best `geo_NMI` of the AE family at
  z = 64 (0.309). The Round-2 z-sweep covers z = 32 / 128. Further pushing
  outside the band (z = 16 or z = 256) is unsupported by any signal we've
  seen.
- **When to revisit.** Only with an a-priori reason — e.g., a different
  downstream task (very-low-dim z = 16 for visualisation; very-high-dim
  z = 256 for fine-grained discrimination) — that justifies the
  hyperparameter range.

#### `ae_z64_w4` — Kendall learned uncertainty weighting

- **Why cut.** Stretch goal in MVP (modeling spec D3 / W4). W2
  inverse-variance weights already give the right ordering — `ae_z64`
  beats `ae_z64_w1` by ~2× on geo_NMI (0.309 vs 0.162; see
  [`01-mvp-modeling-phase.md`](01-mvp-modeling-phase.md) §3). Marginal
  lift from W4 is unlikely to justify the implementation complexity for
  the demo.
- **When to revisit.** If a future researcher wants to harden the W2-vs-W4
  claim for an academic paper. The Kendall-Gal 2018 formulation is ~50
  LOC additional in `losses.py`; one extra training run.

#### Contrastive pretext re-exploration with the same augmentation

- **Why cut.** ND-1 / ND-2 make the structural argument that modality
  dropout on the genre block is the root cause of the Phase 1 failure
  (see [`06-negative-results.md`](06-negative-results.md) §2-§3).
  Re-running with different tau / drop_prob is unlikely to fix the
  root cause; the structural conflict between augmentation primitive
  and downstream task remains.
- **When to revisit.** Only after changing the augmentation primitive
  (e.g. additive Gaussian noise on numerical-only, or genre-preserving
  views via a tabular-specific augmentation library). The pretext is
  not abandoned conceptually — it is abandoned in **this particular
  augmentation regime**.

### §2.C Original-spec exclusions (still excluded)

[`02-clustering-improvements-spec.md`](02-clustering-improvements-spec.md) §1
("Out-of-scope") listed three techniques as out-of-scope for the
clustering-improvements sprint. They remain out-of-scope for the
final-report tier.

#### Larger text embeddings (BGE-large, e5-mistral, text-embedding-3-small)

- **Why cut.** Already justified in the clustering-improvements spec
  §1: requires re-running EDA on raw text (several-hour pass); the raw
  text CSV is gitignored.
- **When to revisit.** If the team has cluster compute and the raw text
  CSV is re-materialised. The pipeline supports it; only the upstream
  embedding cost is the blocker.

#### MMCMAE (Multi-Modal Contrastive Masked AE, CVPR 2025)

- **Why cut.** Two-stage progressive pre-training; ~2-week
  reimplementation. Out of scope per clustering-improvements spec §1.
- **When to revisit.** Academic paper extension; not the final-report
  deliverable.

#### TabClusterNet

- **Why cut.** Replaces encoder with TabNet, invalidates all existing
  checkpoints. Out of scope per clustering-improvements spec §1.
- **When to revisit.** A from-scratch follow-up project; not a
  continuation of CineEmbed.

### §2.D Modality-ablation cuts

These are the F1 / F2 modality-removal runs from the original modeling
spec (D6 in `docs/archive/specs/2026-05-04-modeling-design.md`). They
are defensible-but-deferred — paper-worthy ablations that the demo does
not require.

#### `ae_z64_no_text` (F1 — text modality removed)

- **Why cut.** Would quantify the text-block contribution. The demo
  backbone is already selected; ablation doesn't change the demo
  backbone choice. The MVP-era vanilla-vs-multimodal comparison
  (`vanilla_ae_z64` 0.215 vs `ae_z64` 0.309 by geo_NMI) already provides
  the strongest architectural-claim evidence; a modality-by-modality
  decomposition is icing.
- **When to revisit.** When the report has space for a modality-ablation
  section, or when extending the project to an academic paper.

#### `ae_z64_no_director` (F2 — director modality removed)

- **Why cut.** Same as F1. The director block was already singled out
  with a special G2 mask (D4) to handle the high `has_director_bio = 0`
  rate (96.8% of films lack a bio); the post-hoc missing-data sub-manifold
  finding ([`01-mvp-modeling-phase.md`](01-mvp-modeling-phase.md) §3) is
  the qualitative version of this ablation. F2 would quantify it.
- **When to revisit.** Same as F1.

## §3 Replacement direction — what we did INSTEAD

The team did pursue compensating directions in the time-budget freed by
these cuts. For the report's "scope discipline" narrative, the substitutions
were:

| Cut item | Replacement direction |
|---|---|
| 27 UMAPs across all axes | `genre@5` retrieval metric + eyeball top-5 dump (see [`07-retrieval-vs-nmi-discovery.md`](07-retrieval-vs-nmi-discovery.md)) |
| DEC k-grid | `dec_z64_k21` re-evaluation on retrieval, revealing ND-4 |
| Contrastive HERO RUN promise (Round 1 winner) | `ae_z64` demo backbone selection on retrieval grounds |
| Linear probing all checkpoints | `_retrieval_eval` + `_eyeball_top5` on the two demo candidates only |
| VAE z-sweep | Documented ND-3 as a strong-but-unfalsified posterior-collapse hypothesis with clear remediation directions |
| `ae_z64_w4` | Reaffirmed W2 result as the project's strongest architectural claim — single ablation (W1) does the work |

These substitutions converted three days of grid-search compute into one
day of evaluation-pipeline work that produced the project's headline
methodological finding. The report's framing should acknowledge that
the project's scope discipline turned what could have been "we ran out
of time" into "we found something better than what was on the list".

## §4 Cross-references

- [`06-negative-results.md`](06-negative-results.md) — the empirical basis
  for the §2.B evidence-based cuts (ND-1 / ND-2 motivate the contrastive
  re-exploration cut; ND-3 motivates the VAE z-sweep cut; ND-4 motivates
  the DEC k-grid cut).
- [`07-retrieval-vs-nmi-discovery.md`](07-retrieval-vs-nmi-discovery.md) —
  the metric re-alignment that made the Round 2 retargeting (from DEC to
  AE family) safe and informed the F1 / F2 deferrals.
- [`02-clustering-improvements-spec.md`](02-clustering-improvements-spec.md) §6
  (alias: [`docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`](../superpowers/specs/2026-05-06-clustering-improvement-techniques.md) §5)
  — the original out-of-scope list that this file extends.
- [`01-mvp-modeling-phase.md`](01-mvp-modeling-phase.md) §4 — the MVP-era
  deferral list that this file consolidates.
- [`2026-05-16-two-round-modeling-strategy.md`](../superpowers/specs/2026-05-16-two-round-modeling-strategy.md) §5
  — the explicit-scope-cuts table from the two-round spec, which this
  file consolidates with the post-Round-1 additions.
