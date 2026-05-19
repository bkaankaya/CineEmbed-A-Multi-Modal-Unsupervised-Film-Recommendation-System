# 06 — Negative Results

> Four distinct negative results from the production-modeling work, with
> root-cause analyses. None are apologetic; each is a deliberate contribution
> to the (modest) literature on what NOT to do with multi-modal tabular
> contrastive learning.
>
> Read `00-context-and-goals.md` first.

## §1 Why this file exists

The CineEmbed final report will include a "Negative Findings and Limitations"
section. This file is its primary source. We had four distinct negative
results, each worth its own analysis, each useful as a contribution to the
modest literature on what NOT to do with multi-modal tabular contrastive
learning. Most undergraduate projects gloss over honest negatives; we are
explicitly building them into the report because they were the more
informative half of the empirical work.

Each section gives the original spec's claim, the observed numbers, the
root-cause hypothesis, the falsification path (where applicable), and what
would have to change for the technique to succeed. The "what to do instead"
is deferred to [`08-scope-cuts-future-work.md`](08-scope-cuts-future-work.md);
this file stays in the diagnostic register.

The four items, ordered by depth of insight, not chronology:

| ID | One-line summary |
|---|---|
| ND-1 | Phase 1 contrastive pretext underperformed the +5-12% NMI lift the spec promised. |
| ND-2 | Round 1 AE→DEC fine-tune from contrastive backbones underperformed cold-start MVP DEC. |
| ND-3 | VAE z=64 likely suffered posterior collapse. |
| ND-4 | DEC angular collapse breaks cosine retrieval — the headline negative finding. |

ND-4 is the methodological centrepiece. ND-1 and ND-2 are the same failure
mode at different stages (pretext and fine-tune); they share a root cause
and are presented as a cause-and-falsification pair. ND-3 is the smallest
finding but is included for completeness — the VAE attempt was a single
budgeted run and it failed, and the report needs to say so.

## §2 ND-1: Phase 1 contrastive pretext underperformed the +5-12% NMI lift the spec promised.

### §2.1 The claim from the spec

[`02-clustering-improvements-spec.md`](02-clustering-improvements-spec.md) §2.1
("Contrastive pre-training stage (SimCLR-style, modality-dropout augmentation)")
quotes the 2024-2025 deep-clustering literature (TCSS, SCAN family, sgSDC)
as the basis for an expected payoff: "5-12% NMI lift after 30-60 epochs of
pretext, before any AE/DEC fine-tune". The claim is grounded in published
SimCLR-derived methods that demonstrated this magnitude of lift on related
self-supervised pre-training pipelines.

The hyperparameters chosen, per §2.1:

- `temperature ∈ {0.1, 0.5}` (sweep both; default 0.1 after the 2026-05-16 amendment)
- `projection_dim = 128` (2× latent_dim, SimCLR rule of thumb)
- `drop_prob = 0.3` per modality (each view drops ~2 of 7 modalities on expectation)
- `batch_size = 1024`
- per-row block masking (each row in a batch gets independent block masks)

### §2.2 What we observed

All three Phase 1 configs ended **below** the MVP `ae_z64` baseline on `geo_NMI`.
Numbers from [`10-results-table.md`](10-results-table.md) Phase-1 row:

| Run | tau | drop_prob | gNMI | dNMI | lNMI | geo_NMI |
|---|---:|---:|---:|---:|---:|---:|
| `contrastive_tau0p5_drop0p3` | 0.5 | 0.3 | 0.216 | 0.218 | 0.374 | 0.260 |
| `contrastive_tau0p1_drop0p3` | 0.1 | 0.3 | 0.216 | 0.286 | 0.174 | 0.221 |
| `contrastive_tau0p1_drop0p4` | 0.1 | 0.4 | 0.150 | 0.243 | 0.189 | 0.190 |
| MVP baseline `ae_z64` | — | — | 0.328 | 0.341 | 0.264 | 0.309 |
| MVP baseline `dec_z64_k21` | — | — | 0.332 | 0.342 | 0.294 | 0.323 |

Focus on the best-of-Phase-1 config by gNMI, `contrastive_tau0p1_drop0p3`:
gNMI = 0.216 vs `ae_z64` 0.328 — a **−34% gap**, **not** the promised
+5-12% lift. By `geo_NMI` the Phase 1 winner is `contrastive_tau0p5_drop0p3`
at 0.260, still −16% below `ae_z64`'s 0.309. There is no Phase 1 config in
the sweep that beat the MVP baseline on either axis.

### §2.3 Root-cause hypothesis (validated downstream by ND-2)

The SimCLR-style modality-dropout augmentation creates encoder invariance to
dropped block content. This is the explicit intent of contrastive learning:
two augmented views of the same row must map to nearby latents, regardless
of which modalities were dropped. The encoder therefore learns to ignore
which subset of modalities is present.

Because the genre block (22 dims) is itself a modality block subject to
dropout, the encoder learned **genre-invariant** representations — the
opposite of what the downstream clustering objective on `primary_genre`
needs. This is a structural conflict between the augmentation primitive and
the downstream task.

The relevant literature (Chen et al. 2020, SimCLR) used image-domain
augmentations — random crop, colour jitter, Gaussian blur, horizontal flip —
that do **not** strip the target class signal. Cropping a cat photo still
leaves a cat; jittering colours does not turn a dog into a cat. Modality
dropout on heterogeneous tabular data is fundamentally different: dropping
the genre block strips the target class signal, and the contrastive
objective explicitly rewards the encoder for not caring whether the block
is present.

### §2.4 Secondary observation: tau=0.5 had higher lNMI than tau=0.1

A secondary, weaker observation worth recording: tau=0.5 (looser contrast)
produced higher lNMI (0.374) than tau=0.1 (0.174), even though the broader
deep-clustering literature suggests lower temperatures sharpen the
contrastive objective. Hypothesis: looser contrast lets the encoder lazily
grab strong easy signals — language one-hot is high-cardinality (31 codes)
and stable across views, so a less-discriminative encoder still preserves
language information well. Tighter contrast pushes the encoder to find more
carefully-discriminative representation, at the cost of dropping signal in
the easy-to-encode modalities.

This is **one data point** and we do not overclaim it. It is consistent
with the broader pattern (Phase 1 winner by `geo_NMI` is tau=0.5, driven by
lNMI; Phase 1 best by gNMI is tau=0.1, driven by — well, nothing it could
beat). A two-config sweep at one drop_prob cannot separate "tau effect"
from "drop_prob × tau interaction" or from random seed variance.

### §2.5 Falsifiability

If the pretext encoder genuinely demoted genre, then fine-tuning on the
genre-clustering objective should recover at least partially. The Round 1
DEC fine-tune (HERO RUN, `dec_z64_k21_from_contrastive_t{0p1,0p5}`) is the
falsification test. ND-2 shows that it does not recover; the encoder
remains genre-invariant after fine-tuning. The two failures are coupled —
ND-2 is the empirical confirmation that ND-1's root cause is structural,
not a hyperparameter accident.

## §3 ND-2: Round 1 AE→DEC fine-tune from contrastive backbones underperformed cold-start MVP DEC.

### §3.1 The claim from the two-round spec

[`2026-05-16-two-round-modeling-strategy.md`](../superpowers/specs/2026-05-16-two-round-modeling-strategy.md)
§3 ("Round 1 — Architecture comparison @ z=64") names the HERO row in the
9-run grid as `contrastive_pretext + DEC` and frames it as "the natural
payoff of the clustering-improvement spec". Two HERO RUNs were trained:
`dec_z64_k21_from_contrastive_t0p1` (from tau=0.1 pretext) and
`dec_z64_k21_from_contrastive_t0p5` (from tau=0.5 pretext). Each starts
from the corresponding Phase 1 backbone, gets a fresh DEC head, and is
fine-tuned on the standard DEC objective.

The expected payoff (carried forward from the original
clustering-improvement spec): each HERO RUN should beat the MVP cold-start
`dec_z64_k21` (gNMI = 0.332, geo_NMI = 0.323) by 5-12% NMI.

### §3.2 What we observed

Numbers from [`10-results-table.md`](10-results-table.md) Round-1 row:

| Run | gNMI | dNMI | lNMI | geo_NMI | KMeans-on-AE-latent gNMI | DEC-vs-KMeans delta |
|---|---:|---:|---:|---:|---:|---:|
| `dec_z64_k21_from_contrastive_t0p5` | 0.098 | 0.487 | 0.125 | 0.181 | 0.098 | 0.000 |
| `dec_z64_k21_from_contrastive_t0p1` | 0.120 | 0.641 | 0.016 | 0.107 | 0.119 | +0.001 |
| MVP baseline `dec_z64_k21` | 0.332 | 0.342 | 0.294 | 0.323 | — | — |

By `geo_NMI`: the HERO RUNs landed at **0.181 and 0.107**, vs the MVP cold-start
DEC's **0.323** — a **44-67% gap below the baseline**, not above.

By gNMI specifically: 0.098 and 0.120 vs MVP 0.332 — **63-70% below**.

Decade is hyper-aligned: dNMI = 0.641 (t0p1) and 0.487 (t0p5) vs MVP 0.342.
Language is near-zero: lNMI = 0.016 (t0p1) and 0.125 (t0p5) vs MVP 0.294.
The encoder is no longer producing language-coherent latents under either
pretext source; it has over-aligned on decade (the easiest axis, see
[`01-mvp-modeling-phase.md`](01-mvp-modeling-phase.md) §3) and dropped
everything else.

### §3.3 Diagnostic ruling out the DEC head

The Round-1 notebook (`notebooks/07_round1_finetune.ipynb`) ran a follow-on
diagnostic: it computed KMeans-k21 on the AE-latent (i.e., the
contrastive-pretext-initialized backbone after the AE pretext concluded but
**before** the DEC step) and compared it to DEC argmax on the same backbone
after DEC fine-tune.

| Run | DEC argmax gNMI | KMeans-on-AE-latent gNMI | Delta |
|---|---:|---:|---:|
| `dec_z64_k21_from_contrastive_t0p5` | 0.098 | 0.098 | 0.000 |
| `dec_z64_k21_from_contrastive_t0p1` | 0.120 | 0.119 | +0.001 |

The DEC step did **not** poison an otherwise-fine encoder; the encoder was
already at the failing level before DEC saw a single batch. The DEC head's
contribution to gNMI is at the rounding-error floor (≤ 0.001), well below
the noise we see in DEC training across runs. **The contrastive pretext is
the cause of the failure**, not the DEC fine-tune.

### §3.4 Why fine-tune couldn't recover

A plausible mechanism: the early SGD trajectory was dominated by the
contrastive-pretrained features (decade-, language-strong but
genre-invariant). The AE reconstruction loss doesn't penalise a
genre-invariant encoder heavily — the genre block is 22 dims out of 564
input, and W2 inverse-variance weighting clips at `[0.1, 10.0]` so a
poorly-reconstructed sparse multi-label block contributes at most ~10× the
per-dim weight of a baseline block. The genre block also contains an
inherently sparse signal (multi-label 21-dim, mean ~2 active per row);
reconstruction MSE on a near-zero target does not strongly penalise zero
predictions. Without an explicit genre-aware fine-tune objective, the
encoder stays in the basin of attraction the pretext put it in.

This is consistent with the broader observation in transfer-learning
literature that catastrophic-forgetting protection helps weak signals
survive fine-tuning, while task-aligned features dominate. The contrastive
pretext put the encoder in a "modality-invariant" basin; AE reconstruction
fine-tuning, on this data, is too weak a signal to pull it out.

### §3.5 Implication

Contrastive pretext + AE/DEC fine-tune as a recipe does **not** transfer to
genre-clustering on heterogeneous tabular data with modality-dropout
augmentation. To make it work, one of three changes is needed:

1. **Augmentations that DO NOT touch the target modality.** Additive
   Gaussian noise on numerical features only, leaving genre/language/etc.
   intact. This keeps the contrastive objective from making the encoder
   invariant to the very signal the downstream task needs. The drawback is
   that on tabular data with primarily categorical / one-hot blocks, the
   space of label-preserving augmentations is small.

2. **A two-objective pretext: contrastive + supervised genre classification.**
   But then it is no longer self-supervised — it requires labels at
   pretext time, which collapses the motivation for contrastive learning
   on this dataset.

3. **A different augmentation primitive entirely.** TabClusterNet
   (TabNet-style feature masking) or MMCMAE (multi-modal contrastive
   masked AE) use augmentations designed for tabular data, but both are
   out of scope per [`02-clustering-improvements-spec.md`](02-clustering-improvements-spec.md) §1.

## §4 ND-3: VAE z=64 likely posterior collapse.

### §4.1 What we observed

`vae_z64` was added to Round 1 for methodological completeness, per
[`2026-05-16-two-round-modeling-strategy.md`](../superpowers/specs/2026-05-16-two-round-modeling-strategy.md)
§3 ("the final report needs to answer 'did you try VAE?' honestly"). The
run early-stopped at epoch 12 with best val_loss = 0.2727, producing
gNMI = 0.103, dNMI = 0.358, lNMI = 0.056, `geo_NMI` = 0.127 — below every
MVP baseline including the two non-deep ones, and on the same order of
magnitude as `kmeans_raw_k21`. [`10-results-table.md`](10-results-table.md)
Round-1 row.

The val curve plateau at epoch 12 is suspicious given β was only at 0.1 max
(warmed up linearly over 10 epochs). The encoder ought to have had two more
epochs of clean reconstruction-dominated training to converge usefully
before β reached its plateau, and ten more epochs after at the early-stop
patience. Early-stop firing at epoch 12 means the val loss was not
improving consistently after only 2 epochs of full-β training.

### §4.2 Why we suspect posterior collapse

Posterior collapse is a well-known VAE failure mode (Bowman et al. 2015
"Generating Sentences from a Continuous Space", Razavi et al. 2019
"Preventing Posterior Collapse with delta-VAEs"). The KL term in the ELBO,
`KL(q(z|x) || N(0, I))`, can drive the encoder to produce a posterior
indistinguishable from the prior — effectively setting `q(z|x) = N(0, I)`
for every input. This causes:

- The latent encodes nothing about the input.
- The decoder ignores z and reconstructs a mean image / "constant" output.
- Val loss plateaus at a level where the KL term is ~0 (encoder matches
  prior) and the reconstruction is whatever the decoder can do
  prior-free.

Our config: β_max = 0.1, 10-epoch warmup, lr = 1e-3 (same as AE), z = 64.
β = 0.1 is not extreme by VAE standards, but with a 64-dim latent and a
154-dim concat input, the KL term can dominate the reconstruction term once
the warmup completes — particularly because the W2 inverse-variance scheme
in our codebase clips block weights at `[0.1, 10.0]`, which limits how
strongly any individual block can override the KL.

### §4.3 Caveats / uncertainty

We did **not** instrument the recon-vs-KL split per epoch — that would be
the test that distinguishes posterior collapse from other failure modes.
The early-stop at epoch 12 is consistent with posterior collapse but could
also be explained by:

- An over-aggressive learning rate (1e-3 is the AE default; VAEs are
  often more delicate and trained at 1e-4 or 5e-4).
- The β warmup schedule colliding with the early-stop patience: if val_loss
  rises during β warmup (expected — adding KL increases the loss), the
  patience counter accumulates from the wrong moment, and patience-10
  fires before β-warmup-and-recover completes.
- Decoder under-capacity given the 154-dim concat target. The VAE decoder
  in `heads.VAEHead` mirrors the AE decoder shape — adequate for an AE
  but possibly under-capacity when forced to share variance with a KL term.

Mark this hypothesis as **strong but not falsified**. The behaviour is
consistent with posterior collapse and the config is in the regime where
collapse is known to occur, but the diagnostic (per-epoch KL/recon split)
that would confirm it was not in the run logs.

### §4.4 Implication for Round 2

The original two-round spec said "if VAE wins Round 1, do `vae_z32` /
`vae_z128` in Round 2". VAE z=64 did not win — it lost to every MVP
baseline. The VAE z-sweep is therefore explicitly skipped (see
[`08-scope-cuts-future-work.md`](08-scope-cuts-future-work.md) §2.A). VAE
was given one chance and demonstrably failed to outperform AE at z=64; we
will not chase it. The report's "did you try VAE?" answer is now "yes, it
underperformed, and we hypothesise posterior collapse — diagnostic is
future work".

## §5 ND-4: DEC angular collapse breaks cosine retrieval — the headline negative finding.

### §5.1 What we observed

The MVP champion `dec_z64_k21` has the **highest** `geo_NMI` of any model
in the project (0.323) and the **lowest** cosine-retrieval quality among
the demo-candidate backbones (`genre@5` = 0.557, vs `ae_z64`'s 0.714 — a
22-point gap). Numbers from
[`10-results-table.md`](10-results-table.md) Retrieval-eval row and the two
inference manifests `artifacts/inference/{ae_z64,dec_z64_k21}/manifest.json`.

The eyeball top-5 for every query has all five neighbours at cosine ≈ 1.000
— i.e. angularly identical to the query — so the ranking within a query's
cluster is a random tie-break. Concrete examples from
`artifacts/inference/dec_z64_k21/manifest.json`:

- Inception → top-5 cosines: 0.99992, 0.99989, 0.99986, 0.99986, 0.99986.
- The Godfather → 0.99994, 0.99993, 0.99992, 0.99991, 0.99990.
- The Shawshank Redemption → 0.99999, 0.99999, 0.99998, 0.99998, 0.99998.
- The Matrix → 0.99999, 0.99999, 0.99999, 0.99999, 0.99999.

The difference between top-1 and top-5 is at the floating-point precision
floor (≤ 1e-4 in 32-bit). For any DEC cluster (~15-17k films on average),
the cluster is effectively a single point in cosine space.

By contrast, `ae_z64` (lower `geo_NMI` 0.309) has top-5 cosines spanning a
meaningful angular gradient. From `artifacts/inference/ae_z64/manifest.json`:

- Inception → top-5 cosines: 0.969, 0.966, 0.961, 0.952, 0.951.
- The Godfather → 0.964, 0.963, 0.946, 0.942, 0.935.
- Spirited Away → 0.982, 0.968, 0.963, 0.962, 0.958.

These are recognisable angular distances (gap of ~0.02-0.04 between top-1
and top-5), and the order is meaningful — closer neighbours are more
similar.

### §5.2 Root cause

The DEC objective combines reconstruction (with `lambda_recon = 0.1`) and
KL between the soft Student-t cluster assignment `q` and the sharpened
target distribution `p`:

```
q_ij = (1 + ||z_i - μ_j||²)^(-α+1/2) / Σ_j' (1 + ||z_i - μ_j'||²)^(-α+1/2)
p_ij = q_ij² / f_j ;  f_j = Σ_i q_ij  ;  p_ij ← p_ij / Σ_j' p_ij'
L = KL(P || Q) + λ_recon · MSE_recon
```

The KL term explicitly **pulls intra-cluster vectors towards the cluster
centroid** in the q-space. Through the Student-t kernel this manifests as
cosine-1 alignment within a cluster: every z_i in the cluster is pulled
towards μ_j_assigned, and the kernel reward is monotone in `||z_i - μ_j||`.
After 21 epochs of DEC training, the within-cluster spread shrinks to the
floating-point precision floor.

NMI does **not** see this collapse: cluster assignments are
unchanged (each z_i still maps to its argmax cluster), and gNMI/dNMI/lNMI
all measure agreement between cluster ID and label. The clustering metric
NMI is blind to the geometric structure inside each cluster. Cosine
retrieval, on the other hand, **only** measures geometric structure inside
each cluster, and that is exactly what DEC destroyed.

### §5.3 Implication

**Clustering-objective embeddings can poison retrieval.** This is the core
methodological finding of the project. Cosine retrieval needs intra-cluster
angular variance; the DEC objective minimises it. Whenever the downstream
task is **ranking** (not bucketing), AE-style continuous-manifold
embeddings beat DEC-style cluster-aware embeddings — even if DEC has higher
NMI by every standard clustering metric.

### §5.4 What to report

This is paper-worthy. We arrived at it accidentally by adding the
retrieval-task metric to `scripts/build_index.py:236` (the `_retrieval_eval`
function — see [`07-retrieval-vs-nmi-discovery.md`](07-retrieval-vs-nmi-discovery.md))
as a sanity check before deploying the demo backbone. Recommend the final
report devote a section to this finding; the narrative form lives in
[`07-retrieval-vs-nmi-discovery.md`](07-retrieval-vs-nmi-discovery.md) and
should drive the analysis section.

## §6 Summary table

A compact cross-listing. "Falsifying evidence" is the diagnostic that ruled
out alternative explanations.

| ID | Failure | Root cause | Falsifying evidence | What would have to change |
|---|---|---|---|---|
| ND-1 | Phase 1 contrastive pretext lost on geo_NMI: 0.190-0.260 vs MVP `ae_z64` 0.309 (−16-39%). | Modality-dropout augmentation makes encoder invariant to dropped blocks, including the genre block — the target class signal is dropped during pretext. | The downstream HERO RUNs cannot recover the lost signal (ND-2), confirming the failure is in the encoder, not the eval. | Augmentations that do not touch the target modality (e.g. additive Gaussian noise on numerical-only); or two-objective pretext (contrastive + supervised); or a different augmentation primitive (TabClusterNet / MMCMAE). |
| ND-2 | Round 1 HERO RUNs landed at geo_NMI 0.107-0.181 vs MVP `dec_z64_k21` 0.323 (−44-67%). | Encoder remained genre-invariant after AE/DEC fine-tune; the basin-of-attraction from the contrastive pretext was not pulled out by the reconstruction objective. | DEC-vs-KMeans-on-AE-latent delta is ≤ 0.001 on both fine-tunes — proves the DEC step did not poison the encoder; the encoder was already broken. | Genre-aware fine-tune signal (e.g. supervised auxiliary head); or stronger encoder-perturbation regularisation; or stronger augmentation-target alignment at pretext time. |
| ND-3 | VAE z=64 early-stopped at epoch 12, gNMI 0.103, geo_NMI 0.127 — below every MVP baseline. | Likely posterior collapse: at β_max = 0.1 with z = 64 and lr = 1e-3, KL term dominates after warmup; encoder collapses to prior. | Not falsified — per-epoch recon/KL split was not logged. Hypothesis status: strong but not confirmed. | KL-annealing schedule (slower β warmup); free-bits constraint (Kingma et al. 2016); lower lr (1e-4); decoder capacity audit. |
| ND-4 | `dec_z64_k21` has highest geo_NMI (0.323) but lowest genre@5 (0.557); all in-cluster cosines collapse to 1.000. | DEC's KL(P‖Q) explicitly pulls intra-cluster vectors towards cluster centroid; through the Student-t kernel this is cosine-1 alignment. NMI is blind to intra-cluster geometry; retrieval is not. | The angular-spread sanity statistics: `dec_z64_k21` random-pair `cos` std 0.421 (looks healthy by pair-level), but in-cluster top-5 collapse to ≤ 1e-4 spread (collapse only manifests within a cluster). | Replace DEC with a clustering objective that preserves angular variance (e.g. spherical-KMeans + smoothness regulariser); or hold out an AE backbone for retrieval and a DEC backbone for bucketing; or evaluate on retrieval task during training. |

## §7 What we do NOT claim

To avoid overreaching from these four findings:

- We do **not** claim contrastive pretext is universally bad for multi-modal
  data. We claim **modality-dropout-as-augmentation** conflicts with
  downstream tasks targeting the dropped modalities. Different
  augmentation primitives may behave differently.

- We do **not** claim VAE is bad in general. We claim our particular
  config (β_max = 0.1, 10-epoch warmup, lr = 1e-3, z = 64) was prone to
  posterior collapse on this dataset, and we did not have the
  per-epoch KL/recon diagnostic to confirm.

- We do **not** claim DEC is bad in general. We claim **DEC produces
  angular-collapsed embeddings that are unfit for cosine retrieval**.
  DEC remains a fine clustering algorithm by the metric it was designed
  to optimise; the issue is the mismatch between the clustering metric
  (label-set agreement) and the deployed task (cosine ranking).

- We do **not** generalise these results outside heterogeneous
  multi-modal tabular data. SimCLR works fine for natural images (Chen
  et al. 2020 paper). DEC works fine when downstream is bucketing
  (which is what DEC was designed for). The findings here are specific
  to the intersection: heterogeneous tabular + modality-dropout
  augmentation + cosine-retrieval downstream.

## §8 Cross-references

- [`02-clustering-improvements-spec.md`](02-clustering-improvements-spec.md) —
  the spec that made the +5-12% NMI claim that ND-1 falsifies.
- [`04-phase1-contrastive-sweep.md`](04-phase1-contrastive-sweep.md) —
  Phase 1 execution detail. `[verify]` (not yet written).
- [`05-round1-architecture-comparison.md`](05-round1-architecture-comparison.md) —
  Round 1 execution detail. `[verify]` (not yet written).
- [`07-retrieval-vs-nmi-discovery.md`](07-retrieval-vs-nmi-discovery.md) —
  the pivot driven by ND-4.
- [`08-scope-cuts-future-work.md`](08-scope-cuts-future-work.md) —
  what we will not try given these results.
- [`10-results-table.md`](10-results-table.md) — every number cited above
  in source-of-truth form.
- `scripts/build_index.py:236` (`_retrieval_eval`) — the function whose
  output revealed ND-4.
- `notebooks/07_round1_finetune.ipynb` — the notebook with the
  DEC-vs-KMeans diagnostic that confirmed ND-2's root cause.
