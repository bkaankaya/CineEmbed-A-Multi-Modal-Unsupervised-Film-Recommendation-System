# 11 — Metrics deep dive

Every metric used in CineEmbed: what it measures, why we chose it, how it
relates to the project's goals, where in the codebase it's computed, and
where in the journal it appears. Includes the metrics we introduced
mid-project — `geo_NMI` (composite, May 2026-05-09) and `genre@k` (retrieval,
2026-05-17) — and a critical look at the metrics that turned out to be
*wrong proxies* for the demo task.

## Quick-reference inventory

| Metric | Type | Implementation | First appeared | Used for | Notes |
|---|---|---|---:|---|---|
| `genre_nmi` (gNMI) | clustering | `src/cineembed/eval.py:evaluate_run` | MVP | model selection (MVP era) | sklearn `normalized_mutual_info_score` |
| `decade_nmi` (dNMI) | clustering | same | MVP | sanity / "decade is easy" axis | |
| `lang_nmi` (lNMI) | clustering | same | MVP | minority-axis sanity | |
| `genre_ari` (gARI) | clustering | `evaluate_run` | MVP | partition-quality complement to NMI | sklearn `adjusted_rand_score` |
| `decade_ari`, `lang_ari` | clustering | same | MVP | same per axis | |
| `genre_ami` (gAMI) | clustering | `evaluate_run` | spec §2.4 (2026-05-06) | chance-adjusted NMI | sklearn `adjusted_mutual_info_score` |
| `decade_ami`, `lang_ami` | clustering | same | spec §2.4 | same per axis | |
| `per_axis_k.{genre,decade,lang}_nmi_kN` | clustering | `evaluate_run_per_axis_k` | spec §2.2 | axis-matched-k re-eval | `DEFAULT_AXIS_K = {21, 12, 11}` |
| `multilabel_genre_macro_nmi.macro_nmi` | clustering | `multilabel_macro_nmi` | spec §2.5 | per-genre binary NMI, macro avg | uses genre one-hot block directly |
| `geo_NMI` | composite | `wandb_integration.log_eval` + `journal/10-results-table.md` | W&B integration (2026-05-09) | balanced selection metric | (gNMI·dNMI·lNMI)^(1/3) |
| `genre@k` mean/median/std | retrieval | `scripts/build_index.py:_retrieval_eval` | 2026-05-17 | demo-task quality | the metric that picked `ae_z64` |
| Eyeball top-k | retrieval | `scripts/build_index.py:_eyeball_top5` | 2026-05-17 | demo-perception sanity | curated query list |
| Random-pair cosine stats | diagnostic | same | 2026-05-17 | angular-spread / collapse detection | 5 000-pair sample |
| W2 weighted recon | training loss | `losses.weighted_recon_loss` | MVP | AE training objective | inverse-variance, clipped [0.1, 10] |
| G2 director-masked recon | training loss | `losses.director_block_loss` | MVP | director-block has-bio mask | spec §5.2 D4 |
| ELBO recon+βKL | training loss | `losses.vae_elbo` | Round 1 | VAE objective | β warmup over 10 epochs to 0.1 |
| DEC KL(P‖Q) + λ·recon | training loss | `losses.dec_loss` | MVP | DEC clustering loss | λ_recon=0.1 |
| InfoNCE | training loss | `losses.info_nce_loss` | Phase 1 | contrastive pretext | τ default 0.1 (was 0.5) |
| n_reinit | diagnostic | `heads.DECHead.reinit_collapsed_centers` | MVP | cluster-collapse rate per epoch | size_floor=0.001 |
| L2 norm mean (latent) | diagnostic | `scripts/build_index.py` | 2026-05-17 | sanity post-normalisation | should == 1.000 |
| Latent dim-wise std | diagnostic | local re-eval logs | 2026-05-17 | global angular health | dec_z64_k21 ≈ 1.59 |

## §1 Why this file exists

CineEmbed made roughly **seven distinct metric choices**, each one tied to a
specific decision in the project:

1. **NMI as the original clustering objective** — inherited from the modeling
   design spec (D7 in `docs/adr/0001-modeling-hybrid-architecture.md`).
2. **ARI alongside NMI** — partition-similarity belt-and-braces (same ADR).
3. **AMI added** — chance-adjustment for our imbalanced `primary_genre`
   labels (clustering-improvements spec §2.4).
4. **Per-axis-k NMI added** — fix the "decade/lang k-mismatch" bias
   (clustering-improvements spec §2.2).
5. **Multi-label macro-NMI added** — honest treatment of multi-genre films
   (clustering-improvements spec §2.5).
6. **`geo_NMI` introduced** — composite to drive the two-round strategy's
   selection rule away from single-axis over-fitting (W&B integration,
   2026-05-09).
7. **`genre@k` retrieval metric introduced** — when the deployable product
   became a cosine recommender rather than a clustering report
   (2026-05-17, decisive).

The journal documents *what* we measured elsewhere; this file documents
*why* and how the metrics interact with each other. It is the technical
companion to `07-retrieval-vs-nmi-discovery.md` (which captures the
moment we learned NMI was a wrong proxy).

## §2 Clustering metrics — NMI, ARI, AMI

The project's three target axes are three different held-out *partitions* of
the 329 044 films:

- **`primary_genre`** — first-mentioned genre of each film. 21 distinct
  classes (after dropping rare ones), with a heavy long tail. The "real" goal
  axis: clusters should align with this.
- **`decade_bin`** — release decade. ~12 classes (1920s through 2020s plus
  "unknown" for missing release dates). Easy to learn because the input
  features include a `decade_norm` column with high information content.
- **`lang_top10`** — top-10 spoken languages + "other". 11 classes. Easy
  because language is one-hot encoded in the input.

For each axis we report three scalars from a single set of cluster
assignments. All three sit in roughly the same scale (0 = no information /
random alignment, 1 = perfect alignment), but they answer slightly different
questions.

### NMI — Normalised Mutual Information

`NMI(C, L) = 2·I(C; L) / (H(C) + H(L))`

where `I` is mutual information, `H` is entropy. sklearn's
`normalized_mutual_info_score` with default `average_method='arithmetic'`.

- **Range:** [0, 1]. 1 means C and L are deterministic functions of each
  other.
- **Why we chose it:** standard deep-clustering benchmark. The 2024-2025
  literature we surveyed (TCSS, SCAN family, sgSDC) all report NMI as the
  primary headline.
- **Project-specific use:** primary objective for the MVP and the eligibility
  criterion for declaring the deep approach a success (spec D9 success
  criterion: ≥ 10% lift over best non-DL baseline; we cleared this by
  +205% — `dec_z64_k21` gNMI=0.332 vs `kmeans_raw_k21` 0.109).
- **Limitation:** **NMI is upward-biased when the reference partition is
  imbalanced** (Vinh et al. JMLR 2010). Our `primary_genre` has a long tail
  — Drama is ≈ 30% of films, while "TV Movie" and "Foreign" are < 1%. The
  bias inflates NMI on partitions that match large clusters and ignores
  small ones. AMI corrects for this.

### ARI — Adjusted Rand Index

`ARI = (RI − E[RI]) / (max(RI) − E[RI])`

where RI is the fraction of agreeing pairs (same-cluster + different-cluster)
adjusted for chance.

- **Range:** approximately [−1, 1]. 0 = random alignment. Negative = worse
  than random.
- **Why we chose it:** ARI is more conservative than NMI and emphasises the
  *pair-wise* structure rather than the marginal-information overlap.
  Stable companion to NMI.
- **Project-specific use:** sanity-check that an NMI improvement is real and
  not just a rebalancing. Reported alongside NMI in every `evaluate_run`
  call. When NMI ↑ and ARI ↓ together, we suspect an artefact (it didn't
  happen for our deployed runs, but the safeguard was there).

### AMI — Adjusted Mutual Information

`AMI(C, L) = (MI(C, L) − E[MI(C, L)]) / (max(MI) − E[MI(C, L)])`

- **Range:** approximately [−1, 1]. 0 = random alignment.
- **Why we chose it:** specifically because of the long-tailed-genre
  problem. AMI removes the upward bias NMI has for imbalanced partitions.
  It is the "fair" number to report when the reference partition is skewed
  (Vinh et al. 2010, JMLR 11:2837-2854).
- **Project-specific observation:** on our latents AMI sits consistently
  within 0.01-0.02 of NMI. The bias correction is small in absolute terms,
  but the gesture matters for the report — including AMI demonstrates that
  we're aware of NMI's limitation.
- **Where:** added 2026-05-06 in the clustering-improvements spec §2.4,
  landed in commit `8097685` as an additive key in `evaluate_run` output
  (`{short}_ami` keys appear alongside `{short}_nmi` / `{short}_ari`).

### Per-axis-k NMI — fixing the k-mismatch bias

Original eval ran KMeans with `k=21` for *all three* axes. But `decade_bin`
has only ~12 classes and `lang_top10` has 11. Forcing k=21 means the cluster
partition has too many clusters to align cleanly with these axes — a known
NMI penalty.

`evaluate_run_per_axis_k(z, labels, axis_k={'genre': 21, 'decade': 12, 'lang': 11})`
runs KMeans three times, one per axis with axis-matched k, and reports the
NMI/ARI/AMI from each. Output keys: `genre_nmi_k21`, `decade_nmi_k12`,
`lang_nmi_k11`, etc.

- **Why we chose it:** measurement-honesty fix. Doesn't change the model,
  just stops penalising axes with low cardinality.
- **Project-specific finding:** on the MVP `dec_z64_k21` latent the per-axis
  numbers are 0.333 / 0.329 / 0.330 — remarkably *uniform* across axes when
  evaluated at the correct k. This is part of why DEC was the NMI champion
  before the retrieval pivot.
- **Where:** clustering-improvements spec §2.2, commit `8097685`.

### Multi-label macro-NMI — honest treatment of multi-genre films

`primary_genre` collapses every multi-genre film to its first listed genre.
A drama-thriller becomes drama; the thriller signal is discarded. Our 21-dim
genre one-hot block (`genre_*` columns in the feature matrix) preserves the
full multi-label structure.

`multilabel_macro_nmi(cluster_ids, genre_onehot)` computes, for each of the
21 genres, the binary NMI between cluster membership and "is this film
labelled with this genre?", then macro-averages across genres.

- **Range:** [0, 1], same scale as NMI.
- **Why we chose it:** more honest than `primary_genre` NMI because it
  rewards the model for capturing the secondary genres of multi-label films.
- **Project-specific observation:** macro NMI is *similar in absolute value*
  to the primary-genre NMI on our data, which is itself diagnostic — it
  means our latents don't preferentially capture primary over secondary
  genres.
- **Where:** clustering-improvements spec §2.5, commit `8097685`.

## §3 Composite metric — `geo_NMI`

After the W&B integration landed, we needed a single scalar to *order* runs
on the dashboard for the two-round selection rule. The candidates were:

1. **gNMI alone** — single-axis. Allows over-fitting one axis.
2. **mean(gNMI, dNMI, lNMI)** — sum / 3. Linear; high one-axis score can
   mask low others.
3. **min(gNMI, dNMI, lNMI)** — pessimistic. Punishes any one weak axis.
4. **(gNMI · dNMI · lNMI)^(1/3)** — geometric mean. Drops to 0 if any axis
   is 0; less extreme than min.

We chose option 4: **`geo_NMI = (gNMI · dNMI · lNMI)^(1/3)`**.

- **Why this shape:**
  - Geometric mean is the natural composition operator for ratios — and NMI
    is a ratio of information quantities. Adding NMI values across axes is
    not principled; multiplying them and taking a root is.
  - Sensitive enough: if a model achieves great gNMI but lNMI=0.05 (a
    Phase 1 contrastive pattern), the (0.3·0.3·0.05)^(1/3) ≈ 0.17 reflects
    the imbalance.
  - Not as harsh as `min` — a model that's strong on two axes and weak on
    one doesn't get zeroed.
- **Where it's computed:**
  - `src/cineembed/wandb_integration.py:log_eval` auto-computes it when all
    three of `genre_nmi`, `decade_nmi`, `lang_nmi` are present, and pushes
    it to wandb as `{prefix}geo_nmi`.
  - `docs/journal/10-results-table.md` includes it for every run.
  - `scripts/build_index.py` and `notebooks/07_round1_finetune.ipynb`
    surface it as `headline_*_geo_nmi` keys in the wandb run summary.
- **Project-specific use:** locked as the selection metric in the two-round
  strategy spec. Worked correctly: Round 1's "winner" by geo_NMI would have
  been `dec_z64_k21_from_contrastive_t0p5` (0.181) — but this was negative
  evidence (everything underperformed MVP), and the demo backbone choice
  pivoted to `ae_z64` via retrieval-task evaluation (§5).

## §4 Training losses — the metrics that *drive* the latent

The losses are not "metrics" in the sense of reported KPIs, but they shape
every reported number. Each one is a per-block-or-per-row scalar averaged
over the minibatch, optimised by gradient descent.

### W2 inverse-variance reconstruction loss

`L_recon = Σ_b w_b · MSE(decoded_b, target_b)`

where `w_b = clip(1 / total_variance(block_b), 0.1, 10.0)`.

- **Why this shape:** without weighting, the high-variance text block (384
  dims of float-typed sentence embedding) would dominate the loss; the
  binary genre block would be near-invisible.
- **Project-specific observation:** W2 beats W1 (uniform weights) by ~2×
  on gNMI (ae_z64 0.328 vs ae_z64_w1 0.165). Decisive evidence for the
  per-block re-weighting choice.
- **Where:** `losses.weighted_recon_loss`. AE / VAE / DEC all use it.

### G2 director-block masked loss

`L_dir = 0.5 · w_dir · (loss_bio + loss_other)`

where `loss_bio` zeroes out the 64 PCA dims for rows with `has_bio=0`.

- **Why this shape:** the director block has a "missing-data sub-manifold"
  — films without a director biography (10 458 of 329 044, ~3%) would
  otherwise be penalised for failing to reconstruct an all-zero bio vector.
- **Project-specific observation:** without G2, the encoder learns a
  spurious cluster for has_bio=0 films. G2 removes this artefact.
- **Where:** `losses.director_block_loss`, called by `weighted_recon_loss`.

### ELBO with β warmup (VAE)

`L_vae = L_recon + β · KL(N(μ, σ²) ‖ N(0, I))`

with β ramping linearly from 0 to `β_max=0.1` over the first 10 epochs.

- **Why this shape:** standard β-VAE objective. The warmup is a Bowman et
  al. 2015 trick to prevent posterior collapse at the start of training.
- **Project-specific outcome:** the Round 1 `vae_z64` run early-stopped at
  epoch 12 — within 2 epochs of β reaching its max — strongly suggesting
  posterior collapse was happening anyway. Without per-step KL
  instrumentation we can't be certain (see ND-3 in
  `06-negative-results.md`).
- **Where:** `losses.vae_elbo`.

### DEC KL(P‖Q) + reconstruction grounding

`L_dec = KL(P ‖ Q) + λ_recon · L_recon`

where Q is the soft Student-t cluster assignment and P is its sharpened
batch-wise version.

- **Why this shape:** the DEC paper (Xie et al. 2016). The `λ_recon · L_recon`
  term is the spec D10 modification — keeps the encoder from collapsing the
  latent geometry while the KL term pulls vectors towards cluster centres.
- **Project-specific outcome:** the DEC objective produces the highest gNMI
  in the project but the *worst* cosine retrieval — angular collapse (see
  §5 below and `07-retrieval-vs-nmi-discovery.md`). The reconstruction
  grounding wasn't enough to prevent intra-cluster vectors from collapsing
  to the cluster centroid direction.
- **Where:** `losses.dec_loss`, λ_recon=0.1 by default.

### InfoNCE — contrastive pretext

`L_NCE = symmetric_cross_entropy(softmax(z·z^T / τ))` over 2B paired views.

- **Why this shape:** standard SimCLR (Chen et al. 2020). For each row in a
  batch, the augmented partner view is the positive, all other rows are
  negatives.
- **Project-specific knob: temperature τ.**
  - Spec §2.1 originally specified `τ = 0.5` (SimCLR's default for natural
    images).
  - Amendment 2026-05-16 changed the default to `τ = 0.1`. Rationale: in
    heterogeneous tabular data the latent geometry is denser than in image
    embeddings, so a tighter contrastive signal helps the encoder
    discriminate. Phase 1 kept both values in the sweep grid.
  - Phase 1 result: τ=0.1 and τ=0.5 produced **nearly identical gNMI**
    (0.216 both) but very different dNMI/lNMI distribution. τ=0.5 lets the
    encoder grab the language one-hot signal lazily (lNMI=0.374 vs 0.174);
    τ=0.1 spreads the signal more evenly. Neither beat the MVP. The
    amendment was correct in spirit but didn't rescue the approach.
- **Where:** `losses.info_nce_loss`. Used only in
  `scripts/train_contrastive.py` (Phase 1) and the Round 1 pretext-loading
  pipeline. Discarded after pretext per the SimCLR-paper convention.

## §5 Retrieval metrics — `genre@k`, eyeball, angular-spread

The decisive metric set, introduced 2026-05-17 in `scripts/build_index.py`
when the deployed product became a cosine recommender. These metrics
discovered the angular-collapse problem in `dec_z64_k21` — see
`07-retrieval-vs-nmi-discovery.md` for the full narrative.

### `genre@k` — top-k retrieval precision

For a random subset of N query films (default N=500, k=5, seed=42):

1. Compute cosine similarity from the query latent to all 329 044 film
   latents (the latents are L2-normalised, so this is a single matmul).
2. Take the top k+1 by similarity (the first is the query itself).
3. Drop the self-match. The remaining k are the top-k nearest neighbours.
4. Fraction of those k that share the query's `primary_genre`.

Report mean / median / std across the query subset.

- **Why we chose it:** this is what the demo user *experiences*. If genre@5
  is 0.7, three or four of the five "similar films" shown to the user
  share its primary genre — visibly relevant. If it's 0.3, the demo feels
  broken.
- **Project-specific finding:** ae_z64 = 0.714 (median 0.800), dec_z64_k21 =
  0.557 (median 0.600). 28% relative gap, in the opposite direction of
  the NMI ranking — `dec_z64_k21` is the NMI champion but the retrieval
  loser. This finding is the basis of the demo-backbone pivot to `ae_z64`.
- **Where:** `scripts/build_index.py:_retrieval_eval` (lines ≈ 220-265).
  Persisted to each backbone's `artifacts/inference/<run>/manifest.json`.

### Eyeball top-k — human-readable sanity

A curated list of 10 well-known query titles (Inception, The Godfather,
Toy Story, The Shawshank Redemption, Pulp Fiction, The Matrix, Interstellar,
Forrest Gump, The Dark Knight, Spirited Away). For each: fuzzy-match the
title to the dataset, compute top-5 nearest neighbours, print and persist.

- **Why we chose it:** numbers can lie; eyeball doesn't. If "Toy Story" →
  Toy Story 2, WALL·E, Ratatouille, that's *visibly* a competent recommender.
  If it returns five tied-at-cosine-1.000 films from the same large cluster
  in random order, the demo will look broken to any user regardless of
  what genre@5 reports.
- **Project-specific finding:** the eyeball was decisive. The numbers said
  dec_z64_k21 had genre@5=0.557 — not terrible. The eyeball revealed *every*
  top-5 cosine was 1.000 — angular collapse. genre@5 of 0.557 was the
  consequence of random tie-breaks happening to land in the right cluster
  ~56% of the time, not because the model was actually ranking films.
- **Where:** `scripts/build_index.py:_eyeball_top5` (lines ≈ 290-345).
  Persisted to manifest.

### Random-pair cosine distribution — angular-spread diagnostic

Sample 5 000 random pairs of films, compute cosine similarity, report mean /
std / min / max. For a healthy 64-dim sphere distribution the cosine
distribution should be roughly centred at 0 with non-trivial spread.

- **Why we chose it:** if the latent is angularly collapsed, *every* pair
  will have cosine close to 1 and the std will be very low. Conversely if
  the latent is too spread out (random initialization), pair cosines will
  cluster near 0 with low std. The metric flags both pathologies.
- **Project-specific finding:** ae_z64 has random_pair_cos mean=0.303 std=0.299
  range [-0.537, 1.000] — healthy. dec_z64_k21 has mean=0.096 std=0.421 —
  *wider* than ae_z64. The apparent paradox is the angular-collapse
  signature: between random film pairs DEC looks healthy, but *within* any
  single cluster (≈ 16k films) the vectors are angularly identical, so the
  cluster-internal ranking is random tie-break.
- **Where:** `scripts/build_index.py:_retrieval_eval`, embedded inside the
  manifest under `retrieval.angular.{mean, std, min, max}`.

## §6 Diagnostic metrics — the supporting cast

These don't appear in the headline results table but were essential to
diagnosing failures.

- **`n_reinit` per epoch (`heads.DECHead.reinit_collapsed_centers`).** Counts
  the number of cluster centres re-initialised at each epoch's end because
  their assigned-sample count fell below `size_floor=0.001` (0.1% of N).
  This is the spec D10 mitigation against the original DEC paper's
  cluster-collapse failure mode. Persisted in `history['n_reinit']`.
- **Latent L2 norm mean post-normalisation.** `np.linalg.norm(z, axis=1).mean()`
  after L2 normalisation; should be 1.000 to floating-point precision. A
  divergence would mean a bug in `_encode_all`. Always sanity-checked at
  the head of `build_index.py`.
- **Latent dim-wise std.** `z_all.std(axis=0).mean()` and `.min()`. Tells you
  whether the latent has collapsed dimensions. dec_z64_k21 had dim-wise mean
  std ≈ 1.59, min ≈ 1.07 — globally healthy, which was the diagnostic clue
  that the angular collapse was *cluster-local* rather than *global*.
- **Backbone parameter count.** `sum(p.numel() for p in bb.parameters())`.
  Always logged by build_index. 58 780 for the production
  `MultiModalBackbone(block_dims, hidden_dim=128, latent_dim=64,
  proj_dims=DEFAULT_PROJ_DIMS)`. Useful as a checkpoint identity check.
- **Checkpoint SHA-256.** First 64 hex chars in the inference manifest.
  Reproducibility anchor; lets the demo backend confirm the same model that
  generated the embeddings is what's claimed.

## §7 The big lesson — NMI is the wrong proxy for retrieval

This is the single most important methodological finding of the project,
and the reason this file exists.

**Clustering metrics (NMI, ARI, AMI, geo_NMI) measure how well the predicted
cluster IDs agree with held-out labels.** They are computed from a
*partition* of the data, not from the *embedding distances*.

**Retrieval metrics (genre@k, eyeball, angular spread) measure how the
embedding distances rank the neighbours of a query.** They are computed
from the *latent geometry* directly.

These two metric families *can disagree completely*, and on our data they
did. The DEC objective optimises the former at the cost of the latter:
intra-cluster vectors are pulled towards the cluster centroid, which is
exactly the angular-collapse signature.

**Pre-conditions for the disagreement (so future work knows when to watch
for it):**

- The downstream task uses **cosine** or **L2** distance ranking, not
  cluster ID assignment.
- The training objective contains an explicit **cluster-centroid pull**
  (DEC's KL term, k-means style assignments, contrastive losses with
  cluster-aware sampling, etc.).
- The encoder has **enough capacity** to collapse intra-cluster geometry
  without sacrificing inter-cluster separation (any non-trivial deep model
  on heterogeneous data).

**Mitigation:** evaluate the deployed task's metric *directly* in every
sweep, not as a post-hoc sanity check. If the demo is a recommender,
genre@k (or any cosine-based metric) should be in the eval pipeline from
day one.

**Why we didn't catch this earlier:** the project inherited NMI as the
selection metric from the original modeling design spec (2026-05-04), and
the web-app pivot decision (2026-05-16) didn't trigger a metric re-design.
This is itself a process lesson — when the deployed task changes, the
selection metric must change with it.

## §8 What we considered and did NOT use

A short tour of metrics we evaluated and rejected, for the record.

- **F-measure on binary genre classification.** Considered for "is film X
  a Drama?" style probing. Subsumed by `multilabel_macro_nmi` which gives
  per-genre NMI numbers we can then macro-aggregate or break out.
- **Davies-Bouldin / Silhouette score.** Internal cluster-quality metrics
  with no ground truth. Rejected for the deployed task (we have ground
  truth on the three axes) but they would be useful for unlabelled or
  weakly-labelled extensions; flag them as future-work metrics.
- **Recall@k on item-to-item retrieval with synthetic positives.** Would
  require a query-document benchmark which we don't have. The eyeball test
  is the human-judgement version.
- **Coverage / diversity metrics on the recommendations.** Important for
  production recommenders (preventing the "top-5 are all Marvel films"
  failure mode) but out of scope for the demo. Future work.
- **Calibration / probabilistic metrics on the soft cluster assignments.**
  Would be relevant if the demo exposed a probability ("78% confident this
  is similar"). Our demo gives raw cosine which the UI may translate to a
  similarity bar; no calibration argument needed.
- **NDCG or MRR.** Information-retrieval ranking metrics. Suitable
  successors to `genre@k` for a more sophisticated ranking evaluation, but
  they require a graded relevance signal (more relevant > less relevant >
  not relevant) which our `primary_genre` label doesn't provide. Future
  work.

## §9 Recommendations for the future

For the report:

- Lead the methodology section with the **two-family metric framing**
  (clustering metrics + retrieval metrics) so the eventual disagreement is
  pre-stated rather than presented as a surprise.
- Cite **Vinh et al. JMLR 2010** for the imbalance-bias argument that
  motivates AMI.
- Cite **Xie et al. 2016** for the DEC objective and note the angular
  collapse as a known failure mode of cluster-centroid-pulling losses for
  retrieval applications.
- Use **`geo_NMI` for the clustering comparison table** and `genre@5` for
  the retrieval comparison table. Do not merge them.

For an extended version of this project:

- Add **NDCG@k** if the dataset gains a graded relevance label (e.g. via
  user-side click data).
- Instrument **per-epoch KL split** for VAE runs to detect posterior
  collapse before the val curve plateaus.
- Add a **retrieval-task loss** alongside (or in place of) DEC's KL — a
  triplet loss or contrastive sampling against same-genre positives —
  if the deployed task is retrieval-first.
- Add **inter-cluster cosine** as a diagnostic alongside the
  random-pair cosine to disambiguate cluster-internal collapse from
  global collapse.

## §10 Cross-references

- `00-context-and-goals.md` — project definitions, including the metric
  glossary.
- `02-clustering-improvements-spec.md` — the 2026-05-06 spec that
  introduced AMI, per-axis-k, multilabel macro-NMI.
- `03-tooling-wandb-integration.md` — where `geo_NMI` first appeared
  (`log_eval` auto-composes it).
- `07-retrieval-vs-nmi-discovery.md` — the narrative of the metric pivot,
  with the same numbers from a different angle.
- `06-negative-results.md` — ND-4 is the same finding viewed as a failure
  mode of DEC.
- `10-results-table.md` — every metric value for every run.
