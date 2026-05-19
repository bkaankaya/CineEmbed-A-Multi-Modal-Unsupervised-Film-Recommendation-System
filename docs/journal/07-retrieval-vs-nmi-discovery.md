# 07 — Retrieval vs NMI: the decisive pivot

> The single most important methodological finding of the project: a model's
> clustering NMI does not predict its retrieval quality on the deployed
> task. We discovered this by accident on 2026-05-17 while preparing the
> inference index for the web-app demo, and it overturned the selection
> metric that had been locked in `2026-05-16-two-round-modeling-strategy.md`.
>
> Read `00-context-and-goals.md` first. Pair with
> [`06-negative-results.md`](06-negative-results.md) §5 (ND-4) for the
> failure-mode angle.

## §1 Why this file exists

The CineEmbed final report leans on this finding harder than any other.
The narrative the report needs to make is:

1. We selected the NMI champion (`dec_z64_k21`) as the default demo backbone.
2. We ran a retrieval evaluation as a sanity check before deploying.
3. The retrieval evaluation revealed the NMI champion was **unusable** for
   cosine top-N retrieval because of angular collapse.
4. We selected the second-place backbone (`ae_z64`) instead, on the basis
   of the retrieval metric and an eyeball test.
5. The generalisable lesson — clustering metrics do not predict
   retrieval quality on collapse-prone embeddings — is paper-worthy.

This file is the long-form version of that narrative. It also documents
the change-of-mind in the demo-backbone-selection spec, the
re-targeting of Round 2 from the DEC family to the AE family, and the
process lesson about choosing an eval that matches the deployed task.

## §2 How the retrieval-task metric was introduced

The demo-pivot ADR D14 ([`docs/superpowers/specs/2026-05-16-web-app-demo-design.md`](../superpowers/specs/2026-05-16-web-app-demo-design.md))
specified that the demo would serve cosine-similarity retrieval over the
64-dim L2-normalized latent. While implementing `scripts/build_index.py`
to materialise that latent for the FastAPI backend, we added three
sanity-check outputs to the manifest that were not in the original spec.
None of them were intended as a "model selection" mechanism — they were
preflight checks. They turned into the model selection mechanism.

### §2.1 `_retrieval_eval` — `genre@k` over a random query subset

`scripts/build_index.py:236-296` defines `_retrieval_eval(z_norm, films_df, k, n_queries, seed)`.
The function:

1. Picks a random subset of films (default 500, seed = 42) as queries.
2. Filters out queries that have no `primary_genre` label (the first
   element of the multi-label `genres` list). Effective n_queries is
   typically ~316 of 500 because ~37% of films lack a genre label.
3. For each remaining query, computes `sims = z_norm @ z_norm[qi]` —
   cosine similarity to all 329 044 films (the L2-norm guarantees this
   is exactly cosine).
4. Picks the top-k+1 by `argpartition`, drops the self-match, keeps top k.
5. Computes `genre@k` as the fraction of top-k neighbours sharing the
   query's `primary_genre`.
6. Returns mean / median / std of `genre@k` across the query subset.

The implementation is a straight forward dense matmul on the laptop —
329 044 × 64 = ~21M floats, ~80 MB, BLAS handles it in milliseconds per
query. No FAISS, no annoy, no vector DB needed.

### §2.2 `_eyeball_top5` — well-known query inspection

`scripts/build_index.py:299-352` defines `_eyeball_top5(z_norm, films_df, queries, k)`,
called with the curated list at `scripts/build_index.py:58-69`:

```python
EYEBALL_QUERIES = [
    'Inception',
    'The Godfather',
    'Toy Story',
    'The Shawshank Redemption',
    'Pulp Fiction',
    'The Matrix',
    'Interstellar',
    'Forrest Gump',
    'The Dark Knight',
    'Spirited Away',
]
```

For each query the function does a fuzzy title lookup (exact-match-first,
falling back to substring; ties broken by highest `popularity`), then
computes the top-5 nearest neighbours and prints / persists each
neighbour's title, year, genres, and cosine similarity. The output
appears both on stdout (for inspection during the build) and in
`manifest.json["eyeball"]` (for permanent reproducibility).

The choice of queries reflects three coverage axes:

- **Genre breadth**: action/sci-fi (Inception, Matrix, Dark Knight,
  Interstellar), drama (Godfather, Shawshank, Forrest Gump),
  animation/family (Toy Story, Spirited Away), Tarantino-coded
  (Pulp Fiction).
- **Decade breadth**: 1970s (Godfather), 1990s (Pulp Fiction, Forrest
  Gump, Matrix, Shawshank, Toy Story), 2000s (Spirited Away, Dark Knight),
  2010s (Inception, Interstellar).
- **Director coherence**: Nolan films cluster (Inception, Interstellar,
  Dark Knight). Studio Ghibli expected to cluster (Spirited Away neighbours).
  Pixar expected to cluster (Toy Story neighbours).

The Studio Ghibli case is the cleanest "is this real" test — if Spirited
Away's nearest neighbours include other Ghibli films, the embedding
captured something genuine; if they're random anime, it didn't.

### §2.3 Angular-spread sanity statistics

`scripts/build_index.py:276-296` computes a third sanity check: cosine
similarity between 5 000 random film pairs. The intent is:

- **`random_pair_cos_mean`** — for a healthy 64-dim sphere this should
  be ≈ 0 (random points on a unit sphere are typically orthogonal in
  high dimensions).
- **`random_pair_cos_std`** — the spread; very low std means the latent
  has collapsed (all vectors point in the same direction).
- **`random_pair_cos_min/max`** — extremes; the max can hit ≈ 1 if
  there are near-duplicate films, the min can hit ≈ -1 in a healthy
  symmetric distribution.

This statistic catches the most obvious failure mode (all vectors
point at one direction), but it does **not** catch the more subtle
failure mode that DEC produces (vectors are well-spread *between*
clusters but collapsed *within* clusters). See §5.

All three outputs land in `<out>/manifest.json` alongside the
checkpoint SHA-256 (32 hex chars, line 442 of build_index.py), so the
eval is reproducible from a single file.

## §3 The two manifests side-by-side

Both `ae_z64.pt` and `dec_z64_k21.pt` were processed through
`scripts/build_index.py` on 2026-05-17 with identical flags
(`--retrieval-eval --eyeball`). The summary numbers from each manifest:

### `artifacts/inference/ae_z64/manifest.json`

```json
"retrieval": {
  "k": 5,
  "n_queries": 316,
  "genre_at_k_mean":   0.7139240506329114,
  "genre_at_k_median": 0.8,
  "genre_at_k_std":    0.3492248839188716,
  "angular": {
    "random_pair_cos_mean":  0.3032774031162262,
    "random_pair_cos_std":   0.2985352873802185,
    "random_pair_cos_min": -0.537011981010437,
    "random_pair_cos_max":  0.9998890161514282
  }
}
```

### `artifacts/inference/dec_z64_k21/manifest.json`

```json
"retrieval": {
  "k": 5,
  "n_queries": 316,
  "genre_at_k_mean":   0.5569620253164557,
  "genre_at_k_median": 0.6,
  "genre_at_k_std":    0.3737201092755255,
  "angular": {
    "random_pair_cos_mean":  0.0961848646402359,
    "random_pair_cos_std":   0.4214240312576294,
    "random_pair_cos_min": -0.7553672194480896,
    "random_pair_cos_max":  0.9999998807907104
  }
}
```

### Head-to-head

| Statistic | `ae_z64` | `dec_z64_k21` | Note |
|---|---:|---:|---|
| `genre@5` mean | **0.714** | 0.557 | Retrieval-task quality — AE +28%. |
| `genre@5` median | **0.800** | 0.600 | Same. |
| `genre@5` std | 0.349 | 0.374 | Spread similar. |
| `random_pair_cos_mean` | 0.303 | **0.096** | Closer-to-zero is "healthier" by this stat — DEC looks better. |
| `random_pair_cos_std` | 0.299 | **0.421** | Wider spread — DEC looks better. |
| `random_pair_cos_max` | 0.99989 | **0.99999** | DEC has tighter near-duplicates. |
| In-cluster top-5 cosines (eyeball) | 0.93 – 0.99 | **1.000 (collapsed)** | Where DEC actually fails. |

Here is the paradox: by **pair-level** angular spread, `dec_z64_k21`
looks healthier — mean closer to 0, wider std. Between random films, the
DEC latent is more spread out than the AE latent. The collapse is **not
a global collapse** — it is a **within-cluster collapse**. Random pairs
mostly land in different clusters and look healthy; but **inside any
single cluster** (≈ 15-17k films) the vectors are angularly
indistinguishable to ~1e-4 precision.

The retrieval task, by construction, operates **inside** a cluster (top-N
neighbours of a query are usually in the same cluster). So the pair-level
sanity statistic systematically under-detects the failure mode that
matters for retrieval. **The eyeball test was the diagnostic that caught
the actual issue.**

## §4 The eyeball comparison

The most instructive cases from the two manifests. Cosine similarities
quoted verbatim from the JSON.

### §4.1 Inception (2010)

`ae_z64` returns a coherent Nolan + sci-fi-blockbuster grouping:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | Interstellar | 2014 | 0.969 |
| 2 | Avengers: Age of Ultron | 2015 | 0.966 |
| 3 | The Avengers | 2012 | 0.961 |
| 4 | The Dark Knight Rises | 2012 | 0.952 |
| 5 | Dunkirk | 2017 | 0.951 |

The 0.969 → 0.951 gradient is real and the ordering is meaningful — Nolan
films (Interstellar, Dark Knight Rises, Dunkirk) anchor the list along
with two Marvel blockbusters; an embedding-quality eyeball "pass".

`dec_z64_k21` returns:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | The Dark Knight Rises | 2012 | 0.99992 |
| 2 | Avengers: Age of Ultron | 2015 | 0.99989 |
| 3 | Guardians of the Galaxy | 2014 | 0.99986 |
| 4 | The Avengers | 2012 | 0.99986 |
| 5 | Star Wars: The Force Awakens | 2015 | 0.99986 |

Same cluster (modern action-blockbuster) but **all at cosine = 1.000**. The
order is a random tie-break — top-2 through top-5 differ by ≤ 0.00003,
which is below the floating-point precision floor at fp32. The ranking is
not meaningful; another DEC inference run with a different RNG state would
return them in a different order.

### §4.2 Toy Story (1995)

`ae_z64`:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | Toy Story 2 | 1999 | 0.986 |
| 2 | WALL·E | 2008 | 0.978 |
| 3 | Ratatouille | 2007 | 0.974 |
| 4 | Spaceballs | 1987 | 0.956 |
| 5 | Mary and Max | 2009 | 0.956 |

Pixar coherence: Toy Story 2, WALL·E, Ratatouille is exactly the cluster a
human would name. Spaceballs and Mary and Max are looser but defensible
(family / animation). Strong eyeball pass.

`dec_z64_k21`:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | Toy Story 2 | 1999 | 0.99994 |
| 2 | Mary and Max | 2009 | 0.99992 |
| 3 | WALL·E | 2008 | 0.99991 |
| 4 | **On the Waterfront** | **1954** | 0.99989 |
| 5 | Ratatouille | 2007 | 0.99989 |

On the Waterfront (1954, Crime/Drama, Elia Kazan) is in the top-5 for Toy
Story. The cluster DEC put Toy Story into contains On the Waterfront, and
the tied cosines mean it surfaces above Ratatouille as the 4th-place
neighbour. This is incoherent. This is the kind of result that would
embarrass the demo if a course evaluator queried Toy Story and got On
the Waterfront as a "similar" film.

### §4.3 Spirited Away (2001)

The clearest "this is real" case.

`ae_z64`:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | Princess Mononoke | 1997 | 0.982 |
| 2 | Nausicaä of the Valley of the Wind | 1984 | 0.968 |
| 3 | Pom Poko | 1994 | 0.963 |
| 4 | Battle Royale | 2000 | 0.962 |
| 5 | Kiki's Delivery Service | 1989 | 0.958 |

**Four of five neighbours are Studio Ghibli films** (Mononoke, Nausicaä,
Pom Poko, Kiki). The fifth (Battle Royale) is a 2000 Japanese film, which
is still semantically reasonable as a Japanese-cinema neighbour. The
embedding has clearly captured something real — director (Miyazaki),
studio (Ghibli), country (Japan), animation style.

`dec_z64_k21`:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | Cowboy Bebop: The Movie | 2001 | 0.99994 |
| 2 | Princess Mononoke | 1997 | 0.99993 |
| 3 | Ghost in the Shell | 1995 | 0.99992 |
| 4 | Big Nothing | 2006 | 0.99992 |
| 5 | Dragon Ball Z: Broly | 1993 | 0.99992 |

This is now an "anime films" cluster, with the Ghibli signal substantially
weaker — only Princess Mononoke (top-2) is a clear Ghibli match. Cowboy
Bebop, Ghost in the Shell, Dragon Ball Z are anime but not Ghibli; Big
Nothing (2006, British crime comedy) appears to be a tied-cosine accident.

### §4.4 The Shawshank Redemption (1994)

The "drama at a prison" semantic check.

`ae_z64`:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | The Silence of the Lambs | 1991 | 0.987 |
| 2 | The Usual Suspects | 1995 | 0.986 |
| 3 | American History X | 1998 | 0.986 |
| 4 | Boyz n the Hood | 1991 | 0.984 |
| 5 | Reservoir Dogs | 1992 | 0.984 |

90s crime / dark-drama: Silence of the Lambs, Usual Suspects, American
History X, Boyz n the Hood, Reservoir Dogs. Semantically coherent.

`dec_z64_k21`:

| # | Title | Year | Cosine |
|---:|---|---:|---:|
| 1 | Scent of a Woman | 1992 | 0.99999 |
| 2 | Elizabeth | 1998 | 0.99999 |
| 3 | The Lion King | 1994 | 0.99998 |
| 4 | Beauty and the Beast | 1991 | 0.99998 |
| 5 | American History X | 1998 | 0.99998 |

The Lion King (1994 Disney animation) and Beauty and the Beast (1991
Disney animation) appear in the top-5 for The Shawshank Redemption (1994
prison drama). This is wrong in a way no human would tolerate. The DEC
cluster that contains Shawshank also contains Lion King and Beauty and
the Beast — likely because the cluster encodes "released early 1990s,
high vote-count, broad appeal" rather than genre-coherent semantics — and
the tied cosines surface them as top-5.

## §5 Why angular collapse happens in DEC

Mechanically (re-stated from [`06-negative-results.md`](06-negative-results.md) §5.2):
DEC minimises

```
L_DEC = KL(P || Q) + λ_recon · MSE_recon
```

where Q is the soft Student-t cluster assignment:

```
q_ij = (1 + ||z_i - μ_j||²)^(-α+1/2)  / Σ_j' (1 + ||z_i - μ_j'||²)^(-α+1/2)
```

and P is the sharpened target distribution `p_ij = q_ij² / f_j` (with row
re-normalisation). Both Q and P are functions of `||z_i - μ_j||` only,
and both reward small intra-cluster distances. After enough epochs, the
within-cluster Euclidean (and therefore angular) distances shrink to
near-zero — exactly the gradient signal the loss provides.

This is **a feature for clustering** (purity / NMI / ARI go up because
cluster ID becomes perfectly aligned with the centroid's region of
attraction) and **a bug for retrieval** (no ranking signal remains
inside a cluster — every z_i ≈ μ_j_assigned).

It is invisible to clustering metrics. NMI/ARI work on cluster ID, not
on embedding distance. A clustering algorithm that maps every cluster to
a single point gives the same NMI as a clustering algorithm that
preserves intra-cluster geometry, **as long as the cluster assignments
are the same**.

`lambda_recon = 0.1` is too weak to counter the KL signal at this point.
A higher recon weight would push back, but at the cost of cluster
purity — there is no free lunch.

## §6 Why AE doesn't have this problem

AE's reconstruction objective acts as an implicit regulariser:

```
L_AE = Σ_b w_b · MSE(decoder_b(z_i), x_i,b)
```

Every input row `x_i` must be uniquely reconstructable from its `z_i`, so
distinct rows must have distinct `z_i` (up to the decoder's representational
capacity). The latent geometry stays smooth — there is no force pulling
in-cluster vectors towards a centroid, because there is no centroid in
the objective.

The W2 inverse-variance weighting (D3) keeps low-variance blocks
(language one-hot, decade, genre) contributing meaningfully to the
reconstruction signal, so each modality leaves its angular trace in z.

The resulting top-5 cosine gradient (0.93-0.99 range) is what cosine
retrieval needs.

## §7 The decision and its consequences

### §7.1 Demo backbone selected

**Demo backbone is `artifacts/models/ae_z64.pt`.** SHA-256[:32] =
`e7326ef3d6f6988a6ebd7e6502c9f0c4` (verified by build_index.py at the
inference build).

This is a deliberate departure from the locked selection metric in
[`2026-05-16-two-round-modeling-strategy.md`](../superpowers/specs/2026-05-16-two-round-modeling-strategy.md)
§2 ("Round-1 architectures are ranked by the geometric mean of NMI across
the three label axes"). The new evidence (genre@5 + eyeball) justifies
overriding the prior selection rule. The web-app spec was amended on
2026-05-17 to capture this — see
[`2026-05-16-web-app-demo-design.md`](../superpowers/specs/2026-05-16-web-app-demo-design.md)
"Amendment — 2026-05-17 — Demo backbone selected".

### §7.2 Round 2 retargeted

The two-round spec was also amended on 2026-05-17 — see
[`2026-05-16-two-round-modeling-strategy.md`](../superpowers/specs/2026-05-16-two-round-modeling-strategy.md)
"Amendment — 2026-05-17 — Round 1 outcomes + Round 2 retargeting". Round
2 originally ran on the geo_NMI winner; with the demo-backbone selection
now made on retrieval grounds, Round 2 runs `ae_z32` and `ae_z128`
(cold-start, no contrastive prereq) — the AE family, not the DEC family.

This is a strict simplification. Original Round 2 would have required
either fresh contrastive pretext at z = 32 / 128 (~50 min Colab plus
pretext-fragility risk; given Phase 1 already failed, the risk of
repeating that failure at z = 32 / 128 is high) or a VAE z-sweep (only
if VAE had won Round 1, which it did not). Cold-start AE z-sweep is
~20 min Colab.

### §7.3 Compute halved

Round 2 wall-clock estimate dropped from ~50 min to ~20 min on free
Colab T4. The team's T4-pod-day budget is now well within reach
without further pivots.

## §8 The general lesson

**Choose the evaluation metric that matches the deployed task.**

For a recommender (cosine ranking), NMI was the wrong proxy. The
project inherited the clustering metric from the original modeling
design spec, which was written before the demo pivot. The demo pivot on
2026-05-16 [`docs/superpowers/specs/2026-05-16-web-app-demo-design.md`](../superpowers/specs/2026-05-16-web-app-demo-design.md)
should have triggered a metric re-design at the same time — but it
didn't. We continued to select Round 1's winner by `geo_NMI` even
though the **deployed** task was cosine retrieval. We discovered the
mismatch only by accident on 2026-05-17 when adding the retrieval
sanity-check.

Flag this as a **process lesson, not just a methodological one**: when
the project's deliverable changes, audit the evaluation metric for
alignment with the new deliverable. The "select by `geo_NMI`" decision
in the two-round spec should have been re-opened the moment the demo
pivot was decided. It wasn't, and we paid for it by nearly deploying
the wrong backbone.

The genuine methodological insight is the **mechanism** — clustering
metrics are blind to intra-cluster geometry, retrieval is not, and
embeddings produced by clustering objectives are systematically
collapse-prone in the way that breaks retrieval. This is the part the
report should lean on.

## §9 What the final report should say

A draft paragraph for the final report's "Methodology" or "Discussion"
section:

> Standard clustering evaluation (NMI / ARI against held-out axis
> labels) is the wrong proxy for recommender quality. We empirically
> observed that the highest-NMI model in our sweep — a Deep Embedded
> Clustering checkpoint at z = 64 — suffered intra-cluster angular
> collapse (all top-5 cosine similarities saturated at 1.000), rendering
> it unsuitable for cosine-based retrieval. A retrieval-task metric
> (`genre@5`) and a manual eyeball test on well-known queries are the
> right complement to clustering metrics when the deployed task is
> ranking rather than bucketing. The model finally deployed is the
> auto-encoder backbone `ae_z64` (genre@5 = 0.714, median 0.800), not
> the NMI champion `dec_z64_k21` (genre@5 = 0.557, median 0.600,
> intra-cluster cosines collapsed to 1.000).

A draft paragraph for the report's "Limitations / Lessons" section:

> The clustering metric was inherited from an earlier project framing
> in which the deliverable was a clustering analysis, not a recommender
> demo. The deliverable changed on 2026-05-16 to a web-app demo running
> cosine retrieval, but the selection metric was not re-aligned. The
> mismatch was caught by a routine sanity-check during the inference
> build; in a less-instrumented pipeline it would have shipped. We
> recommend that future projects audit their evaluation pipeline as
> part of any scope change.

## §10 Cross-references

- [`06-negative-results.md`](06-negative-results.md) §5 (ND-4) — the
  same finding presented in the failure-mode register.
- [`05-round1-architecture-comparison.md`](05-round1-architecture-comparison.md) —
  the Round-1 outcome that led to revisiting selection. `[verify]` (not yet written).
- [`08-scope-cuts-future-work.md`](08-scope-cuts-future-work.md) —
  the revised Round 2 plan.
- [`09-operational-incidents.md`](09-operational-incidents.md) —
  GMM singular-cov on dec_z64_k21 latent is another symptom of the same
  angular collapse. `[verify]` (not yet written).
- [`10-results-table.md`](10-results-table.md) — retrieval-eval row at
  the bottom, with the side-by-side comparison.
- `scripts/build_index.py` — the source of the metric, particularly
  `_retrieval_eval` at line 236, `_eyeball_top5` at line 299, the
  `EYEBALL_QUERIES` list at line 58, and the manifest schema at
  line 433.
- `artifacts/inference/ae_z64/manifest.json` — the AE retrieval-eval ground truth.
- `artifacts/inference/dec_z64_k21/manifest.json` — the DEC retrieval-eval
  ground truth.
- [`2026-05-16-web-app-demo-design.md`](../superpowers/specs/2026-05-16-web-app-demo-design.md)
  — Amendment 2026-05-17 captures the demo-backbone selection on
  retrieval grounds.
- [`2026-05-16-two-round-modeling-strategy.md`](../superpowers/specs/2026-05-16-two-round-modeling-strategy.md)
  — Amendment 2026-05-17 captures the Round 2 retargeting.
