# ADR 0001 — CineEmbed Modeling Phase: Hybrid Multi-Modal Architecture

**Status:** Active (D1–D15 locked; last updated 2026-05-17)
**Authors:** Baran Dinçoğuz (with Claude)
**Course:** SENG 474 — Deep Learning · TED University · Spring 2026
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya
**Phase:** AE/VAE/DEC modeling (follows EDA `eda_v2.ipynb`)

> Living document. Decisions appended chronologically as the brainstorming
> progresses. Final spec will live at `docs/superpowers/specs/`.

---

## Context

The EDA phase produced a `(329044, 564)` feature matrix with seven modality blocks (parent + extension):

| Block | Dim | Scaling | Coverage / notes |
|---|---:|---|---|
| numerical | 6 | StandardScaler | log-popularity, log-vote_count, runtime_norm, vote_average_norm, has_vote, has_engagement |
| genre | 22 | bypass (one-hot) | top-20 + Unknown + has_genre. has_genre=63.1% |
| language | 31 | bypass (one-hot) | top-30 + 'other'. Coverage near-100% |
| decade | 2 | StandardScaler | decade_norm + has_release_date. has_release_date=92.4% |
| awards | 6 | StandardScaler | log1p of prior_*. Awards merge rate 22% (most films lack any prior win) |
| text_overview | 384 | L2-normalize per row | multilingual sentence embedding |
| director_profile | 113 | mixed (L2 for bio_pca + bypass for one-hot/flags) | bio coverage 3.2%, lang coverage 100% |

Verified empirical observations from `pipeline_version.json` and figures:
- Modality variance ratio (text+bio_pca vs others): **0.046** — severe imbalance
- Bio PCA explained variance: 83.3% (above 75% threshold)
- Bio embeddings cluster meaningfully by country (F3 figure)
- Director-lang vs film-lang off-diagonal mass: ~15-20% (modest non-redundancy, F4 figure)
- Feature matrix MD5: `e99cee84b6891ea352a7b44d5d7d0ee4` (single-run reproducibility unverified)

---

## Decision Log (chronological)

### 2026-05-04 D1 — Modeling scope: Hybrid (C-structure × D-architecture)

**Decision:** Build an empirical comparison of three models (AE, VAE, DEC) where all three share a multi-modal encoder backbone. The shared backbone has modality-specific projection layers feeding into a fused latent space.

**Alternatives considered:**

| Option | Description | Verdict |
|---|---|---|
| A | DEC only (direct clustering) | Rejected — too thin for "deep learning" course |
| B | AE → DEC sequential | Rejected — doesn't address modality imbalance; misses VAE pedagogy |
| C | AE + VAE + DEC vanilla comparison | Rejected — predictable, ignores actual data structure (modality variance 0.046) |
| D | Multi-modal AE + DEC, single model | Rejected — no controlled ablation, riskier for 4-6 week timeline |
| **Hybrid (chosen)** | C's comparison structure with D's multi-modal architecture | Accepted — addresses modality imbalance at architecture level AND provides empirical comparison framework |

**Rationale:**
1. The modality variance ratio (0.046) is THE central data problem. C ignores it; D solves it but offers no comparison; hybrid solves it AND compares.
2. SENG 474 grading favors: clear story (data → architecture → empirical results), rigorous comparison, working implementation.
3. 3-person team: backbone implemented by 1 person, three model heads by 3 people in parallel.
4. Risk mitigation: vanilla "single FC encoder" fallback always available if multi-modal backbone fails.

**Architecture sketch:**

```
                    Modality-specific projection layers
                  ┌──────────────────────────────────────┐
   numerical (6)   ──→ FC(6→16)    ──┐
   genre (22)      ──→ FC(22→16)   ──┤
   language (31)   ──→ FC(31→16)   ──┤
   decade (2)      ──→ FC(2→4)     ──┼──→ concat (~196)
   awards (6)      ──→ FC(6→16)    ──┤        │
   text (384)      ──→ FC(384→64)  ──┤        ▼
   director (113)  ──→ FC(113→32)  ──┘    FC backbone
                  └──────────────────────────────────────┘    │
                                                              ▼
                                                          latent z
                                                              │
                              ┌───────────────┬───────────────┤
                              ▼               ▼               ▼
                          AE head         VAE head        DEC head
                          (decoder        (μ,σ →          (cluster
                          + MSE)         reparam +        centroids
                                         decoder +        + KL on
                                         ELBO)            soft P)
```

Three heads, one shared backbone. Each model trains the backbone but outputs differently.

**Tradeoffs accepted:**
- Backbone failure cascades to all 3 heads → mitigated by vanilla fallback
- More moving parts than C → but parallelizable across team
- Less focused than D → but C-style comparison gives course-grade safety

**Pending decisions:**
- ~~Latent dim z~~ → see D2
- ~~Loss weighting strategy~~ → see D3
- Modality projection sizes (16/16/16/4/16/64/32 — first cut, may tune)
- Training schedule (sequential AE → VAE → DEC, or parallel runs)
- Bio coverage handling (gating vs flag-only)
- Evaluation metrics suite (NMI/ARI vs genre, reconstruction MSE, latent visualization, ablations)
- Ablation matrix (vanilla concat baseline, leave-one-modality-out, etc.)

---

### 2026-05-04 D2 — Latent dim: ablation grid {32, 64, 128}

**Decision:** Train each model (AE, VAE, DEC) at three latent dims: **z ∈ {32, 64, 128}**. 9 main training runs.

**Rationale:**
- 22 genre classes × ~3 dims/class baseline → 64 is the sweet spot, 32 is tight, 128 is loose.
- DEC literature converges around 10-32 for ~10 cluster problems; we have 22 → bias toward 64.
- Free Colab T4 budget: ~15 min/run × 9 runs ≈ 2-3 hours total compute, splittable across two sessions.
- Reproducibility: all 9 runs use seed=42 + fixed deterministic init.

**Compute budget:** Free T4, no Pro required. ~3-5 hours wall-clock total including eval.

---

### 2026-05-04 D3 — Loss weighting: W2 main, W1 ablation, W4 stretch

**Decision:** Three-tier loss-weighting strategy:
1. **Main runs (all 9)**: W2 — block-level inverse-variance weighting.
2. **Ablation (1 run)**: W1 — uniform (no weighting), z=64 AE only, to demonstrate W2's necessity.
3. **Stretch (1 optional run)**: W4 — Kendall et al. 2018 learned uncertainty weighting, z=64 AE. **If W4 yields measurable improvement over W2, highlight specifically in the final report as a state-of-the-art bonus contribution.**

**W2 formula** (main strategy):
```python
# Computed once on training data, fixed during training
block_variances = {b: X_blocks[b].var(axis=0).sum() for b in BLOCK_ORDER}
block_weights = {b: 1.0 / max(v, 1e-6) for b, v in block_variances.items()}
# In training loop:
loss = sum(block_weights[b] * F.mse_loss(decoded[b], input[b]) for b in blocks)
```

**Expected weights** (preliminary, derived from EDA stats):
| Block | Total variance | Weight w_b |
|---|---|---|
| numerical | ~6 | 0.17 |
| genre | ~3 | 0.33 |
| language | ~1 | 1.0 |
| decade | ~2 | 0.5 |
| awards | ~6 | 0.17 |
| text | ~1 (L2-norm) | 1.0 |
| director | ~2-3 | 0.4 |

After weighting, each block contributes ~1.0 to total reconstruction loss → balanced gradient signal.

**Rejected alternatives:**
- W3 (per-feature inverse-std): one-hot rare classes have std≈0 → divide-by-zero risks, gain not worth complexity.
- W5 (block-norm matching): isomorphic to W2 in practice; W2 simpler.

**Total run count: 9 main + 1 ablation + 1 optional stretch = 10-11 runs.**

---

### 2026-05-04 D4 — Bio coverage: G2 (masked loss on bio_pca cols only)

**Decision:** For rows where `has_director_bio == 0` (96.8% of films), the 64 `dir_bio_pca_*` columns are EXCLUDED from the reconstruction loss. Other modalities (genre, language, decade, etc.) are NOT masked even when their `has_*` flag is 0.

**Implementation sketch:**
```python
bio_mask = has_director_bio[:, None]   # (B, 1) broadcasts across 64 bio_pca dims
recon_bio_pca = ((decoded_bio_pca - input_bio_pca) ** 2 * bio_mask).sum() / bio_mask.sum().clamp_min(1) / 64
recon_other_dirprofile = ((decoded[:, 64:] - input[:, 64:]) ** 2).mean()
loss_director = w_director * 0.5 * (recon_bio_pca + recon_other_dirprofile)
```

**Rationale:**
- 96.8% missing for bio_pca makes masking essential — without it, bio dims produce trivial constant-zero output and gradient signal is wasted.
- Genre/decade missing rates (37%, 7.6%) are not severe enough to warrant masking — model can learn the implicit "no genre"/"no date" patterns as features.
- The `has_director_bio` flag remains an input feature (in director_profile block, dim index 64) — the encoder uses it to gate bio info.
- Ablation enabled: with G2, "with vs without director bio modality" comparison is meaningful (vs G1 where it would be vacuous).

---

### 2026-05-04 D5 — Training schedule + code organization: S2 (per-model notebooks + shared package)

**Decision:** Three model-specific notebooks (`02_train_ae`, `03_train_vae`, `04_train_dec`) plus a smoke-test notebook and a results notebook. All notebooks import from a shared `src/cineembed/` Python package containing the multi-modal backbone, model heads, losses, data loader, and eval helpers.

**Repository layout:**

```
src/cineembed/                 ← installable Python package
├── __init__.py
├── backbone.py                ← multi-modal encoder (D1 architecture)
├── heads.py                   ← AE/VAE/DEC head modules
├── losses.py                  ← W2 weighted MSE, ELBO, DEC KL, G2 mask
├── data.py                    ← feature_matrix.npz loader, train/val split
└── eval.py                    ← NMI/ARI/MSE/UMAP helpers

notebooks/
├── 01_smoke_test.ipynb        ← backbone parse + dummy forward
├── 02_train_ae.ipynb          ← AE × {32,64,128} + W1 baseline + W4 stretch
├── 03_train_vae.ipynb         ← VAE × {32,64,128}
├── 04_train_dec.ipynb         ← DEC × {32,64,128}, loads pretrained AE
└── 05_results.ipynb           ← cross-run table + latent viz + final figs

artifacts/models/
├── ae_z{32,64,128}.pt
├── vae_z{32,64,128}.pt
└── dec_z{32,64,128}.pt
```

**Dependency graph:**
```
feature_matrix.npz
        │
        ├──→ 01_smoke_test (validation only)
        │
        ├──→ 02_train_ae   ───┐  (parallel)
        ├──→ 03_train_vae  ───┤
        │                     │
        │                     ▼
        │                artifacts/models/ae_z*.pt
        │                     │
        └─────────────────────┴──→ 04_train_dec
                                       │
                                       ▼
                                  05_results
```

**Team parallelism:** Baran → AE, Arda → VAE, Kaan → DEC. Backbone owned jointly (defined in `src/cineembed/backbone.py`).

**Colab integration:** `!pip install -e /content/src` at top of each notebook + `from cineembed import ...`. Or use `sys.path.insert(0, '/content/src')` shortcut.

**Rationale:**
- DRY: backbone code lives in ONE place; updates propagate to all notebooks via reimport.
- Team parallelism: 3 notebooks → 3 owners.
- Test isolation: backbone smoke test runs in 01, model training in 02-04, comparison in 05.
- Sequential dependency made explicit: DEC notebook reads AE checkpoint paths.
- Consistent with parent EDA spec §6 ("Pipeline Import Bridge"): EDA stays in `eda_v2.ipynb`, modeling code lives in `src/`.

---

### 2026-05-04 D6 — Evaluation suite: Tier 2 (recommended)

**Decision:** Compute the following metrics for all 9 main runs + 1 W1 baseline + 1 optional W4 stretch + 2 modality ablations = **13 runs total**.

**Always-on metrics (free, computed during/after every run):**
- Per-block reconstruction MSE (track which modalities each model captures)
- Total weighted MSE
- NMI vs primary genre (genre[0] from each film's genre list)
- ARI vs primary genre
- 2D UMAP visualization × 3 colorings (genre, decade, language) → 3 PNGs/run
- VAE-only: ELBO breakdown (reconstruction vs KL term tracking per epoch)
- DEC-only: cluster purity per genre, soft-assignment confidence histogram

**Tier 2 add-ons (compute cost: minimal, value: high):**
- **Linear probing** — train a linear classifier on the latent z to predict primary_genre (22 classes) and decade_bin. Run for each of the 9 main models. Probe MSE/accuracy quantifies "does the latent space genuinely encode useful info?" → critical for course narrative.
- **Modality leave-one-out ablation** — at z=64, AE only:
  - **F1**: text block zeroed at input + decoder output → "does the 384-dim text embedding actually help?"
  - **F2**: director_profile block zeroed → "does the low-coverage bio modality pay off?"
  Each is +1 training run (~15-20 min). Total +2 runs.

**Rejected (Tier 3):**
- Spearman correlations (latent dim × continuous features) — interesting but lower payoff
- 4 additional leave-one-out ablations (numerical, genre, language, awards) — diminishing returns
- DEC cluster center evolution viz — eye-candy without strong signal

**Total compute budget:** 13 runs × ~15-20 min = 3-4 hours wall-clock on free Colab T4. Splittable across 2-3 sessions.

**Reporting:** A single results table with rows={AE, VAE, DEC} × {z=32, 64, 128} × 3 metrics (MSE/NMI/ARI), plus modality ablation deltas, plus W1/W4 ablation deltas. ~30 numbers total — paper-quality density.

---

### 2026-05-04 D7 — Ground truth labels: L4 (three orthogonal axes)

**Decision:** Compute NMI and ARI against THREE distinct ground-truth axes per model, giving each evaluation three answers to "what does the latent space encode?":

| Axis | Source | Class count | Computation |
|---|---|---:|---|
| **Genre (primary)** | `df['genres'].str.split('|').str[0].fillna('Unknown')` | 21 | First genre in pipe-delimited list |
| **Decade** | `df['decade']` | 13-14 | 1900s, 1910s, ..., 2020s, 'Unknown' |
| **Language (top-10)** | top-10 of `original_language` + 'other' | 11 | Collapsed from top-30 for eval-clarity |

**Eval flow (per model):**
```python
# For AE/VAE: latent z → KMeans → cluster_id
# For DEC: soft-assignment argmax → cluster_id

for axis_name, labels in [('genre', primary_genre),
                           ('decade', decade_bin),
                           ('lang', lang_top10)]:
    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    ari = adjusted_rand_score(labels, cluster_assignments)
```

**Rationale:**
- Three axes reveal whether the latent space is genre-biased (high genre-NMI, low decade-NMI), temporally-organized, or linguistically-clustered.
- Tells a richer story than a single number: "AE captures genre well, weak on decade. DEC explicit cluster optimization improves genre NMI by Δ but loses decade structure."
- L1 (single primary-genre) would be a defensible minimum but loses much of the latent space narrative.
- L3 (multi-label NMI) requires non-standard implementation (`scikit-multilearn` or hand-rolled), with limited additional insight over L4 — overengineered for course scope.
- Implementation cost is zero — all three label vectors derive from the existing `feature_matrix.npz` and `movies_eda_final.csv`.

---

### 2026-05-04 D8 — DEC cluster count: K2 (k-sweep {10, 21, 30})

**Decision:** Train DEC at three k values per latent dim: **k ∈ {10, 21, 30}**. Combined with z ∈ {32, 64, 128}, this yields **9 DEC runs** (vs 3 for K1).

**Hypotheses being tested:**
- **k=10**: Latent space organizes into ~10 broad genre-families (action+adventure+war; drama+romance+history; comedy+family; etc.). If k=10 wins on NMI, the model is discovering meta-genre structure.
- **k=21**: Direct match to primary_genre class count → fair NMI/ARI vs the L4 genre axis.
- **k=30**: Over-segmentation. If NMI(k=30) > NMI(k=21), the encoder discovered sub-genre granularity beyond TMDB's labeling.

**Rationale:**
- Single k = lucky guess risk; sweep distinguishes encoder quality from k-sensitivity.
- Compute cost: +6 DEC runs × ~10 min = +1 hour, well within free Colab T4 budget.
- Course narrative gains: "DEC's behavior at different k values reveals the latent space's natural cluster granularity" — paper-worthy finding regardless of outcome.

**Implementation note:** All three k values share the same pretrained AE checkpoint per latent dim (no extra AE training needed). Only the DEC head's cluster centroids and KL-target distribution differ.

---

### 2026-05-04 D9 — Peer-review revisions: baselines, clipping, β warmup, relative success criteria

**Trigger:** External LLM peer review of consolidated design (2026-05-04). Four substantive change requests, all accepted.

**(1) Vanilla baselines added (3 new runs)**

Without baselines, the multi-modal backbone hypothesis is unfalsifiable. Three baselines now part of the run matrix:

| Baseline | Method | Compute |
|---|---|---|
| `kmeans_raw` | `KMeans(n=21).fit_predict(X)` directly on 564-dim feature matrix | <1 min (sklearn) |
| `kmeans_pca` | PCA(564→64) → KMeans(n=21) | <1 min |
| `vanilla_ae_z64` | Single `Linear(564→64)` encoder, full reconstruction; no modality projection | ~15 min training |

Each is evaluated against the same L4 axes (genre/decade/lang). The multi-modal backbone's win is now testable as `multi_modal_AE_z64.NMI > vanilla_ae_z64.NMI × 1.05`.

**(2) Success criteria reframed: relative + absolute floor**

Replaced absolute thresholds with relative-improvement framing:
- `best_deep_NMI_genre > best_baseline_NMI_genre × 1.10` (10% relative gain)
- AND `best_deep_NMI_genre > 0.15` (absolute floor — guards against "improved over noise")
- `multi_modal_AE_z64.NMI > vanilla_ae_z64.NMI × 1.05` (architecture validation)
- `W2.NMI > W1.NMI × 1.05` (weighting validation)
- `F1.NMI ≠ AE_main_z64.NMI` (text contribution measurable)
- `F2.NMI ≠ AE_main_z64.NMI` (director contribution measurable)

This avoids the "0.30 absolute target seems aggressive" concern — frame all comparisons against in-experiment baselines.

**(3) W2 weight clipping**

```python
w_raw = 1.0 / max(block_var, 1e-6)
w_b = np.clip(w_raw, 0.1, 10.0)   # 100× range cap
```

Prevents low-variance blocks (e.g., language one-hot in skewed datasets) from receiving extreme weights and dominating training. Negligible cost, robust safety net.

**(4) VAE β warmup as default, not mitigation**

β=1 from epoch 0 invites trivial KL collapse on heterogeneous tabular data. Baseline VAE training schedule:
```python
β(epoch) = min(epoch / 10, 1.0) * 1.0   # linear warmup over 10 epochs
```
Epoch 0: pure reconstruction. Epoch 10+: full ELBO. Standard technique from β-VAE literature.

**(5) Reframed §1 Overview** — academic honesty:

> "This study does NOT claim a single best deep clustering model. It compares how three increasingly complex representation-learning approaches (AE, VAE, DEC) handle a heterogeneous multi-modal feature matrix. The hypothesis is architectural — modality-aware backbone outperforms vanilla concat. The clustering metrics are probes of latent structure across three axes (genre/decade/language), not leaderboard scores."

**Updated total run count: 18-19 → 21-22 runs.** Compute budget unchanged (~5-6 hours wall-clock) because baselines are cheap.

---

### 2026-05-04 D10 — Second peer-review pass: blocking technical fixes

**Trigger:** External LLM peer review of consolidated design (second pass, 2026-05-04). Four blocking technical bugs identified and fixed before plan-writing.

**(1) Loss double-counting fixed**

The original VAE/DEC loss formulas had:
```python
recon = sum(w_b * mse(b) for b in BLOCKS)
recon += director_block_loss(...)   # director already counted in BLOCKS sum!
```

If `BLOCKS` includes `'director'`, this double-counts. Consolidated into a canonical `weighted_recon_loss(...)` helper (spec §5.2.1) that excludes director from the generic sum and adds it via `director_block_loss(...)` with G2 mask. AE main, VAE, DEC, and W1 ablation all use this canonical helper.

**(2) Validation split contradiction resolved**

Original spec said "100% data, no holdout" while also specifying `early_stop_patience=10` on validation MSE — incompatible. Reframed to **90/10 split with `random_state=42` for early-stopping** while embedding extraction and cluster evaluation still use all 329,044 films. Linear probing keeps its separate 80/20 split for the linear classifier itself.

**(3) DEC P target distribution clarified as batch-wise**

Original spec said both "P updated every T=100 mini-batches" and showed batch-wise P computation in code — inconsistent. Adopted **batch-wise P as a deliberate practical approximation** of the original DEC paper's full-dataset P. Removed `target_update_T` from `DEC_EXTRA` hyperparameters and documented the choice as a known simplification. This is standard for academic reproductions and avoids the engineering overhead of caching full-dataset P every T epochs.

**(4) Colab install path corrected**

Original snippet: `!pip install -e /content/cineembed-repo/src/cineembed` — wrong because `pyproject.toml` lives at the repo root with `src/` layout. Corrected to `!pip install -e /content/cineembed-repo`. Added a `sys.path` fallback in case modern setuptools auto-discovery fails on Colab.

**(5) Optional: report figure-count guidance**

27 UMAP figures is too many for a final report. Spec now explicitly states: 9 main figures (best AE/VAE/DEC × 3 axes) + 3 baseline genre-axis UMAPs = **12 figures in main report**, the remaining 18+ go to supplementary/appendix. All 27 are still produced for completeness.

**Status:** Spec finalized. No further peer-review revisions expected before plan-writing.

---

## Final Architecture Summary

After D1-D9 decisions:

- **Architecture**: Multi-modal projection backbone + 3 heads (AE, VAE, DEC)
- **Latent dims**: {32, 64, 128} ablation grid
- **Loss weighting**: W2 inverse-variance + clipping; W1 + W4 ablations
- **Bio coverage**: G2 masked loss on 64 bio_pca dims when has_director_bio=0
- **Code**: Per-model notebooks + shared `src/cineembed/` package
- **Evaluation**: Tier 2 — 3 axes (genre/decade/lang) × NMI/ARI + linear probing + 2 modality ablations
- **DEC k**: {10, 21, 30} sweep
- **Baselines**: KMeans-raw, PCA+KMeans, vanilla concat-AE
- **Success criteria**: Relative improvement over baselines + absolute floor
- **VAE training**: β warmup 0→1 over 10 epochs
- **Runs**: 21-22 total (~5-6h compute on free Colab T4)
- **Framing**: Comparative study, not winner-declaration

---

### 2026-05-06 D11 — Clustering-improvement techniques landed

**Trigger:** Post-MVP feasibility audit identified five established techniques
that could push numbers higher with bounded engineering effort.

**Decision:** Land five additive techniques in a single sprint, no breaking
changes to existing APIs. Cross-ref:
`docs/superpowers/specs/2026-05-06-clustering-improvement-techniques.md`
(commit `8097685`).

1. **InfoNCE contrastive pretext** (§2.1) — `info_nce_loss`,
   `ContrastiveHead`, `ContrastivePairDataset`, `make_contrastive_dataloader`.
   The existing `MultiModalBackbone.forward(blocks, block_mask)` already
   accepts modality dropout, which is the natural augmentation primitive.
2. **Per-axis k-sweep evaluation** (§2.2) — `evaluate_run_per_axis_k`:
   axis-matched k per label axis (decade k=12, lang k=11). Stops penalizing
   non-genre axes for partition-cardinality mismatch.
3. **Soft / non-KMeans clustering** (§2.3) — `cluster_assignments_gmm`,
   `cluster_assignments_spectral`, `cluster_assignments_hdbscan`.
4. **AMI keys** (§2.4) — `*_ami` added to `evaluate_run` output for
   chance-adjusted reporting alongside NMI (Vinh et al. 2016).
5. **Multi-label macro-NMI** (§2.5) — `multilabel_macro_nmi` over the genre
   block; per-genre breakdown serves as a diagnostic.

**Acceptance:** All 68 tests pass. No existing public API breaks.

---

### 2026-05-16 D12 — Per-row block masking + InfoNCE τ=0.1 default

**Trigger:** Follow-on review of the contrastive-pretext implementation
before launching Phase 1 on Colab.

**Decision (two amendments to D11 / spec §2.1):**

**(1) Masking granularity: per-row, not per-batch.**

Original implementation used a single scalar mask per batch ("shatters
negatives" concern when every row in a view shares the same dropped modalities).
The backbone forward is amended to accept either:
- legacy scalar `float` (F1/F2 ablation paths), or
- `Tensor (B, 1)` per-row mask (contrastive views).

Per-row masking prevents batch-level co-adaptation and produces stronger
negatives within each batch — each row in a view has its own independent
dropout pattern.

**(2) InfoNCE temperature default τ=0.5 → τ=0.1.**

SimCLR's natural-image default of τ=0.5 is calibrated for image-embedding
geometry. Heterogeneous tabular signal here is denser than natural-image
embeddings — a lower temperature sharpens the contrastive objective more
effectively. The Phase 1 sweep retains τ=0.5 as a comparison run, but the
spec's documented default is now τ=0.1.

**Implementation:** commit `20472d4`. Spec amended inline.

---

### 2026-05-16 D13 — Two-round modeling strategy supersedes exhaustive ablation

**Trigger:** Wall-clock budget audit with four working days to deadline; the
21-22 run matrix from D2/D6/D8 is not feasible alongside the web-app pivot
(D14).

**Decision:** Replace the exhaustive ablation grid with a two-round strategy
producing the same headline narrative at ~20% of the compute. Cross-ref
`docs/superpowers/specs/2026-05-16-two-round-modeling-strategy.md`.

**Selection metric (locked):**

```
geo_NMI = (gNMI · dNMI · lNMI)^(1/3)
```

Geometric mean across the three label axes — penalizes a model that wins one
axis and tanks another.

**Round 1 — architecture comparison @ z=64 (~30 min Colab, 9 rows):**
6 MVP carry-over runs + 3 new (`vae_z64`, best of `phase-1-sweep`,
`contrastive_pretext + DEC`).

**Round 2 — z-sweep on the Round-1 winner only (~20 min Colab):**
Winner re-trained at z=32 and z=128 → 3-row z-dim sensitivity sub-table.

**Explicit scope cuts (justified as future work in the final report):**
- `ae_z32`, `ae_z128` — covered by Round-2 winner z-sweep.
- `ae_z64_no_text` (F1), `ae_z64_no_director` (F2) — modality ablation, deferred.
- `ae_z64_w4` — Kendall learned weighting, marginal expected gain over W2.
- `dec_z32_*`, `dec_z64_k10/30`, `dec_z128_*` — k=21 won MVP; k-grid deferred.
- `vae_z32`, `vae_z128` — only run if VAE wins Round 1.

Total new training compute = 2 (Round 1) + 2 (Round 2) = 4 runs (~50 min Colab),
plus the 3 Phase 1 contrastive runs.

---

### 2026-05-16 D14 — Project deliverable pivot to web app demo

**Trigger:** SENG 474 final deliverable clarification — a working demo
significantly outweighs a more thorough report at the course-grade margin.

**Decision:** Final deliverable is a **working web app**, not just a report.
Cross-ref `docs/superpowers/specs/2026-05-16-web-app-demo-design.md`.

- **Backend:** FastAPI REST. Endpoints:
  - `GET /api/films/search?q=...` → list (id, title, year)
  - `GET /api/films/{id}/similar?top=N` → top-N nearest neighbours
  - `GET /api/films/random?n=...` → random films (UI cold start)
- **Inference:** cosine similarity over L2-normalized 64-dim latents
  (Round-2 winner). 329k × 64 = ~80 MB float32 in RAM. <10 ms per query
  via numpy matmul + topk. No FAISS.
- **Pre-compute (`scripts/build_index.py`):** backbone-agnostic; reads the
  winner's `state_dict`, encodes all 329k rows, L2-normalizes, saves
  `artifacts/inference/embeddings.npy` + `artifacts/inference/films.parquet`.
- **Frontend:** minimal static HTML/JS. Search box, top-N selector
  (2/5/10), result cards. Posters DEFERRED — TMDb on-demand via the `id`
  column in `movies_eda_final.csv` is an open decision.
- **Deployment:** `uvicorn cineembed.api:app --reload` on localhost:8000;
  static served from `/static/` mount.
- **Report tier:** "half-academic" — between bare demo and full ablation
  paper. Deadline 2026-05-20.

This re-prioritizes the remaining work order to: (1) finish models, (2)
inference pipeline + REST API, (3) frontend UI, (4) posters & polish.

### 2026-05-17 D15 — Demo backbone locked to `ae_z32` after Round 2 z-sweep

**Trigger:** Round 2 z-sweep on the AE family (z ∈ {32, 64, 128}) revealed
a U-curve, not a monotonic "bigger is better" or "smaller is better"
relationship. The earlier 2026-05-17 (AM) decision that selected `ae_z64`
over `dec_z64_k21` (retrieval over NMI) needed a second-stage refinement
within the AE family.

**Decision:** The demo backbone is **`artifacts/models/ae_z32/ae.pt`**
(multi-modal AE at z=32, hidden_dim=128). All deployment artifacts move
to `artifacts/inference/ae_z32/{embeddings.npy, films.parquet, manifest.json}`.

**Evidence (sweep complete, see `docs/journal/12-z-sweep-ae-z32-discovery.md`):**

| Backbone | gNMI | dNMI | lNMI | geo_NMI | `genre@5` mean | `pair_cos_std` | `dim_std_min` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`ae_z32`** | **0.334** | 0.295 | 0.216 | 0.277 | **0.723** | 0.301 | 0.117 |
| `ae_z64` (was demo) | 0.328 | 0.341 | 0.264 | 0.309 | 0.715 | 0.299 | 0.062 |
| `ae_z128` | 0.273 | 0.275 | 0.272 | 0.274 | 0.722 | 0.289 | **0.025** |

**Tiebreaker logic:** `ae_z32` and `ae_z128` are tied on `genre@5` within
noise (0.723 vs 0.722). Five independent signals all favour z=32:

1. `gNMI` higher by 6.1 absolute points (z=128 collapses on genre axis).
2. `dim_std_min` healthier (0.117 vs 0.025) — z=128 has a near-dead dim.
3. `pair_cos_std` wider (0.301 vs 0.289) — z=128 angular-collapsing.
4. Reconstruction loss lower (best_val 0.0223 vs 0.0237).
5. Occam: 32-dim model is smaller, faster inference, less RAM.

**Why not z=64?** Best `geo_NMI` (0.309) and best `dNMI`+`lNMI`, but loses
both demo-relevant axes (`genre@5`, `gNMI`) to z=32. Composite metric
includes redundant modalities (decade, language) that the demo doesn't
care about.

**Interpretation:** z=32 is the **information-bottleneck sweet spot** for
this task. The encoder is forced to concentrate capacity on the
highest-entropy modalities (384-d text embedding, 113-d director PCA),
demoting redundant ones (decade, language). z=128 sits past the sweet
spot in the other direction — no compression pressure, dim allocation
diffuses, gNMI collapses. This is the project's **second methodological
finding** (first: clustering NMI does not predict retrieval quality —
2026-05-17 AM amendment).

**Affects:**

- `docs/superpowers/specs/2026-05-16-web-app-demo-design.md` — amendment
  appended (2026-05-17 PM).
- `scripts/build_index.py` example commands — default checkpoint path
  updated.
- Backend / frontend deployment artifact location: `artifacts/inference/ae_z32/`.
- API contract: unchanged. Embedding dim is internal.

**Out-of-scope (deferred):** z=16 ablation. Cheap (~10 min train + eval)
and would establish whether z=32 is the U-curve minimum or merely better
than tested alternatives. Deferred to "Future work" in the report.

---

## Lessons learned (MVP, 2026-05-05; expanded as Phase 1 / Round 1 land)

Concise empirical takeaways from the 6 MVP runs (full detail in `docs/FINDINGS.md`):

- **Decade is the easy axis.** All three architectures (vanilla, multi-modal,
  W1) recover decade at NMI ≈ 0.34–0.37. Year-correlated patterns are ordinal
  and single-valued — KMeans gets there for free.
- **W2 inverse-variance weighting is critical.** W1 (uniform) loses -50% genre
  / -73% language NMI vs W2. Asymmetric collapse: small blocks (decade) survive
  uniform weighting because StandardScaler already normalized them; large
  high-dim blocks (text 384, language 31) lose all gradient signal.
- **Missing release_date forms a coherent sub-manifold.** Not predicted by
  H1–H3. UMAP shows `decade_bin = 0` films isolated as a cluster across all
  four architectures; DEC compresses them most explicitly. A representation-
  learning interpretability win that emerged post-hoc.
- **Multi-modal architecture beats vanilla concat.** +178% lang_NMI, +14%
  genre_NMI; trade-off: -7.6% decade_NMI. Modality-specific projection
  allocates capacity to text/director blocks at slight cost on the trivially-
  encoded decade signal.
- **No single architecture wins all axes.** Vanilla wins decade_NMI +
  genre_ARI; multi-modal wins decade_ARI; DEC wins genre_NMI + lang_NMI +
  lang_ARI. This non-uniformity is the principled-trade-off story and the
  motivation for the `geo_NMI` composite metric (D13).


## Constraints (locked)

- Compute: Google Colab T4 GPU (free tier or Pro), Mac CPU for development
- Timeline: ~4-6 weeks to course end
- Team size: 3
- Reproducibility: seed=42 throughout, MD5-trackable artifacts, single training notebook per model
- Input: `artifacts/feature_matrix.npz` from EDA phase, no re-running upstream pipeline
- Out of scope: REST API, frontend, additional data sources

---

## Open Questions Pipeline

All questions resolved (D1-D10 above). Spec finalized at [`docs/superpowers/specs/2026-05-04-modeling-design.md`](../superpowers/specs/2026-05-04-modeling-design.md).
