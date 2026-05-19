# CineEmbed — AI Handoff Brief: Intermediate Report + Presentation

> **For the AI assistant taking over this project.** This single file contains everything you need to produce the deliverables. Read top to bottom, then execute. Do not ask the user clarifying questions unless something here is genuinely ambiguous — the project is mature, all decisions have been made, and the data is final.
>
> **Author of this brief:** Claude (the previous AI). I have full context of the entire project. The user is handing the report+presentation work to a teammate now.

---

## 0. Your Mission

Produce **two deliverables**, both committed to this repo and pushed to GitHub `main`:

| # | Artifact | Path | Format | Length |
|---|---|---|---|---|
| 1 | **Intermediate Report** | `docs/report/INTERMEDIATE_REPORT.md` | Markdown (Pandoc-friendly for PDF export) | 6–8 page equivalent (~3000–4500 words) |
| 2 | **Presentation Slides** | `docs/presentation/PRESENTATION.md` | Marp markdown (auto-renders to PDF/PPTX/HTML) | 10–14 slides, 16:9 ratio |

Both artifacts must be **self-contained**: figures embedded with relative paths from the repo root, all numerical claims sourced from `artifacts/eval/results.json`, no broken references.

**When done:** commit + push, then tell the user the deliverables are ready.

---

## 1. Project Identity

- **Title:** *CineEmbed — Multi-Modal Unsupervised Embedding of 329,044 Films*
- **Course:** SENG 474 — Deep Learning
- **Institution:** TED University, Spring 2026
- **Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya
- **Phase:** Modeling MVP (intermediate report scope) — full study deferred for final report
- **Goal in one sentence:** Train unsupervised 64-dim representations of movie metadata via multi-modal AE/DEC architectures, and demonstrate that the latent geometry recovers three orthogonal label axes (primary genre, decade, language) substantially better than non-deep KMeans baselines.

---

## 2. Repository Setup

```bash
git clone https://github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System.git
cd CineEmbed-
```

Repo is **public**. All artifacts you need are vendored in `artifacts/` (models, figures, eval results) — except the 700 MB `feature_matrix.npz`, which is regenerable from the EDA pipeline and not strictly needed for report writing.

---

## 3. Required Reading (~10 minutes total — do this BEFORE writing anything)

These files contain the canonical truth. If anything in this brief contradicts them, **the source files win**.

| Priority | File | What you'll learn |
|---|---|---|
| 🔴 critical | `docs/FINDINGS.md` | All 9 hero findings with numbers, hypotheses status, figure index |
| 🔴 critical | `artifacts/eval/results.json` | Raw NMI/ARI for all 6 runs — your single source of truth for numbers |
| 🟡 important | `docs/adr/0001-modeling-hybrid-architecture.md` | The 10 design decisions (D1–D10) |
| 🟡 important | `docs/superpowers/specs/2026-05-04-modeling-design.md` | Full design spec with rationale |
| 🟢 reference | `docs/PRESENTATION_PROMPTS.md` | Pre-written Codex prompts for diagrams (reuse these!) |
| 🟢 reference | `docs/PROGRESS.md` | Phase status + what's deferred for final report |
| 🟢 reference | `mermaid/*.md` | 5 already-rendered Mermaid diagrams (PNGs in `figures/`) |

You don't need to read the source code (`src/cineembed/`) to write the report — the architecture is fully documented in the spec.

---

## 4. Canonical Numerical Results (memorize / cite directly)

### 4.1 All-Runs Comparison Table (z=64 latent, k=21 KMeans)

```
| Tier                  | Run                | genre_NMI | genre_ARI | decade_NMI | decade_ARI | lang_NMI | lang_ARI |
|-----------------------|--------------------|----------:|----------:|-----------:|-----------:|---------:|---------:|
| Non-deep baseline     | kmeans_raw_k21     |    0.109  |    0.063  |    0.233   |    0.093   |   0.075  |   0.026  |
| Non-deep baseline     | pca_kmeans_k21     |    0.084  |    0.061  |    0.224   |    0.085   |   0.094  |   0.042  |
| Simple deep baseline  | vanilla_ae_z64     |    0.287  |  **0.247**|  **0.369** |    0.175   |   0.095  |   0.030  |
| Ablation (W1 uniform) | ae_z64_w1          |    0.165  |    0.094  |    0.367   |    0.176   |   0.070  |   0.026  |
| Multi-modal AE        | ae_z64             |    0.328  |    0.229  |    0.341   |  **0.211** |   0.264  |   0.090  |
| **DEC (BEST)**        | **dec_z64_k21**    |  **0.332**|    0.244  |    0.342   |    0.210   | **0.294**| **0.090**|
```

**Bold = column winner.** No model wins all 6 metrics — this is the principled-trade-off story.

### 4.2 Headline Comparisons (use these in the abstract / conclusion)

| Comparison | Numbers | Relative gain |
|---|---|---:|
| DEC vs `kmeans_raw` (genre) | 0.332 vs 0.109 | **+205%** |
| DEC vs `pca_kmeans` (genre) | 0.332 vs 0.084 | **+295%** |
| DEC vs `pca_kmeans` (lang) | 0.294 vs 0.094 | **+213%** |
| Multi-modal vs vanilla (lang) | 0.264 vs 0.095 | **+178%** |
| W2 vs W1 (genre) | 0.328 vs 0.165 | **+99%** |
| W2 vs W1 (lang) | 0.264 vs 0.070 | **+277%** |
| DEC vs ae_z64 (ARI on genre) | 0.244 vs 0.229 | **+6.6%** (compactness gain) |

### 4.3 Pre-Registered Hypotheses Status (all PASS)

- **H1** — DEC > AE on genre_NMI: ✅ PASS (0.332 > 0.328, marginal NMI but +6.6% ARI)
- **H2** — Best deep > best non-deep baseline by ≥10%: ✅ PASS *massively* at +205%
- **H3** — Best deep NMI > 0.15 absolute floor: ✅ PASS (0.332 ≫ 0.15)

### 4.4 Training Dynamics

| Run | Epochs (early-stopped) | Final val_loss | Notes |
|---|---:|---:|---|
| vanilla_ae_z64 | 58 | 0.0126 | Plateaued cleanly — but lowest val_loss yet middling NMI = "useful representation ≠ perfect copy" |
| ae_z64 (best) | 69 | 0.0208 | Longest training — careful learning with proper W2 weighting |
| ae_z64_w1 | 37 | 0.0453 | Patience exhausted early — diagnostic of W1 collapse |
| dec_z64_k21 | 21 | 0.127† | †KL+recon combined; not comparable to AE pure-recon. **0 cluster reinits — healthy throughout.** |

---

## 5. The 9 Hero Findings (use as report section anchors)

Each finding is a self-contained claim with evidence. Convert each into a paragraph in the Results section of the report.

### Finding 1 — Architecture's biggest win is on LANGUAGE (+178%)
Multi-modal projection layers are essential for capturing low-frequency one-hot blocks. Vanilla concat-AE lang_NMI = 0.095; multi-modal ae_z64 lang_NMI = 0.264 (**+178%**). Genre also wins (+14%) but decade slightly loses (-7.6%). **Hero figure:** `artifacts/figures/umap/umap_ae_z64_lang.png` (clear language micro-clusters visible).

### Finding 2 — W2 inverse-variance weighting is critical (asymmetric collapse)
Without W2 weighting, high-dim low-variance modalities (text 384, language 31) lose all gradient signal. W1 vs W2 on genre: -50%; on lang: -73%. Small blocks like decade (2 dims) are immune because StandardScaler already gave them per-feature variance ≈ 1. **Hero figure:** `artifacts/figures/umap/umap_ae_z64_w1_genre.png` (diffuse blob, no fine structure).

### Finding 3 — Reconstruction loss ≠ clustering quality
`vanilla_ae` had the lowest val_loss (0.0126) but middling genre_NMI (0.287); `ae_z64` had higher val_loss (0.0208) but the best AE genre_NMI (0.328). Classical "useful representation vs. perfect copy" tension — vivid empirical demonstration.

### Finding 4 — Decade is the strongest natural axis (NMI ≈ 0.34–0.37 universal)
All architectures (even W1 ablation) recover decade at NMI ~0.35. Movie metadata has strong year-correlated patterns that emerge regardless of architecture. **Implication:** genre is harder than decade because real-world genres are multi-label and overlap heavily.

### Finding 5 — Pareto trade-off across axes
Multi-modal is *not uniformly superior* to vanilla:
- Wins big on lang (+178%)
- Wins moderately on genre (+14%)
- **Loses slightly on decade (-7.6%)**
This is a **principled trade-off** — modality projection allocates capacity to text/director, slightly reducing fidelity on the trivially-encoded decade signal.

### Finding 6 — Deep models outperform non-deep baselines by 3–5×
| | kmeans_raw | pca_kmeans | dec_z64_k21 | DEC vs raw / pca |
|---|---:|---:|---:|---:|
| genre_NMI | 0.109 | 0.084 | 0.332 | **+205% / +295%** |
| genre_ARI | 0.063 | 0.061 | 0.244 | **+287% / +300%** |
| lang_NMI | 0.075 | 0.094 | 0.294 | **+292% / +213%** |

Non-deep clustering on raw 564-dim features recovers ~⅓ of the structure that the multi-modal pipeline finds. **PCA-64 + KMeans is *worse* than raw-KMeans on genre** (PCA discards genre-discriminative variance) but slightly better on language (PCA cleans noise). H2's strongest single claim.

### Finding 7 — DEC sharpens cluster boundaries (ARI > NMI improvement)
DEC was initialized from `ae_z64` and trained 21 KL+recon epochs:
- genre_NMI: 0.328 → 0.332 (+1.2%)
- **genre_ARI: 0.229 → 0.244 (+6.6%)**
- **lang_NMI: 0.264 → 0.294 (+11.4%)**
- decade_NMI: flat

DEC's contribution is **cluster compactness, not new structural information**. The larger ARI gain than NMI gain is the diagnostic signature. `total_reinit = 0` across all 21 epochs (healthy training). DEC closes the vanilla/multi-modal ARI gap (0.244 vs 0.247 — essentially tied) while keeping multi-modal's lang advantage. **DEC = best of both worlds.**

### Finding 8 — Latent topology evolves dramatically (blobs → islands → tight islands)
UMAP visualization reveals a **qualitative topology shift** that quantitative metrics alone don't fully convey:
- **vanilla**: 2 mega-blobs with dense Unknown-genre mass
- **ae_z64**: dozens of small islands with central genre-coherent micro-clusters
- **dec_z64_k21**: even more atomized, tighter islands
- **ae_z64_w1**: mid-sized blobs, no fine structure

This is paper-quality visual evidence that (1) modality projection creates structural diversity, (2) inverse-variance weighting keeps it stable, (3) explicit clustering sharpens it into discrete partitions. **Hero figure:** `artifacts/figures/umap/umap_comparison_genre.png` (3-panel side-by-side — the single most informative figure in the entire study).

### Finding 9 — Films with missing release_date form a coherent latent sub-manifold
In all 4 architectures, films with `decade_bin = 0` (missing release_date, ~7.4% of subsample) form a **clearly isolated cluster** in the latent — DEC compresses them into the cleanest partition. **The model wasn't *forced* to encode missingness as structural** — `has_release_date` is one feature among 564. Yet missingness emerges as a *dimension of latent geometry, not just a flag*. **Post-hoc discovery** — not predicted by H1–H3, an interpretability win. **Hero figure:** `artifacts/figures/umap/umap_dec_z64_k21_decade.png` (red cluster top-right).

---

## 6. Available Visual Assets

### 6.1 UMAP Figures (`artifacts/figures/umap/` — 13 PNGs)

Tier-ranked for the report:

| Tier | File | Use in report |
|---|---|---|
| 🥇 hero | `umap_comparison_genre.png` | Architecture progression — Finding 8 |
| 🥇 hero | `umap_dec_z64_k21_decade.png` | Missing-data manifold — Finding 9 |
| 🥈 strong | `umap_dec_z64_k21_lang.png` | Language clustering — Finding 1 |
| 🥈 strong | `umap_ae_z64_w1_genre.png` | W1 collapse visual — Finding 2 |
| supporting | `umap_ae_z64_genre.png`, `umap_ae_z64_lang.png` | Method illustration |
| supporting | `umap_vanilla_ae_z64_*.png` (3 files) | Baseline contrast |
| reference | `umap_dec_z64_k21_genre.png`, `umap_ae_z64_w1_decade.png`, `umap_ae_z64_w1_lang.png`, `umap_ae_z64_decade.png` | Appendix completeness |

### 6.2 EDA Figures (`artifacts/figures/` — 21 PNGs)

For the data engineering / methodology section. Most useful:
- `genre_distribution.png` — class imbalance motivation
- `multilingual_coverage.png` — language sparsity motivation
- `director_bio_coverage.png` — G2 mask motivation (96.8% missing bio)
- `correlation_heatmap.png` / `correlation_pearson.png` — feature interactions
- `modality_balance_before_after.png` — W2 weighting motivation
- `clusterability_pca.png` — baseline clusterability check
- `embedding_analysis.png` — high-level pipeline view

### 6.3 Pre-Rendered Architecture Diagrams (`figures/` — root level)

These are already rendered from the Mermaid prompts in `docs/PRESENTATION_PROMPTS.md`:
- `figures/high-level-architecture.png` — pipeline overview (Section 1 prompt)
- `figures/data-engineering-pipeline.png` — data pipeline (Section 2 prompt)
- `figures/architecture_multimodal.png` — model schematic (Section 3 prompt)

For diagrams not yet rendered (taxonomy, evaluation methodology), the Mermaid source is in `mermaid/` directory — render at `mermaid.live` if needed.

### 6.4 Mermaid Sources (`mermaid/`)

5 ready-to-render diagrams. If you want fresh PNGs, paste each `.md` into mermaid.live and export.

---

## 7. Methodology Snapshot (for the Method section)

### 7.1 Data
- **329,044 films** from TMDB merged with external awards data and Wikipedia director bios
- **564-dim feature matrix**, organized into **7 block-contiguous modalities**:
  - `numerical` (6): log_popularity, log_vote_count, runtime_norm, vote_average_norm, has_vote, has_engagement
  - `genre` (22): one-hot indicators + has_genre flag
  - `language` (31): one-hot original_language top-N
  - `decade` (2): decade_norm + has_release_date
  - `awards` (6): prior log-counts (Oscar/BAFTA/Cannes wins+nominations)
  - `text` (384): all-MiniLM-L6-v2 sentence embeddings of overview/title
  - `director` (113): bio_pca_64 + has_director_bio + dir_lang_30 + dir_country_18 + has_director_lang

### 7.2 Architecture
- **Multi-Modal Backbone:** 7 modality-specific `_BlockProjection` layers compress each block to its proj_dim:
  - numerical→16, genre→16, language→16, decade→4, awards→16, text→64, director→32 (total 164 dims)
  - Concat → `Linear(164→128)` + ReLU + Dropout(0.2) → `Linear(128→64)` = **z ∈ ℝ⁶⁴**
- **Heads:** AEHead (deterministic), VAEHead (μ/σ + reparameterization, **not used in MVP**), DECHead (Student-t kernel + KL on soft assignments)
- **Vanilla baseline:** single `Linear(564→128) → Linear(128→64)` — no modality projection

### 7.3 Loss Functions
- **W2 (canonical):** `L = Σ_b w_b · MSE(X_b, X̂_b)`, where `w_b = 1 / Var(X_b)` clipped to [0.1, 10.0]
- **W1 (ablation):** uniform weights (`w_b = 1` for all blocks)
- **G2 (director bio masking):** the bio_pca_64 reconstruction loss is masked by `has_director_bio` (96.8% of films have NO Wikipedia bio → those rows contribute 0 loss for that block)
- **DEC:** Student-t kernel `q_ij = (1+||z_i-μ_j||²/α)^(-(α+1)/2)`, normalized; target `p_ij = q_ij² / Σq_ij²` computed per batch (D10 decision); loss = γ·KL(P||Q) + recon, γ=0.1

### 7.4 Training
- AdamW optimizer, lr=1e-3, weight_decay=1e-5
- Early stopping: patience=10, min_delta=1e-4 on val_loss
- Train/val split: 90/10 random, seed=42
- Batch size 1024, all training on Colab T4 GPU
- Total wallclock: ~4 hours for all 6 runs

### 7.5 Evaluation
- **3-axis ground truth:** primary_genre (21 classes), decade_bin (~12), lang_top10 (11)
- **Cluster assignment:** KMeans (k=21, n_init=20, seed=42) on full-data latents
- **Metrics:** NMI (information overlap, robust to label permutations) + ARI (cluster agreement, harder to satisfy)
- **No single axis privileged** — all three reported simultaneously per the principled-trade-off story

### 7.6 Decisions made (referenced in ADR)
| ID | Decision |
|---|---|
| D1 | Hybrid C-structure × D-architecture — modality-specific projections + shared backbone |
| D2 | Latent dim z=64 (z=32 and z=128 deferred for final report) |
| D3 | W2 inverse-variance as main + W1 ablation, W4 stretch deferred |
| D4 | G2 director bio masking (96.8% missing bio) |
| D5 | Per-model notebooks + shared `src/cineembed` package |
| D6 | Tier-2 evaluation: 3-axis NMI/ARI as core, linear probing deferred |
| D7 | L4 ground truth: 3-axis evaluation (genre/decade/lang) |
| D8 | DEC k=21 (k-sweep deferred) |
| D9 | Peer-review fixes: 3 baselines, weight clipping, β warmup, relative criteria |
| D10 | Batch-wise DEC P target (not full-dataset) — 10× speedup, no quality regression |

---

## 8. Deliverable A: `INTERMEDIATE_REPORT.md`

### Structure (use these section headings exactly)

```markdown
# CineEmbed: Multi-Modal Unsupervised Embedding of 329,044 Films

**SENG 474 — Deep Learning, Spring 2026, TED University**
**Team:** Baran Dinçoğuz, Arda Arvas, Kaan Kaya
**Date:** [today's date]

---

## Abstract (~150 words)

[Problem, approach, headline result, contribution. Lead with the +205% deep-vs-baseline claim and the 9 findings preview.]

## 1. Introduction (~400 words)

- The clustering problem on heterogeneous metadata
- Why deep representations beat non-deep clustering on multi-modal features
- Our three contributions:
  1. A multi-modal AE architecture with W2 inverse-variance weighting
  2. A 3-axis evaluation methodology that exposes principled trade-offs
  3. Empirical findings on cluster compactness via DEC + a post-hoc latent-geometry discovery (Finding 9)
- Roadmap of the report

## 2. Related Work (~250 words)

- Multi-modal autoencoders for heterogeneous tabular+text data (cite the spec's references)
- Deep Embedded Clustering (Xie et al. 2016) — Student-t kernel + KL self-training
- Why we extend DEC: explicit modality structure + 3-axis evaluation
- Difference from prior single-axis movie clustering work

## 3. Data and Features (~500 words, 1 figure)

- 329,044 films, sources, scale
- 7 modalities table (Section 7.1 of this brief)
- Class imbalance + missingness handling (G2 director bio masking)
- Sparsity stats (language 99% zero per row, etc.)
- **Figure 1:** `figures/data-engineering-pipeline.png` (or `artifacts/figures/multilingual_coverage.png` if pipeline diagram preferred elsewhere)

## 4. Methodology (~700 words, 2 figures)

### 4.1 Multi-Modal Backbone Architecture
- Modality-specific projection layers (Section 7.2)
- Concat → FC backbone → z=64
- **Figure 2:** `figures/architecture_multimodal.png`

### 4.2 Heads
- AEHead, DECHead (mention VAE deferred)

### 4.3 Loss Functions
- W2 inverse-variance weighting with clipping [0.1, 10.0]
- G2 bio masking
- DEC KL+recon

### 4.4 Three-Axis Evaluation
- Why 3 axes (Section 5 reasoning)
- KMeans k=21 → NMI+ARI per axis
- **Figure 3:** Use mermaid `mermaid/05-three-axis-evaluation-methodology.md` (render to PNG first)

### 4.5 Six Runs Trained
- 4 deep + 2 baselines (full table from Section 4.1 of this brief)

## 5. Results (~1000 words, 5–6 figures)

### 5.1 Main Comparison Table
- The full results table (Section 4.1 of this brief), in markdown
- Bold winners

### 5.2 Headline: Deep models beat non-deep baselines by 3–5× (Finding 6)
- The +205% / +295% claim
- **Figure 4:** Bar chart from `docs/PRESENTATION_PROMPTS.md` Section 6 prompt — IF NOT YET RENDERED, render it via the prompt and save to `figures/results_bar_chart.png`. Otherwise reference whichever exists.

### 5.3 Multi-Modal Architecture Wins on Language (Finding 1) + Pareto Trade-Off (Finding 5)
- The +178% lang gain
- The decade -7.6% loss as principled trade-off
- **Figure 5:** `artifacts/figures/umap/umap_ae_z64_lang.png` showing language micro-clusters

### 5.4 W2 Weighting is Critical (Finding 2)
- 99% / 277% gains over W1
- W1 collapse story
- **Figure 6:** `artifacts/figures/umap/umap_ae_z64_w1_genre.png` showing diffuse W1 latent

### 5.5 DEC Sharpens Cluster Compactness (Finding 7)
- ARI > NMI improvement pattern
- Cluster health: 0 reinits

### 5.6 Latent Topology Evolution (Finding 8)
- Blobs → islands → tight islands
- **Figure 7:** `artifacts/figures/umap/umap_comparison_genre.png` (the single most important figure)

### 5.7 Discovery: Missing-Data Manifold (Finding 9)
- Post-hoc finding
- **Figure 8:** `artifacts/figures/umap/umap_dec_z64_k21_decade.png` red cluster

## 6. Discussion (~400 words)

- Why H2 succeeded so dramatically: heterogeneous features really do benefit from multi-modal projection
- Why DEC's gain is "ARI not NMI": KL pressure tightens, doesn't restructure
- The principled trade-off: no architecture wins all 6 metrics
- Implication for downstream tasks (recommender, search): multi-modal+DEC captures lang+genre richly, vanilla still better for decade-only queries

## 7. Limitations and Future Work (~250 words)

What's deferred for the final report (be transparent):
- VAE family (z=32, 64, 128) — not yet trained
- Additional AE z dims (32, 128)
- F1/F2 modality ablations (no-text, no-director-profile)
- DEC k-sweep (we ran only z=64×k=21 of 9 planned)
- W4 (Kendall learned uncertainty) — optional stretch
- Linear probing on z=64 frozen latents
- Full reproducibility audit

## 8. Conclusion (~150 words)

- All three pre-registered hypotheses (H1, H2, H3) PASS
- Bonus: post-hoc discovery (Finding 9)
- The 3-axis methodology revealed that *no single architecture wins all 6 metrics* — multi-modal makes principled trade-offs
- Best overall model: `dec_z64_k21` with genre_NMI=0.332

## References (5–10 items)

[At minimum: DEC paper (Xie et al. 2016), MiniLM paper, UMAP paper, sklearn, PyTorch. Consult the spec for any others cited there.]

## Appendix (optional — figures + supplementary tables)

- All UMAP figures organized by axis
- Per-block dimensions table
- D1–D10 decision rationale (or link to ADR)
```

### Writing Rules

1. **Numbers:** every numerical claim must come from `artifacts/eval/results.json`. Do not round inconsistently — present 3 decimal places throughout (`0.332` not `0.33` or `0.3320`).
2. **Tone:** academic but not stiff. Past tense for what was done, present tense for general claims and figure captions.
3. **Figures:** embed via relative path (`![caption](artifacts/figures/umap/umap_xxx.png)`). Each figure needs a numbered caption like *"Figure 4: Latent topology comparison across three architectures..."*.
4. **Don't invent.** If you can't source a claim from the repo files, leave it out.
5. **Length discipline:** target word counts in the structure above are *budgets*, not minimums. Tight is better than padded.

---

## 9. Deliverable B: `PRESENTATION.md` (Marp format)

### Setup
Use **Marp** (`marp.app`) for slides — it converts markdown to PDF/PPTX/HTML and is friendlier than reveal.js for academic talks.

Frontmatter:
```yaml
---
marp: true
theme: default
size: 16:9
paginate: true
header: "CineEmbed — SENG 474 Intermediate Report"
footer: "Baran Dinçoğuz · Arda Arvas · Kaan Kaya · TED University · 2026"
style: |
  section { font-size: 22px; }
  h1 { color: #C2410C; }
  h2 { color: #2563EB; }
---
```

### Slide Structure (12 slides target)

| # | Title | Content | Key visual |
|---|---|---|---|
| 1 | Title | Project name, team, course, date, one-line tagline | `figures/high-level-architecture.png` (small) |
| 2 | Problem & Motivation | Why heterogeneous movie metadata is hard for clustering; what 3-axis evaluation buys you | `artifacts/figures/genre_distribution.png` (class imbalance) |
| 3 | Data Pipeline | 329K films → 564-dim 7-block matrix | `figures/data-engineering-pipeline.png` |
| 4 | Multi-Modal Architecture | Modality projections → concat → z=64 → decoders, with W2 + G2 callouts | `figures/architecture_multimodal.png` |
| 5 | Three-Tier Model Taxonomy | Non-deep / simple deep / multi-modal — explain why we trained 6 models | mermaid `04-three-tier-model-taxonomy.md` rendered |
| 6 | 3-Axis Evaluation | KMeans k=21 → NMI+ARI on 3 orthogonal labels | mermaid `05-three-axis-evaluation-methodology.md` rendered |
| 7 | Results — Main Comparison Table | The full table (4.1) with 6 winners highlighted | (table only, no figure) |
| 8 | Headline: Deep beats Non-Deep by 3-5× | Bar chart of NMI across all models on 3 axes | `figures/results_bar_chart.png` (render via PRESENTATION_PROMPTS.md §6 prompt) |
| 9 | Architecture Wins on Language (+178%) | Finding 1 + Finding 5 (Pareto trade-off) | `artifacts/figures/umap/umap_ae_z64_lang.png` |
| 10 | DEC Sharpens Cluster Boundaries | Finding 7 + Finding 8 — show the topology evolution | `artifacts/figures/umap/umap_comparison_genre.png` (3-panel) |
| 11 | Discovery: Missing-Data Manifold | Finding 9 — the post-hoc surprise | `artifacts/figures/umap/umap_dec_z64_k21_decade.png` |
| 12 | Conclusion + Future Work | H1–H3 all PASS, deferred items list, what's next | (text only) |

### Rendering Marp

```bash
# Install (one-time)
npm install -g @marp-team/marp-cli

# Render to PDF
marp docs/presentation/PRESENTATION.md --pdf --output docs/presentation/presentation.pdf

# Or PPTX for editing
marp docs/presentation/PRESENTATION.md --pptx --output docs/presentation/presentation.pptx
```

### Slide Style Rules

- **One claim per slide.** If a slide has more than 3 bullets, it's too dense.
- **Numbers on slides should be HUGE** — the +205% / +178% / +99% are the headline numbers; make them stand out (font-size 48+).
- **Every results slide needs a figure** — never just text + numbers.
- **Slide 7 is dense** (the table) — that's OK, it's a reference slide. The presenter says "and here's the full picture" and moves on.
- **Slide 11 (Finding 9)** should land as a "look what we accidentally discovered" surprise — frame it as bonus content.
- **Speaker notes:** add Marp speaker notes (`<!-- ... -->`) for each slide so whoever presents has a script.

### Optional polish (if time permits)
- Consistent color palette (use the team colors): primary `#C2410C` (orange), secondary `#2563EB` (blue), accent `#15803D` (green) — these match the mermaid diagrams already rendered
- Footer with logo or repo URL on every slide
- Animations: keep minimal (built-in Marp transitions are enough)

---

## 10. Style + Brand Guidelines

### Colors (use consistently across report + slides)
- Primary: `#C2410C` (deep orange) — for hero / winner highlights
- Secondary: `#2563EB` (blue) — for analytical / methodology elements
- Accent: `#15803D` (green) — for "PASS" indicators / positive outcomes
- Neutral: `#64748B` (slate) — body text, secondary content
- Background: `#FFFFFF`

### Tone
- **Academic** but **not stilted**. Use "we" not "the authors". Don't pretend to false neutrality.
- **Honest** about limits. Finding 5 (Pareto trade-off) and the DEC marginal NMI gain are framed as *features*, not bugs — this is honest science.
- **Confident** about the 9 findings — they're empirically grounded.

### Language

- **Report:** English only.
- **Slides:** English by default. If the team prefers Turkish for in-class presentation, the friend can translate slide titles + bullet points after rendering — leave a comment block at the top of `PRESENTATION.md` saying "TODO: optional Turkish translation pass before in-class delivery".

---

## 11. Constraints (DO NOT do these)

1. ❌ **Do not run new experiments.** All training is done. The 6 runs in `results.json` are the canonical evidence. Do not add VAE, additional z-dims, or new ablations — those are explicitly deferred for the final report.
2. ❌ **Do not change the numbers.** Report what `results.json` says, even if a number seems "off" (e.g., genre_ARI inversion between vanilla/multi-modal — that's Finding 7, leave it).
3. ❌ **Do not add MeMoji-grade conclusions.** No "in conclusion, deep learning is amazing" platitudes. Stick to what the data shows.
4. ❌ **Do not regenerate UMAPs.** The 13 figures in `artifacts/figures/umap/` are the canonical visualizations. Use them.
5. ❌ **Do not reorganize the repo.** Just add files in `docs/report/` and `docs/presentation/`.
6. ❌ **Do not invoke any subagent or skill** for "creative work" — the design is already done, you're just *writing* the deliverables. No brainstorming, no plan-phase, no spec-phase. Just write.

---

## 12. Validation Checklist (before declaring done)

### For `INTERMEDIATE_REPORT.md`:
- [ ] All 9 findings appear in Results section, each with at least one numerical claim and one figure
- [ ] Abstract leads with the +205% deep-vs-baseline result and previews the 3 hypotheses
- [ ] Figure paths use **relative** references and resolve from repo root
- [ ] Numbers match `artifacts/eval/results.json` exactly (3 decimal places)
- [ ] Section 7 (Limitations) explicitly lists deferred items — be transparent
- [ ] No invented citations or unverified claims
- [ ] Pandoc-renders cleanly: `pandoc INTERMEDIATE_REPORT.md -o report.pdf` (test this)

### For `PRESENTATION.md`:
- [ ] 10–14 slides
- [ ] Marp frontmatter valid (`marp: true`, theme, size 16:9)
- [ ] Every results slide has a figure
- [ ] Slide 12 covers H1–H3 PASS + deferred items
- [ ] Speaker notes (`<!-- ... -->`) present for each slide
- [ ] Renders to PDF: `marp PRESENTATION.md --pdf` (test this)

### Final
- [ ] Both files committed in a single commit (or two adjacent commits) on `main`
- [ ] Push to GitHub successful
- [ ] Tell the user: "Intermediate report and presentation are ready at `docs/report/INTERMEDIATE_REPORT.md` and `docs/presentation/PRESENTATION.md`. Latest commit: [hash]."

---

## 13. If You Get Stuck

- **Question about a number?** → Open `artifacts/eval/results.json` and `docs/FINDINGS.md`. Both files are aligned.
- **Question about a design choice?** → `docs/adr/0001-modeling-hybrid-architecture.md` has the rationale for D1–D10.
- **Question about figure choice?** → Section 6.1 of this brief has a tier-ranked list.
- **Question about scope?** → Section 11 of this brief lists what NOT to do. If something isn't in this brief or the source files, it's out of scope.
- **Diagram needs rendering?** → Open the corresponding `mermaid/*.md`, paste into `mermaid.live`, export PNG, save to `figures/`.

If after consulting these you still need user input, ask a tightly scoped question with options (A/B/C). Do not ask open-ended questions.

---

## 14. Final Notes from the Previous AI

The user (Baran) has been an excellent collaborator — clear feedback, fast iteration, willing to make trade-offs (e.g., choosing single-PR over many-small-PRs, accepting principled trade-off framing for Finding 5). Trust the project's existing decisions; the methodology is sound and the results are clean.

The intermediate report is **the deliverable that closes the MVP and earns the grade for the modeling phase**. The full study (with deferred items) will continue toward the final report later. So: the intermediate report should feel like a *complete, self-contained study at the MVP scope* — not a "we plan to do more" placeholder. The 6 runs + 9 findings are enough for a strong intermediate.

Good luck. The data is in your favor — these results genuinely are this good.

— Claude (previous AI assistant)
