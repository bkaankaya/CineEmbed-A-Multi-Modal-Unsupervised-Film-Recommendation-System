# CineEmbed — Presentation Diagram Prompts (for Codex)

> **Kullanım:** Her bölüm bağımsız bir Codex prompt'u. Master Context Block (§0) tüm prompt'ların başına eklenebilir veya her prompt zaten yeterince self-contained.
>
> **Tüm prompt'lar İngilizce** (Codex İngilizce'de daha iyi). Slayttaki başlıklar/etiketler Türkçe olabilir — diyagramı oluşturduktan sonra metni düzenle.
>
> **Önerilen format:** Mermaid (akış diyagramları için) + matplotlib (data viz için) + opsiyonel Excalidraw (hand-drawn academic feel için).

---

## 0. Master Project Context Block

**Use this as a preamble for any prompt below. Or paste once at conversation start in Codex, then reference "the project described above".**

```
PROJECT CONTEXT — CineEmbed

CineEmbed is an unsupervised representation learning study on movie metadata, completed for the SENG 474 (Deep Learning) course at TED University, Spring 2026. Team: Baran Dinçoğuz, Arda Arvas, Kaan Kaya.

GOAL: learn a 64-dimensional latent embedding for 329,044 films using multi-modal autoencoders, and evaluate whether the latent geometry recovers three orthogonal label axes — primary_genre (21 classes), decade_bin (~12 classes), lang_top10 (11 classes) — better than non-deep baselines.

DATA: 329,044 films × 564-dim feature matrix, organized into 7 modality blocks (block-contiguous columns):
  • numerical (6 dims): log_popularity, log_vote_count, runtime_norm, vote_average_norm, has_vote, has_engagement
  • genre (22 dims): one-hot genre indicators + has_genre flag
  • language (31 dims): one-hot original_language top-N
  • decade (2 dims): decade_norm + has_release_date
  • awards (6 dims): prior log-counts (Oscar wins/noms, BAFTA, Cannes, etc.)
  • text (384 dims): all-MiniLM-L6-v2 sentence embedding of overview/title
  • director (113 dims): bio_pca_64 + has_director_bio + dir_lang_30 + dir_country_18 + has_director_lang

MODELS TRAINED (z=64 latent for all):
  • vanilla_ae_z64       — concat-AE (single FC encoder, no modality projection) — simple-deep baseline
  • ae_z64               — multi-modal AE with W2 inverse-variance loss weighting (the proposed architecture)
  • ae_z64_w1            — same as ae_z64 but uniform loss weighting (W1 ablation, shows W2 is critical)
  • dec_z64_k21          — Deep Embedded Clustering (initialized from ae_z64, then KL+recon optimization)

BASELINES (non-deep): kmeans_raw_k21 on 564-dim raw features, pca_kmeans_k21 on PCA-reduced 64-dim.

EVALUATION: Cluster all film latents with KMeans (k=21), compute NMI and ARI against three label axes simultaneously. No single label axis is privileged — all three are reported.

KEY RESULTS:
  • dec_z64_k21 wins genre_NMI=0.332 and lang_NMI=0.294 (best deep model)
  • Deep models beat non-deep baselines by +205% to +295% on genre_NMI (massive H2 PASS)
  • Multi-modal architecture beats vanilla concat by +178% on lang_NMI (modality projection critical)
  • W2 inverse-variance weighting beats W1 uniform by +99% on genre_NMI (peer-review-suggested ablation validates the choice)

TECHNICAL STACK: PyTorch 2.10+, scikit-learn, sentence-transformers (consumed), UMAP for visualization, Google Colab T4 GPU for training, GitHub for code, Drive for artifacts.

FILE STRUCTURE:
  data/                  raw + intermediate CSVs
  src/cineembed/         data, backbone, heads, losses, train, eval modules (7 modules, 30 pytest)
  notebooks/             00_colab_setup, 02_train_ae, 04_train_dec, 05_results, 06_umap
  docs/                  FINDINGS.md (results), PROGRESS.md (status), adr/ (decisions), superpowers/ (specs+plans)
  artifacts/             feature_matrix.npz, models/*.pt, eval/results.json, figures/
```

---

## 1. Hero High-Level Architecture (slide 1 / introduction)

**Slayt amacı:** Tek bir bakışta projenin ne yaptığını anlatan paper-style hero figure.

**Tool:** Mermaid `graph LR` (or Excalidraw if you prefer hand-drawn).

**Prompt:**

```
Generate a Mermaid diagram (graph LR direction) titled "CineEmbed — Multi-modal AE/DEC Pipeline" showing the full pipeline at a high level for an academic presentation slide. Components:

1. LEFT: a single rounded box "329,044 films × 564-dim multi-modal feature matrix" — this is the input.

2. CENTER-LEFT: split the input into 7 parallel small boxes labeled with their dim counts:
   numerical(6), genre(22), language(31), decade(2), awards(6), text(384), director(113).
   Group these visually in a subgraph titled "Modality Blocks".

3. CENTER: an orange/highlighted box labeled "Multi-Modal Backbone (modality-specific projections → 164-dim concat → FC → z=64)". Each of the 7 modality blocks should have an arrow pointing into this box.

4. CENTER-RIGHT: from the Backbone box, a single arrow to a box labeled "Latent z ∈ ℝ⁶⁴" (style this prominently — it's the main output).

5. RIGHT: from the Latent box, three parallel arrows to three boxes labeled:
   - "KMeans (k=21) → primary_genre" with metric NMI=0.332
   - "KMeans (k=21) → decade_bin" with metric NMI=0.342
   - "KMeans (k=21) → lang_top10" with metric NMI=0.294
   Group these in a subgraph titled "3-Axis Evaluation".

6. Add a small dashed-line box BELOW the Backbone reading "Variants: vanilla concat-AE (no projection), W1 uniform weights (ablation), DEC (explicit cluster optimization)".

Style: clean academic look, high contrast (works on projector), use 2-3 colors maximum. The diagram should fit in a 16:9 slide.
```

**Çıktı:** Mermaid kodu — `mermaid.live`'de render et → SVG/PNG indir.

---

## 2. Data Pipeline (raw → feature matrix)

**Slayt amacı:** Veri tarafının nasıl 564-dim'e ulaştığını göstermek.

**Tool:** Mermaid `graph TD` (top-down flow).

**Prompt:**

```
Generate a Mermaid graph TD diagram titled "CineEmbed — Data Engineering Pipeline" showing how raw movie data was transformed into the 564-dim feature matrix used by all models.

Flow:
1. TOP: three parallel source boxes (different colors):
   - "TMDB metadata (329K films): title, overview, genres, runtime, popularity, vote_count, vote_average, original_language, release_date"
   - "External awards (Oscars, BAFTA, Cannes, ...): prior counts merged via fuzzy title match"
   - "Wikipedia director bios: scraped, embedded with sentence-transformers"

2. MIDDLE LAYER: three parallel transformation boxes, one per source:
   - "Cleaning + log/standard scaling + missingness flags"
   - "Award counts → log + has_X flags (6 features)"
   - "Bio embeddings → PCA-64 + has_director_bio mask (G2 masked loss)"

3. CENTER: a SINGLE box labeled "Block-contiguous concatenation" that all three flows merge into.

4. BELOW concat: a wide box showing the 7 blocks side-by-side with their dim counts, like a horizontal bar:
   numerical(6) | genre(22) | language(31) | decade(2) | awards(6) | text(384) | director(113) = 564 dims

5. BOTTOM: arrow from the wide box to a final highlighted box "feature_matrix.npz (329,044 × 564, float32, ~700 MB)".

Style: vertical flow (TD), use color to distinguish sources (blue=TMDB, gold=awards, green=director). Keep it clean — this is for an undergraduate ML presentation.
```

**Çıktı:** Mermaid → render → PNG.

---

## 3. Multi-Modal Model Architecture (encoder + decoder)

**Slayt amacı:** Method / Architecture slaytında — ana modeli net göster.

**Tool:** Matplotlib + custom rectangles (or Mermaid if simpler is OK).

**Prompt:**

```
Write Python matplotlib code that draws a clean academic-style multi-modal autoencoder architecture diagram for "CineEmbed — Multi-Modal Backbone (W2 inverse-variance loss)".

Layout (left to right):

ENCODER SIDE (left half):
- 7 input blocks stacked vertically with their dims and colors:
  numerical [6 dims]    — light blue
  genre [22 dims]       — orange
  language [31 dims]    — green
  decade [2 dims]       — gray
  awards [6 dims]       — gold
  text [384 dims]       — purple
  director [113 dims]   — red

- Arrows from each block to a "_BlockProjection" layer (one per modality), with output dims:
  numerical → 16, genre → 16, language → 16, decade → 4, awards → 16, text → 64, director → 32
  (these are the DEFAULT_PROJ_DIMS values)

- The 7 projection outputs concatenate into a "Concat: 164-dim" box.

- Then 2 FC layers shown sequentially:
  Linear(164 → 128) + ReLU + Dropout(0.2)
  Linear(128 → 64)
  Output box labeled "Latent z ∈ ℝ⁶⁴" (highlighted, this is the bottleneck).

DECODER SIDE (right half, mirror image):
- From z (64), an FC stack: Linear(64 → 128) + ReLU → Linear(128 → 164) → split into 7 modality-specific decoders.
- Each modality has its own _BlockDecoder reconstructing back to its original dim.
- Show the 7 reconstructed outputs (X̂_numerical, X̂_genre, ...) on the far right.

LOSS BLOCK (below the figure):
- Show "L = Σ_b w_b · MSE(X_b, X̂_b)" where w_b are the W2 inverse-variance weights.
- Annotate w_b clipped to [0.1, 10.0] in small text.
- Special note: "G2 mask: director bio loss only on rows with has_director_bio=1 (96.8% of rows masked out)".

Style: use rectangles with rounded corners, arrows with arrowheads, monospace font for dim labels. Save as 'figures/architecture_multimodal.png' at 200 DPI. Figure size 14x6 inches.

Vanilla baseline contrast (small inset in bottom-right corner):
- A tiny version showing just: 564-dim concat → Linear(564→128) → Linear(128→64), labeled "Vanilla concat-AE (no modality projection)".
```

**Çıktı:** matplotlib PNG. Yoğun bir diyagram — gerekirse Codex'e "split into encoder-only and decoder-only versions" dedirt.

---

## 4. Three-Tier Model Taxonomy (comparison)

**Slayt amacı:** Karşılaştırma slaytında — neden bu kadar çok model denedik?

**Tool:** Mermaid `graph TD` veya matplotlib pyramid/tier diagram.

**Prompt:**

```
Generate a Mermaid graph TD diagram titled "Three-Tier Model Taxonomy — CineEmbed" with three horizontal layers (tiers) of boxes from bottom to top:

TIER 1 (BOTTOM, gray fill) — "Non-Deep Baselines":
  - Box: "kmeans_raw_k21 (564 dim)" — genre_NMI=0.109
  - Box: "pca_kmeans_k21 (PCA-64)"  — genre_NMI=0.084

TIER 2 (MIDDLE, light blue) — "Simple Deep Baseline":
  - Box: "vanilla_ae_z64 (concat-AE)" — genre_NMI=0.287

TIER 3 (TOP, orange/highlighted) — "Multi-Modal Deep Models":
  - Box: "ae_z64_w1 (W1 uniform — ablation)" — genre_NMI=0.165
  - Box: "ae_z64 (W2 inverse-variance)" — genre_NMI=0.328
  - Box: "dec_z64_k21 (DEC — explicit clustering)" — genre_NMI=0.332 ★ BEST

Add diagonal arrows from lower tiers to upper tiers labeled with relative gains:
  - Tier 1 (best=0.109) → Tier 2 (0.287): "+163% (deep beats non-deep)"
  - Tier 2 (0.287) → Tier 3 best (0.332): "+15.8% (multi-modal beats simple deep)"
  - Tier 1 best (0.109) → Tier 3 best (0.332): "+205% TOTAL"

Below the diagram add a small caption:
"H2 spec criterion: best deep > best baseline by ≥10%. PASS at +205%."

Style: tiers stacked vertically with subtle background colors per tier, clean text, no clutter.
```

**Çıktı:** Mermaid render. Bu hikaye anlatıcı bir slayt için ideal.

---

## 5. Evaluation Methodology (3-axis ground truth)

**Slayt amacı:** "Nasıl ölçtük?" slaytı — yöntemi savun.

**Tool:** Mermaid + small caption box.

**Prompt:**

```
Generate a Mermaid graph LR diagram titled "Three-Axis Evaluation Methodology — Why We Don't Privilege One Label" for an academic ML presentation.

Flow:
1. LEFT: input box "Latent embeddings z ∈ ℝ^(N × 64) for all 329,044 films".

2. CENTER: a single box "KMeans clustering (k=21, n_init=20, seed=42)" that takes z as input and outputs a "cluster_id ∈ {0..20}" for each film.

3. RIGHT: three parallel evaluation paths from cluster_id, each going to a labeled axis:
   - Path 1: "Compare clusters vs primary_genre (21 classes from genres column)" → metrics box "NMI(genre), ARI(genre)"
   - Path 2: "Compare clusters vs decade_bin (~12 classes, derived from release_date)" → metrics box "NMI(decade), ARI(decade)"
   - Path 3: "Compare clusters vs lang_top10 (11 classes, top languages + Other)" → metrics box "NMI(lang), ARI(lang)"

4. BELOW the three metric boxes, add a small justification panel listing 4 reasons we use 3 axes (one bullet each):
   - "Genre is multi-label and overlaps heavily — single-label NMI underestimates true clustering quality"
   - "Decade is ordinal and easy — high NMI doesn't prove the model learned semantics, just chronology"
   - "Language is sparse (~99% zero per row) — only multi-modal models capture it"
   - "Reporting all three reveals the principled trade-offs each architecture makes"

Style: horizontal flow, parallel branches in distinct colors (blue/orange/green for the three axes). Clean academic look.
```

---

## 6. Results Hero Bar Chart (NMI comparison across models × axes)

**Slayt amacı:** Results slaytı — sayısal kanıt.

**Tool:** Matplotlib grouped bar chart.

**Prompt:**

```
Write Python matplotlib code that produces a grouped bar chart titled "CineEmbed — NMI Across All Models and Axes (z=64)" for an academic presentation.

Data (hardcode this in the script — these are the actual results):

    models = ['kmeans_raw',   'pca_kmeans',   'vanilla_ae',   'ae_z64_w1',   'ae_z64',   'dec_z64_k21']
    genre_nmi  = [0.109, 0.084, 0.287, 0.165, 0.328, 0.332]
    decade_nmi = [0.233, 0.224, 0.369, 0.367, 0.341, 0.342]
    lang_nmi   = [0.075, 0.094, 0.095, 0.070, 0.264, 0.294]

Layout:
- 6 model groups along the x-axis, 3 bars per group (genre / decade / lang).
- Use a colorblind-friendly palette for the 3 axes (suggest: tab:blue, tab:orange, tab:green).
- Y-axis: NMI (0 to 0.4 range), grid on for readability.
- Bar values shown above each bar (rotated 90 degrees if needed).
- Legend in upper-left.
- Mark the winning bar per axis with a small star (★) above it:
    genre winner: dec_z64_k21
    decade winner: vanilla_ae
    lang winner:  dec_z64_k21

Visual emphasis:
- Group the bars visually into "Non-Deep | Deep" with a vertical dashed separator after pca_kmeans.
- Background-shade the "Non-Deep" region a very light gray to signal the divide.
- Add a horizontal dashed line at NMI=0.15 labeled "H3 absolute floor".

Save as 'figures/results_bar_chart.png' at 200 DPI, figure size 12x6 inches. Use tight_layout. Title in 16pt, axis labels 12pt, tick labels 10pt.
```

**Çıktı:** PNG. Bu **rapor + sunum için MUST** bir figür.

---

## 7. Project Directory Tree (codebase organization)

**Slayt amacı:** "Mühendislik tarafında ne yaptık?" — kod organizasyonunu göster.

**Tool:** ASCII tree veya Mermaid `graph TD`.

**Prompt:**

```
Generate a Mermaid graph TD diagram titled "CineEmbed — Repository Structure (Reproducibility-First Engineering)" showing the file/directory layout of the project. Use rectangles for directories (light blue) and rounded boxes for files (white). Annotate each node with a one-line role description.

Tree:

CineEmbed-/  (repo root)
├── data/                           — raw + intermediate CSVs (movies_eda_final.csv 329K rows)
├── src/cineembed/                  — Python package, 7 modules, 30 pytest passing
│   ├── data.py                     — feature loading + 3-axis label extraction
│   ├── backbone.py                 — MultiModalBackbone (modality projections + FC)
│   ├── heads.py                    — AEHead, VAEHead, DECHead
│   ├── losses.py                   — W2 inverse-variance + G2 director mask
│   ├── train.py                    — generic train_model + checkpointing
│   └── eval.py                     — KMeans clustering + 3-axis NMI/ARI + UMAP
├── notebooks/
│   ├── 00_colab_setup.ipynb        — Drive mount + git clone + verify
│   ├── 02_train_ae.ipynb           — 3 AE runs (vanilla + W2 + W1)
│   ├── 04_train_dec.ipynb          — DEC training from ae_z64 init
│   ├── 05_results.ipynb            — KMeans baselines + final comparison
│   └── 06_umap.ipynb               — UMAP visualizations
├── docs/
│   ├── FINDINGS.md                 — empirical results, hero claims, hypothesis status
│   ├── PROGRESS.md                 — phase tracking
│   ├── adr/0001-modeling-hybrid-architecture.md  — D1-D10 decisions
│   └── superpowers/specs+plans/    — design contracts
├── artifacts/                      — feature_matrix.npz, models/*.pt, eval/results.json, figures/
├── tests/                          — 30 pytest covering all 7 modules
└── README.md

Format the diagram as a tree structure where each directory contains its children visually. Use a left-to-right layout (or top-down) — whichever fits a 16:9 slide better. Style: clean academic, monospace font for paths, sans-serif for descriptions.
```

**Çıktı:** Mermaid. Slayt için gerekirse ASCII tree alternatifi de iste.

---

## Hangi prompt hangi slayt için?

| Slayt | Prompt | Süre |
|---|---|---|
| 1. Title / Intro | §1 (Hero Architecture) | tek render, 5 dk |
| 2. Problem & Data | §2 (Data Pipeline) | tek render, 5 dk |
| 3. Method / Model | §3 (Multi-Modal Architecture) + §4 (Tier Taxonomy) | iki figür, 15 dk |
| 4. Evaluation | §5 (3-Axis Methodology) | tek render, 5 dk |
| 5. Results | §6 (Bar Chart) — **MUST** + UMAP figürlerin (06_umap.ipynb'den) | 10 dk |
| 6. Discussion | §6 yeniden vurgu + FINDINGS.md'deki hero claims | metin slaydı |
| 7. Engineering / Reproducibility (opsiyonel) | §7 (Directory Tree) | 5 dk |

**Toplam:** ~50 dk Codex iterasyonu + revizyon. Her diyagramı render ettikten sonra "make this more academic / less cluttered / use this color palette" gibi follow-up prompt'larla iyileştir.

---

## Codex'e nasıl yaklaşmalı

1. **Master Context'i (§0) tek seferde paste et**, sonra her diyagram için sadece o bölümün spesifik prompt'unu yaz.
2. **İlk render'dan sonra revize et** — "make the arrows thicker", "swap colors X and Y", "remove this label".
3. **Çıktı hep aynı format olsun**:
   - Mermaid → `mermaid.live`'de render → "Export PNG" ya da "Export SVG"
   - Matplotlib → script'i çalıştır, PNG'yi al
4. **Tüm figürleri `artifacts/figures/presentation/` altına koy**, slaytlara oradan al.

---

## Bonus: tek bir master figure (opsiyonel)

Eğer tek bir "everything in one diagram" hero figure istersen, §1'deki Hero Architecture'a §6'daki bar chart'ı ekleyerek bir 2-panel composite oluşturabilirsin. Bu README/poster için de iyi:

```
Generate a 2-panel matplotlib figure for a research poster:
- Top panel (60%): the architecture diagram from prompt §3 (multi-modal AE schematic)
- Bottom panel (40%): the bar chart from prompt §6 (NMI across models)
- Shared title at the top: "CineEmbed: Multi-Modal Embedding for 329K Films — SENG 474, TED University, 2026"
- Add team names ("Baran Dinçoğuz, Arda Arvas, Kaan Kaya") in small font under the title.
- Save as figures/poster_master.png at 300 DPI, figure size 16x10 inches.
```
