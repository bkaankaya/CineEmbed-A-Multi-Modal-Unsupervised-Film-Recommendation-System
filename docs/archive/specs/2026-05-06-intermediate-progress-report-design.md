# Design — Intermediate Progress Report (LaTeX) + Custom-Designed PPTX

**Date:** 2026-05-06
**Authors:** Ahmet Baran Dinçoğuz, Arda Arvas, Bertan Kaan Kaya
**Course:** SENG 474 — Deep Learning, Spring 2026, TED University
**Status:** APPROVED (user picked B style + pdflatex + python-pptx slides)

---

## 1. Goals

Replace the prior `INTERMEDIATE_REPORT.md` / Marp deck with two professionally-designed deliverables:

1. A **LaTeX progress report** styled as a formal project status update (Style B from brainstorming) — `docs/report/intermediate-progress-report.tex` → `intermediate-progress-report.pdf`.
2. A **custom-designed PPTX deck** built via `python-pptx` (not Marp) — `docs/presentation/intermediate-progress-presentation.pptx`.

Both deliverables consume the canonical empirical evidence already on disk (`artifacts/eval/results.json`, the 13 UMAP figures, the 5 architecture diagrams). No new experiments. No regenerated figures.

## 2. Non-goals

- Not regenerating any figure or rerunning any model.
- Not maintaining the prior `INTERMEDIATE_REPORT.md` markdown — it stays in repo as historical artifact, the `intermediate-progress-report.pdf` is the deliverable.
- Not maintaining the prior Marp deck — replaced.
- Not migrating any existing infrastructure or notebooks.

## 3. Tone

**Project status report**, not research paper. Past tense for completed phases ("the data engineering phase concluded"), present-progressive for in-progress work ("the team is currently writing"), future tense for planned work ("the VAE family will be trained in the next phase"). Third-person where natural ("the team verified", "evaluation produced") rather than first-person plural ("we discovered").

The empirical results from the modeling MVP appear as **evidence of work completed**, not as "research findings". Numbers are reported with their meaning ("genre_NMI=0.332, exceeding the H3 threshold of 0.150 by a wide margin"), not as discoveries.

---

## 4. LaTeX progress report

### 4.1 Engine, class, files

| Item | Value |
|---|---|
| Engine | `pdflatex` (TeX Live 2026) |
| Class | `article`, 11pt, A4 paper |
| Build driver | `latexmk -pdf` via checked-in `docs/report/.latexmkrc` |
| Source | `docs/report/intermediate-progress-report.tex` |
| Bibliography | `docs/report/references.bib` |
| Output (committed) | `docs/report/intermediate-progress-report.pdf` |
| Auxiliary cleanup | `latexmk -c` removes `.aux/.log/.out/.toc/.bbl/.bcf/.run.xml/.fls/.fdb_latexmk` |

### 4.2 Package list

```latex
\usepackage[a4paper, margin=25mm]{geometry}
\usepackage{microtype}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}      % handles common diacritics in pdflatex
\usepackage{lmodern}              % Latin Modern (default) for body
\usepackage[scaled=0.92]{helvet}  % sans for headings
\usepackage{titlesec}             % section heading restyling
\usepackage{booktabs}             % professional tables
\usepackage{array}
\usepackage{tabularx}             % auto-width columns
\usepackage{colortbl}             % row coloring
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}             % loaded last where possible
\usepackage[capitalise]{cleveref} % \cref / \Cref
\usepackage{enumitem}             % tighter lists
\usepackage{amsmath, amssymb}
\usepackage{tcolorbox}            % callout boxes
\tcbuselibrary{skins,breakable}
\usepackage{pgfgantt}             % timeline Gantt
\usepackage{fancyhdr}             % headers/footers
\usepackage[backend=biber, style=numeric, sorting=none]{biblatex}
\addbibresource{references.bib}
```

If `helvet` clashes with body font readability, fall back to default `cmr` everywhere — single-font document is acceptable.

### 4.3 Visual design

**Color palette** (defined once at preamble):

```latex
\definecolor{brandPrimary}{HTML}{C2410C}   % orange — section headings only
\definecolor{brandSecondary}{HTML}{2563EB} % blue — hyperlinks, status:planned
\definecolor{brandAccent}{HTML}{15803D}    % green — status:complete
\definecolor{brandAmber}{HTML}{B45309}     % amber — status:in-progress
\definecolor{brandSlate}{HTML}{64748B}     % slate — status:deferred + body de-emphasis
\definecolor{rowAlt}{HTML}{F8FAFC}         % alternating row shading
```

**Status-to-color mapping (canonical, used in both LaTeX and PPTX):**

| Status badge | Color |
|---|---|
| `COMPLETE` | brandAccent (green `#15803D`) |
| `IN PROGRESS` | brandAmber (`#B45309`) |
| `PLANNED` | brandSecondary (blue `#2563EB`) |
| `DEFERRED` | brandSlate (`#64748B`) |

**Section heading style** (via `titlesec`):

- `\section`: large sans, brandPrimary color, with thin underline rule.
- `\subsection`: medium sans, brandSecondary color.
- `\subsubsection`: small sans, body color.

**Custom commands:**

```latex
\newcommand{\status}[1]{...}  % rendered as colored pill: COMPLETE / IN PROGRESS / DEFERRED / PLANNED
\newcommand{\headline}[2]{...}  % big-number callout (e.g., +205% on genre_NMI vs baseline)
\newcommand{\evidence}[1]{...}  % small-italic "Evidence:" prefix
```

**Tables:**

- `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) — never any vertical lines.
- Alternating row shading via `\rowcolors{2}{rowAlt}{white}`.
- All numerical columns right-aligned (`r`); descriptive columns left-aligned.

**Callout boxes (tcolorbox):**

- "Headline result" box — orange-tinted background, large number left, description right.
- "Risk" box — amber border, plain background.

**Headers/footers (fancyhdr):**

- Even left + odd right: page x of y.
- Even right + odd left: "SENG 474 · CineEmbed".
- Plain page style on chapter/title pages.

**Title page** — single `\maketitle`-replacement page with vertical centering:

```
[ vertical fill ]
                  CineEmbed
   Multi-Modal Unsupervised Embedding of 329,044 Films

           Intermediate Progress Report — v1.0

         Ahmet Baran Dinçoğuz · Arda Arvas
                Bertan Kaan Kaya

       SENG 474 — Deep Learning · Spring 2026
              TED University · 2026-05-06

         Repository: github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System
[ vertical fill ]
```

### 4.4 Section outline

| § | Section | Target pages | Content |
|---|---|---:|---|
| — | Title page | 1 | as above |
| — | Executive Summary | 0.5 | RAG status badge "ON TRACK"; 3-bullet status; headline `+205%` callout box; H1/H2/H3 hypothesis tracker (3-row mini-table) |
| 1 | Project Overview | 1 | 1.1 Goal; 1.2 Scope (329K films, 564 dims, 7 modalities); 1.3 Team responsibilities (3-row table) |
| 2 | Schedule & Milestones | 1 | 2.1 Timeline (`pgfgantt` chart spanning project start → final report); 2.2 Milestone status table with `\status{...}` badges |
| 3 | Work Completed to Date | 3-4 | 3.1 Data engineering phase — describes outputs (564-dim matrix, 7 blocks), Fig 1 (multilingual coverage); 3.2 Architecture design phase — multi-modal backbone, W2 weighting, G2 masking, Fig 2 (architecture diagram); 3.3 Modeling MVP phase — 6 runs trained, 3-axis evaluation, Fig 3 (eval methodology), main results table, headline callout, Fig 4 (3-panel UMAP), Fig 5 (missing-data manifold), Fig 6 (W1 collapse), Fig 7 (language win) |
| 4 | Work In Progress | 0.5 | 4.1 Active workstreams; 4.2 This deliverable's status; expected completion dates |
| 5 | Plan to Final Report | 1 | 5.1 Deferred experiments table (item · why deferred · target completion · owner); 5.2 Final report timeline |
| 6 | Risks & Mitigation | 0.5 | 3-4 RAG-rated risks (compute, scope, time) in `tcolorbox` callouts |
| 7 | Deliverables Status | 0.5 | "Planned vs delivered" checklist table covering all artifacts in scope for the intermediate phase |
| — | References | 0.25 | 5-7 entries via `biblatex`/numeric (DEC, MiniLM, UMAP, sklearn, PyTorch, AAE/IDEC, tabular-DL) |
| A | Appendix A — Per-run results | 1 | Verbatim numerical results from `artifacts/eval/results.json` (3 decimals) |
| B | Appendix B — Decision log summary | 0.5 | D1–D10 one-line summaries (link to ADR) |

**Total page target: 8–10 pages including title page and appendices.**

### 4.5 Numerical claims sourcing

Every number cited in the document MUST come from `artifacts/eval/results.json` at 3-decimal precision, with the exception of derived percentage gains, which are computed as `(deep − baseline) / baseline × 100` rounded to integer percent. Computed values for canonical citations (computed once and cached in the `.tex` file as comments next to each occurrence):

```
+205% = (0.332 - 0.109) / 0.109 × 100  [DEC vs kmeans_raw, genre_NMI]
+295% = (0.332 - 0.084) / 0.084 × 100  [DEC vs pca_kmeans, genre_NMI]
+287% = (0.244 - 0.063) / 0.063 × 100  [DEC vs kmeans_raw, genre_ARI]
+213% = (0.294 - 0.094) / 0.094 × 100  [DEC vs pca_kmeans, lang_NMI]
+178% = (0.264 - 0.095) / 0.095 × 100  [multi-modal vs vanilla, lang_NMI]
+99%  = (0.328 - 0.165) / 0.165 × 100  [W2 vs W1, genre_NMI]
+277% = (0.264 - 0.070) / 0.070 × 100  [W2 vs W1, lang_NMI]
```

### 4.6 Figure list (all already on disk, relative paths from `docs/report/`)

| Fig | Path | Caption (short) |
|---|---|---|
| 1 | `../../artifacts/figures/multilingual_coverage.png` | Long-tailed language distribution motivating the language block's sparsity-aware treatment. |
| 2 | `../../figures/architecture_multimodal.png` | Multi-modal backbone: 7 modality projections → 164-dim concat → 64-dim latent. |
| 3 | `../../figures/three-exis-eval-method.png` | Three-axis evaluation: KMeans (k=21) on the latent, scored independently against genre, decade, and language. |
| 4 | `../../artifacts/figures/umap/umap_comparison_genre.png` | Latent topology evolution: vanilla (mega-blobs) → multi-modal (islands) → DEC (tighter islands). |
| 5 | `../../artifacts/figures/umap/umap_dec_z64_k21_decade.png` | Films with missing release-date form an isolated sub-manifold (red cluster, upper right). |
| 6 | `../../artifacts/figures/umap/umap_ae_z64_w1_genre.png` | W1 ablation: uniform loss weighting collapses fine-structure into diffuse blobs. |
| 7 | `../../artifacts/figures/umap/umap_ae_z64_lang.png` | Multi-modal latent colored by language: distinct micro-clusters per language family. |

### 4.7 References (bibliography)

Minimum 7 entries in `references.bib`:
- Xie et al. 2016 (DEC, ICML)
- Guo et al. 2017 (IDEC)
- Reimers & Gurevych 2019 (MiniLM/SentenceBERT)
- McInnes et al. 2018 (UMAP, arXiv)
- Pedregosa et al. 2011 (scikit-learn, JMLR)
- Paszke et al. 2019 (PyTorch, NeurIPS)
- Gorishniy et al. 2021 (tabular DL, NeurIPS)

### 4.8 Build instructions

`.latexmkrc`:

```perl
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;
$out_dir = '.';
@default_files = ('intermediate-progress-report.tex');
```

User commands:

```bash
cd docs/report/
latexmk            # build PDF
latexmk -c         # clean intermediates (keep PDF)
latexmk -C         # clean everything including PDF
```

---

## 5. Custom-designed PPTX deck

### 5.1 Approach

A Python script (`docs/presentation/build_presentation.py`) using `python-pptx` programmatically constructs the deck with full control over:

- Slide masters (custom one-master-per-slide-type pattern — title slide, content slide, full-bleed image slide, divider)
- Typography — Calibri Body / Calibri Light Headings (matches PowerPoint native, available everywhere)
- Color palette identical to the LaTeX report
- Custom shapes — branded color bar at slide top, page-number footer, status-pill renderer

This produces `docs/presentation/intermediate-progress-presentation.pptx` — a **native, fully editable** PowerPoint file.

### 5.2 Build script structure

```
docs/presentation/build_presentation.py     # entry point
docs/presentation/_slides/
    __init__.py
    theme.py                  # colors, fonts, sizes, helpers
    components.py             # status pill, headline number, table renderer, image cropper
    slides.py                 # one function per slide (slide_01_title, ..., slide_12_close)
docs/presentation/requirements.txt           # python-pptx>=0.6.23, Pillow>=10.0
```

`build_presentation.py` is a thin orchestrator: builds the `Presentation()` object, calls each slide builder in order, saves to disk. Re-running overwrites the output deterministically.

### 5.3 Slide outline (12 slides, 16:9, 1920×1080)

| # | Type | Title | Content |
|---|---|---|---|
| 1 | Title | CineEmbed — Intermediate Progress Report | Project name, subtitle, team, course, date, repo URL |
| 2 | Status | Project Status: ON TRACK | RAG indicator + 3 status bullets + small headline number |
| 3 | Content | Project Goal & Scope | 329K films, 564 dims, 7 modalities table, dataset image right |
| 4 | Content | Schedule & Milestones | Mini-Gantt as embedded image (or shape-drawn), milestone bullets with status pills |
| 5 | Content | Work Completed: Data Engineering | "Phase complete" pill + bullets + Fig (multilingual_coverage) |
| 6 | Content | Work Completed: Architecture | "Phase complete" pill + bullets + Fig (architecture_multimodal) |
| 7 | Content | Work Completed: Modeling MVP | "Phase complete" pill + main results table |
| 8 | Headline | Headline Result: +205% over Non-Deep Baseline | Big "+205%" number, two-row supporting evidence below |
| 9 | Content | Modeling Evidence — Latent Topology | UMAP 3-panel comparison (full-bleed) + 1-sentence caption |
| 10 | Content | Bonus Finding — Missing-Data Manifold | UMAP DEC-decade figure + 2-bullet narrative |
| 11 | Content | Plan to Final Report | Deferred items table (item / why / target completion) |
| 12 | Close | Status Summary & Q&A | Hypotheses tracker + repo URL + thanks |

### 5.4 Visual design language (slides)

| Element | Spec |
|---|---|
| Slide size | 16:9, 13.333" × 7.5" (default PowerPoint widescreen) |
| Margins | 0.5" all around for content; full-bleed allowed for hero images |
| Top color bar | 6pt-tall brand-primary bar across full width (visual anchor on every non-title slide) |
| Title typography | Calibri Light, 36pt, brand-primary, top-left |
| Section subtitle | Calibri, 18pt, brand-slate, immediately under title |
| Body typography | Calibri Body, 18pt, near-black (#1E293B); 14pt for tertiary content |
| Bullet style | • (custom-rendered, brand-secondary), 12pt indent |
| Headline numbers | Calibri Light, **120pt**, brand-primary, paired with 18pt slate caption below |
| Status pills | rounded-rectangle shape, 6pt radius; fill = status color; text = white, Calibri 11pt bold |
| Tables | thin top/mid/bottom borders only (no internal grid); header row fill = #F1F5F9 |
| Page footer | 10pt slate, "SENG 474 · CineEmbed · Slide n / 12", left-aligned, 0.25" from bottom |
| Figure cropping | Pillow used to crop figures to slide-aspect-friendly bounds where needed |

### 5.5 Build and verification

```bash
cd docs/presentation/
pip install -r requirements.txt
python3 build_presentation.py
# → produces intermediate-progress-presentation.pptx
```

The script must be **near-idempotent**: running it twice produces a `.pptx` whose slides, shapes, and content are identical (deterministic shape ordering by inserting in fixed sequence; no timestamp metadata written into custom fields). Internal XML revision IDs and `pptx` core property timestamps are allowed to differ — that's a python-pptx implementation detail outside our control.

After build, the `.pptx` opens cleanly in Microsoft PowerPoint, Apple Keynote, and Google Slides without warnings or layout breakage.

---

## 6. Names & spellings (canonical)

| Canonical | Alt forms allowed |
|---|---|
| Ahmet Baran Dinçoğuz | "Baran" in informal contexts; ASCII fallback "Ahmet Baran Dincoguz" used as `\author` and PPTX text |
| Arda Arvas | — |
| Bertan Kaan Kaya | "Kaan" in informal contexts |

**Decision:** use ASCII-only forms (`Dincoguz`) in both LaTeX `\author` and PPTX text fields per user instruction, since pdflatex Turkish-character handling adds friction without benefit.

## 7. Out of scope

- The previously committed `docs/report/INTERMEDIATE_REPORT.md`, `docs/presentation/PRESENTATION.md`, `CineEmbed_Presentation.pptx`, and `CineEmbed_Presentation.pdf` are **superseded but not deleted** in this work — the new deliverables live alongside. A separate cleanup commit (after user approves the new outputs) can remove them.
- No Beamer slides — user picked python-pptx.
- No Marp work — user picked python-pptx.

## 8. Acceptance criteria

The work is "done" when all of the following are true:

- [ ] `docs/report/intermediate-progress-report.tex` exists and `latexmk` produces a clean PDF with **zero warnings or errors**.
- [ ] PDF is 8–10 pages including title page and appendices.
- [ ] Every numerical claim in the PDF cites `artifacts/eval/results.json` and matches at 3-decimal precision.
- [ ] All 7 figures render at appropriate size with numbered captions, all `\cref{...}` references resolve.
- [ ] `docs/report/.latexmkrc` and `docs/report/references.bib` checked in.
- [ ] `docs/presentation/intermediate-progress-presentation.pptx` opens cleanly in PowerPoint and Keynote.
- [ ] PPTX has 12 slides matching the outline in §5.3.
- [ ] All headline numbers on slides match the LaTeX report numbers.
- [ ] `docs/presentation/build_presentation.py` is near-idempotent (running it twice produces slide/shape/content-identical `.pptx` files; only python-pptx-internal revision IDs may differ).
- [ ] Both deliverables committed and pushed to `main`.
