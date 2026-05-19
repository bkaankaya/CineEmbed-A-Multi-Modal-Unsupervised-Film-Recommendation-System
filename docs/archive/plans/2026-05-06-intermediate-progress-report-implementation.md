# Intermediate Progress Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a formal LaTeX progress report (PDF) and a custom-designed native `.pptx` deck for SENG 474 intermediate review, replacing the prior Marp/markdown deliverables.

**Architecture:** LaTeX side is a single-document `pdflatex` build via `latexmk` (8–10 page A4 article with custom status/headline/evidence commands, `biblatex` bibliography, `pgfgantt` timeline, `tcolorbox` callouts). PPTX side is a Python build script using `python-pptx`, factored into theme/components/slides modules so each slide is a small function over reusable primitives. Both deliverables share one canonical color palette, one set of figures, and the same source-of-truth numbers from `artifacts/eval/results.json`.

**Tech Stack:** LaTeX (`pdflatex`, `latexmk`, `biblatex` + `biber`, `pgfgantt`, `tcolorbox`, `booktabs`, `titlesec`, `fancyhdr`), Python 3 (`python-pptx >= 0.6.23`, `Pillow >= 10`), bash for builds.

**Spec:** `docs/superpowers/specs/2026-05-06-intermediate-progress-report-design.md`

**Files created or touched:**

- `docs/report/intermediate-progress-report.tex` — main LaTeX document
- `docs/report/references.bib` — bibliography database
- `docs/report/.latexmkrc` — build configuration
- `docs/report/intermediate-progress-report.pdf` — build output (committed)
- `docs/presentation/build_presentation.py` — entry-point script
- `docs/presentation/_slides/__init__.py` — package marker
- `docs/presentation/_slides/theme.py` — palette, fonts, EMU helpers
- `docs/presentation/_slides/components.py` — reusable shape primitives (status pill, headline number, table renderer)
- `docs/presentation/_slides/slides.py` — one builder function per slide
- `docs/presentation/requirements.txt` — pinned Python deps
- `docs/presentation/intermediate-progress-presentation.pptx` — build output (committed)

**Canonical numbers (cite verbatim from `artifacts/eval/results.json`):**

| Run | gNMI | gARI | dNMI | dARI | lNMI | lARI | epochs | val_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_ae_z64` | 0.287 | 0.247 | 0.369 | 0.175 | 0.095 | 0.030 | 58 | 0.013 |
| `ae_z64` | 0.328 | 0.229 | 0.341 | 0.211 | 0.264 | 0.090 | 69 | 0.021 |
| `ae_z64_w1` | 0.165 | 0.094 | 0.367 | 0.176 | 0.070 | 0.026 | 37 | 0.045 |
| `dec_z64_k21` | 0.332 | 0.244 | 0.342 | 0.210 | 0.294 | 0.090 | 21 | 0.127 |
| `kmeans_raw_k21` | 0.109 | 0.063 | 0.233 | 0.093 | 0.075 | 0.026 | — | — |
| `pca_kmeans_k21` | 0.084 | 0.061 | 0.224 | 0.085 | 0.094 | 0.042 | — | — |

**Derived headline percentages:** +205% (DEC vs `kmeans_raw`, gNMI), +295% (DEC vs `pca_kmeans`, gNMI), +287% (DEC vs `kmeans_raw`, gARI), +213% (DEC vs `pca_kmeans`, lNMI), +178% (`ae_z64` vs `vanilla`, lNMI), +99% (W2 vs W1, gNMI), +277% (W2 vs W1, lNMI).

---

## Phase 1 — LaTeX foundation

### Task 1.1: Bibliography database

**Files:**
- Create: `docs/report/references.bib`

- [ ] **Step 1: Create the bib file**

Write `docs/report/references.bib`:

```bibtex
@inproceedings{xie2016dec,
  title     = {Unsupervised Deep Embedding for Clustering Analysis},
  author    = {Xie, Junyuan and Girshick, Ross and Farhadi, Ali},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  pages     = {478--487},
  year      = {2016}
}

@inproceedings{guo2017idec,
  title     = {Improved Deep Embedded Clustering with Local Structure Preservation},
  author    = {Guo, Xifeng and Gao, Long and Liu, Xinwang and Yin, Jianping},
  booktitle = {Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {1753--1759},
  year      = {2017}
}

@inproceedings{reimers2019sbert,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2019}
}

@article{mcinnes2018umap,
  title   = {{UMAP}: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author  = {McInnes, Leland and Healy, John and Melville, James},
  journal = {arXiv preprint arXiv:1802.03426},
  year    = {2018}
}

@article{pedregosa2011sklearn,
  title   = {Scikit-learn: Machine Learning in {P}ython},
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}

@inproceedings{paszke2019pytorch,
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS)},
  year      = {2019}
}

@inproceedings{gorishniy2021tabular,
  title     = {Revisiting Deep Learning Models for Tabular Data},
  author    = {Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year      = {2021}
}
```

- [ ] **Step 2: Commit**

```bash
cd "<repo-root>"
git add docs/report/references.bib
git commit -m "docs(report): add references.bib for intermediate progress report"
```

---

### Task 1.2: latexmk configuration

**Files:**
- Create: `docs/report/.latexmkrc`

- [ ] **Step 1: Create the config**

Write `docs/report/.latexmkrc`:

```perl
# latexmk configuration for intermediate-progress-report.tex
# Run from inside docs/report/:  latexmk        → builds PDF
#                                latexmk -c     → cleans intermediates
#                                latexmk -C     → cleans everything including PDF

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;
$out_dir = '.';
@default_files = ('intermediate-progress-report.tex');

# Extra files to clean with -c
$clean_ext = 'bbl bcf run.xml fls fdb_latexmk synctex.gz';
```

- [ ] **Step 2: Commit**

```bash
git add docs/report/.latexmkrc
git commit -m "docs(report): add latexmk config for intermediate progress report build"
```

---

### Task 1.3: LaTeX preamble + title page + minimal buildable skeleton

**Files:**
- Create: `docs/report/intermediate-progress-report.tex`

Goal: produce a buildable 1-page document (title page only) that confirms the toolchain works end-to-end, including biber.

- [ ] **Step 1: Write the preamble + title page**

Write `docs/report/intermediate-progress-report.tex`:

```latex
\documentclass[11pt,a4paper]{article}

% ===== Geometry & typography =====
\usepackage[a4paper, margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[scaled=0.92]{helvet}
\usepackage{microtype}

% ===== Tables, math, figures =====
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}
\usepackage{pgfgantt}

% ===== Bibliography =====
\usepackage[backend=biber, style=numeric, sorting=none]{biblatex}
\addbibresource{references.bib}

% ===== Brand colors =====
\definecolor{brandPrimary}{HTML}{C2410C}
\definecolor{brandSecondary}{HTML}{2563EB}
\definecolor{brandAccent}{HTML}{15803D}
\definecolor{brandAmber}{HTML}{B45309}
\definecolor{brandSlate}{HTML}{64748B}
\definecolor{rowAlt}{HTML}{F8FAFC}
\definecolor{nearBlack}{HTML}{1E293B}

% ===== Hyperref (load late) =====
\usepackage[hidelinks, colorlinks=true, linkcolor=brandSecondary, urlcolor=brandSecondary, citecolor=brandSecondary]{hyperref}
\usepackage[capitalise]{cleveref}

% ===== Section heading style =====
\titleformat{\section}
  {\sffamily\Large\bfseries\color{brandPrimary}}
  {\thesection}{0.6em}
  {}[\vspace{2pt}\hrule height 0.5pt]
\titleformat{\subsection}
  {\sffamily\large\bfseries\color{brandSecondary}}
  {\thesubsection}{0.5em}{}
\titleformat{\subsubsection}
  {\sffamily\normalsize\bfseries\color{nearBlack}}
  {\thesubsubsection}{0.4em}{}

% ===== Headers / footers =====
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily\color{brandSlate} SENG 474 \textbullet{} CineEmbed}
\fancyhead[R]{\small\sffamily\color{brandSlate} Intermediate Progress Report}
\fancyfoot[C]{\small\sffamily\color{brandSlate} Page \thepage{} of \pageref{LastPage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\usepackage{lastpage}

% ===== Custom commands =====
% \status{complete|inprogress|planned|deferred}
\newcommand{\status}[1]{%
  \ifstrequal{#1}{complete}{\colorbox{brandAccent}{\textcolor{white}{\sffamily\bfseries\small\,COMPLETE\,}}}{%
  \ifstrequal{#1}{inprogress}{\colorbox{brandAmber}{\textcolor{white}{\sffamily\bfseries\small\,IN PROGRESS\,}}}{%
  \ifstrequal{#1}{planned}{\colorbox{brandSecondary}{\textcolor{white}{\sffamily\bfseries\small\,PLANNED\,}}}{%
  \ifstrequal{#1}{deferred}{\colorbox{brandSlate}{\textcolor{white}{\sffamily\bfseries\small\,DEFERRED\,}}}{}}}}%
}
\usepackage{etoolbox} % for \ifstrequal
\usepackage{xparse}

% Headline-number callout box
\newtcolorbox{headlinebox}{%
  colback=brandPrimary!8!white,
  colframe=brandPrimary,
  boxrule=0.6pt,
  arc=2pt,
  left=10pt, right=10pt, top=8pt, bottom=8pt,
}

% Risk callout box
\newtcolorbox{riskbox}[1]{%
  colback=brandAmber!6!white,
  colframe=brandAmber,
  boxrule=0.6pt,
  arc=2pt,
  title={#1},
  fonttitle=\sffamily\bfseries\small\color{white},
  coltitle=white,
  colbacktitle=brandAmber,
  left=8pt, right=8pt, top=6pt, bottom=6pt,
}

% Evidence prefix
\newcommand{\evidence}[1]{\textit{\color{brandSlate}\small Evidence: }\small #1}

% Big number on a line (used in headline boxes)
\newcommand{\bignumber}[1]{{\fontsize{36pt}{42pt}\sffamily\bfseries\color{brandPrimary}#1}}

% ===== Document =====
\begin{document}

% ----- Title page (no header/footer) -----
\thispagestyle{empty}
\vspace*{4cm}
\begin{center}
{\sffamily\Huge\bfseries\color{brandPrimary} CineEmbed} \\[0.4cm]
{\sffamily\Large Multi-Modal Unsupervised Embedding of 329{,}044 Films} \\[1.6cm]
{\sffamily\large\color{brandSecondary} Intermediate Progress Report \textbullet{} v1.0} \\[2.5cm]
{\large Ahmet Baran Dincoguz \quad\textbullet\quad Arda Arvas \quad\textbullet\quad Bertan Kaan Kaya} \\[1.0cm]
{\sffamily\normalsize SENG 474 --- Deep Learning, Spring 2026} \\
{\sffamily\normalsize TED University} \\[0.6cm]
{\sffamily\normalsize 2026-05-06} \\[1.5cm]
{\small\color{brandSlate}Repository: \href{https://github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System}{github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System}}
\end{center}
\vfill
\clearpage

% Body sections will be added in subsequent tasks.
\section*{Body sections to be inserted by subsequent tasks}

\printbibliography

\end{document}
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report"
latexmk
```

Expected: `intermediate-progress-report.pdf` is created in `docs/report/`. Build completes with exit code 0. PDF is at least 1 page.

If `\ifstrequal` errors, ensure `\usepackage{etoolbox}` is loaded BEFORE `\newcommand{\status}`. Move it earlier in the preamble if needed.

- [ ] **Step 3: Verify page count**

```bash
pdfinfo intermediate-progress-report.pdf 2>/dev/null | grep Pages
# expected: Pages: 2  (title + placeholder + bibliography section)
```

If `pdfinfo` is unavailable on the system, skip — the build success itself is the gate.

- [ ] **Step 4: Commit**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): scaffold intermediate-progress-report with title page + preamble"
```

Note: do NOT commit the PDF yet — it will be committed at the very end after all sections land.

---

## Phase 2 — LaTeX content sections

Each task in this phase replaces the placeholder line `\section*{Body sections to be inserted by subsequent tasks}` (or the body region between the title page and `\printbibliography`) with the relevant content. **Add content; do not remove the bibliography or other earlier-added content** when applying a task.

### Task 2.1: Executive Summary

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Replace the body placeholder with the executive summary**

Locate `\section*{Body sections to be inserted by subsequent tasks}` and replace it with:

```latex
% ===== Executive Summary =====
\section*{Executive Summary}
\addcontentsline{toc}{section}{Executive Summary}

\noindent\status{complete}\ \textbf{Modeling MVP phase delivered on schedule.} The intermediate phase of the CineEmbed project is on track: all three pre-registered hypotheses (H1, H2, H3) have been validated, six models have been trained and evaluated against three orthogonal label axes, and the deferred work for the final report has been scoped and prioritized.

\begin{itemize}[leftmargin=1.2em, itemsep=2pt, topsep=2pt]
  \item \textbf{Data engineering:} 564-dimensional feature matrix over 329{,}044 films, organized into seven block-contiguous modalities, frozen at v1.2.
  \item \textbf{Modeling:} six runs trained spanning three model tiers (non-deep baselines, simple deep, multi-modal deep + DEC fine-tune); evaluation completed against \texttt{primary\_genre}, \texttt{decade\_bin}, and \texttt{lang\_top10}.
  \item \textbf{Hypotheses:} H1 (DEC > AE on \texttt{genre\_NMI}), H2 (best-deep $\geq 1.10\times$ best-non-deep), H3 (NMI $> 0.15$ floor) all PASS; H2 by an order of magnitude.
\end{itemize}

\vspace{4pt}
\begin{headlinebox}
\noindent\bignumber{$+205\%$} \hfill \begin{minipage}[b]{0.65\linewidth}\raggedleft\small relative gain on \texttt{genre\_NMI} of the best deep model (\texttt{dec\_z64\_k21}: 0.332) over the strongest non-deep baseline (\texttt{kmeans\_raw\_k21}: 0.109). Pre-registered threshold for H2 was $+10\%$.\end{minipage}
\end{headlinebox}

\vspace{4pt}
\noindent\textbf{Hypothesis tracker:}

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabularx}{\linewidth}{@{}l X r l@{}}
\toprule
\textbf{ID} & \textbf{Statement} & \textbf{Result} & \textbf{Status} \\
\midrule
H1 & DEC \texttt{genre\_NMI} > AE \texttt{genre\_NMI}              & $0.332 > 0.328$ & \status{complete} PASS \\
H2 & Best-deep \texttt{genre\_NMI} $\geq 1.10\times$ best-non-deep  & $+205\%$        & \status{complete} PASS \\
H3 & Best-deep \texttt{genre\_NMI} > 0.15 absolute floor            & $0.332 \gg 0.15$& \status{complete} PASS \\
\bottomrule
\end{tabularx}
\end{center}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report"
latexmk
```

Expected: builds with exit code 0. PDF page count grows to 3 (title + summary page + bib).

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add executive summary with H1-H3 hypothesis tracker"
```

---

### Task 2.2: §1 Project Overview

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Insert §1 immediately after the executive summary `\clearpage`**

Insert this block (and keep everything that follows it untouched):

```latex
% ===== 1. Project Overview =====
\section{Project Overview}
\label{sec:overview}

\subsection{Goal}
The CineEmbed project trains unsupervised 64-dimensional representations of movie metadata via multi-modal autoencoder and Deep Embedded Clustering (DEC) architectures, with the objective of demonstrating that the learned latent geometry recovers three orthogonal label axes --- primary genre, decade, and original language --- substantially better than non-deep KMeans baselines.

\subsection{Scope}
The dataset is a corpus of 329{,}044 films assembled from TMDB metadata, awards records (Oscar / BAFTA / Cannes), and Wikipedia-derived director biographies. The feature matrix has 564 dimensions organized into seven block-contiguous modalities:

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l r p{8.4cm}@{}}
\toprule
\textbf{Modality block} & \textbf{Dim} & \textbf{Contents} \\
\midrule
\texttt{numerical} & 6   & log\_popularity, log\_vote\_count, runtime\_norm, vote\_average\_norm, has\_vote, has\_engagement \\
\texttt{genre}     & 22  & 21-way one-hot + \texttt{has\_genre} flag \\
\texttt{language}  & 31  & top-30 original\_language one-hot + flag (sparse: $\sim$99\% zero per row) \\
\texttt{decade}    & 2   & decade\_norm + has\_release\_date \\
\texttt{awards}    & 6   & log-counts of prior Oscar / BAFTA / Cannes wins and nominations \\
\texttt{text}      & 384 & \texttt{all-MiniLM-L6-v2} sentence embeddings of overview + title \\
\texttt{director}  & 113 & bio\_pca\_64 + has\_director\_bio + dir\_lang\_30 + dir\_country\_18 + has\_director\_lang \\
\bottomrule
\end{tabular}
\end{center}

\noindent See \cref{fig:multilingual-coverage} for the long-tailed language distribution that motivates the language block's sparsity-aware treatment.

\begin{figure}[h]
\centering
\includegraphics[width=0.78\linewidth]{../../artifacts/figures/multilingual_coverage.png}
\caption{Multilingual coverage of the 329{,}044-film corpus. Top-3 languages dominate; the long tail is what motivates the modality-specific projection design (\cref{sec:work-completed}).}
\label{fig:multilingual-coverage}
\end{figure}

\subsection{Team and Responsibilities}
\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l p{10cm}@{}}
\toprule
\textbf{Member} & \textbf{Primary responsibility} \\
\midrule
Ahmet Baran Dincoguz & Architecture design, modeling pipeline, evaluation, repository ownership \\
Arda Arvas           & Data engineering, awards merge, feature blocks \\
Bertan Kaan Kaya     & Director-bio extension, UMAP analysis, presentation deliverables \\
\bottomrule
\end{tabular}
\end{center}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds, page count is 4. Verify Figure 1 renders (open the PDF or use `pdftotext` and grep for "Multilingual coverage").

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §1 Project Overview with feature-block table and Fig 1"
```

---

### Task 2.3: §2 Schedule & Milestones

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Insert §2 after §1**

```latex
% ===== 2. Schedule & Milestones =====
\section{Schedule and Milestones}
\label{sec:schedule}

\subsection{Project Timeline}

\noindent The project is organized into four sequential phases, of which three have completed and one (Final Report) is upcoming. The Gantt chart below shows the planned and actual schedule.

\begin{center}
\begin{ganttchart}[
  hgrid, vgrid,
  title height=1, title label font=\sffamily\small,
  bar height=0.55,
  bar/.append style={fill=brandSecondary, draw=brandSecondary},
  bar label font=\sffamily\small,
  group/.append style={fill=brandPrimary, draw=brandPrimary},
  group label font=\sffamily\bfseries\small\color{white},
  milestone/.append style={fill=brandAccent, draw=brandAccent},
  milestone label font=\sffamily\small,
  x unit=0.85cm,
  y unit chart=0.6cm,
]{1}{12}
  \gantttitle{Apr}{4} \gantttitle{May}{4} \gantttitle{Jun}{4} \\
  \gantttitlelist{1,...,12}{1} \\
  \ganttbar{Data engineering}{1}{3} \\
  \ganttbar{Architecture design}{3}{5} \\
  \ganttbar{Modeling MVP}{5}{8} \\
  \ganttbar{Intermediate report}{8}{9} \\
  \ganttmilestone{MVP delivered}{9} \\
  \ganttbar[bar/.append style={fill=brandAmber, draw=brandAmber}]{VAE + ablations}{9}{11} \\
  \ganttbar[bar/.append style={fill=brandAmber, draw=brandAmber}]{Final report}{11}{12} \\
  \ganttmilestone{Final delivery}{12}
\end{ganttchart}
\end{center}

\subsection{Milestone Status}

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l l l@{}}
\toprule
\textbf{Milestone} & \textbf{Target window} & \textbf{Status} \\
\midrule
Feature matrix v1.2 frozen        & Apr 2026  & \status{complete} \\
Multi-modal architecture finalized& Apr 2026  & \status{complete} \\
Six runs trained and evaluated    & May 2026  & \status{complete} \\
Pre-registered hypotheses tested  & May 2026  & \status{complete} \\
UMAP latent analysis              & May 2026  & \status{complete} \\
Intermediate progress report      & May 2026  & \status{inprogress} \\
VAE family training (z = 32/64/128) & Jun 2026 & \status{planned} \\
F1 / F2 modality ablations        & Jun 2026  & \status{planned} \\
DEC k-sweep (9-cell grid)         & Jun 2026  & \status{planned} \\
Final report                      & Jun 2026  & \status{planned} \\
\bottomrule
\end{tabular}
\end{center}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds, page count grows by 1. Gantt chart renders without "package not found" errors. If `pgfgantt` is missing, install:

```bash
tlmgr install pgfgantt
```

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §2 Schedule and Milestones with Gantt timeline"
```

---

### Task 2.4: §3 Work Completed — intro + §3.1 Data Engineering

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Insert §3 intro and §3.1**

```latex
% ===== 3. Work Completed to Date =====
\section{Work Completed to Date}
\label{sec:work-completed}

\noindent Three phases have completed since project kickoff. Each subsection below summarizes the phase scope, the deliverables produced, and the empirical evidence supporting completion.

\subsection{Data engineering phase \hfill \status{complete}}
\label{sec:data-engineering}

\noindent The team assembled a unified feature matrix from three input sources:

\begin{itemize}[leftmargin=1.2em, itemsep=2pt, topsep=2pt]
\item \textbf{TMDB} --- 329{,}044 films, basic metadata (title, overview, runtime, language, genres, votes, popularity, release date).
\item \textbf{Awards records} --- Oscar, BAFTA, Cannes wins and nominations, merged on title and release year, used to compute prior log-counts as the \texttt{awards} block.
\item \textbf{Wikipedia director bios} --- raw text scraped per credited director, embedded with \texttt{all-MiniLM-L6-v2}, then projected to 64 components via PCA. Coverage is partial: 96.8\% of films lack a Wikipedia bio for any of their credited directors.
\end{itemize}

\noindent Three engineering decisions are worth noting because they shape the modeling phase:
\begin{enumerate}[leftmargin=1.4em, itemsep=2pt, topsep=2pt]
\item Sparse modalities (language, director-language, director-country) preserved as one-hot blocks rather than embedded, to keep them interpretable.
\item Missing release dates encoded as a binary flag (\texttt{has\_release\_date}) rather than imputed --- this turned out to be structurally relevant (see \cref{sec:modeling-mvp}).
\item Director-bio reconstruction loss masked by \texttt{has\_director\_bio} (G2 masking) so that 96.8\% of films contribute zero loss for that block, preventing the autoencoder from learning to predict zeros from zeros.
\end{enumerate}

\noindent\evidence{The frozen 564-dim feature matrix, EDA notebook (\texttt{notebooks/01\_eda.ipynb}), and 21 supporting figures under \texttt{artifacts/figures/}.}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds. `\cref{sec:modeling-mvp}` will resolve once §3.3 is added; for now, latexmk should complete with one cross-reference warning. That is acceptable.

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §3 Work Completed intro + §3.1 Data engineering phase"
```

---

### Task 2.5: §3.2 Architecture Design phase

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Insert §3.2 after §3.1**

```latex
\subsection{Architecture design phase \hfill \status{complete}}
\label{sec:architecture-design}

\noindent The team specified and implemented three model heads (deterministic AE, VAE, DEC) on a shared multi-modal backbone. The MVP uses AE and DEC; VAE is implemented but its training is deferred to the final report.

\subsubsection*{Multi-modal backbone}

The encoder applies seven independent \texttt{\_BlockProjection} layers, one per modality, each compressing its block to a fixed projection dimension before concatenation. The 164-dimensional concatenation feeds a shared backbone --- $\mathrm{Linear}(164{\to}128) \to \mathrm{ReLU} \to \mathrm{Dropout}(0.2) \to \mathrm{Linear}(128{\to}64)$ --- yielding the 64-dimensional latent $\mathbf{z}$. \cref{fig:architecture-multimodal} shows the schematic.

\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{../../figures/architecture_multimodal.png}
\caption{Multi-modal backbone. Seven modality-specific projection blocks reduce each feature group to its proj\_dim; the resulting 164-dim concatenation feeds a shared backbone with a 64-dim bottleneck. Decoder heads mirror the input structure.}
\label{fig:architecture-multimodal}
\end{figure}

\subsubsection*{Loss functions}

\noindent\textbf{W2 inverse-variance weighting (canonical).} Per-block reconstruction losses are reweighted by the inverse of each block's variance, with clipping for numerical stability:
\[
\mathcal{L}_{\mathrm{W2}} = \sum_{b=1}^{7} w_b \cdot \mathrm{MSE}(X_b, \hat{X}_b), \qquad w_b = \mathrm{clip}\!\left(\tfrac{1}{\mathrm{Var}(X_b)},\, 0.1,\, 10.0\right).
\]
The clip bounds $[0.1, 10.0]$ are a peer-review-driven safety measure that prevents the language block (with very low average variance) from dominating early training.

\noindent\textbf{W1 (ablation).} Uniform weights $w_b = 1$ for all blocks. Identical architecture and training schedule as W2. This run isolates the contribution of inverse-variance weighting from modality projection.

\noindent\textbf{G2 director-bio masking.} Element-wise multiplication of the \texttt{bio\_pca\_64} reconstruction loss by \texttt{has\_director\_bio} so that bio-less rows contribute zero loss for that sub-block.

\noindent\textbf{DEC fine-tuning.} Soft assignments $q_{ij} = (1 + \lVert z_i - \mu_j \rVert^2)^{-1}$ over $k=21$ centroids initialized via KMeans++ on pre-trained AE latents; auxiliary target $p_{ij} = q_{ij}^2 / \sum_i q_{ij}$ computed per batch (10$\times$ speedup over full-dataset target with no measurable quality regression). Total loss $\gamma \cdot \mathrm{KL}(P\,\|\,Q) + \mathcal{L}_{\mathrm{recon}}$ with $\gamma = 0.1$.

\noindent\evidence{Implementation in \texttt{src/cineembed/}, design rationale in \texttt{docs/adr/0001-modeling-hybrid-architecture.md} (decisions D1--D10).}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds, Figure 2 renders.

- [ ] **Step 3: Commit**

```bash
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §3.2 Architecture design phase with W2/W1/G2/DEC loss specs"
```

---

### Task 2.6: §3.3 Modeling MVP phase

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

This is the largest single task. It contains the main results table, the headline callout, and four figures.

- [ ] **Step 1: Insert §3.3 after §3.2**

```latex
\subsection{Modeling MVP phase \hfill \status{complete}}
\label{sec:modeling-mvp}

\noindent Six models were trained and evaluated against three orthogonal label axes (\texttt{primary\_genre}, \texttt{decade\_bin}, \texttt{lang\_top10}). All four deep models share a single training recipe: AdamW ($\mathrm{lr}=10^{-3}$, weight\_decay $=10^{-5}$), early stopping (patience $=10$, $\Delta_{\min} = 10^{-4}$ on val\_loss), 90/10 random split (seed=42), batch size 1024, on a single Colab T4 GPU. Total wallclock for all six runs was approximately four hours.

\subsubsection*{Three-axis evaluation methodology}

For each trained model, the full-data 64-dim latent is clustered with KMeans (\texttt{k=21, n\_init=20, seed=42}) and the resulting partition is independently scored with NMI (information overlap, label-permutation invariant) and ARI (cluster agreement, harder to satisfy) against the three axes (\cref{fig:eval-method}). No axis is privileged.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{../../figures/three-exis-eval-method.png}
\caption{Three-axis evaluation methodology. The single 64-dim latent is clustered once with KMeans (k=21); the resulting partition is independently scored with NMI/ARI against three orthogonal label axes.}
\label{fig:eval-method}
\end{figure}

\subsubsection*{Main results}

\Cref{tab:main-results} reports all six runs across all six (axis $\times$ metric) cells. Six column winners are split across four different models --- no architecture dominates --- which is the principled-trade-off result documented in \cref{sec:plan-final}.

\begin{table}[h]
\centering\small
\rowcolors{2}{rowAlt}{white}
\caption{Main comparison: six runs evaluated against three orthogonal label axes (NMI and ARI per axis). All deep models use $z=64$; KMeans variants use $k=21$. Bold cells are column winners.}
\label{tab:main-results}
\begin{tabular}{@{}l l rrrrrr@{}}
\toprule
\textbf{Tier} & \textbf{Run} & \textbf{gNMI} & \textbf{gARI} & \textbf{dNMI} & \textbf{dARI} & \textbf{lNMI} & \textbf{lARI} \\
\midrule
Non-deep      & \texttt{kmeans\_raw\_k21}  & 0.109 & 0.063 & 0.233 & 0.093 & 0.075 & 0.026 \\
Non-deep      & \texttt{pca\_kmeans\_k21}  & 0.084 & 0.061 & 0.224 & 0.085 & 0.094 & 0.042 \\
Simple deep   & \texttt{vanilla\_ae\_z64}  & 0.287 & \textbf{0.247} & \textbf{0.369} & 0.175 & 0.095 & 0.030 \\
Ablation (W1) & \texttt{ae\_z64\_w1}       & 0.165 & 0.094 & 0.367 & 0.176 & 0.070 & 0.026 \\
Multi-modal   & \texttt{ae\_z64}           & 0.328 & 0.229 & 0.341 & \textbf{0.211} & 0.264 & 0.090 \\
\textbf{Best (DEC)} & \texttt{\textbf{dec\_z64\_k21}} & \textbf{0.332} & 0.244 & 0.342 & 0.210 & \textbf{0.294} & \textbf{0.090} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection*{Headline result --- H2}

\begin{headlinebox}
\noindent\bignumber{$+205\%$} \hfill \begin{minipage}[b]{0.65\linewidth}\raggedleft\small relative gain on \texttt{genre\_NMI} of \texttt{dec\_z64\_k21} (0.332) over the strongest non-deep baseline \texttt{kmeans\_raw\_k21} (0.109). Same comparison on \texttt{genre\_ARI}: $+287\%$. Same comparison on \texttt{lang\_NMI} vs \texttt{pca\_kmeans\_k21}: $+213\%$.\end{minipage}
\end{headlinebox}

\subsubsection*{Architecture-level evidence}

\noindent\textbf{Modality projection wins on language ($+178\%$).} Multi-modal \texttt{ae\_z64} produces \texttt{lang\_NMI} = 0.264 versus the vanilla concat-AE's 0.095, a relative gain of $+178\%$. \cref{fig:umap-ae-lang} shows distinct language micro-clusters in the multi-modal latent.

\begin{figure}[h]
\centering
\includegraphics[width=0.62\linewidth]{../../artifacts/figures/umap/umap_ae_z64_lang.png}
\caption{UMAP projection of multi-modal \texttt{ae\_z64} latents, colored by \texttt{lang\_top10}. Distinct micro-clusters per language family are visible, supporting the lang\_NMI=0.264 measurement.}
\label{fig:umap-ae-lang}
\end{figure}

\noindent\textbf{Inverse-variance weighting is critical ($+99\%$ on gNMI, $+277\%$ on lNMI).} The W1 ablation (uniform weights) collapses fine-structure into diffuse blobs (\cref{fig:umap-w1}); the same architecture under W2 produces dozens of genre-coherent islands (\cref{fig:umap-comparison} center). The W1 run also exhibits early-stopping at epoch 37 versus W2's 69, a model-health signal that the optimizer cannot escape modality imbalance.

\begin{figure}[h]
\centering
\includegraphics[width=0.62\linewidth]{../../artifacts/figures/umap/umap_ae_z64_w1_genre.png}
\caption{UMAP projection of \texttt{ae\_z64\_w1} (uniform weighting) latents, colored by genre. Diffuse blobs, no fine genre structure --- compare with the W2 panel of \cref{fig:umap-comparison}.}
\label{fig:umap-w1}
\end{figure}

\noindent\textbf{DEC sharpens cluster boundaries (ARI $>$ NMI gain pattern).} DEC was initialized from \texttt{ae\_z64}'s encoder and trained for 21 epochs of KL+reconstruction. \texttt{genre\_NMI} improves by $+1.2\%$ but \texttt{genre\_ARI} improves by $+6.6\%$ and \texttt{lang\_NMI} by $+11.4\%$. The diagnostic signature is that ARI gain $>$ NMI gain: information content is similar between the AE and DEC latents, but DEC's partition is more crisp. Cluster health was monitored throughout: \texttt{total\_reinit = 0} across all 21 DEC epochs, meaning all 21 KMeans++ centroids survived KL training without collapse.

\subsubsection*{Latent-topology evolution}

\Cref{fig:umap-comparison} captures the architecture progression visually: vanilla concat-AE produces two mega-blobs, multi-modal \texttt{ae\_z64} produces dozens of small genre-coherent islands, and \texttt{dec\_z64\_k21} produces even tighter atomized islands with sharper inter-cluster gaps.

\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{../../artifacts/figures/umap/umap_comparison_genre.png}
\caption{Latent topology comparison (UMAP projection, 15K subsample, cosine metric). Left: \texttt{vanilla\_ae\_z64} (two mega-blobs). Center: multi-modal \texttt{ae\_z64} (dozens of islands). Right: \texttt{dec\_z64\_k21} (tighter atomized islands). All three panels use the same projection settings; only the underlying 64-dim latent differs.}
\label{fig:umap-comparison}
\end{figure}

\subsubsection*{Bonus finding --- missing-data sub-manifold}

Coloring the DEC latent by \texttt{decade\_bin} reveals an unexpected pattern (\cref{fig:umap-missing}): films with \texttt{decade\_bin = 0} (the $\sim$7.4\% of the corpus with missing release dates) form an isolated cluster in the upper-right of the latent. The same red points are visible in vanilla and multi-modal decade plots, but DEC's KL pressure compresses them into the cleanest partition. Importantly, the model was not forced to encode missingness as latent geometry --- \texttt{has\_release\_date} is one binary input among 564. Yet across all four deep architectures, "no release date" emerges as a dimension of latent geometry, not just a flag. This finding was not predicted by the pre-registered hypotheses and is recorded as a representation-learning interpretability win.

\begin{figure}[h]
\centering
\includegraphics[width=0.62\linewidth]{../../artifacts/figures/umap/umap_dec_z64_k21_decade.png}
\caption{UMAP projection of \texttt{dec\_z64\_k21} latents, colored by \texttt{decade\_bin}. The isolated red cluster in the upper-right corresponds to films with missing release date --- a coherent sub-manifold that emerged without being engineered.}
\label{fig:umap-missing}
\end{figure}

\noindent\evidence{Six trained checkpoints under \texttt{artifacts/checkpoints/}, full numerical results in \texttt{artifacts/eval/results.json}, 13 UMAP visualizations under \texttt{artifacts/figures/umap/}, and reproducibility notebook \texttt{notebooks/06\_umap.ipynb}.}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds, all four figures (umap-ae-lang, umap-w1, umap-comparison, umap-missing) render. Page count increases by 4–5.

- [ ] **Step 3: Commit**

```bash
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §3.3 Modeling MVP phase — main results, evidence, 4 UMAPs"
```

---

### Task 2.7: §4 Work In Progress + §5 Plan to Final Report

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Insert §4 and §5 after §3.3**

```latex
% ===== 4. Work In Progress =====
\section{Work In Progress}
\label{sec:in-progress}

\noindent Two workstreams are active during the intermediate-report window:

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l p{7.5cm} l@{}}
\toprule
\textbf{Workstream} & \textbf{Description} & \textbf{Status} \\
\midrule
This document       & Compiling the intermediate progress report (LaTeX), the source-of-truth tables, and the presentation deck. & \status{inprogress} \\
UMAP analysis writeup & Documenting the latent-topology and missing-data findings (\cref{sec:modeling-mvp}) for inclusion in the final report's interpretability section. & \status{inprogress} \\
\bottomrule
\end{tabular}
\end{center}

\noindent Both are expected to complete by 2026-05-09.

% ===== 5. Plan to Final Report =====
\section{Plan to the Final Report}
\label{sec:plan-final}

\noindent The intermediate phase deliberately scopes the modeling MVP to the strongest set of empirical claims achievable within the timeline. The following items are deferred to the final report; each has an owner and a target completion window.

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l p{6.0cm} l l@{}}
\toprule
\textbf{Item} & \textbf{Why deferred} & \textbf{Target} & \textbf{Owner} \\
\midrule
VAE family ($z = 32, 64, 128$)        & Probabilistic head implemented but not trained; needed for AE-vs-VAE comparison. & 2026-06-05 & A.B. Dincoguz \\
AE latent-dim sweep ($z = 32, 128$)   & D2 fixed $z=64$ for the MVP; sweep informs the final hyperparameter discussion.   & 2026-06-05 & A.B. Dincoguz \\
F1 modality ablation (no text)        & Quantifies the contribution of the 384-dim sentence-embedding block.            & 2026-06-08 & A. Arvas \\
F2 modality ablation (no director profile) & Quantifies the value of the (sparse, 96.8\%-missing) Wikipedia bio block.    & 2026-06-08 & B.K. Kaya \\
DEC k-sweep (9-cell grid $z \times k$) & MVP runs only the $z{=}64{\times}k{=}21$ cell.                                   & 2026-06-10 & A.B. Dincoguz \\
W4 (Kendall learned uncertainty)      & Optional stretch loss; per-block weights as parameters rather than fixed.        & 2026-06-12 & A.B. Dincoguz \\
Linear probing on frozen latents      & Supplementary evaluation against held-out classifiers per axis.                  & 2026-06-12 & A. Arvas \\
5-seed confidence intervals           & MVP results are single-seed; CIs needed for the final report's robustness claim. & 2026-06-14 & B.K. Kaya \\
Full reproducibility audit            & Single deterministic run script that reproduces all numbers from a frozen feature matrix. & 2026-06-15 & A.B. Dincoguz \\
\bottomrule
\end{tabular}
\end{center}

\noindent The final report submission target is \textbf{2026-06-16}, leaving one day of buffer for typesetting and integration.

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §4 Work In Progress + §5 Plan to Final Report"
```

---

### Task 2.8: §6 Risks & Mitigation + §7 Deliverables Status

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Insert §6 and §7 after §5**

```latex
% ===== 6. Risks and Mitigation =====
\section{Risks and Mitigation}
\label{sec:risks}

\begin{riskbox}{Risk 1 --- Compute budget for the deferred VAE / k-sweep workload}
The deferred experiments add roughly 9 additional training runs (3 VAE-z plus 6 DEC-k cells beyond the MVP). At the MVP's observed wallclock of $\sim$40 minutes per deep run, this is 6 GPU-hours --- within Colab's free-tier daily quota but tight. \textbf{Mitigation:} batch experiments overnight and use checkpointing so any single run can be resumed if Colab disconnects.
\end{riskbox}

\begin{riskbox}{Risk 2 --- Director-bio coverage limits F2 ablation interpretability}
With 96.8\% of films missing a Wikipedia director bio, the F2 ablation (no-director-profile) tests removal of a block that is mostly already masked away by G2. The expected effect size is small. \textbf{Mitigation:} explicitly report the $\sim$3.2\% sub-population effect size separately and document the limitation in the final-report results section.
\end{riskbox}

\begin{riskbox}{Risk 3 --- Single-seed results may overstate stability}
MVP results are computed from a single seed (42). The 5-seed CI work is scheduled but if it produces wide intervals, some headline gains may need softer wording. \textbf{Mitigation:} run the 5-seed sweep early (target 2026-06-04, ahead of the 2026-06-14 deadline) so results can be reframed if needed before the final write-up.
\end{riskbox}

% ===== 7. Deliverables Status =====
\section{Deliverables Status}
\label{sec:deliverables}

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l l l@{}}
\toprule
\textbf{Deliverable} & \textbf{Location} & \textbf{Status} \\
\midrule
Frozen 564-dim feature matrix v1.2     & \texttt{artifacts/features/}             & \status{complete} \\
Six trained model checkpoints          & \texttt{artifacts/checkpoints/}          & \status{complete} \\
Evaluation results JSON                & \texttt{artifacts/eval/results.json}     & \status{complete} \\
13 UMAP visualizations                 & \texttt{artifacts/figures/umap/}         & \status{complete} \\
Findings document                      & \texttt{docs/FINDINGS.md}                & \status{complete} \\
Architecture decision record (D1--D10) & \texttt{docs/adr/0001-modeling\ldots.md} & \status{complete} \\
Intermediate progress report (this doc) & \texttt{docs/report/intermediate-progress-report.pdf} & \status{inprogress} \\
Intermediate presentation deck         & \texttt{docs/presentation/intermediate-progress-presentation.pptx} & \status{inprogress} \\
\bottomrule
\end{tabular}
\end{center}

\clearpage
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/report" && latexmk
```

Expected: build succeeds, three risk callouts render with amber border.

- [ ] **Step 3: Commit**

```bash
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add §6 Risks and Mitigation + §7 Deliverables Status"
```

---

### Task 2.9: Appendices A and B + ensure bibliography renders

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex`

- [ ] **Step 1: Replace the bibliography stub with Appendix A, Appendix B, then bibliography**

Locate the line `\printbibliography` near end-of-document and replace it with:

```latex
% ===== Appendix A — Per-run results =====
\appendix
\section{Per-Run Numerical Results}
\label{app:results}

\noindent Reproduced verbatim from \texttt{artifacts/eval/results.json} at 3-decimal precision.

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l rrrrrr rr@{}}
\toprule
\textbf{Run} & \textbf{gNMI} & \textbf{gARI} & \textbf{dNMI} & \textbf{dARI} & \textbf{lNMI} & \textbf{lARI} & \textbf{epochs} & \textbf{val\_loss} \\
\midrule
\texttt{vanilla\_ae\_z64} & 0.287 & 0.247 & 0.369 & 0.175 & 0.095 & 0.030 & 58 & 0.013 \\
\texttt{ae\_z64}          & 0.328 & 0.229 & 0.341 & 0.211 & 0.264 & 0.090 & 69 & 0.021 \\
\texttt{ae\_z64\_w1}      & 0.165 & 0.094 & 0.367 & 0.176 & 0.070 & 0.026 & 37 & 0.045 \\
\texttt{dec\_z64\_k21}    & 0.332 & 0.244 & 0.342 & 0.210 & 0.294 & 0.090 & 21 & 0.127 \\
\texttt{kmeans\_raw\_k21} & 0.109 & 0.063 & 0.233 & 0.093 & 0.075 & 0.026 & --- & --- \\
\texttt{pca\_kmeans\_k21} & 0.084 & 0.061 & 0.224 & 0.085 & 0.094 & 0.042 & --- & --- \\
\bottomrule
\end{tabular}
\end{center}

\noindent The DEC \texttt{val\_loss} of 0.127 combines reconstruction and KL terms and is not directly comparable to the AE pure-reconstruction val\_loss values.

% ===== Appendix B — Decision log summary =====
\section{Decision Log Summary}
\label{app:decisions}

\noindent The complete decision rationale is in \texttt{docs/adr/0001-modeling-hybrid-architecture.md}. One-line summaries:

\begin{center}\small
\rowcolors{2}{rowAlt}{white}
\begin{tabular}{@{}l p{11cm}@{}}
\toprule
\textbf{ID} & \textbf{Decision} \\
\midrule
D1  & Hybrid C-structure $\times$ D-architecture: modality-specific projections + shared backbone. \\
D2  & Latent dim $z=64$ for the MVP ($z=32$ and $z=128$ deferred to the final report). \\
D3  & W2 inverse-variance weighting + W1 ablation; W4 deferred as optional stretch. \\
D4  & G2 director-bio masking (96.8\% of films lack a Wikipedia bio). \\
D5  & Per-model notebooks plus a shared \texttt{src/cineembed} package. \\
D6  & Tier-2 evaluation: 3-axis NMI/ARI as core; linear probing deferred. \\
D7  & L4 ground truth: three-axis (genre, decade, language) labels. \\
D8  & DEC $k=21$; k-sweep deferred. \\
D9  & Peer-review fixes: 3 baselines, weight clipping, $\beta$ warmup, relative criteria. \\
D10 & Batch-wise DEC $P$ target rather than full-dataset (~10$\times$ speedup, no quality regression). \\
\bottomrule
\end{tabular}
\end{center}

% ===== References =====
\printbibliography[heading=bibintoc, title={References}]
```

Note: the existing `\printbibliography` near end-of-document is replaced by this whole block. Make sure `\appendix` precedes the appendix `\section{...}`s.

- [ ] **Step 2: Build (twice for biber)**

```bash
cd "<repo-root>/docs/report" && latexmk
```

`latexmk` automatically runs biber and reruns pdflatex as needed; one invocation should suffice. Expected: build succeeds with zero errors. Bibliography renders with all 7 entries.

- [ ] **Step 3: Verify final page count**

```bash
pdfinfo intermediate-progress-report.pdf 2>/dev/null | grep Pages
# expected: Pages: 9 to 11
```

If page count is outside 8–12, inspect for orphan pages or runaway floats; otherwise accept.

- [ ] **Step 4: Commit (LaTeX source only, no PDF yet)**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex
git commit -m "docs(report): add Appendices A and B + finalize bibliography"
```

---

## Phase 3 — LaTeX final integration

### Task 3.1: Final build, citation insertion, PDF commit

**Files:**
- Modify: `docs/report/intermediate-progress-report.tex` (insert citations)
- Add: `docs/report/intermediate-progress-report.pdf`

- [ ] **Step 1: Insert citations into §3.2 and §3.3 to ensure all bib entries are cited (so biblatex doesn't drop them)**

In §3.2 Architecture design, locate the line ending in `Total loss $\gamma \cdot \mathrm{KL}...` and append `~\cite{xie2016dec,guo2017idec}` to the end of that sentence (before the period).

In the same subsection, append `~\cite{reimers2019sbert}` to the sentence that mentions \texttt{all-MiniLM-L6-v2} sentence embeddings (it lives in §3.1 actually — locate "all-MiniLM-L6-v2" and append the citation).

In §3.3 Modeling MVP, near the "UMAP projection" caption introduction (figure caption text containing "UMAP"), add `~\cite{mcinnes2018umap}` once.

In §1.2 Scope, after "scikit-learn" mentions or in the methodology paragraph that mentions KMeans (§3.3 first paragraph), append `~\cite{pedregosa2011sklearn,paszke2019pytorch}`.

In §3.2 Architecture design before the W2 equation, append `~\cite{gorishniy2021tabular}` to the sentence about modality-specific projections.

If any of these locations are ambiguous after content shifts, add the citations at the end of the section paragraph as `\nocite{xie2016dec,guo2017idec,reimers2019sbert,mcinnes2018umap,pedregosa2011sklearn,paszke2019pytorch,gorishniy2021tabular}` placed just before §3.3's `\evidence{...}` line. This guarantees all 7 entries appear in the bibliography.

- [ ] **Step 2: Final build**

```bash
cd "<repo-root>/docs/report" && latexmk -gg
```

`-gg` forces a full rebuild from scratch. Expected: zero errors, no missing-citation warnings.

- [ ] **Step 3: Inspect for warnings**

```bash
grep -E "Warning|Error|undefined" intermediate-progress-report.log | head -30
```

Expected: ideally zero. Acceptable to ignore: `Underfull \hbox` warnings (cosmetic line-break issues), `Overfull \hbox` warnings under 5pt, `Package microtype Warning: Unknown font feature ...`.

NOT acceptable: `LaTeX Warning: Citation '...' undefined`, `LaTeX Warning: Reference '...' undefined`, `LaTeX Error: ...`, `Missing $ inserted`.

If any unacceptable warning exists, fix before continuing.

- [ ] **Step 4: Commit the PDF**

```bash
cd "<repo-root>"
git add docs/report/intermediate-progress-report.tex docs/report/intermediate-progress-report.pdf
git commit -m "docs(report): finalize citations and commit built PDF"
```

---

## Phase 4 — PPTX foundation

### Task 4.1: Python deps + verify python-pptx installs

**Files:**
- Create: `docs/presentation/requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```
python-pptx>=0.6.23,<1.0
Pillow>=10.0
```

- [ ] **Step 2: Install**

```bash
cd "<repo-root>"
python3 -m pip install -r docs/presentation/requirements.txt
python3 -c "import pptx; from PIL import Image; print('pptx', pptx.__version__, '| Pillow OK')"
```

Expected: prints version like `pptx 0.6.23 | Pillow OK` (version may vary).

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/requirements.txt
git commit -m "docs(pptx): pin python-pptx + Pillow for presentation builder"
```

---

### Task 4.2: Theme module — colors, fonts, EMU helpers

**Files:**
- Create: `docs/presentation/_slides/__init__.py`
- Create: `docs/presentation/_slides/theme.py`

- [ ] **Step 1: Create the package marker**

Write `docs/presentation/_slides/__init__.py`:

```python
"""Slide-builder package for the CineEmbed intermediate progress presentation."""
```

- [ ] **Step 2: Write the theme module**

Write `docs/presentation/_slides/theme.py`:

```python
"""Centralized theme: colors, fonts, sizes, layout helpers.

All numeric layout constants are in EMU (English Metric Units) via Inches/Pt.
The slide canvas is 13.333" x 7.5" (16:9 widescreen, PowerPoint default).
"""
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor


# ----- Slide canvas -----
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ----- Brand palette (matches LaTeX report) -----
class Colors:
    PRIMARY     = RGBColor(0xC2, 0x41, 0x0C)  # orange — section headings, headline numbers
    SECONDARY   = RGBColor(0x25, 0x63, 0xEB)  # blue — links, planned status
    ACCENT      = RGBColor(0x15, 0x80, 0x3D)  # green — complete status
    AMBER       = RGBColor(0xB4, 0x53, 0x09)  # in-progress status
    SLATE       = RGBColor(0x64, 0x74, 0x8B)  # body de-emphasis, deferred status
    NEAR_BLACK  = RGBColor(0x1E, 0x29, 0x3B)  # body text
    ROW_ALT     = RGBColor(0xF8, 0xFA, 0xFC)  # alternating row shading
    HEADER_FILL = RGBColor(0xF1, 0xF5, 0xF9)  # table header fill
    WHITE       = RGBColor(0xFF, 0xFF, 0xFF)


# ----- Typography -----
class Fonts:
    # Calibri ships with PowerPoint and is universally available across
    # Windows, macOS, and Office on the web. We use Light for headings
    # (clean, formal) and the body cut for everything else.
    HEADING = "Calibri Light"
    BODY    = "Calibri"


# ----- Type scale (in points) -----
class Sizes:
    TITLE        = Pt(36)   # slide title
    SUBTITLE     = Pt(20)   # title-page subtitle
    SECTION_LBL  = Pt(14)   # tiny brand label above title
    BODY         = Pt(18)   # standard bullet body
    BODY_SMALL   = Pt(14)   # tertiary content
    CAPTION      = Pt(12)   # figure captions
    PILL         = Pt(11)   # status pill text
    HEADLINE_BIG = Pt(180)  # big "+205%" number
    FOOTER       = Pt(10)   # page footer


# ----- Layout regions (positions / sizes) -----
class Layout:
    # Color bar at the top of every content slide
    BAR_LEFT   = Emu(0)
    BAR_TOP    = Emu(0)
    BAR_WIDTH  = SLIDE_W
    BAR_HEIGHT = Inches(0.083)  # 6pt-ish

    # Slide title bounds
    TITLE_LEFT   = Inches(0.5)
    TITLE_TOP    = Inches(0.30)
    TITLE_WIDTH  = Inches(12.333)
    TITLE_HEIGHT = Inches(0.75)

    # Subtitle (under title, on content slides)
    SUBTITLE_LEFT   = Inches(0.5)
    SUBTITLE_TOP    = Inches(1.0)
    SUBTITLE_WIDTH  = Inches(12.333)
    SUBTITLE_HEIGHT = Inches(0.4)

    # Body content region
    BODY_LEFT   = Inches(0.5)
    BODY_TOP    = Inches(1.55)
    BODY_WIDTH  = Inches(12.333)
    BODY_HEIGHT = Inches(5.5)

    # Footer
    FOOTER_LEFT   = Inches(0.5)
    FOOTER_TOP    = Inches(7.15)
    FOOTER_WIDTH  = Inches(12.333)
    FOOTER_HEIGHT = Inches(0.25)


# ----- Project repo paths (relative to docs/presentation/) -----
REPO_ROOT          = "../../"
FIG_UMAP_DIR       = REPO_ROOT + "artifacts/figures/umap/"
FIG_EDA_DIR        = REPO_ROOT + "artifacts/figures/"
FIG_DIAGRAM_DIR    = REPO_ROOT + "figures/"
```

- [ ] **Step 3: Smoke test**

```bash
cd "<repo-root>"
python3 -c "from docs.presentation._slides import theme; print(theme.Colors.PRIMARY, theme.SLIDE_W)"
```

Expected: prints something like `RGB(C2, 41, 0C) 12192000`. (Actual numeric EMU value will vary; just verify no import errors.)

If the import path errors due to the dot in `docs.presentation`, instead test by changing into the directory:

```bash
cd "<repo-root>/docs/presentation"
python3 -c "from _slides import theme; print(theme.Colors.PRIMARY)"
```

- [ ] **Step 4: Commit**

```bash
cd "<repo-root>"
git add docs/presentation/_slides/__init__.py docs/presentation/_slides/theme.py
git commit -m "docs(pptx): theme module — palette, fonts, layout regions"
```

---

### Task 4.3: Components module — reusable shape primitives

**Files:**
- Create: `docs/presentation/_slides/components.py`

- [ ] **Step 1: Write the components module**

```python
"""Reusable PowerPoint shape primitives.

All functions take a `slide` and return the shape they added (so callers can
tweak position if needed). Keep this module purely cosmetic — no business
content lives here.
"""
from typing import Iterable, Sequence
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from . import theme


# ---------- Slide chrome ----------

def add_top_bar(slide):
    """Brand-primary color bar across the top of the slide."""
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        theme.Layout.BAR_LEFT, theme.Layout.BAR_TOP,
        theme.Layout.BAR_WIDTH, theme.Layout.BAR_HEIGHT,
    )
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = theme.Colors.PRIMARY
    return bar


def add_title(slide, text: str):
    tb = slide.shapes.add_textbox(
        theme.Layout.TITLE_LEFT, theme.Layout.TITLE_TOP,
        theme.Layout.TITLE_WIDTH, theme.Layout.TITLE_HEIGHT,
    )
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.name = theme.Fonts.HEADING
    run.font.size = theme.Sizes.TITLE
    run.font.bold = False
    run.font.color.rgb = theme.Colors.PRIMARY
    return tb


def add_subtitle(slide, text: str):
    tb = slide.shapes.add_textbox(
        theme.Layout.SUBTITLE_LEFT, theme.Layout.SUBTITLE_TOP,
        theme.Layout.SUBTITLE_WIDTH, theme.Layout.SUBTITLE_HEIGHT,
    )
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.name = theme.Fonts.BODY
    run.font.size = theme.Sizes.SUBTITLE
    run.font.color.rgb = theme.Colors.SLATE
    return tb


def add_footer(slide, page_num: int, total_pages: int):
    tb = slide.shapes.add_textbox(
        theme.Layout.FOOTER_LEFT, theme.Layout.FOOTER_TOP,
        theme.Layout.FOOTER_WIDTH, theme.Layout.FOOTER_HEIGHT,
    )
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = f"SENG 474  ·  CineEmbed  ·  Slide {page_num} / {total_pages}"
    run.font.name = theme.Fonts.BODY
    run.font.size = theme.Sizes.FOOTER
    run.font.color.rgb = theme.Colors.SLATE
    return tb


# ---------- Status pills ----------

_STATUS_COLORS = {
    "complete":   theme.Colors.ACCENT,
    "inprogress": theme.Colors.AMBER,
    "planned":    theme.Colors.SECONDARY,
    "deferred":   theme.Colors.SLATE,
}
_STATUS_LABELS = {
    "complete":   "COMPLETE",
    "inprogress": "IN PROGRESS",
    "planned":    "PLANNED",
    "deferred":   "DEFERRED",
}


def add_status_pill(slide, status: str, left, top, width=Inches(1.4), height=Inches(0.32)):
    """Rounded-rectangle status pill. `status` ∈ {complete, inprogress, planned, deferred}."""
    if status not in _STATUS_COLORS:
        raise ValueError(f"unknown status: {status!r}")
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    pill.adjustments[0] = 0.5  # full pill curvature
    pill.line.fill.background()
    pill.fill.solid()
    pill.fill.fore_color.rgb = _STATUS_COLORS[status]
    tf = pill.text_frame
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = _STATUS_LABELS[status]
    run.font.name = theme.Fonts.BODY
    run.font.size = theme.Sizes.PILL
    run.font.bold = True
    run.font.color.rgb = theme.Colors.WHITE
    return pill


# ---------- Headline number ----------

def add_headline_number(slide, big_text: str, caption_text: str,
                        left=Inches(0.5), top=Inches(2.0),
                        width=Inches(12.333), height=Inches(3.5)):
    """Big primary-color number with smaller slate caption underneath."""
    # Big number
    big = slide.shapes.add_textbox(left, top, width, Inches(2.6))
    tf = big.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = big_text
    run.font.name = theme.Fonts.HEADING
    run.font.size = theme.Sizes.HEADLINE_BIG
    run.font.bold = False
    run.font.color.rgb = theme.Colors.PRIMARY

    # Caption
    cap = slide.shapes.add_textbox(left, top + Inches(2.7), width, Inches(0.8))
    tfc = cap.text_frame
    tfc.margin_left = tfc.margin_right = Emu(0)
    tfc.margin_top = tfc.margin_bottom = Emu(0)
    tfc.word_wrap = True
    pc = tfc.paragraphs[0]
    pc.alignment = PP_ALIGN.CENTER
    runc = pc.add_run()
    runc.text = caption_text
    runc.font.name = theme.Fonts.BODY
    runc.font.size = theme.Sizes.BODY
    runc.font.color.rgb = theme.Colors.SLATE
    return big, cap


# ---------- Bullets ----------

def add_bullets(slide, items: Sequence[str],
                left=Inches(0.5), top=Inches(1.7),
                width=Inches(12.333), height=Inches(5.0),
                font_size=None, color=None):
    """Add a textbox with bullet items. Bullet marker is rendered as a literal `•` so
    the rendering is consistent across PowerPoint and Keynote (PPTX bullet XML is finicky)."""
    font_size = font_size or theme.Sizes.BODY
    color = color or theme.Colors.NEAR_BLACK
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = f"•  {item}"
        run.font.name = theme.Fonts.BODY
        run.font.size = font_size
        run.font.color.rgb = color
    return tb


# ---------- Table renderer ----------

def add_table(slide, header: Sequence[str], rows: Sequence[Sequence[str]],
              left, top, width, height,
              header_fill=None, alt_row_fill=None,
              col_aligns: Sequence[str] = None):
    """Render a clean booktabs-style table with header fill and alternating row shading."""
    header_fill = header_fill or theme.Colors.HEADER_FILL
    alt_row_fill = alt_row_fill or theme.Colors.ROW_ALT
    n_rows = len(rows) + 1  # +1 for header
    n_cols = len(header)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = tbl_shape.table

    # Header row
    for j, h in enumerate(header):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.text_frame.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if (col_aligns is None or col_aligns[j] == "l") else PP_ALIGN.RIGHT
        run = p.add_run()
        run.text = h
        run.font.name = theme.Fonts.BODY
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = theme.Colors.NEAR_BLACK

    # Body rows with alternating shading
    for i, row in enumerate(rows, start=1):
        is_alt = (i % 2 == 0)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            if is_alt:
                cell.fill.solid()
                cell.fill.fore_color.rgb = alt_row_fill
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = theme.Colors.WHITE
            cell.text_frame.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (col_aligns is None or col_aligns[j] == "l") else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(val)
            run.font.name = theme.Fonts.BODY
            run.font.size = Pt(12)
            run.font.color.rgb = theme.Colors.NEAR_BLACK
    return tbl_shape


# ---------- Image ----------

def add_image(slide, path: str, left, top, width=None, height=None):
    """Insert an image from disk. If only width or only height given, aspect is preserved."""
    return slide.shapes.add_picture(path, left, top, width=width, height=height)
```

- [ ] **Step 2: Smoke test**

```bash
cd "<repo-root>/docs/presentation"
python3 -c "from _slides import components, theme; print('components OK')"
```

Expected: prints `components OK`.

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/presentation/_slides/components.py
git commit -m "docs(pptx): components module — bar/title/pill/headline/table primitives"
```

---

### Task 4.4: Slides skeleton + entry-point

**Files:**
- Create: `docs/presentation/_slides/slides.py`
- Create: `docs/presentation/build_presentation.py`

- [ ] **Step 1: Write the slides skeleton**

Write `docs/presentation/_slides/slides.py` with stub builders for all 12 slides. Each builder is replaced with real content in Phase 5.

```python
"""One builder function per slide. Each builder accepts a `prs` (Presentation)
and the index/total used by the footer, and adds a fully-built slide to `prs`.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from . import theme, components as C

TOTAL = 12

def _new_blank_slide(prs):
    """Add a blank-layout slide (layout index 6 in default master)."""
    blank_layout = prs.slide_layouts[6]
    return prs.slides.add_slide(blank_layout)


def _content_slide(prs, idx: int, title: str, subtitle: str = ""):
    """Standard content slide: top bar, title, optional subtitle, footer."""
    s = _new_blank_slide(prs)
    C.add_top_bar(s)
    C.add_title(s, title)
    if subtitle:
        C.add_subtitle(s, subtitle)
    C.add_footer(s, idx, TOTAL)
    return s


# Stub builders — replaced in Phase 5
def slide_01_title(prs):       _content_slide(prs, 1, "TODO 01")
def slide_02_status(prs):      _content_slide(prs, 2, "TODO 02")
def slide_03_goal(prs):        _content_slide(prs, 3, "TODO 03")
def slide_04_schedule(prs):    _content_slide(prs, 4, "TODO 04")
def slide_05_data(prs):        _content_slide(prs, 5, "TODO 05")
def slide_06_arch(prs):        _content_slide(prs, 6, "TODO 06")
def slide_07_mvp_table(prs):   _content_slide(prs, 7, "TODO 07")
def slide_08_headline(prs):    _content_slide(prs, 8, "TODO 08")
def slide_09_topology(prs):    _content_slide(prs, 9, "TODO 09")
def slide_10_bonus(prs):       _content_slide(prs, 10, "TODO 10")
def slide_11_plan(prs):        _content_slide(prs, 11, "TODO 11")
def slide_12_close(prs):       _content_slide(prs, 12, "TODO 12")


BUILDERS = [
    slide_01_title, slide_02_status, slide_03_goal, slide_04_schedule,
    slide_05_data, slide_06_arch, slide_07_mvp_table, slide_08_headline,
    slide_09_topology, slide_10_bonus, slide_11_plan, slide_12_close,
]
```

- [ ] **Step 2: Write the entry-point**

Write `docs/presentation/build_presentation.py`:

```python
#!/usr/bin/env python3
"""Build the CineEmbed intermediate progress presentation.

Run:
    cd docs/presentation/
    python3 build_presentation.py

Output:
    docs/presentation/intermediate-progress-presentation.pptx
"""
from pathlib import Path
import sys

from pptx import Presentation
from pptx.util import Inches

# Local package import
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _slides import slides as S
from _slides import theme

OUT = HERE / "intermediate-progress-presentation.pptx"


def build() -> Path:
    prs = Presentation()
    prs.slide_width  = theme.SLIDE_W
    prs.slide_height = theme.SLIDE_H
    for builder in S.BUILDERS:
        builder(prs)
    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
```

- [ ] **Step 3: Run the builder (stubs)**

```bash
cd "<repo-root>/docs/presentation"
python3 build_presentation.py
```

Expected: prints `wrote .../intermediate-progress-presentation.pptx (NNNN bytes)`. File exists. Even though slides are stubs, the deck has 12 slides with top bars, titles, and footers.

- [ ] **Step 4: Verify slide count**

```bash
python3 -c "from pptx import Presentation; p = Presentation('intermediate-progress-presentation.pptx'); print('slides:', len(p.slides))"
```

Expected: `slides: 12`.

- [ ] **Step 5: Commit**

```bash
cd "<repo-root>"
git add docs/presentation/_slides/slides.py docs/presentation/build_presentation.py
git commit -m "docs(pptx): slide skeleton + build entry-point — 12 stubbed slides"
```

Do NOT commit the `.pptx` yet — committed at the end after content is in.

---

## Phase 5 — PPTX slide content

Each task in this phase replaces a group of stub builders with real content. After each task, re-run `build_presentation.py` and verify by opening the output file. (Use `qlmanage -p` on macOS for a quick preview, or open in PowerPoint.)

### Task 5.1: Slides 1–3 (Title, Status, Project Goal)

**Files:**
- Modify: `docs/presentation/_slides/slides.py`

- [ ] **Step 1: Replace `slide_01_title`, `slide_02_status`, `slide_03_goal`**

Find the three stub functions and replace each with the implementations below.

```python
def slide_01_title(prs):
    s = _new_blank_slide(prs)
    # No top bar on title slide — replace with full-bleed brand-primary background block on top half
    accent = s.shapes.add_shape(
        # filled rectangle for the top "ribbon"
        1,  # MSO_SHAPE.RECTANGLE
        Emu(0), Emu(0), theme.SLIDE_W, Inches(2.0),
    )
    from pptx.enum.shapes import MSO_SHAPE
    accent.line.fill.background()
    accent.fill.solid()
    accent.fill.fore_color.rgb = theme.Colors.PRIMARY

    # Title — large, white-on-orange centered in the ribbon
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.55), Inches(12.333), Inches(1.0))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = "CineEmbed"
    run.font.name = theme.Fonts.HEADING
    run.font.size = Pt(60)
    run.font.color.rgb = theme.Colors.WHITE

    # Subtitle inside the ribbon
    sub = s.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(12.333), Inches(0.5))
    tfs = sub.text_frame
    tfs.margin_left = tfs.margin_right = Emu(0)
    ps = tfs.paragraphs[0]
    runs = ps.add_run()
    runs.text = "Multi-Modal Unsupervised Embedding of 329,044 Films"
    runs.font.name = theme.Fonts.BODY
    runs.font.size = Pt(20)
    runs.font.color.rgb = theme.Colors.WHITE

    # Document label
    label = s.shapes.add_textbox(Inches(0.5), Inches(2.6), Inches(12.333), Inches(0.5))
    tfl = label.text_frame
    tfl.margin_left = tfl.margin_right = Emu(0)
    pl = tfl.paragraphs[0]
    runl = pl.add_run()
    runl.text = "Intermediate Progress Report  ·  v1.0"
    runl.font.name = theme.Fonts.BODY
    runl.font.size = Pt(20)
    runl.font.color.rgb = theme.Colors.SECONDARY

    # Authors
    auth = s.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(0.6))
    tfa = auth.text_frame
    tfa.margin_left = tfa.margin_right = Emu(0)
    pa = tfa.paragraphs[0]
    runa = pa.add_run()
    runa.text = "Ahmet Baran Dincoguz   ·   Arda Arvas   ·   Bertan Kaan Kaya"
    runa.font.name = theme.Fonts.BODY
    runa.font.size = Pt(22)
    runa.font.color.rgb = theme.Colors.NEAR_BLACK

    # Course / institution
    course = s.shapes.add_textbox(Inches(0.5), Inches(4.95), Inches(12.333), Inches(0.5))
    tfc = course.text_frame
    pc = tfc.paragraphs[0]
    runc = pc.add_run()
    runc.text = "SENG 474 — Deep Learning · Spring 2026 · TED University · 2026-05-06"
    runc.font.name = theme.Fonts.BODY
    runc.font.size = Pt(16)
    runc.font.color.rgb = theme.Colors.SLATE

    # Repo URL
    repo = s.shapes.add_textbox(Inches(0.5), Inches(6.85), Inches(12.333), Inches(0.4))
    tfr = repo.text_frame
    pr = tfr.paragraphs[0]
    runr = pr.add_run()
    runr.text = "github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System"
    runr.font.name = theme.Fonts.BODY
    runr.font.size = Pt(14)
    runr.font.color.rgb = theme.Colors.SECONDARY


def slide_02_status(prs):
    s = _content_slide(prs, 2, "Project Status: ON TRACK",
                       "Modeling MVP delivered. Hypotheses H1, H2, H3 all PASS.")
    # Big status pill
    C.add_status_pill(s, "complete", left=Inches(0.5), top=Inches(1.55),
                      width=Inches(2.0), height=Inches(0.45))
    # Key bullets
    C.add_bullets(s,
        items=[
            "Six runs trained, evaluated against 3 orthogonal label axes (genre · decade · language).",
            "Best deep model dec_z64_k21 reaches genre_NMI = 0.332, +205 % over best non-deep baseline.",
            "Bonus finding: films with missing release date form a coherent latent sub-manifold.",
            "Final-report scope (VAE family, k-sweep, F1/F2 ablations) defined and on schedule.",
        ],
        left=Inches(0.5), top=Inches(2.25), width=Inches(12.333), height=Inches(3.5),
    )
    # Headline number callout (compact)
    C.add_headline_number(s, big_text="+205 %",
        caption_text="DEC genre_NMI vs KMeans on raw features (0.332 vs 0.109)",
        left=Inches(7.5), top=Inches(2.20), width=Inches(5.4), height=Inches(3.2),
    )


def slide_03_goal(prs):
    s = _content_slide(prs, 3, "Project Goal & Scope",
                       "What we set out to do, on what data.")
    # Left column — bullets
    C.add_bullets(s,
        items=[
            "Train unsupervised 64-dim representations of movie metadata.",
            "329,044 films from TMDB + awards records + Wikipedia director bios.",
            "564-dim feature matrix organized into 7 modality blocks.",
            "Recover 3 orthogonal label axes without labels: genre, decade, language.",
            "Evaluate via KMeans clustering of the latent → NMI / ARI.",
        ],
        left=Inches(0.5), top=Inches(1.7), width=Inches(7.5), height=Inches(5.0),
    )
    # Right column — modality table
    C.add_table(s,
        header=["Block", "Dim", "Notes"],
        rows=[
            ["numerical", "6",   "popularity, runtime, votes"],
            ["genre",     "22",  "21-way one-hot + flag"],
            ["language",  "31",  "top-30 langs (~99% zero)"],
            ["decade",    "2",   "decade_norm + flag"],
            ["awards",    "6",   "Oscar/BAFTA/Cannes counts"],
            ["text",      "384", "MiniLM-L6-v2 embeddings"],
            ["director",  "113", "bio_pca_64 + lang/country"],
        ],
        left=Inches(8.2), top=Inches(1.65), width=Inches(4.7), height=Inches(3.6),
        col_aligns=["l", "r", "l"],
    )
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/presentation"
python3 build_presentation.py
qlmanage -p intermediate-progress-presentation.pptx >/dev/null 2>&1 &
sleep 3
killall qlmanage 2>/dev/null || true
```

Open the file in PowerPoint or Keynote (or use `open -a "Microsoft PowerPoint" intermediate-progress-presentation.pptx`) and verify slides 1–3 render correctly:
- Slide 1: title page with orange ribbon, authors, repo URL
- Slide 2: status pill, four bullets, headline +205% on the right
- Slide 3: bullets left, modality table right

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/presentation/_slides/slides.py
git commit -m "docs(pptx): build slides 1-3 — title page, status, project goal"
```

---

### Task 5.2: Slides 4–7 (Schedule, Data, Architecture, MVP)

**Files:**
- Modify: `docs/presentation/_slides/slides.py`

- [ ] **Step 1: Replace `slide_04_schedule` through `slide_07_mvp_table`**

```python
def slide_04_schedule(prs):
    s = _content_slide(prs, 4, "Schedule & Milestones",
                       "Three phases complete. Final-report phase ahead.")
    C.add_table(s,
        header=["Milestone",                                  "Window",   "Status"],
        rows=[
            ["Feature matrix v1.2 frozen",                    "Apr 2026", "COMPLETE"],
            ["Multi-modal architecture finalized",            "Apr 2026", "COMPLETE"],
            ["Six runs trained and evaluated",                "May 2026", "COMPLETE"],
            ["Pre-registered hypotheses tested",              "May 2026", "COMPLETE"],
            ["UMAP latent analysis",                          "May 2026", "COMPLETE"],
            ["Intermediate progress report",                  "May 2026", "IN PROGRESS"],
            ["VAE family training (z = 32 / 64 / 128)",       "Jun 2026", "PLANNED"],
            ["F1 / F2 modality ablations",                    "Jun 2026", "PLANNED"],
            ["DEC k-sweep (9-cell grid)",                     "Jun 2026", "PLANNED"],
            ["Final report",                                  "Jun 2026", "PLANNED"],
        ],
        left=Inches(0.5), top=Inches(1.65), width=Inches(12.333), height=Inches(5.2),
        col_aligns=["l", "l", "l"],
    )


def slide_05_data(prs):
    s = _content_slide(prs, 5, "Work Completed: Data Engineering",
                       "564-dim feature matrix, 7 modality blocks, frozen v1.2.")
    C.add_status_pill(s, "complete", left=Inches(0.5), top=Inches(1.55),
                      width=Inches(2.0), height=Inches(0.4))
    C.add_bullets(s,
        items=[
            "Three sources merged: TMDB, awards records, Wikipedia director bios.",
            "Sparse modalities preserved as one-hot (interpretable).",
            "Missing release date encoded as binary flag — turned out to be structurally relevant (slide 10).",
            "Director-bio reconstruction loss masked by has_director_bio flag (G2 masking).",
        ],
        left=Inches(0.5), top=Inches(2.25), width=Inches(7.0), height=Inches(4.5),
    )
    # Right: figure
    C.add_image(s, theme.FIG_EDA_DIR + "multilingual_coverage.png",
                left=Inches(7.7), top=Inches(2.05), width=Inches(5.2))
    cap = s.shapes.add_textbox(Inches(7.7), Inches(6.4), Inches(5.2), Inches(0.4))
    tfc = cap.text_frame
    pc = tfc.paragraphs[0]
    pc.alignment = PP_ALIGN.CENTER
    runc = pc.add_run()
    runc.text = "Multilingual coverage — long tail motivates sparsity-aware design."
    runc.font.name = theme.Fonts.BODY
    runc.font.size = theme.Sizes.CAPTION
    runc.font.italic = True
    runc.font.color.rgb = theme.Colors.SLATE


def slide_06_arch(prs):
    s = _content_slide(prs, 6, "Work Completed: Architecture Design",
                       "Multi-modal backbone, W2 inverse-variance loss, G2 bio masking, DEC head.")
    C.add_status_pill(s, "complete", left=Inches(0.5), top=Inches(1.55),
                      width=Inches(2.0), height=Inches(0.4))
    # Architecture image — large, centered
    C.add_image(s, theme.FIG_DIAGRAM_DIR + "architecture_multimodal.png",
                left=Inches(0.6), top=Inches(2.1), width=Inches(8.5))
    # Bullets on the right
    C.add_bullets(s,
        items=[
            "7 modality projections → concat (164-dim) → backbone → z=64.",
            "W2: per-block inverse-variance weighting (clipped to [0.1, 10]).",
            "G2: mask bio reconstruction loss by has_director_bio.",
            "DEC head: Student-t soft assignment, k=21, γ=0.1 on KL.",
        ],
        left=Inches(9.4), top=Inches(2.25), width=Inches(3.5), height=Inches(4.5),
        font_size=Pt(15),
    )


def slide_07_mvp_table(prs):
    s = _content_slide(prs, 7, "Work Completed: Modeling MVP — Six Runs",
                       "Three tiers, six metrics, four different column winners.")
    C.add_status_pill(s, "complete", left=Inches(0.5), top=Inches(1.55),
                      width=Inches(2.0), height=Inches(0.4))
    C.add_table(s,
        header=["Run",                "gNMI", "gARI", "dNMI", "dARI", "lNMI", "lARI"],
        rows=[
            ["kmeans_raw_k21",         "0.109","0.063","0.233","0.093","0.075","0.026"],
            ["pca_kmeans_k21",         "0.084","0.061","0.224","0.085","0.094","0.042"],
            ["vanilla_ae_z64",         "0.287","0.247","0.369","0.175","0.095","0.030"],
            ["ae_z64_w1 (W1 ablation)","0.165","0.094","0.367","0.176","0.070","0.026"],
            ["ae_z64",                 "0.328","0.229","0.341","0.211","0.264","0.090"],
            ["dec_z64_k21 (BEST)",     "0.332","0.244","0.342","0.210","0.294","0.090"],
        ],
        left=Inches(0.5), top=Inches(2.25), width=Inches(12.333), height=Inches(4.0),
        col_aligns=["l", "r", "r", "r", "r", "r", "r"],
    )
    # Caption
    cap = s.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(12.333), Inches(0.4))
    tfc = cap.text_frame
    pc = tfc.paragraphs[0]
    runc = pc.add_run()
    runc.text = ("z = 64, KMeans k = 21. No model wins all six metrics — "
                 "the principled-trade-off result.")
    runc.font.name = theme.Fonts.BODY
    runc.font.size = theme.Sizes.CAPTION
    runc.font.italic = True
    runc.font.color.rgb = theme.Colors.SLATE
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/presentation"
python3 build_presentation.py
```

Open the deck and inspect slides 4–7. Expected:
- Slide 4: 10-row milestones table with status text in last column
- Slide 5: complete pill + 4 bullets + multilingual_coverage figure on right
- Slide 6: complete pill + architecture diagram + bullets
- Slide 7: complete pill + 6-row results table

- [ ] **Step 3: Commit**

```bash
cd "<repo-root>"
git add docs/presentation/_slides/slides.py
git commit -m "docs(pptx): build slides 4-7 — schedule, data, architecture, MVP table"
```

---

### Task 5.3: Slides 8–10 (Headline, Topology, Bonus)

**Files:**
- Modify: `docs/presentation/_slides/slides.py`

- [ ] **Step 1: Replace `slide_08_headline` through `slide_10_bonus`**

```python
def slide_08_headline(prs):
    s = _content_slide(prs, 8, "Headline Result — H2",
                       "Best deep model vs best non-deep baseline on genre_NMI.")
    # Big number in the middle
    C.add_headline_number(s, big_text="+205 %",
        caption_text="dec_z64_k21: 0.332     vs     kmeans_raw_k21: 0.109",
        left=Inches(0.5), top=Inches(1.9), width=Inches(12.333), height=Inches(3.4),
    )
    # Three-up reinforcement strip
    strip_top = Inches(5.5)
    items = [
        ("+295 %", "DEC vs PCA-KMeans (gNMI)"),
        ("+287 %", "DEC vs raw-KMeans (gARI)"),
        ("+213 %", "DEC vs PCA-KMeans (lNMI)"),
    ]
    col_w = Inches(4.0)
    for i, (big, cap_text) in enumerate(items):
        left = Inches(0.7 + i * 4.1)
        # number
        nb = s.shapes.add_textbox(left, strip_top, col_w, Inches(0.7))
        tf = nb.text_frame
        tf.margin_left = tf.margin_right = Emu(0)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = big
        run.font.name = theme.Fonts.HEADING
        run.font.size = Pt(40)
        run.font.color.rgb = theme.Colors.PRIMARY
        # caption
        cap = s.shapes.add_textbox(left, strip_top + Inches(0.75), col_w, Inches(0.5))
        tfc = cap.text_frame
        tfc.margin_left = tfc.margin_right = Emu(0)
        pc = tfc.paragraphs[0]
        pc.alignment = PP_ALIGN.CENTER
        runc = pc.add_run()
        runc.text = cap_text
        runc.font.name = theme.Fonts.BODY
        runc.font.size = Pt(14)
        runc.font.color.rgb = theme.Colors.SLATE


def slide_09_topology(prs):
    s = _content_slide(prs, 9, "Latent Topology Evolves",
                       "vanilla → multi-modal → DEC: blobs → islands → tight islands.")
    # Full-width 3-panel UMAP
    C.add_image(s, theme.FIG_UMAP_DIR + "umap_comparison_genre.png",
                left=Inches(0.4), top=Inches(1.7), width=Inches(12.5))
    # Caption
    cap = s.shapes.add_textbox(Inches(0.4), Inches(6.4), Inches(12.5), Inches(0.5))
    tfc = cap.text_frame
    pc = tfc.paragraphs[0]
    pc.alignment = PP_ALIGN.CENTER
    runc = pc.add_run()
    runc.text = ("UMAP, 15K subsample, cosine metric. Same projection settings; "
                 "only the underlying 64-dim latent differs.")
    runc.font.name = theme.Fonts.BODY
    runc.font.size = theme.Sizes.CAPTION
    runc.font.italic = True
    runc.font.color.rgb = theme.Colors.SLATE


def slide_10_bonus(prs):
    s = _content_slide(prs, 10, "Bonus Finding — Missing-Data Manifold",
                       "Films with no release date form a coherent latent sub-manifold.")
    # Image left
    C.add_image(s, theme.FIG_UMAP_DIR + "umap_dec_z64_k21_decade.png",
                left=Inches(0.5), top=Inches(1.7), width=Inches(7.5))
    # Bullets right
    C.add_bullets(s,
        items=[
            "~7.4 % of films have no release date (decade_bin = 0).",
            "These films form an isolated red cluster (upper-right) across all 4 deep architectures.",
            "DEC compresses them into the cleanest partition.",
            "Not predicted by H1–H3 — post-hoc interpretability win.",
            "Practical use: latent-space queries naturally cluster missing-metadata films for triage.",
        ],
        left=Inches(8.2), top=Inches(1.95), width=Inches(4.7), height=Inches(5.0),
        font_size=Pt(15),
    )
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/presentation"
python3 build_presentation.py
```

Open and inspect:
- Slide 8: big "+205 %" centered, 3-up reinforcement strip below (+295/+287/+213)
- Slide 9: full-width UMAP comparison figure with caption
- Slide 10: UMAP decade figure left, 5 bullets right

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/_slides/slides.py
git commit -m "docs(pptx): build slides 8-10 — headline, topology, bonus finding"
```

---

### Task 5.4: Slides 11–12 (Plan, Close)

**Files:**
- Modify: `docs/presentation/_slides/slides.py`

- [ ] **Step 1: Replace `slide_11_plan` and `slide_12_close`**

```python
def slide_11_plan(prs):
    s = _content_slide(prs, 11, "Plan to the Final Report",
                       "Deferred experiments with target completion windows.")
    C.add_table(s,
        header=["Item",                              "Why deferred",                                    "Target"],
        rows=[
            ["VAE family (z = 32 / 64 / 128)",       "Probabilistic head not yet trained.",            "2026-06-05"],
            ["AE z-dim sweep (z = 32, 128)",         "MVP fixed z = 64.",                              "2026-06-05"],
            ["F1 ablation (no text)",                "Quantify text-block contribution.",              "2026-06-08"],
            ["F2 ablation (no director profile)",    "Quantify director-block contribution.",          "2026-06-08"],
            ["DEC k-sweep (9-cell z×k grid)",        "MVP runs only z=64 × k=21.",                     "2026-06-10"],
            ["W4 (Kendall learned uncertainty)",     "Stretch loss; learnable per-block weights.",     "2026-06-12"],
            ["Linear probing on frozen latents",     "Held-out classifier evaluation per axis.",       "2026-06-12"],
            ["5-seed confidence intervals",          "MVP single-seed → CIs needed.",                  "2026-06-14"],
            ["Reproducibility audit",                "Single deterministic run script.",               "2026-06-15"],
        ],
        left=Inches(0.5), top=Inches(1.7), width=Inches(12.333), height=Inches(5.0),
        col_aligns=["l", "l", "l"],
    )
    # Footer note
    note = s.shapes.add_textbox(Inches(0.5), Inches(6.75), Inches(12.333), Inches(0.4))
    tfn = note.text_frame
    pn = tfn.paragraphs[0]
    runn = pn.add_run()
    runn.text = "Final report submission target: 2026-06-16."
    runn.font.name = theme.Fonts.BODY
    runn.font.size = Pt(14)
    runn.font.italic = True
    runn.font.color.rgb = theme.Colors.SLATE


def slide_12_close(prs):
    s = _content_slide(prs, 12, "Status Summary  ·  Q & A",
                       "All three pre-registered hypotheses PASS.")
    # Hypothesis tracker
    C.add_table(s,
        header=["ID",  "Statement",                                          "Result",            "Status"],
        rows=[
            ["H1", "DEC genre_NMI > AE genre_NMI",                           "0.332 > 0.328",     "COMPLETE"],
            ["H2", "Best-deep ≥ 1.10 × best-non-deep on genre_NMI",          "+205 %",            "COMPLETE"],
            ["H3", "Best-deep genre_NMI > 0.15 absolute floor",              "0.332 ≫ 0.15",      "COMPLETE"],
        ],
        left=Inches(0.5), top=Inches(1.85), width=Inches(12.333), height=Inches(2.4),
        col_aligns=["l", "l", "l", "l"],
    )
    # Pull quote
    q = s.shapes.add_textbox(Inches(0.5), Inches(4.6), Inches(12.333), Inches(1.2))
    tfq = q.text_frame
    tfq.word_wrap = True
    pq = tfq.paragraphs[0]
    pq.alignment = PP_ALIGN.CENTER
    runq = pq.add_run()
    runq.text = ("Best model: dec_z64_k21  —  genre_NMI = 0.332,  "
                 "lang_NMI = 0.294,  decade_NMI = 0.342")
    runq.font.name = theme.Fonts.HEADING
    runq.font.size = Pt(22)
    runq.font.color.rgb = theme.Colors.PRIMARY

    # Repo + thanks
    foot = s.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(12.333), Inches(0.5))
    tff = foot.text_frame
    pf = tff.paragraphs[0]
    pf.alignment = PP_ALIGN.CENTER
    runf = pf.add_run()
    runf.text = "github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System     ·     Thank you. Questions?"
    runf.font.name = theme.Fonts.BODY
    runf.font.size = Pt(16)
    runf.font.color.rgb = theme.Colors.SLATE
```

- [ ] **Step 2: Build and verify**

```bash
cd "<repo-root>/docs/presentation"
python3 build_presentation.py
```

Open and inspect:
- Slide 11: 9-row deferred-items table with target dates, footer note
- Slide 12: 3-row hypothesis tracker, pull quote, thanks line

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/_slides/slides.py
git commit -m "docs(pptx): build slides 11-12 — plan to final, status summary"
```

---

## Phase 6 — Final integration and push

### Task 6.1: Idempotency check, deck verification, commit PPTX

**Files:**
- Add: `docs/presentation/intermediate-progress-presentation.pptx`

- [ ] **Step 1: Run the build twice and confirm the slide count and structural shape**

```bash
cd "<repo-root>/docs/presentation"
python3 build_presentation.py
cp intermediate-progress-presentation.pptx /tmp/deck1.pptx
python3 build_presentation.py
cp intermediate-progress-presentation.pptx /tmp/deck2.pptx
python3 - <<'PY'
from pptx import Presentation
for p in ["/tmp/deck1.pptx", "/tmp/deck2.pptx"]:
    prs = Presentation(p)
    shape_counts = [len(s.shapes) for s in prs.slides]
    print(p, "slides:", len(prs.slides), "shape counts:", shape_counts)
PY
```

Expected: both decks have 12 slides; per-slide shape counts are identical between the two runs (this is the near-idempotency check from the spec). Acceptable: pptx revision IDs may differ — that's outside our control.

- [ ] **Step 2: Smoke-open in PowerPoint or Keynote**

```bash
open -a "Microsoft PowerPoint" "<repo-root>/docs/presentation/intermediate-progress-presentation.pptx" 2>/dev/null \
  || open -a "Keynote"            "<repo-root>/docs/presentation/intermediate-progress-presentation.pptx" 2>/dev/null \
  || echo "Open it manually for visual verification."
```

Tab through every slide. Acceptance: no missing-image red Xs, no overflow, all status pills render with correct color, all tables fit on the slide, all figures fit and are not stretched.

- [ ] **Step 3: Commit the built PPTX**

```bash
cd "<repo-root>"
git add docs/presentation/intermediate-progress-presentation.pptx
git commit -m "docs(pptx): build and commit intermediate-progress-presentation.pptx"
```

---

### Task 6.2: Final push

- [ ] **Step 1: Show what's about to be pushed**

```bash
cd "<repo-root>"
git log --oneline origin/main..HEAD
git status
```

Expected: a sequence of commits from this plan. Working tree clean (modulo the pre-existing dirty entries from before this work).

- [ ] **Step 2: Push**

```bash
git push origin main
```

Expected: push succeeds with no rejection or force needed.

- [ ] **Step 3: Sanity URL check (optional)**

```bash
echo "Open in browser to verify upload:"
echo "https://github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System/tree/main/docs/report"
echo "https://github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System/tree/main/docs/presentation"
```

---

## Self-review checklist (run after writing the plan above)

**Spec coverage:**

- §4.1 LaTeX engine + class + build → covered by Tasks 1.2, 1.3, 3.1
- §4.2 Package list → covered by Task 1.3
- §4.3 Visual design (colors, headings, custom commands, callouts, headers/footers, title page) → Task 1.3 (preamble + title), Task 2.1+ (using `\status`/`\headline` etc.)
- §4.4 Section outline 1–7 + appendices → Tasks 2.1 through 2.9
- §4.5 Numerical claims sourcing → numbers embedded in the plan tasks
- §4.6 Figure list (7 figures) → all 7 referenced in Tasks 2.2, 2.5, 2.6
- §4.7 References (7 entries) → Task 1.1
- §4.8 Build instructions → Task 1.2 (.latexmkrc), Task 3.1 (final build)
- §5.1 PPTX approach → Tasks 4.1–4.4
- §5.2 Build script structure → Tasks 4.2, 4.3, 4.4
- §5.3 Slide outline (12 slides) → Tasks 5.1–5.4
- §5.4 Visual design language → Task 4.2 (theme.py), Task 4.3 (components.py)
- §5.5 Build and verification → Tasks 4.1 (install), 6.1 (idempotency)
- §6 Names canonical (ASCII) → applied in Tasks 1.3 and 5.1
- §7 Out of scope (no deletion of old Marp deliverables) → respected throughout
- §8 Acceptance criteria → covered by per-task verifications + Phase 6

No gaps identified.

**Placeholder scan:** No `TBD`/`TODO`/`fill in details` in code or content. Stub `def slide_NN_*: TODO NN` exists deliberately in Task 4.4 and is replaced in Phase 5 — that is the staged-development pattern, not a placeholder.

**Type/name consistency check:**

- `\status{...}` defined in Task 1.3 with arguments `complete | inprogress | planned | deferred` — used consistently in Tasks 2.1, 2.3, 2.7, 2.8, 5.x.
- `Colors.*` and `Fonts.*` defined in Task 4.2 — used in Task 4.3 components and Phase 5 slides.
- `BUILDERS` list in Task 4.4 — function names match the implementations in Phase 5.
- `theme.SLIDE_W`, `theme.SLIDE_H` referenced consistently.

No naming drift.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-06-intermediate-progress-report-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because tasks are tightly scoped and the visual outputs benefit from review checkpoints.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster end-to-end but the main context grows over the run.

**Which approach?**
