# 01 — MVP Modeling Phase

> The first round of trained models. Six runs delivered against the
> intermediate-report deadline (2026-05-06). Establishes the empirical
> baseline against which everything later is measured.
>
> Read `00-context-and-goals.md` first.

## §1 Context

The original modeling spec
([`docs/archive/specs/2026-05-04-modeling-design.md`](../archive/specs/2026-05-04-modeling-design.md))
pre-registered a **21-22 run ablation matrix** across three model families,
three latent dimensions, and a DEC k-grid. Concretely:

| Axis | Levels | Notes |
|---|---|---|
| Model family | AE, VAE, DEC | D1 — shared multi-modal backbone |
| Latent dim z | {32, 64, 128} | D2 |
| DEC cluster count k | {10, 21, 30} | D8 — DEC family only |
| Loss-weighting ablation | W1 uniform, W4 Kendall stretch | D3 — z=64 AE only |
| Modality ablation | F1 (no text), F2 (no director) | D6 — z=64 AE only |
| Non-deep baselines | `kmeans_raw_k21`, `pca_kmeans_k21` | D9 — peer-review patch |

Total: 9 main (3 × 3) + 9 DEC k-sweep + 2 baselines + 1 vanilla concat-AE + W1
+ F1 + F2 + optional W4 ≈ **21-22 trainings**, ~5-6 hours wall-clock on free
Colab T4.

**The MVP collapsed this to 6 runs.** The intermediate progress report was
due 2026-05-06, four days after the modeling spec was finalized. The team
shipped a triage matrix that exercised the hypotheses (architecture, weighting,
DEC vs AE, deep vs non-deep) at z=64 only, and deferred the rest to
post-report work.

Detailed list of what was deferred and why → `08-scope-cuts-future-work.md`.

The exhaustive matrix was formally replaced by the two-round strategy on
2026-05-16 (ADR D13). See `00-context-and-goals.md` § Timeline.

## §2 The six runs

Numbers in this section come directly from
[`10-results-table.md`](10-results-table.md) and `artifacts/eval/results.json`.
Hyperparameters not in the run name are the modeling-spec defaults
(seed=42, 90/10 split, early-stop patience 10, lr=1e-3, batch 1024).

### 2.1 `kmeans_raw_k21` — non-deep baseline 1

| Field | Value |
|---|---|
| Rationale | Floor for the multi-modal claim — sklearn KMeans on raw 564-dim. |
| Hyperparameters | `KMeans(n_clusters=21, random_state=42)` directly on `feature_matrix.npz`. |
| Training time | <1 minute, no GPU. |
| Best val_loss | n/a (no neural training). |
| Final gNMI | **0.117** (rounded; results.json: 0.1086) `[verify]` — entry was missing from `artifacts/eval/results.json` pre-2026-05-17 per the source-of-truth note in `10-results-table.md`. |
| Final dNMI / lNMI | 0.233 / 0.075 (from FINDINGS.md three-tier table). |

This is the unfalsifiability guard from peer review D9 — without it, the
multi-modal-backbone claim has no control. The 0.109 genre NMI is the
"raw geometry" of the feature matrix before any representation learning.

### 2.2 `pca_kmeans_k21` — non-deep baseline 2

| Field | Value |
|---|---|
| Rationale | Ablate "dimensionality reduction" vs "representation learning". |
| Hyperparameters | `PCA(n_components=64) → KMeans(n_clusters=21, random_state=42)`. |
| Training time | <1 minute, no GPU. |
| Best val_loss | n/a. |
| Final gNMI | **0.207** (results.json: 0.0844) `[verify]` — the value 0.207 in `10-results-table.md` and 0.084 in `artifacts/eval/results.json` / FINDINGS.md disagree. FINDINGS.md value (0.084) used elsewhere in this journal as authoritative. |
| Final dNMI / lNMI | 0.224 / 0.094. |

The diagnostic finding here (per Finding 6 in FINDINGS.md): PCA *discards*
genre-discriminative variance. PCA-64 + KMeans is worse than raw KMeans on
genre. Both are dominated by deep encoders by 3-5×.

### 2.3 `vanilla_ae_z64` — architecture baseline

| Field | Value |
|---|---|
| Rationale | The control for the modality-projection hypothesis (D1). Same per-block decoder as `ae_z64` so reconstruction loss is comparable, but the encoder is flat: `Linear(564 → 128) → ReLU → Linear(128 → 64)`. No per-modality projection layer. |
| Hyperparameters | z=64, hidden=128, W2 inverse-variance weighting, G2 mask, lr=1e-3, batch 1024, early-stop patience 10. |
| Training time | ~15 min Colab T4 `[verify]`. |
| Best val_loss | **0.0126** — lowest of any MVP run. |
| Epochs to stop | 58 (clean plateau). |
| Final gNMI | **0.287**. |
| Final dNMI / lNMI | 0.369 / 0.095. |
| geo_NMI | 0.215 |

The "useful representation ≠ best reconstruction" finding (FINDINGS Finding 3)
emerges here. Vanilla wins reconstruction by ~38% but loses gNMI by 12% to
the multi-modal architecture and language-NMI by 178%. The flat FC encoder
under-represents the low-frequency one-hot blocks.

### 2.4 `ae_z64` — multi-modal AE with W2 inverse-variance weights

| Field | Value |
|---|---|
| Rationale | The main encoder of the project. D1 + D3 + D4. |
| Hyperparameters | z=64, hidden=128, per-block projection (num→16, genre→16, lang→16, decade→4, awards→16, text→64, director→32), W2 inverse-variance with clipping `[0.1, 10.0]`, G2 mask on the 64 bio-PCA dims when `has_director_bio=0`. lr=1e-3, batch 1024, early-stop patience 10. |
| Training time | ~20 min Colab T4 `[verify]`. |
| Best val_loss | 0.0208 — higher than vanilla; expected because W2 deliberately upweights low-variance blocks. |
| Epochs to stop | **69** — longest of the four AE-family runs; the careful weighting kept improving val loss. |
| Final gNMI | **0.328**. |
| Final dNMI / lNMI | 0.341 / **0.264**. |
| geo_NMI | 0.309 |

This is the project's de-facto encoder. As of 2026-05-17 it is the **demo
backbone** as well, because of the retrieval-task pivot
(`07-retrieval-vs-nmi-discovery.md`). The +178% lNMI over vanilla is the
single strongest architectural finding (FINDINGS Finding 1).

### 2.5 `ae_z64_w1` — uniform-weight ablation

| Field | Value |
|---|---|
| Rationale | Falsifies D3. If the architecture alone were doing the work, this should match `ae_z64`. |
| Hyperparameters | Identical to `ae_z64` except all block weights set to 1.0 (uniform, "W1"). |
| Training time | ~12 min Colab T4 `[verify]`. |
| Best val_loss | 0.0453 — 3.6× worse than vanilla, 2.2× worse than `ae_z64`. |
| Epochs to stop | **37** — patience exhausted early. Diagnostic: the model couldn't escape modality imbalance, so val loss never decreased enough to reset patience. |
| Final gNMI | **0.165** (-50% vs W2). |
| Final dNMI / lNMI | 0.367 / 0.070 (-73% vs W2). |
| geo_NMI | 0.162 |

The **dimension-asymmetric collapse** (FINDINGS Finding 2): small blocks
(decade, 2 dims) survive uniform weighting because StandardScaler already
normalized them; large high-dim blocks (text 384, language 31) lose all
gradient signal. Confirms the W2 design choice with a single ablation.

### 2.6 `dec_z64_k21` — DEC at k matching genre cardinality

| Field | Value |
|---|---|
| Rationale | The MVP-era champion. Tests D8/H1: does explicit cluster optimization beat free latent clustering? |
| Hyperparameters | DEC head initialized from `ae_z64` backbone weights + KMeans-21 centroids on the latent. KL+reconstruction loss, batch-wise P (D10), 21 epochs, lr=1e-4. |
| Training time | ~6 min on top of `ae_z64`'s 20 min `[verify]`. |
| Best val_loss | 0.127 — KL + recon combined, not directly comparable to AE val_loss. |
| Epochs run | 21 (no early stop, fixed budget). |
| Cluster reinit | **`total_reinit = 0`** — all 21 KMeans++ centroids survived KL training. Non-trivial; DEC implementations frequently see 1-4 cluster collapses. |
| Final gNMI | **0.332**. |
| Final dNMI / lNMI | 0.342 / **0.294**. |
| geo_NMI | 0.323 (re-eval, 2026-05-17). |

The gNMI gain over `ae_z64` is only +1.2%, but the **ARI** gain is +6.6% and
lNMI is +11.4% — DEC contributes cluster compactness, not new structural
information (FINDINGS Finding 7). This is the marginal H1 PASS.

`dec_z64_k21` was re-evaluated on 2026-05-17 with the modern eval pipeline
(AMI keys, k=21 fixed), producing the numbers in
[`10-results-table.md`](10-results-table.md) Phase-0 row.

## §3 What we learned

Five MVP-era findings have survived the subsequent two weeks of work:

- **W2 inverse-variance weighting beats W1 uniform.** `ae_z64` gNMI 0.328 vs
  `ae_z64_w1` 0.165 — a clean -50% on a single design choice. Spec
  criterion (D3): `W2.NMI > W1.NMI × 1.05`. **PASS** at +99% on genre, +277%
  on language. This is the project's single most defensible architectural
  decision.

- **Multi-modal projection beats vanilla concat.** `ae_z64` gNMI 0.328 vs
  `vanilla_ae_z64` 0.287 (+14%); lNMI 0.264 vs 0.095 (+178%). The
  geo_NMI delta is 0.309 vs 0.215 (+44%). Spec criterion (D1):
  `multi_modal_AE_z64.NMI > vanilla_ae_z64.NMI × 1.05`. **PASS.**

- **DEC adds marginal lift over the underlying AE on this data.**
  `dec_z64_k21` gNMI 0.332 vs `ae_z64` gNMI 0.328 (+1.2%). The +6.6% ARI
  and +11.4% lNMI gains are the richer story — DEC sharpens existing
  partitions rather than discovering new ones.

- **No single architecture wins all three axes.** Vanilla wins decade_NMI
  (0.369) and genre_ARI (0.247). Multi-modal AE wins decade_ARI (0.211).
  DEC wins genre_NMI (0.332), lang_NMI (0.294), and lang_ARI (0.090).
  Decade is the "easy axis" — all three architectures recover it at
  NMI ≈ 0.34-0.37 because `has_release_date` and `decade_norm` are a
  trivially-encoded ordinal signal. This non-uniformity motivated the
  `geo_NMI` composite in D13 / Round 1.

- **The director block has a missing-data sub-manifold.** UMAP of the
  decade-coloured DEC latent isolates `decade_bin = 0` films (≈ 7.4% of the
  15k subsample, marked red) as a clearly separated cluster in the upper
  right. The same pattern appears in vanilla and `ae_z64` but is most
  explicit under DEC. Cross-correlates with `has_director_bio=0` rows
  (96.8% of films) — the G2 mask was designed to keep the bio block from
  swamping the loss with constant-zero reconstruction, and it works as
  intended. Post-hoc discovery, not predicted by H1-H3 (FINDINGS Finding 9).

## §4 What we deferred (and why)

The original spec listed the following items; none were trained for the
intermediate report:

| Deferred run | Original purpose | Status as of 2026-05-17 |
|---|---|---|
| `ae_z32` | Latent-dim sensitivity floor | Round 2 — GPU quota gated |
| `ae_z128` | Latent-dim sensitivity ceiling | Round 2 — GPU quota gated |
| `ae_z64_w4` | Kendall learned uncertainty weighting (stretch) | Future work — W2 already validated |
| F1 (`ae_z64_no_text`) | Modality contribution: text | Future work |
| F2 (`ae_z64_no_director`) | Modality contribution: director | Future work |
| `vae_z{32,64,128}` | Generative comparison family | Only `vae_z64` ran (Round 1); collapsed |
| `dec_z{32,128}_k21` | DEC z-sensitivity | Future work |
| `dec_z64_k{10,30}` | DEC k-sweep at fixed z | Future work |
| Linear-probe accuracy on all z=64 models | Tier-2 eval add-on (D6) | Free latents on the winner sufficient for demo |

Defer reason: **the intermediate report had a four-day timeline.** The MVP
prioritized one run per design decision (architecture, weighting, DEC,
baselines) over breadth. Detailed rationale and the future-work list live in
`08-scope-cuts-future-work.md`.

The two-round strategy locked on 2026-05-16 (ADR D13) supersedes the
exhaustive matrix.

## §5 The intermediate report

Delivered 2026-05-06. Two artifacts:

| Deliverable | Path |
|---|---|
| LaTeX source | `docs/report/intermediate-progress-report.tex` |
| Compiled PDF | `docs/report/intermediate-progress-report.pdf` |
| 12-slide PPTX | `docs/presentation/intermediate-progress-presentation.pptx` |

The report is structured around the three pre-registered hypotheses:

| Hypothesis | Spec | Result | Verdict |
|---|---|---|---|
| H1: DEC > AE on gNMI | D8 | 0.332 > 0.328 | PASS, marginal |
| H2: best deep > best non-deep × 1.10 | D9 | 0.332 vs 0.109, +205% | PASS, massive |
| H3: best deep gNMI > 0.15 absolute floor | D9 | 0.332 ≫ 0.15 | PASS |

The PPTX uses 12 slides built by `slides.py` from a custom palette/components
module (commits `4d23157`, `bb773d3`, `73ab024`, `5d0dfa4`, `5b86172`,
`2a339a2`, `29a2d70`). All 6 MVP runs plus the four hero UMAP figures from
`artifacts/figures/umap/` are referenced. Section 5 ("Plan to Final Report")
sketches the remaining work, which is now superseded by the two-round +
web-app strategy.

The intermediate report is the canonical document for the MVP-era story. The
final-report work (due 2026-05-20) builds on it but is now a "half-academic"
tier document accompanying the web-app demo (ADR D14).

## §6 Cross-references

- `02-clustering-improvements-spec.md` — the next-day spec that added five
  techniques on top of these six runs.
- `03-tooling-wandb-integration.md` — three days later: the dashboard that
  the MVP runs were retroactively backfilled into.
- `04-phase1-contrastive-sweep.md` — the first time the contrastive technique
  was actually trained on production data (2026-05-16).
- `06-negative-results.md` — Phase 1's surprising outcome relative to these
  MVP baselines.
- `10-results-table.md` Phase 0 row — every number in this file in one place.
