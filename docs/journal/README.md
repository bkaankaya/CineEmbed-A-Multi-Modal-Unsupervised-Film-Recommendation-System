# CineEmbed — Experimental Journal

A structured chronicle of every experiment, test, result, analysis, and
abandoned direction in the CineEmbed project, from the EDA phase through the
final web-app pivot. This journal is the single source of truth for "what we
tried, what worked, what didn't, and why."

It is distinct from the other documentation in the repository:

- `docs/PROGRESS.md` — high-level project status tracker (current state).
- `docs/FINDINGS.md` — canonical run-results inventory.
- `docs/adr/0001-modeling-hybrid-architecture.md` — architectural decisions.
- `docs/superpowers/specs/` — forward-looking design contracts.
- `docs/archive/` — completed-phase historical specs and plans.
- `docs/report/` — academic-report deliverables.
- **`docs/journal/`** (this directory) — the experimental narrative + post-hoc
  analysis. Read this to understand *why* the codebase looks the way it does.

## Reading order

Start here if you have time:

1. **`00-context-and-goals.md`** — project goals, scope, key constraints.
2. **`10-results-table.md`** — every run, every metric, in one place.
3. **`07-retrieval-vs-nmi-discovery.md`** — first methodological finding (NMI ≠ retrieval).
4. **`12-z-sweep-ae-z32-discovery.md`** — second methodological finding (z-sweep U-curve; demo backbone locked at `ae_z32` per ADR D15).
5. **`06-negative-results.md`** — what failed and why.
6. **`08-scope-cuts-future-work.md`** — what we explicitly skipped.

If you only have five minutes: read `07`, `12`, and `10`.

If you're picking up the project to extend it: read `00` then jump to the
phase you need (`01` MVP, `04` Phase 1, `05` Round 1, `12` Round 2).

## File index

| File | Topic | Length |
|---|---|---:|
| `00-context-and-goals.md` | Project context, course, team, goals, timeline | foundational |
| `01-mvp-modeling-phase.md` | The 6 MVP runs (intermediate report deliverable) | history |
| `02-clustering-improvements-spec.md` | 5 techniques landed (InfoNCE + GMM/spectral/HDBSCAN + AMI + multilabel + per-axis-k) | history |
| `03-tooling-wandb-integration.md` | W&B integration, backfill, single-run pattern | history |
| `04-phase1-contrastive-sweep.md` | 3 contrastive pretext configs + results | execution |
| `05-round1-architecture-comparison.md` | VAE + 2 contrastive→DEC fine-tunes + diagnostic | execution |
| `06-negative-results.md` | 4 failure modes with root-cause analysis | analysis |
| `07-retrieval-vs-nmi-discovery.md` | NMI ≠ retrieval; ae_z64 demo backbone selection | analysis |
| `08-scope-cuts-future-work.md` | Things explicitly skipped + justifications | analysis |
| `09-operational-incidents.md` | WANDB key leak, sweep idempotency, GPU quota, GMM singular cov | ops |
| `10-results-table.md` | Every run × every metric in one table | data |
| `11-metrics-deep-dive.md` | Every metric: rationale, math, where used, why we trusted/distrusted it | reference |
| `12-z-sweep-ae-z32-discovery.md` | Round 2 finding (final): U-curve in z; `ae_z32` locked as demo backbone (ADR D15) | analysis |

## Naming convention

Files use `NN-kebab-case.md` where `NN` is the canonical reading order. The
numeric prefix is not meant as strict precedence — files cross-reference each
other freely.

## How this journal stays accurate

If you add a model run, a sweep, or a meaningful experimental result:

1. Add a row to `10-results-table.md`.
2. Append to the phase file that covers it (`04`, `05`, or a new file).
3. If it changes a decision recorded here, add a brief amendment block at the
   top of the relevant analysis file (`07`, `08`).
4. Do not edit settled history — append amendments dated `YYYY-MM-DD`.

## Generated

- **Initial population:** 2026-05-17 (AM), drawn from the conversation log + git
  history on `main` through commit `1e06a41` (build_index
  + initial `ae_z64` selection).
- **Round 2 finalization:** 2026-05-17 (PM), commit `ddb6cbd` — file 12 §1
  headline, §9, §11 rewritten with three-way U-curve result; file 10 Round 2
  sub-table extended; demo backbone re-locked at `ae_z32` per ADR D15.
