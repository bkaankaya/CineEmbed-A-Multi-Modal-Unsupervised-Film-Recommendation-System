# 09 — Operational Incidents

> Factual record of the non-trivial operational problems hit during the
> Phase 1 + Round 1 modeling sweeps and the retrieval-eval pivot. Each
> entry is structured: **symptom → root cause → fix → residual risk →
> lesson**. The audience is the next person who hits the same problem
> and needs to recognise it quickly.

## §1 Why this file exists

Modeling work on a Colab + Google Drive + Git + Weights & Biases +
multi-machine surface accumulates operational debt that doesn't fit
anywhere else in the docs. The MVP phase had relatively little of
this — most of the runs were local, the dataset fit in memory, and
W&B integration came later. The Phase 1 sweep (`04-phase1-contrastive-sweep.md`)
and Round 1 fine-tunes (`05-round1-architecture-comparison.md`) were the
first sustained Colab work in the project, and they surfaced a cluster of
secret-handling, idempotency, sync-divergence, and quota issues that took
real wall-clock to recover from.

This file collects those incidents so the next sweep doesn't replay them.
It is **not** a runbook — see `notebooks/00_colab_setup.ipynb` and the
sweep notebooks themselves for happy-path instructions. It is a record of
the failure modes and their fixes.

## §2 INC-1: `WANDB_API_KEY` committed to a public repo (commit `17d6fbb`)

**Severity:** 1 (credential leak; risk-accepted, see Decision below).

**Symptom.** During Phase 1 sweep setup (2026-05-16), the user pasted

```python
import os
os.environ['WANDB_API_KEY'] = '<redacted-wandb-key>'
```

into a new cell of `notebooks/03_train_contrastive.ipynb` to avoid the
Colab-Secret workflow. The cell (id `508819c8`) was saved by Colab
autosave, the notebook was committed as part of `17d6fbb`
("fix(03_train_contrastive): skip-if-done guard in sweep loop"), and the
commit was pushed to `origin/main`. The W&B API key
was therefore public for the duration the branch was visible on GitHub.

**Detection.** The leak was caught during the journal-prep audit:

```
git log -S "<wandb-key-prefix>" --all --oneline
```

surfaces exactly one commit: `17d6fbb`. The diff for that commit shows
the cell inserted between the existing markdown intro (`03ct002`) and the
"environment + paths" cell that follows; see lines added in the
`diff --git a/notebooks/03_train_contrastive.ipynb` block of the commit.

**Decision: do not revoke.** Free-tier W&B account, no billable resources
attached, and rotating the key would invalidate `~/.netrc` on every
machine. The user accepted the risk. The cell was deleted on disk and a
follow-up commit (`88dbcc4`) removed it from the working tree along with
the three-step detection ladder fix that supersedes it (INC-4 below).

**Residual risk.**

- Anyone who cloned or fetched `main` between the
  `17d6fbb` push and the `88dbcc4` cleanup retains a usable key in their
  reflog. The git history of the remote branch still contains the leak;
  `git log -S` will still find it.
- Automated GitHub secret scanners (TruffleHog, GitGuardian) may have
  indexed the key during its public window. W&B's own secret-scanner
  notification did not fire (no email observed) `[verify]`.
- Force-history rewriting (`git filter-repo` / interactive rebase) would
  remove the leak from the tree but break every existing clone's history
  and is not warranted for an accepted-risk free-tier key.

**Lesson.** The Colab Secret API (`google.colab.userdata.get('WANDB_API_KEY')`)
is the only safe path on Colab. Never paste raw secrets into a notebook
cell that will be saved. If a one-time env-var workaround is unavoidable,
make it a scratch cell, run it, then **delete the cell before saving** —
or keep secrets in an untracked `setup_local.py` and `.gitignore` it.

**Mitigation in place.** Both
`notebooks/03_train_contrastive.ipynb:03ct003` and
`notebooks/07_round1_finetune.ipynb:r1_wandb_cell` now implement the
three-step detection ladder described in INC-4. The detection cell
includes an inline comment warning future contributors not to paste raw
keys into the committed source.

## §3 INC-2: Phase 1 sweep loop not idempotent — run 1 retrained twice

**Severity:** 2 (compute / time waste; one eval.json overwritten).

**Symptom.** During the Phase 1 sweep on 2026-05-16, the user stopped
the sweep cell after run 1 (`contrastive_tau0p1_drop0p3`) finished
because the UMAP rendering was eating 10-15 min per run with no
streaming progress output. UMAP-rendering disabled (`INCLUDE_UMAP = False`)
and the cell was re-executed. The loop did **not** detect that run 1's
`pretext_backbone.pt` + `eval.json` were already on Drive — it kicked
off training run 1 from scratch a second time, overwriting the first
run's `eval.json` in place. The second training took ~25.7 min (vs
~6.7 min the first time) because Colab had reassigned a slower GPU
slot between sessions `[verify]`; the wall-clock divergence is what
made the issue visible to the user.

**Root cause.** The original sweep cell in
`notebooks/03_train_contrastive.ipynb:03ct005` had no skip check at the
head of the loop body. Every iteration unconditionally built the
`python scripts/train_contrastive.py ...` command and invoked it via
`get_ipython().system(cmd)`. The subprocess on its own can't know that
the artifact already exists — the script's `mkdir(parents=True,
exist_ok=True)` happily reuses the directory and `torch.save` overwrites
the checkpoint.

**Fix.** Commit `88dbcc4` (notebooks/03_train_contrastive.ipynb cell
`03ct005`) added a `FORCE = False` flag and a per-iteration check:

```python
for cfg in SWEEP:
    run_dir = ARTIFACTS / 'models' / cfg['run_name']
    ckpt = run_dir / 'pretext_backbone.pt'
    evalj = run_dir / 'eval.json'
    ...
    if (not FORCE) and ckpt.exists() and evalj.exists():
        print(f"SKIP — already trained. ckpt={ckpt.stat().st_size//1024} KB, "
              f"eval={evalj.stat().st_size} B.")
        print(f"      Set FORCE=True to overwrite, or delete {run_dir} to redo.")
        continue
```

Set `FORCE = True` (or `rm -rf <run_dir>`) to redo a config.

Note: the `17d6fbb` commit message claims to have added this guard, but
the actual notebook diff in that commit only adds the leaked-key cell
(see INC-1) and toggles `INCLUDE_UMAP = True → False`. The skip-if-done
guard was first written but didn't make it into the tree until `88dbcc4`
(the commit message there is explicit: *"the prior 'skip' commit `17d6fbb`
landed empty — only output diff, no source change went through"*). This
is itself an operational lesson — `git show <hash> -- <file>` is the
only way to verify what actually changed.

**Recovery.** The run-1 `eval.json` written to Drive was the second
training's, not the first's. The first training's metrics were therefore
lost. Because seed=42 was identical, the two runs converged to within
±0.005 of each other on every NMI/ARI/AMI metric (the residual variation
comes from Colab non-determinism in cuDNN / dataloader workers), so the
substantive loss was negligible.

**Residual risk.** The same shape of bug — re-running a sweep cell after
edits without per-iteration artifact gating — can recur in any notebook
that wraps a training script in a `for cfg in SWEEP:` loop.
`notebooks/07_round1_finetune.ipynb:r1_sweep_cell` inherits the same
guard pattern by design (its cell-level markdown even says "Idempotent —
set `FORCE=True` to retrain"). Any future sweep notebook should copy the
pattern.

## §4 INC-3: `wandb.init` interactive prompt cascade in subprocess

**Severity:** 3 (single-recovery-loop friction; fully fixed).

**Symptom.** The very first Phase 1 Colab dry-run (2026-05-16, before
commit `d0f08ac`) died inside `wandb.init` with `KeyboardInterrupt`
roughly two minutes after the script started. wandb had printed
something like `Enter your choice:` to stderr (re-paginating the
three-way "Use existing W&B account / Create new / Disable" menu) and
then immediately bailed.

**Root cause.** When `WANDB_API_KEY` is unset, `~/.netrc` is absent, and
`WANDB_MODE` is not set, `wandb.init` falls back to a three-choice
interactive login prompt implemented via `click.prompt`. In a Colab
notebook subprocess (`!python scripts/train_contrastive.py ...`) stdin
is not connected to a pty — `click.prompt` immediately raises
`click.Abort`, which the wandb wrapper re-raises as `KeyboardInterrupt`
through its own retry logic. The exception then propagated up through
`with wandb_ctx as run:` in `scripts/train_contrastive.py` and killed
the script before training began.

**Fix.** Commit `d0f08ac` ("fix(wandb): single run per training + safe
offline fallback") rebuilt both halves of the contract:

1. Script side (`scripts/train_contrastive.py:90-92, 254-273`): added
   a `--wandb-mode` argument forwarded to the `wandb_run(...)`
   context manager. The context manager honours `mode='offline'` and
   writes to `<repo>/wandb/offline-run-*/` without ever consulting the
   network.
2. Notebook side
   (`notebooks/03_train_contrastive.ipynb:03ct003`): a three-step
   detection ladder runs **before** the sweep cell, sets
   `os.environ['WANDB_MODE']` explicitly, and passes the same value as
   `--wandb-mode` to the script. The notebook never invokes the
   interactive `wandb login` CLI.

If no key is found, the run lands under `<repo>/wandb/offline-run-*`; the
user can `wandb sync wandb/offline-run-...` later to push it online. The
commit message confirms the offline smoke run: *"Offline smoke run: 1
epoch CPU on the full 329k-row dataset; train, eval, artifact, and
summary logged into a single offline-run-* dir without prompting."*

**Lesson.** Any third-party tool that does interactive auth must be
wrapped with explicit env / config detection **and** a non-interactive
fallback before being run in a subprocess. Don't rely on the tool's
own "auto-detect" path; treat it as an interactive-only escape hatch
and short-circuit it.

## §5 INC-4: W&B Colab Secret vs env var detection inconsistency

**Severity:** 3 (single-loop friction; led directly to INC-1).

**Symptom.** After the dry-run in INC-3 had been worked around with an
inline env-var set, the user noticed the detection cell in
`notebooks/03_train_contrastive.ipynb:03ct003` was still printing
`No WANDB_API_KEY found → using OFFLINE mode.` despite the env var being
populated by the cell above it.

**Root cause.** The detection logic before `88dbcc4` had the shape:

```python
if IN_COLAB:
    try:
        from google.colab import userdata
        _key = userdata.get('WANDB_API_KEY')
        ...  # set env var if present
    except Exception:
        pass
else:
    # local-only fallback
    if Path.home().joinpath('.netrc').exists():
        WANDB_KEY_PRESENT = True
```

When `IN_COLAB` is True, the else branch never runs. The env-var check
lived under the else branch only, so a Colab run that already had
`WANDB_API_KEY` set in the process env (e.g. by an INC-1-style scratch
cell) never resolved to `WANDB_KEY_PRESENT = True`. The two detection
paths were mutually exclusive when the environments actually overlap.

**Fix.** Commit `88dbcc4` (cell `03ct003`) flattened the ladder to three
independent guarded checks:

1. **Colab Secret** (`google.colab.userdata.get('WANDB_API_KEY')`) —
   preferred; never lands in the notebook source.
2. **Process env var** (`os.environ.get('WANDB_API_KEY')`) — works in
   both Colab and local. Set externally; do not paste into the file.
3. **`~/.netrc`** — local-only fallback for users who ran `wandb login`.

If any of the three resolves, `WANDB_KEY_PRESENT = True` and the mode
becomes online. If all three fail, `WANDB_MODE = 'offline'`. Both the
sweep notebook (cell `03ct003`) and the Round 1 notebook
(cell `r1_wandb_cell`) use the identical ladder.

**Lesson.** Auto-detect ladders should never be mutually exclusive when
the environments overlap. Write each check as a guard that mutates a
shared `present` flag and falls through if it can't help; don't use
`if / else` branches that compete for the right to set the result.

## §6 INC-5: GMM singular covariance on the DEC z=64 latent

**Severity:** 2 (compute waste — one re-eval iteration; analysis
implication is more important than the recovery).

**Symptom.** The local CPU re-eval of `dec_z64_k21` (2026-05-17, before
the `dec_z64_k21/eval.json` artifact was written) crashed in
`cluster_assignments_gmm(z, k=21, ...)` with:

```
ValueError: Fitting the mixture model failed because some components have
ill-defined empirical covariance (for instance caused by singleton or
collapsed samples). Try to decrease the number of components, or increase
reg_covar.
```

This was on the same latent that produced perfectly healthy KMeans-k=21
metrics (`gNMI = 0.333`, see `10-results-table.md`), so the crash was
not a "the latent is broken" problem — it was a "GMM-specific assumption
broke down" problem.

**Root cause.** The DEC latent at z=64 has dimension-wise std ≈ 1.59
(healthy at the global scale `[verify]`) but the **cluster-wise**
covariances are degenerate — some of the 21 clusters are small or
angularly collapsed enough that their per-component covariance matrix
becomes singular at the project default `reg_covar=1e-4` (see
`src/cineembed/eval.py:69`). This is consistent with the angular-collapse
finding documented in `07-retrieval-vs-nmi-discovery.md` and ND-4 in
`06-negative-results.md`: inside a single DEC cluster the latent
vectors are aligned to the same direction, so the empirical covariance
has rank deficient with respect to the latent dimensionality.

**Fix.** The local re-eval rebuilt the GMM with `reg_covar=1e-2` (two
orders of magnitude higher); the fit succeeded and the result was
written into `artifacts/models/dec_z64_k21/eval.json` as the
`gmm_k21` block. Values: `genre_nmi = 0.332`, `decade_nmi = 0.346`,
`lang_nmi = 0.283` — within 0.01 of the corresponding KMeans
numbers, which is the expected outcome on a non-collapsed latent
where GMM and KMeans agree.

The project default in `src/cineembed/eval.py:cluster_assignments_gmm`
was **not** changed. The higher `reg_covar` is a workaround for a
collapsed latent, not the right global default — bumping it on healthy
latents (e.g. `ae_z64`) would over-regularise the covariance estimate
and obscure real cluster structure.

**Lesson.** When `sklearn.mixture.GaussianMixture` fails with singular
covariance, the latent has a structural problem — don't just bump
`reg_covar`. Investigate whether the collapse shows up in other
sanity statistics: in this case the `random_pair_cos` distribution
already showed it (`dec_z64_k21` random-pair-cos std = 0.421 vs
`ae_z64`'s 0.299; intra-cluster cosines all 1.000 — see
`10-results-table.md`). The GMM crash was a lagging indicator of the
same underlying angular collapse that the retrieval-eval would later
make blindingly obvious.

## §7 INC-6: Colab free-tier GPU quota exhaustion mid-Round-1

**Severity:** 2 (compute waste; recovered by switching to local CPU).

**Symptom.** During the Round 1 sweep on 2026-05-17 the Colab T4 was
de-allocated and the notebook fell back to CPU. Subsequent operations
were ~10× slower (encode + cluster on 329 044 rows went from ~30 s
to ~5-10 min). The user paused Colab work for the day and continued
on local CPU.

**Root cause.** The project ran roughly 8-10 hours of T4 wall-clock
across 2026-05-16/17 `[verify]`: Phase 1 sweep with UMAP (run 1 had
`INCLUDE_UMAP = True`, ~15 min UMAP each), Round 1 sweep (3 configs at
~10-15 min each), the wasted re-train from INC-2, plus eval iterations
and debugging on top. Colab free-tier rolling quota is a soft 12 hours
of T4 / day `[verify]` and was depleted.

**Workaround.** All retrieval-eval and `build_index` work continued on
the local Mac CPU. `scripts/build_index.py` encodes 329k rows in
under 1 s on the M-series laptop and runs the 500-query genre@5
evaluation in ~10 s `[verify]`. The DEC re-eval (INC-5) and the
`ae_z64`/`dec_z64_k21` retrieval comparison (the basis for the
demo-backbone pivot in `07-retrieval-vs-nmi-discovery.md`) all ran
locally. The pending Round 2 z-sweep is gated on the next free-tier
reset window (~24h cooldown typical).

**Lesson.** Keep the inference path GPU-free. Reserve Colab GPU **only**
for training. Test scripts on CPU locally before launching them on
Colab to confirm they don't accidentally require a GPU for encode /
eval steps that don't need one. `scripts/build_index.py` was designed
to this principle (uses `torch.device('cpu')` by default for the encode
loop) and that's why the demo backbone selection happened on schedule
even after the quota hit.

## §8 INC-7: Drive-vs-local divergence — `dec_z64_k21.pt` only on local Mac

**Severity:** 3 (single-recovery friction; informed the local re-eval
path).

**Symptom.** When running the diagnostic cells added in commit
`19962b8` (`notebooks/07_round1_finetune.ipynb`) on Colab, an
`AssertionError: Missing checkpoint:
/content/drive/MyDrive/CineEmbed/artifacts/models/dec_z64_k21.pt` was
expected during local diagnostic re-eval `[verify]` despite the file
existing locally at
`<repo-root>/.../artifacts/models/dec_z64_k21.pt`. Drive
had `ae_z64.pt`, `vanilla_ae_z64.pt`, `ae_z64_w1.pt` but no MVP DEC
checkpoint.

**Root cause.** The MVP DEC checkpoint was trained locally during the
2026-05-05 MVP run (commit `8097685`) and never uploaded to the Drive
`artifacts/models/` folder. The team's workflow had two canonical
storage surfaces — Drive (for Colab training) and the local repo (for
fast iteration) — but no enforced bidirectional sync. Most MVP
artifacts had been pushed manually after each run; the DEC checkpoint
was missed.

**Fix.** The re-eval was run locally. CPU is fast enough for the full
329k-row encode + KMeans/GMM/per-axis-k pipeline (~15 min total
end-to-end `[verify]`). The `eval.json` was written to
`artifacts/models/dec_z64_k21/eval.json` and a `dec_z64_k21` row added
to `artifacts/eval/results.json`. The Drive copy of
`dec_z64_k21.pt` is still out of date — this is acceptable as long as
the demo backbone is `ae_z64` (which IS on Drive) and the Round 2
re-targeting away from DEC family (see
`07-retrieval-vs-nmi-discovery.md`) means no future Colab run will
need to load DEC.

**Lesson.** Name a single canonical artifact location, or commit
checkpoint hashes to git (via `git lfs` or a tracked
`artifacts/MANIFEST.md` with SHAs), and lint the divergence at the
start of every sweep. A `scripts/check_artifacts.py` that
walks the expected list and prints `(Drive-only / local-only / both /
neither)` for each would have caught this before the diagnostic crash.

## §9 INC-8: `pyarrow` not in project deps — first `build_index.py` run crashed at `to_parquet`

**Severity:** 3 (single-recovery friction).

**Symptom.** The first invocation of `scripts/build_index.py` (commit
`1e06a41`) crashed at line `403` (`films_df.to_parquet(films_path,
index=False)`) with:

```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
A suitable version of pyarrow or fastparquet is required for parquet support.
```

**Root cause.** `pandas.DataFrame.to_parquet` requires either `pyarrow`
or `fastparquet` and neither was in `pyproject.toml`'s dependency list.
The project had been writing JSON / NPZ until this point — parquet was
introduced specifically for the inference-index films table (small but
columnar; loads ~2× faster than JSON for the demo API).

**Fix.** `pyproject.toml` gained a new optional-dependencies group:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.1"]
wandb = ["wandb>=0.16"]
demo = [
    "pyarrow>=15",      # parquet engine for films table
    "fastapi>=0.110",   # web app backend
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
]
```

`pip install -e ".[demo]"` resolves it. The script reruns cleanly.

**Lesson.** When a script introduces a new file-format dependency or a
new transitive runtime (parquet → arrow), add it to the appropriate
optional-deps group **in the same commit** as the script change.
`build_index.py` introducing both parquet output AND a dependency
nobody had installed is exactly the regression cycle to avoid.

## §10 INC-9: Stray secret-bearing cells in notebook source

**Severity:** 3 (caught before commit, but the shape of the bug is INC-1).

**Symptom.** During inspection of `notebooks/03_train_contrastive.ipynb`
(before commit `88dbcc4`), the working tree briefly contained two stray
cells that did not belong in the canonical notebook:

- A `git pull` cell the user had added to refresh the repo inside Colab.
- A scratch cell that set `os.environ['WANDB_API_KEY'] = '<key>'` inline
  (a direct cousin of INC-1's leak).

Both were autosaved by Colab and persisted on disk into the working tree.

**Root cause.** Colab autosaves every cell on focus-change. A
scratch / debug cell typed during a session lands in the working tree
unless the user explicitly removes it before saving. The notebook's
own `nbformat` metadata records cell additions identically to "real"
content cells.

**Fix.** Commit `88dbcc4`'s message explicitly lists this cleanup:
*"Drop two stray cells that were in the working tree (a one-off `git
pull` cell and a scratch cell that had set `WANDB_API_KEY` inline)."*
The cells were removed; the canonical detection ladder in `03ct003`
(see INC-4) supersedes any manual env-set cell.

**Lesson.** Scratch cells with secrets must never live in a notebook
that will be saved. Options that work:

- Use Colab Secrets exclusively (no in-notebook key handling).
- Keep secret-bearing setup in a gitignored sibling file (e.g.
  `notebooks/local_setup.py` that the notebook imports if it exists).
- Run pre-commit `nbstripout` on every notebook — this is the
  industrial standard solution and is **not** currently configured
  in the repo `[verify]`.

## §11 INC-10: `intermediate-progress-presentation.pptx` modified-but-not-committed

**Severity:** 3 (noise / triage friction; no functional impact).

**Symptom.** `git status` has shown
`docs/presentation/intermediate-progress-presentation.pptx` as
**modified** through multiple commit cycles since the presentation
was shipped on 2026-05-06 (W11). The diff is binary, the bytes change
between every `git status` invocation when PowerPoint is open, and
the modifications are most likely autosave / metadata churn (last-open
timestamp, undo state, embedded thumbnail) rather than intentional
content changes.

**Decision.** Leave uncommitted. The intermediate report is shipped;
the canonical version of the deck is the `HEAD` copy from commit
`695bd06` ("docs(pptx): build and commit
intermediate-progress-presentation.pptx"). The working-tree noise is
not actionable.

**Lesson.** Binary artifacts in git that are routinely autosaved by an
external tool produce non-actionable diff noise. Future projects
should keep PPTX **builds** out of the working tree (build under
`dist/` or `build/` and `.gitignore` that path) and commit only the
build script (`docs/presentation/build.py` in this repo's case). The
deck source code is reproducible; the binary output is not.

## §12 INC-11: UMAP single-threaded slowness with `random_state` set

**Severity:** 3 (user-perception "hung notebook" pattern; harmless once
recognised).

**Symptom.** Every Phase 1 contrastive run with `--umap` enabled spent
10-15 min in UMAP rendering after the eval finished. The notebook
emitted no streaming progress during this window, so the first time
the user saw it they thought the kernel had hung and considered
interrupting (which would have invalidated the upstream training).
This was the direct trigger for INC-2 — the user disabled UMAP
mid-sweep and re-ran the cell, only to discover the loop wasn't
idempotent.

**Root cause.** `umap-learn` emits the warning:

> `n_jobs value 1 overridden to 1 by setting random_state`

When a random seed is set, UMAP refuses to parallelise — this is a
deliberate reproducibility decision in the library (multi-threaded
neighbour search is order-dependent). With 329 044 rows × 64 dims on
a single thread, the dimensionality-reduction step itself is the
bottleneck, not the rendering.

**Decision.** For the Phase 1 sweep we kept `--umap` on for the first
config only and disabled it (`INCLUDE_UMAP = False`) for the others.
For Round 1 we left it off entirely. UMAP plots for the report can be
regenerated for any checkpoint as a separate post-sweep step.

**Lesson.** Decouple visual eval from numeric eval. Run UMAP as a
background job after the sweep completes, not inline. If UMAP must be
inline, **either** drop `random_state` (lose reproducibility, gain
multi-threading) **or** print a streaming progress indicator so the
user can distinguish "slow but working" from "hung".

## §13 INC-12: Repository cleanup before the journal

**Not an incident** — flagged here for completeness so the file
inventory in `00-context-and-goals.md` reflects the current state.

During journal preparation (commit `edcbcb6`), the doc audit moved
completed-phase specs/plans under `docs/archive/` and refreshed
`docs/PROGRESS.md`, `docs/FINDINGS.md`, and
`docs/adr/0001-modeling-hybrid-architecture.md` to match the
2026-05-16 state. The `.gitignore` additions in that pass (and the
subsequent `2682 12:49a` commit window) excluded:

- `artifacts/inference/` — regenerable via `scripts/build_index.py`
  in ~10 s on CPU; ~80 MB npy + ~80 MB parquet per backbone.
- `AGENTS.md` — auto-emitted by the claude-mem tooling, not a
  project artifact.
- Root-level duplicates of files already under `data/`
  (`Deep_Learning_EDA_ipynb_adlı_not_defterinin_kopyası.ipynb`,
  `SENG474_Presentation_Team3.pptx`) that had been pasted into the
  repo root during an earlier sync mishap `[verify]`.

This is normal hygiene, not an incident, but it changes what `git
status` reports between journal-prep and any subsequent sweep — a
fresh clone will look different from a clone made two weeks ago.

## §14 Summary table

| ID | Title | Severity | Detected | Fix commit | Status |
|---|---|---:|---|---|---|
| INC-1 | `WANDB_API_KEY` committed to public repo | 1 | journal audit | `88dbcc4` (cleanup) | Risk-accepted; not revoked |
| INC-2 | Sweep loop not idempotent — run 1 retrained twice | 2 | wall-clock divergence | `88dbcc4` (skip-if-done guard) | Fixed |
| INC-3 | `wandb.init` interactive prompt cascade | 3 | dry-run crash | `d0f08ac` (`--wandb-mode` + ladder) | Fixed |
| INC-4 | Colab Secret vs env var detection mutually exclusive | 3 | offline-mode false-positive | `88dbcc4` (flat ladder) | Fixed |
| INC-5 | GMM singular covariance on DEC z=64 | 2 | local re-eval crash | local `reg_covar=1e-2` workaround | Worked around; project default unchanged |
| INC-6 | Colab T4 quota exhaustion | 2 | training fell back to CPU | switched to local CPU + paused Round 2 | Workaround; Round 2 gated on reset |
| INC-7 | `dec_z64_k21.pt` only on local Mac, not Drive | 3 | Colab diagnostic crash | re-eval locally | Workaround; Drive still divergent |
| INC-8 | `pyarrow` missing — `build_index.py` parquet crash | 3 | first script invocation | `pyproject.toml` `[demo]` deps group | Fixed |
| INC-9 | Stray secret-bearing scratch cells in notebook | 3 | inspection before commit | `88dbcc4` (drop cells) | Fixed |
| INC-10 | `.pptx` modified-but-not-committed noise | 3 | persistent `git status` line | not committed (autosave churn) | Triage deferred |
| INC-11 | UMAP single-thread slowness | 3 | first sweep run looked hung | `INCLUDE_UMAP = False` default | Decoupled |

Severity 1 = data/security loss. Severity 2 = compute / time waste.
Severity 3 = friction / single-recovery loop.

## §15 Cross-references

- Timeline of when these incidents fit in the project flow:
  `00-context-and-goals.md` (Timeline section).
- The W&B detection ladder is also documented in
  `03-tooling-wandb-integration.md` from the architecture side.
- The latent angular-collapse that triggered INC-5 (GMM singular
  covariance) is analysed in `07-retrieval-vs-nmi-discovery.md` and
  enumerated as ND-4 in `06-negative-results.md`.
- The pivot to `ae_z64` as demo backbone (which made INC-7's
  Drive-vs-local divergence acceptable) is documented in
  `07-retrieval-vs-nmi-discovery.md`.
- The numbers cited in this file (NMI / std / random_pair_cos) come
  from `10-results-table.md` and the underlying
  `artifacts/models/*/eval.json` + `artifacts/inference/*/manifest.json`.
