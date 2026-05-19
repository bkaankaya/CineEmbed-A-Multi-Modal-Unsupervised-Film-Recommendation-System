# 03 — Tooling: W&B Integration

> Three days after the intermediate report shipped (2026-05-09), W&B was
> added as the dashboard for everything subsequent. This file captures: what
> was built, why this shape, the per-run protocol, and the offline-safe
> contract.
>
> Read `01-mvp-modeling-phase.md` first.

## §1 Why this shape

The module docstring of `src/cineembed/wandb_integration.py:1` opens with a
deliberate set of design trade-offs:

> ```
> Design:
>     - `wandb_run(...)` is a context manager. Pass to train_model as `wandb_run=run`.
>     - If wandb is uninstalled OR `WANDB_MODE=disabled`, it yields None.
>     - All log_* helpers tolerate `run is None` and become no-ops.
>     - Eval helpers in eval.py stay pure: callers compose results, then call
>       `log_eval(run, results)` once, when ready to push.
> ```

Four trade-offs in plain English:

1. **Context manager, not decorator.** `wandb_run(...)` wraps a `try/finally`
   that always calls `wandb.finish()` — even on exception. A decorator
   model would couple the wandb lifecycle to a specific function shape
   (e.g. `train_model`), but the project has multiple sites that need
   wandb (training, eval, backfill script, Round 1 notebook).

2. **`log_eval(...)` is a one-shot push at end of run, not instrumentation
   inside eval functions.** The eval module (`src/cineembed/eval.py`) has
   seven separate composable functions (`cluster_assignments_kmeans/gmm/
   spectral/hdbscan/dec`, `evaluate_run`, `evaluate_run_per_axis_k`,
   `multilabel_macro_nmi`, `linear_probe`, `umap_plot`). Sprinkling
   `wandb.log` inside each would couple them to a network side-effect.
   Notebooks compose freely; one final `log_eval(run, results)` pushes
   the full dictionary.

3. **`log_artifact(...)` versions checkpoints rather than file-uploads.**
   Each training writes a `pretext_backbone.pt` or `<run>.pt` to disk;
   `log_artifact` wraps it as a `wandb.Artifact` so downstream notebooks
   can `wandb.use_artifact(...)` instead of hard-coding Drive paths.
   The Round 1 notebook uses this to load contrastive pretext backbones
   without depending on a brittle filesystem path.

4. **Mode auto-detection rather than hard-coding online/offline.** The
   context manager reads `WANDB_MODE` from the environment if `mode` is
   not passed explicitly, with `'online'` as the documented default. The
   Colab notebooks switch this to `'offline'` when no API key is present,
   which is the offline-safe contract — `wandb.init()` never prompts
   interactively (see §5 below).

Source: `src/cineembed/wandb_integration.py:1-31`.

## §2 The module

`src/cineembed/wandb_integration.py` — 219 LOC. Public surface:

| Symbol | Signature | Purpose |
|---|---|---|
| `wandb_run(config, *, run_name, project, entity, tags, group, notes, mode)` | context manager | Initialize a run; yield `wandb.Run` or `None`. Auto-finish on exit. |
| `log_epoch(run, *, epoch, train_loss, val_loss, lr, extra)` | function | Per-epoch metric push. No-op when `run is None`. |
| `log_eval(run, eval_dict, *, prefix, add_geo_nmi)` | function | One-shot eval push at end of run. Auto-computes `geo_nmi` when the three NMI keys are present. |
| `log_image(run, fig_path, *, key)` | function | Push a PNG/JPG as `wandb.Image`. Raises `FileNotFoundError` if path doesn't exist. |
| `log_artifact(run, path, *, name, type, description)` | function | Version a file (checkpoint, results.json) as a `wandb.Artifact`. |
| `log_table(run, rows, *, columns, key)` | function | Push a tabular result as `wandb.Table`. |

Two constants:

- `DEFAULT_PROJECT = "cineembed"` — the W&B project all CineEmbed runs land
  in.
- `_VALID_MODES: tuple[str, ...] = ("online", "offline", "disabled", "shared")`.

The `geo_NMI` auto-computation in `log_eval` (`src/cineembed/wandb_integration.py:148-168`)
is the **first place this composite metric was wired up** in the codebase.
The formula:

```python
geo_nmi = (genre_nmi · lang_nmi · decade_nmi)^(1/3)
```

is computed when all three keys are present and pushed as `{prefix}geo_nmi`.
This metric later (2026-05-16) becomes the formal selection metric for the
two-round strategy (ADR D13).

Implementation: commit `d2e364c`, 2026-05-09. Tests in
`tests/test_wandb_integration.py` (17 tests) cover:

- `WANDB_MODE=disabled` → `wandb_run()` yields `None`, no network call.
- `FakeRun` stub captures payloads to verify the `geo_nmi` formula against
  the `dec_z64_k21` baseline (genre=0.332, lang=0.294, decade=0.342).
- Edge cases: missing axis, zero axis, prefix handling, `add_geo_nmi=False`.
- `log_image`/`log_artifact` raise `FileNotFoundError` for typos.

## §3 Backfill of 6 MVP runs

`scripts/backfill_wandb.py` — 178 LOC. Commit `31936a5`, 2026-05-09.

The dashboard would otherwise be missing the historical baseline. The
script reads `artifacts/eval/results.json` and creates one W&B run per
entry, tagged with `'backfill'` and `'mvp'` so they can be filtered out of
post-instrumentation sweeps.

**Interface:**

```
.venv/bin/python scripts/backfill_wandb.py
.venv/bin/python scripts/backfill_wandb.py --dry-run        # preview
.venv/bin/python scripts/backfill_wandb.py --entity team3-seng474
.venv/bin/python scripts/backfill_wandb.py --project cineembed
.venv/bin/python scripts/backfill_wandb.py --results artifacts/eval/results.json
```

**What was logged per run** (per `scripts/backfill_wandb.py:118-170`):

| Field | Value |
|---|---|
| `name` | run name from results.json key (e.g. `dec_z64_k21`) |
| `config.run_name` | same |
| `config.z_dim` | from results.json |
| `config.k` | from results.json (DEC only) |
| `config.n_epochs` | from results.json (training runs only) |
| `config.model` | from `HYPERPARAMS` dict (curated by hand from FINDINGS.md / PROGRESS.md) |
| `config.modal` | True for multi-modal, False for vanilla/baselines |
| `config.loss_weights` | `"W2"`, `"W1"`, `"DEC_KL"`, `"equal"` |
| `config.backfilled_from` | string `"artifacts/eval/results.json (Mayıs 5 MVP)"` |
| `tags` | `["mvp", "backfill", "intermediate-report"]` |
| `group` | `"ae-family"` / `"dec-family"` / `"baseline"` (per `GROUPS` dict at `scripts/backfill_wandb.py:72-79`) |
| Metrics: `genre_nmi`, `genre_ari`, `lang_nmi`, `lang_ari`, `decade_nmi`, `decade_ari`, `final_val_loss` | from results.json |
| Metric `geo_nmi` | auto-computed via `_geo_nmi(metrics)` |

The commit message documents the verified dashboard:

> Verified push: dashboard at
> `https://wandb.ai/kaankaya928-ted-university/cineembed` shows all 6
> runs. Best baseline: `dec_z64_k21` with `geo_nmi=0.3217` (genre=0.332,
> lang=0.294, decade=0.342). This becomes the reference line for any future
> sweeps.

(Note: `geo_nmi=0.3217` from the backfill commit message matches
`geo_NMI=0.323` from the 2026-05-17 re-eval in `10-results-table.md` to
within rounding — the 2026-05-17 re-eval also added `*_ami` keys that the
backfill did not capture.)

## §4 Single-run-per-training protocol

After commit `d0f08ac` (2026-05-16,
`fix(wandb): single run per training + safe offline fallback`), the protocol
is: **training + eval + artifact all happen inside one `wandb_run(...)`
context.**

Source: `scripts/train_contrastive.py:256-286`:

```python
from cineembed.wandb_integration import wandb_run as _wandb_run
wandb_ctx = _wandb_run(
    config={...},
    run_name=args.run_name,
    project=args.wandb_project,
    entity=args.wandb_entity,
    group=args.wandb_group,
    tags=args.wandb_tags.split(',') if args.wandb_tags else None,
    mode=args.wandb_mode,
)
with wandb_ctx as run:
    history = train_model(..., wandb_run=run)
    eval_metrics = evaluate_run(cluster_ids, labels)
    log_eval(run, eval_metrics)
    log_artifact(run, str(backbone_path),
                 name=f"pretext_backbone__{args.run_name}",
                 type="model")
```

The earlier version of `train_contrastive.py` opened **two** wandb runs (one
for training, one for eval), which split the metric history across two
separate runs on the dashboard and made comparison hard — the train/eval
metrics didn't share a `run.summary` or a checkpoint pointer. The single-run
pattern fixes this and writes headline numbers (`km_genre_nmi`, `km_geo_nmi`)
to `run.summary` so they appear in the project run list without drilling
into per-step logs.

Round 1 (`notebooks/07_round1_finetune.ipynb`, commit `276f47d`) follows
the same pattern: one wandb run per config, three configs total under
`group=round-1`, with `log_eval` prefixes `km_`, `gmm_`, `axis_`, `dec_`
distinguishing the four cluster-method evaluations on the same backbone.

The pre-2026-05-16 backfill script (`backfill_wandb.py`) does NOT use
`wandb_run(...)` — it calls `wandb.init` and `wandb.finish` directly because
it was written before the context manager existed (same commit-day but
different timing — `d2e364c` defined the context manager, `31936a5` ran the
backfill three minutes later).

## §5 Detection ladder + offline fallback

The `notebooks/03_train_contrastive.ipynb` and
`notebooks/07_round1_finetune.ipynb` use a three-step API-key detection
ladder, source `notebooks/03_train_contrastive.ipynb:143-179`:

```python
WANDB_MODE = None
WANDB_KEY_PRESENT = False

# Step 1: Colab Secret
if WANDB_PROJECT and IN_COLAB:
    try:
        from google.colab import userdata
        _key = userdata.get('WANDB_API_KEY')
        if _key:
            os.environ['WANDB_API_KEY'] = _key
            WANDB_KEY_PRESENT = True
    except Exception:
        pass

# Step 2: process env (also signal-of-presence locally)
elif WANDB_PROJECT and not IN_COLAB:
    if os.environ.get('WANDB_API_KEY') or Path.home().joinpath('.netrc').exists():
        WANDB_KEY_PRESENT = True

# Step 3: classify into a mode
if WANDB_PROJECT is None:
    WANDB_MODE = 'disabled'
elif WANDB_KEY_PRESENT:
    WANDB_MODE = 'online'
    os.environ['WANDB_MODE'] = 'online'
else:
    # Critical: no API key → fall back to offline mode rather than letting
    # wandb.init() prompt interactively (Colab subprocess can't answer).
    WANDB_MODE = 'offline'
    os.environ['WANDB_MODE'] = 'offline'
```

Three sources for the API key, in priority order:

1. **Colab Secret `WANDB_API_KEY`** via
   `google.colab.userdata.get('WANDB_API_KEY')`. The user adds it once via
   Tools → Secrets in the Colab UI. The cell then sets `os.environ['WANDB_API_KEY']`
   so the subprocess inherits it.

2. **Process env `WANDB_API_KEY`** — covers local dev where the key is
   exported in the shell before launching Jupyter.

3. **Local `~/.netrc`** — set by a previous `wandb login` command.

If all three miss, the notebook **switches `WANDB_MODE=offline`** rather
than hitting `wandb.init`'s interactive prompt. The interactive prompt
reads from stdin, which a Colab subprocess cannot answer — it
`KeyboardInterrupt`s and crashes the whole training run.

The fallout from this exact bug is captured in `09-operational-incidents.md`.
The fix (commit `d0f08ac`) is the prior-mode default + the offline-safe
documented contract in the notebook markdown:

| Setup | Behavior |
|---|---|
| Colab Secret `WANDB_API_KEY` present | Online: runs stream to `wandb.ai/<your-entity>/cineembed`. |
| No secret + `WANDB_PROJECT = None` | Disabled. Only console + history.json / eval.json on Drive. |
| No secret + `WANDB_PROJECT` set | Offline. Runs land in `wandb/offline-run-*/`. Sync later with `wandb sync`. |

(From `notebooks/03_train_contrastive.ipynb:107-113`.)

`scripts/train_contrastive.py` accepts `--wandb-mode` to override the
notebook's auto-detection — useful for forcing offline runs locally even when
a `.netrc` is present.

## §6 Group / tag taxonomy

The dashboard at `wandb.ai/<entity>/cineembed` is organised by the
`group` field and free-text `tags`.

| Group | Tags | Runs | Source |
|---|---|---|---|
| `phase-1-sweep` | `['contrastive', 'phase-1']` `[verify]` — tags inferred from spec defaults, exact tag list confirmed by inspecting `scripts/train_contrastive.py` defaults | 3 contrastive pretext runs | `scripts/train_contrastive.py` (commit `5fa95ff`/`d0f08ac`) |
| `round-1` | `['round-1', '<family>']` where family ∈ `{vae, dec-finetune, ae}` `[verify]` | 3 Round 1 architecture-comparison runs | `notebooks/07_round1_finetune.ipynb` (commit `276f47d`) |
| `ae-family` / `dec-family` / `baseline` | `['mvp', 'backfill', 'intermediate-report']` | 6 backfilled MVP runs | `scripts/backfill_wandb.py` (commit `31936a5`) |

The MVP backfill runs use the older `ae-family` / `dec-family` / `baseline`
groups because they were created before the convention of using
`phase-N` / `round-N` group names. They are filterable out of post-MVP
analyses via the `'backfill'` tag — i.e. when reviewing Phase 1 or Round 1
sweeps on the dashboard, filter `tags != backfill`.

## §7 Cross-references

- `04-phase1-contrastive-sweep.md` — the first sweep to use the
  single-run-per-training protocol + the detection ladder in earnest.
- `05-round1-architecture-comparison.md` — uses the same patterns plus the
  `wandb.use_artifact` flow to load pretext backbones.
- `09-operational-incidents.md` — the `KeyboardInterrupt` cascade that
  prompted the offline-safe fallback design.
