REDTEAM

# Task: run provenance, run store (HDF5), and plotting utilities

You are working in `/workspace` (Python 3.14, `uv`, flat `src/` layout with `pythonpath=["src"]`),
on the already-checked-out branch `pipeline-infrastructure` (working tree was clean at hand-off).
Commit to this branch only; never push, never touch `main`.

Read first, in this order: `CLAUDE.md`, `CODING_STYLE.md`, `WRITING_STYLE.md`, `docs/lab-book.md`
("Logging: MLflow"; this file is renamed in the first step below), `docs/implementation_doc.md` ("Logging: `logged_training.py`"),
`docs/capacity-scan-handoff.md`, `scan-list.md`, `TRIAGE.md` (#7, #8, #19), then
`src/run_experiments.py`, `src/logged_training.py`, `src/run_config.py`, `src/train.py`,
`src/dynamics_insights.py`, `src/data.py`, `src/plot_size_scan.py`, and the tests for them.

Follow the `REDTEAM` workflow from `CLAUDE.md`: write the plan, red-team it with a subagent, fix
critical findings, repeat; implement; red-team the implementation at the points listed under
"Phase 6". Every red-team loop is capped at 3 passes (this overrides the open-ended "repeat until
no critical issues" in `CLAUDE.md`; remaining findings go in the final report). The decisions in
"Decisions already made" are final: do not re-litigate them and do not ask about them. Ask the user
only if you are blocked on something not covered here.

## Execution order

- **Phase 0, rename (first commit, before any other change).** `git mv docs/lab-book.md
  docs/old-lab-book.md`. Then change every reference to the old file so it points to
  `old-lab-book.md`: docstrings and comments in `src/` and `tests/` ("See lab-book.md, ..."; ~25
  files), `README.md`, `TRIAGE.md`, `docs/implementation_doc.md`, `docs/capacity-scan-handoff.md`,
  `scan-list.md`, `WRITING_STYLE.md`/`CODING_STYLE.md`/`CLAUDE.md` if they point at it, relative
  Markdown links in both directions (links inside the renamed file, links to it from other docs),
  typst sources and notebooks. Leave the historical `red-team-*.md` reports untouched (they
  describe a past state). Section names quoted next to the file name ("Logging: MLflow") stay
  valid, since the content is unchanged. Do not create the new `docs/lab-book.md` in this phase.
  Verify: `rg "lab-book\.md"` outside the `red-team-*.md` files returns nothing (until Phase 3
  creates the new file), `rg "old-lab-book"` shows each updated reference, `pytest -q` and
  `ruff check` pass, and no markdown link to the old name is left dangling.
- **Phase 1,** `RunConfig` fields. **Phase 2,** `run_store` + tests. **Phase 3,**
  launcher, new lab book, MLflow split, `--workers` + tests. **Phase 4,** `plotting_utils` +
  `plot_size_scan` rewrite + tests. **Phase 5,** legacy import (not committed) and docs.
- **Phase 6, review cadence and final review.** Red-team (subagent, max 3 passes each) after
  Phase 2, after Phase 3, and over the whole branch diff (`git diff main...HEAD`) at the end. After
  the other phases run lint and tests only. At the end, check for bugs and critical downstream
  breakage introduced by *your own changes*; do not stop at "tests pass". At minimum:
  - Every consumer of something you removed or renamed: the old flat `results/*.json` readers and
    writers, `_write_results`, `plot_size_scan.py`, the MLflow JSON artifacts
    (`y_self_per_neuron_*.json`), `RunConfig` constructors and `dataclasses.asdict` users (new
    fields with defaults, MLflow param logging of `None`), `sweep`/`size_scan` name generation,
    commands in `scan-list.md`/`README.md`/handoff docs (run the documented commands' `--help`
    or dry paths; they must still be valid), `typst/` or notebooks that read results.
  - Name uniqueness and file-path assumptions: any code that globs result names, relies on string
    digit keys, or assumes the old `{name}.json` layout.
  - Failure modes: dirty-tree check false positives/negatives (lab-book-only edits, gitignored
    paths), two launches in the same minute, a crash between lab-book append and first run,
    interrupted/partial runs (leftover `.tmp` files), readers opening files while workers write.
  - Docs say what the code does now; no stale command, flag or path remains.
  Fix what you find, add a regression test for each real bug, and list every finding (fixed or
  not) in the final report.

## Why

Result files are currently not attributable to a run. `results/{name}.json` holds only
`accuracy_by_stage` (string digit keys); no config, timestamp, commit, `lr`, seed or class split.
Concretely (verified in `mlflow.db` and `results/`):

- Of 1146 MLflow runs, only 14 carry `mlflow.source.git.commit`. The other 1132 were launched from a
  throwaway worker script outside the repo, and MLflow derives git info from the launching script's
  path. MLflow also never records a dirty tree.
- `initial_classes` / `classes_per_stage` are not fields of `RunConfig`, so they are in neither the
  MLflow params nor any file name. A 1-class-start run and a 5-class-start run of the same
  config/dataset/seed 0 write the same file name.
- Single-seed runs are always seed 0 with no suffix, so reruns silently overwrite (TRIAGE #7). The
  14 legacy files (`lr=0.001`) share names with new-default runs (`lr=0.1`).
- Per-epoch loss and per-neuron `y_self` exist only inside MLflow (SQLite + JSON artifacts), not in
  `results/`.
- A previous MLflow failure in the final `log_text` call caused runs to train and then write zero
  results (see handoff doc).

Goals: (1) every result file is attributable to exactly one run, launch and code state; (2) one
module to write and read run data, in a format that will also hold large activation recordings
later; (3) one module to load and plot that data.

## Decisions already made

1. **Launch** = one invocation of the launcher (e.g. `run_experiments.py --scan size
   --num-seeds 5`). A launch has a `launch_id = YYYY-MM-DD-HH-mm_{git7}` (local time, 7-char commit
   hash). If that directory already exists, fail with a clear message (do not silently reuse or
   suffix).
2. **Refuse to run on a dirty tree.** Dirty = any tracked modification or untracked-but-not-ignored
   file (`git status --porcelain`), except the lab-book file itself (see 4), which is written by the
   launcher and tracked in git. No override flag. Error message must list the offending paths.
   Git provenance is read by our own code via `git` in `REPO_ROOT`, not via MLflow, so it works no
   matter where the launching script lives.
3. **Storage: directory per launch, one HDF5 file per run.**
   ```
   results/{launch_id}/{run_name}.h5
   ```
   `run_id = "{launch_id}/{run_name}"`. Keep the existing run-name scheme unchanged
   (`{config}[_{dataset}][_seed{N}]`): a launch has one dataset and one class split, and
   launches have separate directories, so names cannot collide across launches. Add
   `initial_classes` / `classes_per_stage` to `RunConfig` so they are recorded in `meta_json`,
   MLflow params and the lab-book entry's command line. Replace, don't deprecate: delete the old flat-JSON writer; nothing in the code base
   may write or read `results/*.json` in the old format afterwards.
4. **Lab book:** the existing prose `docs/lab-book.md` is renamed to `docs/old-lab-book.md`
   (see Phase 0 for how). Then a new tracked markdown document `docs/lab-book.md` is created: the
   run log, append-only, one entry per launch. From here on "the lab book" means this new file and
   "the old lab book" means `docs/old-lab-book.md`.
   Auto-filled per entry: `launch_id`, date/time, branch, full commit hash, command line, scan
   type, number of planned runs, dataset, seeds mode, `--workers`. User-supplied and **required**
   CLI flags (fail fast with a clear message if missing or empty): `--goal`, `--method`,
   `--code-changes`. These three are copied verbatim into the entry (the `--code-changes` text is
   how big source changes get recorded; do not auto-generate it).
5. **Run file content.** The whole run's config + provenance is stored *inside* the HDF5 file as one
   JSON string in a scalar string dataset `/meta_json` (a dataset, not an attribute: attributes
   have a 64 KB limit), so attribution can never be separated from the data. No sidecar files.
   `meta_json` contains: `run_id`, `launch_id`, all `RunConfig` fields, `git` (full commit, branch,
   `dirty: false`), `started_at` (ISO-8601 with timezone), `hostname`, `python`/`torch`/`numpy`/
   `h5py` versions, `device`, and `metrics` (final scalars: `final_avg_accuracy`,
   `final_backward_transfer`, `train_seconds`). Arrays are datasets; each dataset carries a `dims`
   attribute (list of axis names), e.g. `accuracy`: `("stage", "digit")`.
   Datasets written by training today: `accuracy` (n_stages x 10, float64, replaces the string-keyed
   dict), `train_loss` (per epoch, global epoch index), `y_self_absmean/{layer}` (per epoch),
   and, when the existing capture flags are on, `y_self_per_neuron_epoch/{layer}` (epochs x
   neurons) and `y_self_per_neuron_batch/{layer}` (batches x neurons), replacing the JSON artifacts.
   No compression and no chunking (not needed at current sizes; HDF5 decompresses transparently
   on read, so it can be enabled later without changing readers).
6. **Writes are atomic, never overwrite, and are independent of MLflow.** If `{run_name}.h5`
   already exists, raise. Otherwise write to `{run_name}.h5.tmp`, then `os.replace` (which
   overwrites by design, hence the explicit check; no race is possible because each run name is
   written by exactly one job per launch). Never open an `.h5` in `"w"` mode.
   The `.h5` is written *before* any end-of-run MLflow logging that can fail, so an MLflow error can
   never lose results. Many worker processes write different files in the same launch
   concurrently (see 11), so there must be no shared mutable file between runs besides the lab
   book, which is written once per launch by the parent process before workers start. Expose
   `begin_launch(...)` (creates the launch dir, checks the tree, appends the lab-book entry,
   returns a launch object that is cheap to pass to worker processes).
7. **MLflow stays, default on**, as a live-supervision option: add `--no-mlflow` (MLflow is then not
   imported or configured at all). When on, tag each MLflow run with `launch_id`, `run_id`, and
   git commit/branch from our own provenance (set the `mlflow.source.git.*` tags explicitly), and
   log `initial_classes` / `classes_per_stage` as params. MLflow is a viewer; the HDF5 files are
   the canonical record. Split `logged_training.py` accordingly: training/recording logic that
   always runs vs. the optional MLflow adapter.
8. **Legacy import (one-off, JSON files only).** Import the 574 data files in `results/` (14 flat
   sweep files dated 2026-08-25 + 560 `sizescan_*` files dated 2026-10-13; `results/plots/` is not
   data) into the new format under one launch directory `results/legacy-import/`. This is a
   throwaway script outside `src/` (e.g. the scratchpad); delete it after use, do not commit it,
   and do not add backward-compatibility code to `src/`. It trains nothing, so it is not a "run
   launcher" in the sense of decision 11 (it is the only exception to that rule). Rules:
   - Source is the `results/*.json` files only. Do **not** read `mlflow.db` or `mlartifacts/`, and
     never modify them or the original `results/*.json` (leave the originals; tell the user they
     can delete them after verifying).
   - Rebuild each file's config from its name with the current builders, as `plot_size_scan.py`
     does today: `size_scan.build_size_scan_configs()` for `sizescan_*` files and
     `sweep.build_configs()` for the 14 flat files; strip the `_fashion_mnist` / `_seed{N}`
     suffixes to get dataset and seed (no suffix = seed 0, mnist). A file whose name matches no
     builder config is skipped and reported, not guessed.
   - The 14 flat files (2026-08-25): `lr=0.001` (the default at the time, per the MLflow params of
     that day and `TRIAGE.md` #20), commit `b5b169174cd004feafab0149cf756321c2896a39` (from the
     MLflow tags of those runs). The 560 size-scan files (2026-10-13): `lr=0.1`, commit `null`
     (unknown; the scan ran on a dirty tree). Infer `initial_classes` from the number of stages
     (10 -> 1, 6 -> 5) with `classes_per_stage=1` assumed. `started_at` = file mtime.
   - Every imported file's `meta_json` gets `legacy: true` and an `inferred_fields` list naming
     everything that was inferred rather than recorded (including all of the above and the config
     itself). Convert string digit keys to the `accuracy` array. Compute `metrics` from it with the
     `metrics` module; `train_seconds` is `null`. No `train_loss` / `y_self_*` datasets (that
     data is MLflow-only and is not imported).
   - Report counts: imported, skipped (with names), and which fields were inferred.
   - Add one lab-book entry for the import (goal: "import pre-pipeline results"; method: describe
     the script; code changes: "n/a") that also states what was lost: per-epoch `train_loss` and
     `y_self` histories of these runs exist only in the gitignored `mlflow.db` / `mlartifacts/`,
     which are the only copy and must not be deleted; commit and `initial_classes` of legacy runs
     are inferred or unknown.
9. **Plotting:** `src/plotting_utils.py`, numpy + matplotlib only (no pandas, no seaborn). It
   replaces all loading/aggregation currently in `src/plot_size_scan.py`; reduce that file to a
   thin script on top of `plotting_utils` + `run_store` that reproduces the two existing figures and
   `scan_summary.csv`. Verify the new numbers equal the current `results/plots/scan_summary.csv`
   (to float tolerance) before replacing; report any difference.
10. Style defaults (from the user's earlier projects). Keep these two functions' behavior and
    defaults; do not make a theme out of anything else:
    ```python
    def plot_setup(fig_dpi=100, save_dpi=1200):
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.rcParams["figure.dpi"] = fig_dpi
        plt.rcParams["figure.figsize"] = (3, 1.5)
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["savefig.dpi"] = save_dpi
        plt.rcParams["savefig.transparent"] = True

    def get_colorcycle(N_colors, cmap="viridis"):
        colormap = plt.get_cmap(cmap)
        return [colormap(i) for i in np.linspace(1 / N_colors, 1 - 1 / N_colors, N_colors)]
    ```
    The small-figure/high-dpi approach is deliberate (it keeps text large relative to the figure).
    Every plotting function must still accept an optional `ax`/`figsize` so callers can override.
    The old hard-coded palette/styling in `plot_size_scan.py` is dropped, not carried over.

11. **`--workers N` and no scratch launchers.** `run_experiments.py` gets `--workers N` (default 1
    = sequential, in-process, behaviour as today; reject N < 1). All runs must be launchable through repo code only: the CLI, or repo
    functions called from `uv run python -c`/pytest. Do **not** create or use helper/driver scripts
    outside the repo (scratchpad, `/tmp`) to launch, fan out or retry runs, including your own
    verification runs. Design (a throwaway driver did this by hand for the 560-run scan, and
    these are the lessons from it, see `docs/capacity-scan-handoff.md`):
    - Unit of work = one config with all its seeds (keeps `replicate_group`'s MLflow parent/child
      nesting inside one process). Jobs are distributed with `concurrent.futures.ProcessPoolExecutor`
      using the `spawn` start method (torch/CUDA are not fork-safe).
    - Seeds are drawn once in the parent before dispatch and passed to the job, so a retry reruns the
      same seeds. Workers build their own datasets/stages from `(dataset, initial_classes,
      classes_per_stage)`; do not pickle datasets.
    - With `workers > 1`, each worker sets `torch.set_num_threads(1)` (otherwise N workers
      oversubscribe the cores). With MLflow on, SQLite writes from many processes hit "database is
      locked" unless the busy timeout is widened (the throwaway driver used 60 s): find the
      supported mechanism and set it in workers.
    - With `workers > 1`, jobs do not print per-stage accuracy matrices (they are in the `.h5`);
      the parent prints one progress line per finished job.
    - A job that raises is retried once, skipping seeds whose `.h5` already exists (atomic writes
      make an existing `.h5` a complete run). After all jobs finish, the parent lists every job that
      still failed with its exception and exits non-zero; completed runs are kept. Never swallow an
      exception silently.
    - The lab-book entry records `--workers`. Nothing in the launch may depend on `workers`
      for results: the same job gives the same numbers sequentially or in parallel (modulo
      nondeterminism that exists today).

## Module contracts

### `src/run_store.py` (write + read)

Public interface declared in one place at the top (`__all__`), per `CODING_STYLE.md`.

- Provenance: a function returning git commit/branch/dirty-paths for `REPO_ROOT` (injectable
  repo path, for tests with a temporary git repo), and a function that raises on dirty tree.
- Launch: `begin_launch(...)`, lab-book append (path injectable), `launch_id` construction.
- Write: create a run file from config + provenance + arrays + scalar metrics (atomic, see 6).
- Read: `open_run(path)` -> small object exposing parsed `meta` (dict) and lazy array access by
  dataset path (HDF5 partial reads, e.g. `run.array("y_self_per_neuron_batch/hidden_layers.0")`
  and slicing without loading everything); `list_runs(root_or_launch_dir)`.
- Design the layout so a future group `/activations/{layer}` with shape like
  (samples, neurons, ...) and a `dims` attr can be added without any format change. Expected scale
  there is ~1000x100x10x10 floats per run (~40 MB float32). **Do not implement activation
  recording** (no speculative features); only make sure nothing in the layout blocks it.
- Fail fast with actionable messages on: missing/corrupt `meta_json`, unknown dataset path,
  existing target file (never overwrite).

### `src/plotting_utils.py`

Keep the main pipeline simple: load -> filter/group -> stack over seeds -> plot.

- `plot_setup`, `get_colorcycle` as in 10.
- Loading/selection helpers, thin: load runs from a launch dir (or several), filter by config
  fields (e.g. `dataset="mnist"`, `model_kind="fcnn"`), group by config key(s), stack a named
  array over the group's seeds into an `ndarray` (seeds x ...), `mean_std` over axis 0. Keep these as
  plain functions on lists of run objects / numpy arrays; mean/std themselves are one-liners,
  so only add helpers that remove real repetition (grouping, stacking, aligning axes).
- Plot functions, each taking arrays + `ax` (create one if `None`) and returning the `Axes`:
  (a) line with mean +/- std band against a swept variable (needed for the size-scan figures,
  log-x option), (b) accuracy-matrix heatmap (stage x digit) for one run or the seed-mean,
  (c) per-epoch curves (loss, `y_self_absmean`) with optional seed band, (d) activity heatmap /
  raster for a (time x neuron) array such as `y_self_per_neuron_batch/*`.
- A save helper that respects `savefig.*` rcParams and creates the output directory.
- Tests use the `Agg` backend; no test may open a window.

## Required changes elsewhere

- `run_experiments.py`: new flags (`--goal`, `--method`, `--code-changes`, `--no-mlflow`, `--workers`,
  `--initial-classes`/`--classes-per-stage` already exist), launch creation, new run writer; drop
  `_write_results`' JSON dump (keep the console summary).
- `run_config.py`: add `initial_classes`, `classes_per_stage`.
- `pyproject.toml`: add `h5py` and declare `numpy` explicitly (now imported directly). Look up
  current stable versions; pin per the global supply-chain rules, matching the specifier style
  already used where a conflict exists. Verify both install and import under Python 3.14 with
  `uv`. (Checked beforehand: `uv pip install --dry-run --only-binary :all: h5py` resolves
  `h5py==3.16.0` as a wheel for this environment, so no fallback is expected; if the real install
  still fails, stop and report to the user before choosing an alternative.)
  Run `pip-audit`/equivalent if available.
- Docs: update `docs/implementation_doc.md` (logging/storage sections), the "Logging: MLflow"
  section of `docs/old-lab-book.md` (what moved where, why), `scan-list.md` (new flags and required
  launch text), `README.md` if it mentions results, `docs/capacity-scan-handoff.md` (mark the
  parts this supersedes), and `TRIAGE.md` #7 and #8 (overwrite problem solved by construction;
  state what remains unsolved: results are still gitignored, hence not reproducible from the
  repo alone, but now they are attributable to a commit).
- Remove dead code you created or left behind (old JSON writer, old loaders, unused imports).
  Flag, don't fix, unrelated problems (e.g. `ty check` pre-existing failures, TRIAGE #3, #11).

## Testing and checks

Behavior, not implementation; test edges and errors; mock only slow/non-deterministic/external
things. Results root and lab-book path must be injectable so no test or smoke run touches the real
`results/` or `docs/lab-book.md`. Cover at least:

- Provenance in a temporary git repo: clean, modified, untracked, ignored-file, lab-book-only change
  (must count as clean).
- Dirty tree refuses; message lists paths. Existing launch dir refuses. Missing/empty
  `--goal/--method/--code-changes` refuses before any training or file creation.
- Round trip: write then read gives identical config, metrics, arrays and `dims`; partial slice read
  works; atomic write leaves no `.tmp` on failure; existing target is never overwritten.
- MLflow failure after the `.h5` is written leaves the `.h5` intact; `--no-mlflow` never imports
  `mlflow` (assert via `sys.modules`).
- `--workers`: argument validation (N < 1); a launch with `workers=2` over tiny configs (tiny model,
  `epochs_per_stage=1`, small data subset, `--no-mlflow` and once with MLflow on a temporary
  tracking URI) produces the same set of run files as `workers=1`; a job that fails once is
  retried and succeeds; a job that always fails is reported at the end with a non-zero exit while
  the other runs' files are kept; seeds are identical on retry.
- `plotting_utils`: grouping/stacking with unequal seed counts, empty selection raises a clear
  error, each plot function returns an `Axes` and runs headless.
- Verify tests catch failures: break the code (e.g. remove the dirty check, drop `os.replace`),
  confirm the relevant test fails, restore.

Run `ruff check`, `ruff format`, `ty check` and `pytest -q` (relevant tests while iterating, full
suite once before the final commit). Zero new warnings. Hard limits from the global standards apply
(<=100 lines/function, complexity <=8, <=5 positional params, 100-char lines, absolute imports,
`CODING_STYLE.md` docs rules).

End-to-end check once the code is committed (a clean tree is required for launches): run a tiny
launch (1-2 configs, `epochs_per_stage=1`) against a temporary results root and lab book, through repo
code only (a pytest test or `uv run python -c` calling repo functions; no scratch scripts), with
MLflow on and with `--no-mlflow`, and with `--workers 1` and `--workers 2`. Open a produced file with
`plotting_utils` and render one of each plot type. Report what you saw, including failures.

## Commits

You may commit to the current branch, one logical change per commit, imperative subject <=72 chars:
order: (0) rename `lab-book.md` -> `old-lab-book.md` + reference updates (Phase 0), (1) `RunConfig` fields, (2) `run_store` + tests, (3) launcher/lab book/
MLflow split/`--workers` + tests, (4) `plotting_utils` + `plot_size_scan` rewrite + tests, (5) docs. The legacy
import runs after (3) and is not committed; its results live in the gitignored `results/`, and
the lab-book entry for it is committed with its own commit.

## Final report to the user

Plain and factual. State: what changed (files), what was verified and how, import counts
(imported / skipped / inferred fields), any difference between the old and new
size-scan numbers, anything that failed or was skipped, every Phase 6 finding (fixed or not, with the regression test
for each fixed one), and open issues found along the way.
