# Implementation documentation

## Data protocol

Class-incremental MNIST or FashionMNIST, cumulative (`data.py`):

- `get_mnist_data(root, dataset="mnist")`: `dataset` selects `"mnist"`
  (default) or `"fashion_mnist"` via the `_DATASETS` lookup (torchvision
  class + that dataset's own normalization mean/std) — everything downstream
  (models, training, metrics) is dataset-agnostic (both are 10-class,
  1x28x28), so this is the only place the choice needs to be made.
- Stage 0: an initial set of classes; stages 1..N: add the rest,
  `classes_per_stage` at a time (default: 5 initial, then 1 at a time,
  i.e. digits {0–4} then 5, 6, 7, 8, 9) — train set is the union of all
  classes introduced so far (not replaced) — per 07-06 notes ("one by one enables
  clearer results with regard to novelty"). `build_stage_digits(num_classes,
  initial_classes, classes_per_stage)` generalizes this split; its default
  output is `STAGE_DIGITS`. The remaining classes don't need to divide
  evenly by `classes_per_stage` — the last stage just takes whatever's left,
  so every class is always introduced exactly once.
- Fixed `epochs_per_stage` epochs per stage — "learnt" is epoch-delimited, not
  loss-delimited (07-14 pre-notes: cost-function conditions may over-restrict
  the loss landscape).
- Eval each stage on the full test set (all 10 classes); the resulting per-digit
  accuracy dict is sliced downstream into a forgetting read-out (already-introduced
  digits) and a generalization read-out (not-yet-introduced ones) — the
  "generalization" check from 07-06.

`build_stages(train_data, test_data, stage_digits=None)` accumulates
`digits_seen` across the loop and delegates each stage's actual construction
to `_build_stage(stage_idx, digits_seen, train_data, test_data)`; `None` (the
sweep's default) uses `STAGE_DIGITS`, else the given split.

## Core module: `autapse.py`

One class, `AutapticLayer`, covering 2 ai2 variants × 2 activation schemes,
with a `T`-step unroll per forward call. `variant` and `scheme` are `Literal` type
aliases exported from the module (`AI2_Variant`, `Activation_Scheme`); see the module's own
docstrings for the exact per-variant/scheme formulas — not duplicated here so this
doc can't drift out of sync with the code again.

The inhibition signal (`y_tilde_prev`, i.e. \tilde y_i(t-1) in old-lab-book.md
notation) starts at `0` regardless of scheme, so `T=1` degenerates to a plain
`Linear` + activation for both schemes and both variants. Each unroll step
(`_step`) delegates to `_get_main_output` (the uninhibited activated linear
output) and `_get_inhibited_output` (the `ai2_diff`/`ai2_gate` branching)
as two separate one-job helpers.

`forward` also stores `self.last_y_self` (the inhibition signal that fed the
final unroll step, detached) after every call, for `dynamics_insights.py` to
read — a plain tensor attribute, not a parameter/buffer, so it's invisible to
`state_dict()`/`.parameters()`/gradients.

## Models: `models.py`

- `FCNN_AI2`: 784-...-10 MLP whose hidden layers are `AutapticLayer`s
  sharing one `(T, ai2_variant, activation_scheme)`; output logits stay a plain
  `Linear`, resized to the last `hidden_sizes` entry. `hidden_sizes:
  tuple[int, ...] = (256, 128)` sets the width (and, since it's arbitrary
  length, the depth) of the hidden stack. `T=1` gives the FCNN controls
  (tests 2–3 in old-lab-book.md's "Implemented tests").
- `CNN`: control architecture, 2×(conv+ReLU+pool) + FC, no self-inhibition.
  `channels: tuple[int, int] = (16, 32)` sets conv1/conv2's output channel
  counts — the CNN control's equivalent of `FCNN_AI2`'s `hidden_sizes` for
  the shrinking-network scan (see `size_scan.py` below).

## Training loop: `train.py`

- `run_incremental_training(model, stages, device, *, epochs_per_stage=5, lr=1e-3,
  batch_size=128, on_epoch_end=None, on_stage_end=None)`: trains `model`
  through the stages in order (plain SGD, momentum 0, cross-entropy loss);
  returns one per-digit accuracy dict per stage (via
  `metrics.evaluate_per_digit_accuracy`), evaluated on the full test set right
  after that stage. `on_epoch_end(stage_index, epoch_index, avg_train_loss)`
  and `on_stage_end(stage_index, per_digit_accuracy)` are optional hooks for
  live-logging progress (see `logged_training.py`) — both default to `None`
  (no-op), so this function stays free of any logging-backend dependency.
- `train_one_epoch`: one forward/backward/step pass over a stage's
  `DataLoader`; returns the epoch's mean training loss, weighted by each
  batch's actual size (the last batch of a stage is generally partial).

## Run configuration: `run_config.py`

`RunConfig`: one sweep entry — everything worth logging about a run:
`name`, `sweep_axis` (tag, not a logged param — which dimension this config
varies relative to its sweep's anchor), `model_kind: Literal["cnn", "fcnn"]`
(explicit discriminator `sweep.build_model` switches on, not inferred from
`ai2_variant is None`), `dataset` (`"mnist"`/`"fashion_mnist"`, see
`data.get_mnist_data`), `initial_classes`/`classes_per_stage` (the class split, see
`data.build_stage_digits`; recorded so a 1-class-start run is never confused with a 5-class-start
run), `epochs_per_stage`, `seed`, `lr`, `batch_size`, plus
the FCNN-only `ai2_variant`, `activation_scheme`, `T`, `hidden_sizes`, and the
CNN-only `channels` (all `None` for the other `model_kind`). Lives in its own
module — `sweep.py` (to build configs), `recorded_training.py`, `logged_training.py` and
`experiment_jobs.py` all need the type, and `run_experiments.py` imports them; putting
`RunConfig` in any of those would risk a circular import. `config_from_dict` rebuilds one from a
run file's `meta["config"]` (JSON lists back to tuples), and `MlflowSettings` (tracking URI,
artifact location, experiment name) lives here so a worker's `Job` can carry it without importing
mlflow.

## Dynamics: `dynamics_insights.py`

`get_y_self_absmeans(model) -> dict[str, float]`: mean `|y_self|` of each
`AutapticLayer`'s last forward call (`autapse.py`'s `last_y_self`), keyed by
submodule name (e.g. `"hidden_layers.0"`). Skips layers with no forward pass
yet and models with no `AutapticLayer`s (e.g. `CNN`) — returns `{}` for
either, not an error, since both are legitimate call states.
`logged_training.py`'s `on_epoch_end` calls this every epoch to log
`y_self_absmean_{layer_name}`; see old-lab-book.md's wishlist (distributed vs.
sparse representations) for why this is worth tracking.

Not a full epoch-weighted average like `train_loss` — it's a snapshot from
the epoch's last batch. Averaging properly would need `train.py` to know
`AutapticLayer` exists, which its callback-based design (model stays a
closure variable in `logged_training.py`, never threaded through
`train.py`'s callback signature) deliberately avoids.

## Metrics: `metrics.py`

Standard CL measures, Avalanche-style (the project's own reference framework):

- Per-stage accuracy matrix (row = stage, col = digit)
- Average accuracy after each stage
- Backward transfer / forgetting: mean(acc at final stage − acc at stage 0),
  over the digits introduced at stage 0 (precondition, not checked: every digit
  passed in must have been introduced at stage 0)

## Experiment sweep: `sweep.py`

Pure config/model-factory definitions — no I/O, no MLflow, no torch training.
Split out of `run_experiments.py` once that file grew to mix "what
experiments exist" with "how to run and report them"; `run_experiments.py`
imports this module, not the other way around.

- `build_configs() -> list[RunConfig]`: 14 configs total, concatenated from 5
  one-job helpers (each returns its own `list[RunConfig]`, closing over the
  module constants, no params):
  - `_build_base_configs()`: 3 controls (`cnn_control`, `fcnn_control_relu`,
    `fcnn_control_sigmoid`, the latter two at `T=1`).
  - `_build_variant_scheme_grid()`: 2 `ai2_variant` × 2 `activation_scheme`
    at `T=3`, `epochs_per_stage=5`, `hidden_sizes=(256, 128)`
    (`BASE_T`/`BASE_EPOCHS_PER_STAGE`/`BASE_HIDDEN_SIZES`).
  - `_build_T_sweep()`, `_build_epochs_sweep()`, `_build_hidden_sizes_sweep()`:
    a one-factor-at-a-time sweep off the `ai2_diff`/`relu_sigmoid` base
    config as anchor (`ANCHOR_VARIANT`/`ANCHOR_SCHEME`) — `T` in
    `T_SWEEP = [2, 5, 8]`, `epochs_per_stage` in `EPOCHS_SWEEP = [2, 10]`,
    `hidden_sizes` in `HIDDEN_SIZES_SWEEP = [(64, 32), (512, 256)]`. `T=1` is
    skipped in the `T` sweep — degenerates to `fcnn_control_relu` regardless
    of variant/scheme, and `relu_sigmoid`+`ai2_diff`+`T=1` is already that
    exact config (documented on `_build_T_sweep`, not `build_configs`, since
    it's specific to that one sweep). See old-lab-book.md, "Logging: MLflow",
    for why one-factor-at-a-time over a full 144-combo grid.
- `build_model(config: RunConfig) -> nn.Module`: `CNN(channels=config.channels)`
  (or `CNN()` if `channels` is unset) for `model_kind="cnn"`; else asserts
  the four FCNN-only fields are set (fail fast on a malformed config —
  `build_configs` always sets all four together, so this guards against a
  `RunConfig` built some other way, not an expected input) and builds
  `FCNN_AI2`.

## Shrinking-network scan: `size_scan.py`

Same pure config/model-factory shape as `sweep.py`, sibling module (not a
`sweep.py` function) so `sweep.build_configs()`'s own base tests stay exactly
what they were. See old-lab-book.md, "ToDos 2026-08-26 or so".

- `FCNN_HIDDEN_SIZES_SCAN`/`CNN_CHANNELS_SCAN`: 8 progressively smaller
  width/channel-count steps for the FCNN and CNN families respectively (same
  length, validated in `build_size_scan_configs`; not meant to match in
  absolute numbers — `CNN`'s default `(16, 32)` is already narrow relative
  to `FCNN_AI2`'s default `(256, 128)`).
- `build_size_scan_configs(hidden_sizes_scan, channels_scan) -> list[RunConfig]`:
  per step, emits the CNN control at that step's channels, the 2 FCNN
  controls (`T=1`) at that step's hidden_sizes, and the `ai2_variant` x
  `activation_scheme` grid (`T=sweep.BASE_T`) at that step's hidden_sizes —
  i.e. every family in `sweep.build_configs()`'s base tests, just shrinking.
  Names are prefixed `sizescan_step{i}_` so they can never collide with
  `sweep.build_configs()`'s own names (run names are `.h5` file stems and MLflow run names —
  see `test_sweep.py`'s own unique-names test for why a collision matters).

## Run provenance and storage: `run_store.py`

One module to write and read run data; torch- and mlflow-free (it imports h5py and numpy only),
so plotting code can import it cheaply. Public interface in `__all__`.

- **Launch** = one invocation of the launcher. `begin_launch(spec, results_root, lab_book,
  repo_root)` validates the required text (`--goal`/`--method`/`--code-changes`, non-blank),
  refuses a dirty git tree (`require_clean_tree`: tracked changes, staged or not, and
  untracked-but-not-ignored files; the lab book is the only exemption, since every launch
  rewrites it), builds `launch_id = YYYY-MM-DD-HH-mm_{git7}` (local time), creates
  `results/{launch_id}/` with `mkdir(exist_ok=False)` (a second launch in the same minute on
  the same commit fails with `LaunchExistsError`; it is never suffixed or merged) and appends
  one entry to the lab book. Everything that can refuse happens before anything is created; if
  the lab-book append fails, the still-empty directory is removed again.
- Git provenance is read by our own code (`get_git_info`, `get_dirty_paths`, all via `git` in
  `repo_root`), not by MLflow, which derives it from the launching script's path and never
  records a dirty tree. `Launch.verify_unchanged()` re-runs the check at the start of every
  job (workers import code from the working tree minutes after launch): committing only the
  lab book is allowed, any other edit or HEAD movement fails the job. Files marked
  `assume-unchanged`/`skip-worktree` and gitignored source files are invisible to it.
- **Lab book** (`docs/lab-book.md`, tracked, append-only): one markdown entry per launch with
  the auto-filled fields (launch id, start time with timezone, branch, full commit, command
  line, scan, planned runs, dataset, seeds mode, class split, workers, MLflow on/off) and the
  user's three texts verbatim. `append_lab_book_entry` is public (the one-off legacy import
  wrote its entry through it).
- **Run file**: `results/{launch_id}/{run_name}.h5`, `run_id = "{launch_id}/{run_name}"`. The
  scalar string dataset `/meta_json` (a dataset, not an attribute: attributes are limited to
  64 KB) holds `run_id`, `run_name`, `launch_id`, `config_name` (the config's name before the
  dataset/seed suffixes, so replicates group without parsing names), `config` (all `RunConfig`
  fields), `git` (`commit`, `branch`, `dirty: false`), `started_at` (ISO-8601 with timezone),
  `hostname`, `python`/`torch`/`numpy`/`h5py` versions, `device`, `torch_num_threads` and `metrics`
  (`final_avg_accuracy`, `final_backward_transfer`, `train_seconds`). Arrays are datasets with a
  `dims` attribute naming their axes; paths are free-form (`y_self_absmean/hidden_layers.0`),
  so a future `/activations/{layer}` group needs no format change (not implemented). No
  compression or chunking; HDF5 decompresses transparently, so it can be switched on later.
- Datasets written by training: `accuracy` (stage, digit; float64, NaN for a digit a stage did
  not evaluate), `train_loss` (epoch, indexed by global epoch), `y_self_absmean/{layer}`
  (epoch) and, with `RunConfig.capture_y_self_per_epoch`/`_per_batch`,
  `y_self_per_neuron_epoch/{layer}` (epoch, neuron) and `y_self_per_neuron_batch/{layer}`
  (batch, neuron). The CNN has no `AutapticLayer`, hence no `y_self_*` datasets.
- **Writes** (`write_run_file`, wrapped by `write_run`) are atomic and never overwrite:
  raise `RunExistsError` if the file exists, else write `{name}.h5.tmp` (mode `"x"`, never
  `"w"`; a stale `.tmp` from a crash is partial by construction and removed first), fsync,
  `os.replace`. A file under its final name is always complete, so readers (and `list_runs`,
  which skips `.tmp`) never see a partial run, and workers need no shared mutable file. Array
  dtypes h5py cannot store are rejected when the `RunArray` is built, i.e. before training
  results exist only in memory, not after.
- **Reads**: `open_run(path) -> RunFile` parses and validates `meta` and closes the file; no
  handle stays open (plotting 560 runs cannot exhaust file descriptors). `run.array(path)`
  returns an `ArrayRef` (`shape`, `dtype`, `dims`; `ref[2:4, 3]` reads just that slice, `read()`
  everything; every access reopens the file). `list_runs(path)` takes a launch directory or a
  results root. Missing/corrupt `meta_json`, unknown dataset path and non-HDF5 files raise
  `RunFileError` with the path and what is available.
- Two writers of the same run name are not supported (one job per name per launch); the stale
  `.tmp` cleanup would let them clobber each other.

## Training recorder: `recorded_training.py`

`train_and_record(config, model, stages, device, hooks=None) -> TrainingRecord` wraps
`train.run_incremental_training` and always records accuracy per stage, loss and y_self absmean
per epoch, and the optional per-neuron histories; `build_arrays(record)` turns that into the
`RunArray`s above. `TrainingHooks` are optional extra callbacks (the MLflow viewer's), called
after the recorder's own. Nothing here imports mlflow. The record's mutating closures follow the
same documented exception to the mutation-avoidance rule as before: the callback-based training
loop is the only place the values become available.

## Experiment runner: `run_experiments.py`, `experiment_jobs.py`

`run_experiments.py` is the launcher (CLI, planning, dispatch, reporting); `experiment_jobs.py`
is what runs inside a worker. `sweep.py`/`size_scan.py` own "what to run."

- `main(*, goal, method, code_changes, num_seeds=1, dataset="mnist", initial_classes=5,
  classes_per_stage=1, workers=1, use_mlflow=True, command=None, env=None)` runs
  `sweep.build_configs()`; `run_size_scan(...)` the same plus `hidden_sizes_scan`/
  `channels_scan` for `size_scan.build_size_scan_configs()`. Both call
  `run_launch(configs, request, scan, env)`, which is also the entry point for an injected
  config list. CLI: `--scan {sweep,size}`, `--num-seeds`, `--dataset`, `--initial-classes`,
  `--classes-per-stage`, `--workers`, `--no-mlflow`, and the required `--goal`, `--method`,
  `--code-changes`. `LaunchEnv` holds results root, lab-book path, repo root, MLflow settings
  and the stage builder (defaults: the repo's; tests inject temporary ones).
- `run_launch` order: reject `workers < 1` -> `plan_jobs` (draw each config's seeds once, in the
  parent, via `_draw_seeds`: `[0]` for `num_seeds == 1`, else `num_seeds` distinct random
  seeds; reject duplicate run names) -> validate the lab-book text -> clean-tree check -> build
  the stages once in the parent (fails fast on a bad class split and pre-downloads the dataset,
  so workers never race on the download) -> with MLflow on, `logged_training.configure` once
  (first use of a fresh database runs schema migrations, which fail when several workers race
  to do it) -> `begin_launch` -> `execute_jobs`. Nothing is created before the last-but-one
  step.
- A **job** (`experiment_jobs.Job`, frozen and picklable; everything a spawned worker needs
  travels inside it) is one config with all its seeds, so MLflow's parent/child nesting stays in
  one process. `run_job` never raises: it returns a `JobOutcome` with the traceback text and any
  swallowed MLflow errors. Per seed (`run_seed`): build the seed variant (`build_seed_variant`:
  name suffixed with the dataset when not `"mnist"` and with `_seed{N}` when replicating),
  check the target does not exist, `torch.manual_seed`, build the model, train and record,
  **write the `.h5`**, then do the optional end-of-run MLflow logging. Workers build their own
  stages from `(dataset, initial_classes, classes_per_stage)` (`build_stages_for`, cached per
  process).
- `execute_jobs(jobs, workers)`: `workers == 1` runs in-process; otherwise a
  `ProcessPoolExecutor` with the `spawn` start method (torch/CUDA are not fork-safe) whose
  workers set `torch.set_num_threads(1)`. A job that fails is retried once, on a fresh pool,
  skipping seeds whose `.h5` exists (atomic writes make an existing file a complete run; on a
  first attempt an existing file is an error). A dead worker (`BrokenProcessPool`) fails every
  unfinished job of that round, so the retry round runs each job in its own single-worker pool
  (`workers` at a time): only a job that really kills its process fails twice. The same seeds are used on retry.
  With `workers > 1` jobs print nothing per stage; the parent prints one progress line per
  finished job. Results should not depend on `workers`, with one known exception: workers run with one torch
  thread (so N workers do not oversubscribe the cores) while an in-process run uses all of them, and
  single-thread reductions sum in a different order. On real MNIST runs this changed final accuracy
  by up to ~0.8 pp (seed 0, same code); with `torch.set_num_threads(1)` in-process the numbers were
  identical. Every run file records `torch_num_threads`, so the two kinds of launch can be told apart;
  compare parallel launches with parallel ones. (The brief demands both the single thread and
  worker-independence; this is the conflict between them.)
- After all jobs: `LaunchError` lists every job that still failed (with its traceback) and every
  swallowed MLflow error; the CLI prints it and exits 1. Exit 2 = refused before starting
  (bad arguments, dirty tree, existing launch directory). Completed runs are always kept.
- `REPO_ROOT` anchoring: `RESULTS_DIR`, `LAB_BOOK` and the data directory are anchored to the
  repo root, so no invocation location gets one right and another wrong.

## Logging: `logged_training.py` (optional MLflow viewer)

MLflow is a viewer for live supervision; the `.h5` files are the canonical record. This module
is imported only when MLflow is on (`--no-mlflow` never imports or configures it; a test checks
`sys.modules` in the parent and in spawned workers). Manual logging, not autolog (autolog only
covers PyTorch Lightning, not this repo's custom training loop; see old-lab-book.md,
"Logging: MLflow").

- **Backend**: `sqlite:///{repo_root}/mlflow.db`, artifacts under `{repo_root}/mlartifacts`, set
  via `MlflowSettings` (`run_config.py`). The experiment `"ai2-mnist"` is created by
  `configure(settings)` with the explicit artifact location the one time it does not exist (a
  new experiment's default artifact location is cwd-relative). View runs with `uv run mlflow ui
  --backend-store-uri sqlite:///mlflow.db`.
- `MlflowSession(settings, launch)` is one job's MLflow state: `begin_group` (parent run for
  seed replicates) / `end_group` (mean/stdev of the replicates' final metrics), `begin_run`
  (child or flat run; returns the live `TrainingHooks` that log `train_loss`,
  `y_self_absmean_{layer}`, per-digit accuracy, average accuracy and a running backward
  transfer), `log_final` (`train_seconds`, final accuracy/BWT, the accuracy-matrix text
  artifact), `cleanup`. Runs are tagged `launch_id`, `run_id`, `sweep_axis` and
  `mlflow.source.git.commit`/`.branch` from our own provenance; all `RunConfig` fields with a
  value (including `initial_classes`/`classes_per_stage`) are params.
- **MLflow can never lose results.** Every logging call is guarded: the first exception prints a
  warning to stderr, disables further MLflow calls for the rest of the job, and is recorded in
  `session.errors`; training and the `.h5` write continue. Teardown is never skipped: runs end
  innermost first, and `cleanup` drains whatever is still open (status FAILED after a crash) so
  a reused worker process cannot leak an active run into its next job.
- **SQLite and many writers**: MLflow 3.15 hard-codes `PRAGMA busy_timeout = 20000` at the start
  of every session (no setting), so only the connection-level timeout can be widened:
  with `workers > 1` the tracking URI gets `?timeout=60` (`with_busy_timeout`). This does not
  make concurrent writers contention-free.
- The per-neuron y_self JSON artifacts (`y_self_per_neuron_*.json`) and `replicate_group` /
  `run_and_log` are gone: the data is in the `.h5`, the grouping in `MlflowSession`.

## Plotting: `plotting_utils.py`, `plot_size_scan.py`

`plotting_utils.py` (numpy + matplotlib only) keeps the pipeline plain: `load_runs` ->
`select_runs`/`group_runs` (config fields first, then top-level meta such as `config_name`) ->
`stack_arrays` (seeds x ...; ragged shapes and empty groups raise with the offending runs) ->
`mean_std` (population std, ddof=0) -> `plot_line_band`, `plot_epoch_curves`,
`plot_accuracy_heatmap`, `plot_activity_heatmap` (each takes arrays and an optional `ax`, returns
the `Axes`) -> `save_figure` (creates the directory, honours `savefig.*` rcParams).
`plot_setup`/`get_colorcycle` set the house style (small figures, high save dpi, transparent).
`plot_size_scan.py` is a thin script on top: size-scan runs with `initial_classes == 1`, grouped
per config and dataset, written as `scan_summary.csv` plus the two figures. It pools seeds
across every launch passed to it. After the legacy import its numbers equal the pre-pipeline
`scan_summary.csv` to float precision (max abs difference 1.1e-16).

## Files

```sh
├── STYLE.md
├── src
│   ├── autapse.py
│   ├── data.py
│   ├── dynamics_insights.py
│   ├── experiment_jobs.py
│   ├── logged_training.py
│   ├── metrics.py
│   ├── models.py
│   ├── plot_size_scan.py
│   ├── plotting_utils.py
│   ├── recorded_training.py
│   ├── run_config.py
│   ├── run_experiments.py
│   ├── run_store.py
│   ├── size_scan.py
│   ├── sweep.py
│   └── train.py
├── tests
│   ├── test_autapse.py          # degeneracy, grad-flow, boundedness checks
│   ├── test_data.py
│   ├── conftest.py              # throwaway git repo + launch fixtures, Agg backend
│   ├── fakes.py                 # tiny datasets/configs, failure-injecting stage builders
│   ├── test_dynamics_insights.py
│   ├── test_experiment_jobs.py
│   ├── test_logged_training.py  # against a real local sqlite backend
│   ├── test_models.py
│   ├── test_plot_size_scan.py
│   ├── test_plotting_utils.py
│   ├── test_recorded_training.py
│   ├── test_run_experiments.py  # includes spawned-worker launches
│   ├── test_run_store.py
│   ├── test_size_scan.py
│   ├── test_sweep.py
│   └── test_train.py
```

Coding conventions for `src/`/`tests/` are in `STYLE.md`, not repeated here.

Flat modules under `src/` -- i.e., this is no installable package.
Sibling imports between them (e.g. `from data import Stage`) resolve because
`pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["src"]` for
tests, and running a script directly (`uv run python src/run_experiments.py`)
auto-adds its own directory to `sys.path`. `tests/` holds only actual test
modules, not implementation code.

## Verification

- `tests/test_autapse.py`: assert `AutapticLayer(..., T=1)` output equals a plain
  `sigmoid(Linear(x))` / `relu(Linear(x))` for both variants × both schemes (degeneracy
  check); assert gradients flow through `T>1` unrolls (no detached state); assert
  `0 <= y <= y_raw` for every variant/scheme combo except `ai2_diff` + `sigmoid`
  (the one combo with no such boundedness guarantee, see the module's docstring).
- `tests/test_data.py`: assert stage `k`'s train set is the superset of stage
  `k-1`'s labels plus exactly one new label; assert test set always contains all 10
  digits; `build_stage_digits` defaults reproduce `STAGE_DIGITS`, cover every
  class exactly once for a range of `initial_classes`/`classes_per_stage`
  combos (including an unevenly-dividing remainder and `initial_classes=1`),
  and reject out-of-range/non-positive args; `get_mnist_data`'s `dataset`
  param picks the right class per name (monkeypatched dataset classes, no
  real download).
- `tests/test_models.py`: `FCNN_AI2` with non-default (including 3-element,
  i.e. depth-varying) `hidden_sizes` produces correctly-shaped output and the
  right number/width of hidden layers; `CNN`'s `channels` param likewise
  produces the right conv channel counts and output shape.
- `tests/test_train.py`: `run_incremental_training` invokes `on_epoch_end`
  exactly `len(stages) * epochs_per_stage` times and `on_stage_end` exactly
  `len(stages)` times, on a tiny fake dataset (no real MNIST, no MLflow); and
  runs fine with both callbacks left at their default `None`.
- `tests/test_dynamics_insights.py`: `get_y_self_absmeans` is `0.0` for
  every layer at `T=1` (degenerate case); nonzero for at least one layer at
  `T>1` with a fixed seed (deterministic, not "usually nonzero"); `{}` for a
  `CNN` and for a model with no forward pass yet.
- `tests/test_sweep.py`: all 14 `build_configs()` names are unique (a
  duplicate would make two runs target the same `.h5` and confuse MLflow's run-name-based
  comparison); every config's `build_model` succeeds
  without raising; CNN/FCNN dispatch is correct, including that the built
  `FCNN_AI2`'s `AutapticLayer`s actually carry the config's `T`/`ai2_variant`/
  `activation_scheme` (not just the right shape — this specific assertion
  was added after mutation testing showed a shape-only check would miss a
  hyperparameter silently getting dropped or swapped).
- `tests/test_run_store.py`: provenance in a throwaway git repo (clean, modified, staged,
  untracked, ignored, lab-book-only, rename, subdirectory root, odd file names); dirty tree,
  blank text and an existing launch directory refuse before anything is created and name the
  offending paths; lab-book append-only behavior; `verify_unchanged` accepts a lab-book-only
  commit and rejects source edits/commits; write/read round trip (meta, arrays, `dims`,
  slices); never overwriting, no `.tmp` after failed writes, stale `.tmp`; corrupt and missing
  `meta_json`, unknown dataset path. Mutation-checked: removing the dirty check, `os.replace`,
  the exists check, the lab-book exemption, `exist_ok=False`, the tmp cleanup or the HEAD check
  each fails a test.
- `tests/test_recorded_training.py`: array shapes/dims/dtypes, per-neuron flags, CNN without
  y_self, NaN for unevaluated digits, hooks.
- `tests/test_logged_training.py`: the MLflow session against a **real** local sqlite backend
  in `tmp_path` (not mocked -- fast and deterministic, so not the case "mock boundaries, not
  logic" is about; failures are injected by patching `mlflow` calls, the external boundary).
  Params/tags (`launch_id`, `run_id`, git commit/branch overriding MLflow's) and metrics; y_self
  metrics for FCNN only; replicate aggregation; an MLflow error mid-training, at start, at the
  end or in teardown is recorded, not raised; `cleanup` ends nested runs innermost first as
  FAILED and a crashed job leaks no active run into the next.
- `tests/test_experiment_jobs.py`: one job in-process: attributable file content, replicates
  and their names, existing target never trained over or overwritten, retry skipping finished
  seeds (also with MLflow aggregation), MLflow failure after training leaves the `.h5` intact,
  source edit after launch fails the job while a lab-book-only commit does not.
- `tests/test_run_experiments.py`: planning (seeds drawn once, unique names), refusals before
  any side effect (blank text, `workers < 1`, dirty tree, bad class split -- the dirty case
  also checks MLflow was not configured), launches with `workers=1` and `workers=2` producing
  the same files, MLflow on a fresh database with 2 workers, retry of a job that failed once
  (also after writing some seeds), a job that always fails reported with the others' files kept,
  a worker that dies, MLflow errors reported with results kept, same-minute launch refusal,
  `--no-mlflow` never importing mlflow in the parent or in spawned workers, CLI parsing and exit
  codes. Spawned workers cannot see the test process's monkeypatches, so failures are injected
  by picklable stage builders in `tests/fakes.py` steered by environment variables.
- `tests/test_plotting_utils.py`, `tests/test_plot_size_scan.py`: loading, filtering, grouping
  with unequal seed counts, ragged stacks, empty selection, each plot function headless
  (`Agg`), `savefig.*` rcParams respected; scan aggregation, CSV and figures from synthetic runs.
- `tests/test_size_scan.py`: mirrors `test_sweep.py`'s checks (unique names,
  every config's `build_model` succeeds without raising) plus: size-scan
  names never collide with `sweep.build_configs()`'s own names; hidden
  sizes/channels actually shrink step over step; every step includes both
  the ai2 grid and the FCNN/CNN controls; built models' shapes match their
  config's `hidden_sizes`/`channels`; mismatched-length
  `hidden_sizes_scan`/`channels_scan` raises.
- `uv run pytest -q` for all of the above.
- `ruff check` / `ruff format --check` / `ty check` clean.
- Smoke-run `run_experiments.py` (a launch needs a clean git tree) for 2 epochs/stages on a
  few configs, confirm no NaNs/crashes, that the `.h5` files open with `plotting_utils`, and
  (with MLflow on) that the runs show up (`mlflow ui --backend-store-uri
  sqlite:///mlflow.db`) before committing to full-length runs. Same for `--scan size` and
  `--dataset fashion_mnist`, and for `--workers 2`.
