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

The inhibition signal (`y_tilde_prev`, i.e. \tilde y_i(t-1) in lab-book.md
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
  (tests 2–3 in lab-book.md's "Implemented tests").
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
`data.get_mnist_data`), `epochs_per_stage`, `seed`, `lr`, `batch_size`, plus
the FCNN-only `ai2_variant`, `activation_scheme`, `T`, `hidden_sizes`, and the
CNN-only `channels` (all `None` for the other `model_kind`). Lives in its own
module — both `sweep.py` (to build configs) and `logged_training.py` (to type
`run_and_log`'s parameter) need the type, and `run_experiments.py` also
imports `logged_training`; putting `RunConfig` in either of those two would
create a circular import.

## Dynamics: `dynamics_insights.py`

`get_y_self_absmeans(model) -> dict[str, float]`: mean `|y_self|` of each
`AutapticLayer`'s last forward call (`autapse.py`'s `last_y_self`), keyed by
submodule name (e.g. `"hidden_layers.0"`). Skips layers with no forward pass
yet and models with no `AutapticLayer`s (e.g. `CNN`) — returns `{}` for
either, not an error, since both are legitimate call states.
`logged_training.py`'s `on_epoch_end` calls this every epoch to log
`y_self_absmean_{layer_name}`; see lab-book.md's wishlist (distributed vs.
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
    it's specific to that one sweep). See lab-book.md, "Logging: MLflow",
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
what they were. See lab-book.md, "ToDos 2026-08-26 or so".

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
  `sweep.build_configs()`'s own names (both write to the same flat
  `results/{name}.json` namespace and MLflow's run-name-based comparison —
  see `test_sweep.py`'s own unique-names test for why that collision matters).

## Experiment runner: `run_experiments.py`

Orchestration/IO only: seeding, training, MLflow logging, result reporting —
`sweep.py`/`size_scan.py` own "what to run."

- `main(num_seeds=NUM_SEEDS, dataset="mnist", initial_classes=5,
  classes_per_stage=1)`: runs `sweep.build_configs()`'s 14 configs.
  `run_size_scan(num_seeds=1, dataset="mnist", initial_classes=5,
  classes_per_stage=1, hidden_sizes_scan=None, channels_scan=None)`: same
  shape, running `size_scan.build_size_scan_configs()`'s scan instead. Both
  are also reachable from the command line via `--num-seeds`/`--dataset`/
  `--initial-classes`/`--classes-per-stage`, plus `--scan {sweep,size}` to
  choose between them (see `_parse_args`).
- `NUM_SEEDS = 1`: the default seed-replicate count. `1` reproduces a plain
  single-seed run exactly — deterministic seed `0`, no MLflow nesting, no
  aggregation, run names unchanged. `>1` draws that many distinct *random*
  seeds (`_draw_seeds`, recorded via the `seed` MLflow param and each
  replicate's results filename) and runs each once, grouped under one
  MLflow parent run per config via `logged_training.replicate_group` (see
  below) — e.g. `num_seeds=3` turns 14 configs into 42 training runs.
- `REPO_ROOT = Path(__file__).resolve().parent.parent`: same anchoring
  pattern as `logged_training.py`'s `REPO_ROOT`, for the same reason —
  `RESULTS_DIR = REPO_ROOT / "results"` (per-run JSON accuracy dumps) and
  `get_mnist_data(root=str(REPO_ROOT / "data"))` (in `_prepare_run`) both
  used to be cwd-relative (`Path("../results")`, `root="data"`) and, since
  one assumed launching from `src/` and the other from the repo root, no
  single invocation location got both right — anchoring both fixes it
  regardless of cwd.
- `_prepare_run(dataset, initial_classes, classes_per_stage)`: shared setup
  for `main`/`run_size_scan` — picks the device, downloads/loads `dataset`,
  builds `stages` off `data.build_stage_digits(initial_classes,
  classes_per_stage)`, creates `RESULTS_DIR`.
- `_run_configs(configs, stages, device, num_seeds, dataset)`: loops
  `_run_config` once per entry in `configs` (either sweep's or the size
  scan's).
- `_run_config(config, stages, device, num_seeds, dataset)`: draws this
  config's seeds via `_draw_seeds`, prints them (for reproducibility even
  without digging into MLflow/results), picks a replicate context via
  `_get_replicate_context`, then loops the drawn seeds calling `_run_seed`
  and appending its result to the collector if replicating.
- `_draw_seeds(num_seeds) -> list[int]`: `[0]` if `num_seeds == 1`
  (unchanged, deterministic default), else `num_seeds` distinct seeds drawn
  via `random.SystemRandom().randrange` — random *only* once multi-seed
  replication is opted into, so the default single-seed sweep's
  reproducibility (lab-book.md: "every run sets a fixed seed... reproducible
  run-to-run") isn't silently lost.
- `_get_replicate_context(config, num_seeds) -> AbstractContextManager[list[tuple[float, float]] | None]`:
  `logged_training.replicate_group(config, num_seeds)` if `num_seeds > 1`,
  else `contextlib.nullcontext()`.
- `_run_seed(config, seed, stages, device, num_seeds, dataset) -> tuple[float, float]`:
  one seed's full pipeline — seeds `torch.manual_seed` *before*
  `sweep.build_model` (covers both weight init and the training loop's
  `DataLoader` shuffling, since nothing else consumes RNG state in between),
  calls `logged_training.run_and_log`, computes `avg_accuracy`/
  `backward_transfer` from the returned per-stage accuracy, writes results,
  and returns the two metrics for the caller to collect if replicating.
- `_build_seed_variant(config, seed, num_seeds, dataset)`: per-seed,
  per-dataset copy of `config` via `dataclasses.replace` — sets
  `seed`/`dataset`, and suffixes `name` with the dataset (only when it's not
  `"mnist"`) and `_seed{seed}` (only when `num_seeds > 1`), so default runs
  keep exactly the names they had before either feature existed, while
  non-default runs never collide with the default's results file/MLflow run
  name.
- `_write_results(name, accuracy_by_stage, avg_acc, bwt)`: dumps
  `RESULTS_DIR/{name}.json` and prints the summary line + accuracy matrix.

## Logging: `logged_training.py`

MLflow adapter, manual (not autolog — autolog only covers PyTorch Lightning,
not this repo's custom training loop; see lab-book.md, "Logging: MLflow").

- **Tracking backend**: `sqlite:///{repo_root}/mlflow.db`. MLflow 3.15 put
  the plain `file:./mlruns` backend into maintenance mode (raises by
  default), so this uses the sqlite backend MLflow itself recommends
  instead. View runs with `uv run mlflow ui --backend-store-uri
  sqlite:///mlflow.db` from the repo root, then open `localhost:5000`.
- **Artifact location**: `file:{repo_root}/mlartifacts`, set explicitly via
  `mlflow.create_experiment(..., artifact_location=...)` in
  `_configure_mlflow()` the one time the `"ai2-mnist"` experiment doesn't
  yet exist — a *new* experiment's default artifact location is cwd-relative
  (`./mlruns`) independent of the tracking URI, so leaving it unset would
  reintroduce the same cwd-dependence the sqlite tracking URI was anchored
  to avoid.
- `run_and_log(config, model, stages, device) -> list[dict[int, float]]`:
  wraps `train.run_incremental_training` in `mlflow.start_run(run_name=
  config.name, nested=mlflow.active_run() is not None)` — flat top-level run
  when called standalone, nested child run when called from inside
  `replicate_group`. Calls `_configure_mlflow()` (see above), then delegates
  to 3 one-job helpers: `_get_loggable_params` (config's fields as params,
  skipping `None`s and `name`/`sweep_axis` — the latter logged as a tag
  instead); `_train_with_logging` (owns
  `accuracy_by_stage`'s full lifecycle, runs training with `on_epoch_end`/
  `on_stage_end` callbacks built by `_make_on_epoch_end`/`_make_on_stage_end`,
  returns the accuracy list plus wall-clock seconds); `_log_final_metrics`
  (`train_seconds`, final accuracy/BWT, the accuracy-matrix text artifact).
  `_make_on_epoch_end` logs `train_loss` and `y_self_absmean_{layer_name}`
  per epoch (`dynamics_insights.py`). `_make_on_stage_end` logs per-digit
  accuracy, average accuracy (seen digits and all digits), and a running
  `backward_transfer` per stage (`0.0` at stage 0 by construction — the
  curve's fixed starting point, not a measurement) — its returned closure
  mutates the caller-owned `accuracy_by_stage` list in place, a documented
  exception to STYLE's mutation-avoidance rule (independent of, not a copy
  of, `replicate_group`'s own exception below — the shapes differ: here a
  factory hands back a closure that mutates its caller's list, there the
  caller declares and appends to its own list). Does not call
  `mlflow.pytorch.log_model` — with `num_seeds > 1`, dozens of runs ×
  checkpoints is needless disk churn for an exploratory sweep.
- `replicate_group(config, num_seeds) -> Iterator[list[tuple[float, float]]]`:
  context manager grouping `num_seeds` seed replicates under one parent
  MLflow run. `_check_num_seeds` fails fast (raises `ValueError`) before any
  MLflow state is touched if `num_seeds < 2`. Yields a
  `list[(avg_accuracy, backward_transfer)]` accumulator the caller appends
  to across its own seed loop (an explicit exception to preferring return
  values over mutation, STYLE.md — there's no return-based shape that
  doesn't restructure the caller's loop just to avoid one `.append()`). On
  clean exit, `_log_replicate_aggregates` asserts the collected count
  matches `num_seeds` (`statistics.stdev` needs ≥2 points, and grouping a
  single replicate isn't meaningful anyway, hence `_check_num_seeds`) and
  logs `mean_final_avg_accuracy`/`std_final_avg_accuracy`/
  `mean_final_backward_transfer`/`std_final_backward_transfer` onto the
  parent run. If the `with` block raises partway through (e.g. one
  replicate's training fails), the exception propagates before this
  function's post-`yield` aggregation code runs at all (no `try`/`finally`
  around `yield`, deliberately) — no misleading partial aggregate ever gets
  logged, and MLflow's own `start_run` marks the parent run FAILED.

## Files

```sh
├── STYLE.md
├── src
│   ├── autapse.py
│   ├── data.py
│   ├── dynamics_insights.py
│   ├── logged_training.py
│   ├── metrics.py
│   ├── models.py
│   ├── run_config.py
│   ├── run_experiments.py
│   ├── size_scan.py
│   ├── sweep.py
│   └── train.py
├── tests
│   ├── test_autapse.py          # degeneracy, grad-flow, boundedness checks
│   ├── test_data.py
│   ├── test_dynamics_insights.py
│   ├── test_logged_training.py  # against a real local sqlite backend
│   ├── test_models.py
│   ├── test_run_experiments.py  # against a real local sqlite backend
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
  collision would silently overwrite `RESULTS_DIR/{name}.json` and confuse
  MLflow's run-name-based comparison); every config's `build_model` succeeds
  without raising; CNN/FCNN dispatch is correct, including that the built
  `FCNN_AI2`'s `AutapticLayer`s actually carry the config's `T`/`ai2_variant`/
  `activation_scheme` (not just the right shape — this specific assertion
  was added after mutation testing showed a shape-only check would miss a
  hyperparameter silently getting dropped or swapped).
- `tests/test_logged_training.py`: exercises `run_and_log` and
  `replicate_group` against a **real** local sqlite backend pointed at
  `tmp_path` (not mocked — fast and deterministic, so not the case
  STYLE.md's "mock boundaries, not logic" guidance is about), via
  `monkeypatch.setattr` on `logged_training.TRACKING_URI`/
  `ARTIFACT_LOCATION`, with an autouse fixture draining any dangling active
  MLflow run between tests. Covers: params/metrics land correctly; y_self
  absmean metrics present for FCNN, absent for CNN; replicate-group
  aggregation matches an independently-computed mean/stdev; a mid-sweep
  failure skips aggregation and leaves the parent run FAILED;
  `replicate_group(config, 1)` raises before touching MLflow.
- `tests/test_run_experiments.py`: exercises `_run_config`/`_run_seed`/
  `_get_replicate_context`/`_draw_seeds` against the same real-sqlite-backend
  pattern as `test_logged_training.py`, plus monkeypatching
  `run_experiments.RESULTS_DIR` (to `tmp_path`) per test. Covers:
  `num_seeds=1` writes one result JSON, a flat (non-nested) MLflow run, and
  always uses seed `0`; `num_seeds=2`/`3` write one result JSON per seed,
  produce a nested parent+children MLflow structure with aggregate metrics
  on the parent, and draw distinct (not sequential `0, 1, 2...`) seeds, each
  recorded on its child run's `seed` param; `_get_replicate_context`
  dispatches to `contextlib.nullcontext` vs. a real `replicate_group`
  context manager correctly; `_run_seed`'s returned `(avg_accuracy,
  backward_transfer)` matches values independently re-derived from its own
  written result JSON; a non-`"mnist"` `dataset` suffixes the result
  filename/MLflow run name and is recorded as the `dataset` param, so it
  can't collide with the same config's `"mnist"` result.
- `tests/test_size_scan.py`: mirrors `test_sweep.py`'s checks (unique names,
  every config's `build_model` succeeds without raising) plus: size-scan
  names never collide with `sweep.build_configs()`'s own names; hidden
  sizes/channels actually shrink step over step; every step includes both
  the ai2 grid and the FCNN/CNN controls; built models' shapes match their
  config's `hidden_sizes`/`channels`; mismatched-length
  `hidden_sizes_scan`/`channels_scan` raises.
- `uv run pytest -q` for all of the above.
- `ruff check` / `ruff format --check` / `ty check` clean.
- Smoke-run `run_experiments.py` for 2 epochs/stages on all 14 configs (or a
  subset), confirm no NaNs/crashes, that the accuracy matrix + BWT numbers
  are produced, and that the MLflow runs show up (`mlflow ui
  --backend-store-uri sqlite:///mlflow.db`) before committing to
  full-length runs. Same for `--scan size` and `--dataset fashion_mnist`
  before committing to full-length runs on either.
