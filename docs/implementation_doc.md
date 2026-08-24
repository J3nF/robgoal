# Implementation documentation

## Data protocol

Class-incremental MNIST, cumulative (`data.py`):

- Stage 0: digits {0–4}
- Stages 1–5: add digit 5, 6, 7, 8, 9 one at a time; train set is the union of all
  digits introduced so far (not replaced) — per 07-06 notes ("one by one enables
  clearer results with regard to novelty").
- Fixed `epochs_per_stage` epochs per stage — "learnt" is epoch-delimited, not
  loss-delimited (07-14 pre-notes: cost-function conditions may over-restrict
  the loss landscape).
- Eval each stage on the full test set (all 10 digits); the resulting per-digit
  accuracy dict is sliced downstream into a forgetting read-out (already-introduced
  digits) and a generalization read-out (not-yet-introduced ones) — the
  "generalization" check from 07-06.

## Core module: `autapse.py`

One class, `AutapticLayer`, covering 2 ai2 variants × 2 activation schemes,
with a `T`-step unroll per forward call. `variant` and `scheme` are `Literal` type
aliases exported from the module (`AI2_Variant`, `Activation_Scheme`); see the module's own
docstrings for the exact per-variant/scheme formulas — not duplicated here so this
doc can't drift out of sync with the code again.

The inhibition signal (`y_tilde_prev`, i.e. \tilde y_i(t-1) in lab-book.md
notation) starts at `0` regardless of scheme, so `T=1` degenerates to a plain
`Linear` + activation for both schemes and both variants.

## Models: `models.py`

- `FCNN_AI2`: 784-256-128-10 MLP whose two hidden layers are `AutapticLayer`s
  sharing one `(T, ai2_variant, activation_scheme)`; output logits stay a plain
  `Linear`. `T=1` gives the FCNN controls (tests 2–3 in lab-book.md's "Implemented
  tests").
- `CNN`: control architecture, 2×(conv+ReLU+pool) + FC, no self-inhibition.

## Training loop: `train.py`

- `run_incremental_training(model, stages, device, epochs_per_stage=5, lr=1e-3,
  batch_size=128)`: trains `model` through the stages in order (plain SGD,
  momentum 0, cross-entropy loss); returns one per-digit accuracy dict per stage
  (via `metrics.evaluate_per_digit_accuracy`), evaluated on the full test set
  right after that stage.
- `train_one_epoch`: one forward/backward/step pass over a stage's `DataLoader`.

## Metrics: `metrics.py`

Standard CL measures, Avalanche-style (the project's own reference framework):

- Per-stage accuracy matrix (row = stage, col = digit)
- Average accuracy after each stage
- Backward transfer / forgetting: mean(acc at final stage − acc at stage 0),
  over the digits introduced at stage 0 (precondition, not checked: every digit
  passed in must have been introduced at stage 0)

## Experiment runner: `run_experiments.py`

- `build_configs()`: the 7 tests — 3 controls (`cnn_control`, `fcnn_control_relu`,
  `fcnn_control_sigmoid`, the latter two at `T=1`) + 2 ai2 variants × 2 activation
  schemes at `T=3`.
- `main()`: downloads MNIST, builds the stages, runs `run_incremental_training`
  per config, writes each config's per-stage accuracy to
  `RESULTS_DIR/{name}.json` (`RESULTS_DIR = Path("../results")`, created if
  missing), and prints average accuracy, backward transfer, and the formatted
  accuracy matrix per config.

## Logging: `logged_training.py`

Stub (`import mlflow`) for the [MLflow PyTorch
integration](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) described
in lab-book.md, "Logging: MLflow". Not yet wired into `train.py` /
`run_experiments.py` — TBW.

## Files

```sh
├── STYLE.md
├── src
│   ├── autapse.py
│   ├── data.py
│   ├── logged_training.py
│   ├── metrics.py
│   ├── models.py
│   ├── run_experiments.py
│   └── train.py
├── tests
│   ├── test_autapse.py # degeneracy, grad-flow, boundedness checks
│   └── test_data.py
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
  digits.
- `uv run pytest -q` for the above.
- `ruff check` / `ty check` clean on new modules.
- Smoke-run `run_experiments.py` for 2 epochs/stages on all 7 configs, confirm no
  NaNs/crashes and that the accuracy matrix + BWT numbers are produced before
  committing to full-length runs.
