# Implementation documentation

## Data protocol

Class-incremental MNIST, cumulative (`data.py`):

- Stage 0: digits {0–4}
- Stages 1–5: add digit 5, 6, 7, 8, 9 one at a time; train set is the union of all
  digits introduced so far (not replaced) — per 07-06 notes ("one by one enables
  clearer results with regard to novelty").
- Fixed `E` epochs per stage — "learnt" is epoch-delimited, not loss-delimited
  (07-14 pre-notes: cost-function conditions may over-restrict the loss landscape).
- Eval each stage on (a) all digits introduced so far → forgetting metrics, and
  (b) the full test set including not-yet-introduced digits → the "generalization"
  check from 07-06.

## Core module: `autapse.py`

One class, `AutapticLayer`, covering 2 inhibition modes × 2 activation schemes,
with a `T`-step unroll per forward call. `mode` and `scheme` are `Literal` type
aliases exported from the module (`Mode`, `Scheme`); see the module's own
docstrings for the exact per-mode/scheme formulas — not duplicated here so this
doc can't drift out of sync with the code again.

The inhibition signal (`y_tilde_prev`, i.e. \tilde y_i(t-1) in lab-book.md
notation) starts at `0` regardless of scheme, so `T=1` degenerates to a plain
`Linear` + activation for both schemes and both modes.

## Metrics: `metrics.py`

Standard CL measures, Avalanche-style (the project's own reference framework):

- Per-stage accuracy matrix (row = stage, col = digit)
- Average accuracy after each stage
- Backward transfer / forgetting: mean(acc at final stage − acc right after that
  digit was first learned), over digits 0–4

## Files

```sh
├── STYLE.md
├── src
│   ├── autapse.py
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── robgoal.jl
│   ├── run_experiments.py
│   └── train.py
├── tests
│   ├── test_autapse.py # T=1 ≡ plain layer (both schemes), shape/grad checks
│   └── test_data.py
```

Coding conventions for `src/`/`tests/` are in `STYLE.md`, not repeated here.

Flat modules under `src/` (no `__init__.py`, no `[build-system]` in
`pyproject.toml` — still no installable package). Sibling imports between them
(e.g. `from data import Stage`) resolve because `pyproject.toml` sets
`[tool.pytest.ini_options] pythonpath = ["src"]` for tests, and running a
script directly (`uv run python src/run_experiments.py`) auto-adds its own
directory to `sys.path`. `tests/` holds only actual test modules, not
implementation code.

## Verification

- `tests/test_autapse.py`: assert `AutapticLayer(..., T=1)` output equals a plain
  `sigmoid(Linear(x))` / `relu(Linear(x))` for both modes × both schemes (degeneracy
  check); assert gradients flow through `T>1` unrolls (no detached state); assert
  `mult_gate` under `relu_sigmoid` never sign-flips (gate strictly in `(0,1)`).
- `tests/test_data.py`: assert stage `k`'s train set is the superset of stage
  `k-1`'s labels plus exactly one new label; assert test set always contains all 10
  digits.
- `uv run pytest -q` for the above.
- `ruff check` / `ty check` clean on new modules.
- Smoke-run `run_experiments.py` for 1 epoch/stage on all 7 configs, confirm no
  NaNs/crashes and that the accuracy matrix + BWT numbers are produced before
  committing to full-length runs.
