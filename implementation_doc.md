# Implementation documentation

## Data protocol

Class-incremental MNIST, cumulative (`data.py`):

- Stage 0: digits {0–4}
- Stages 1–4: add digit 5, 6, 7, 8, 9 one at a time; train set is the union of all
  digits introduced so far (not replaced) — per 07-06 notes ("one by one enables
  clearer results with regard to novelty").
- Fixed `E` epochs per stage — "learnt" is epoch-delimited, not loss-delimited
  (07-14 pre-notes: cost-function conditions may over-restrict the loss landscape).
- Eval each stage on (a) all digits introduced so far → forgetting metrics, and
  (b) the full test set including not-yet-introduced digits → the "generalization"
  check from 07-06.

## Core module: `autapse.py`

One module, 2 modes × 2 schemes, `T`-step unroll per forward call. `inhibition_prev`
starts at `0` regardless of scheme, so `T=1` degenerates to a plain layer for both
schemes and all 3 modes:

```python
class AutapticLayer(nn.Module):
    def __init__(self, in_f, out_f, T, mode, scheme):
        # mode:   "output_diff" | "bias_diff" | "mult_gate"
        # scheme: "sigmoid" | "relu_sigmoid_gate"
        self.linear = nn.Linear(in_f, out_f)
        self.T, self.mode, self.scheme = T, mode, scheme

    def forward(self, x):
        act = torch.sigmoid if self.scheme == "sigmoid" else torch.relu
        inhibition_prev = torch.zeros(x.shape[0], self.linear.out_features, device=x.device)
        for _ in range(self.T):
            if self.mode == "bias_diff":
                raw = x @ self.linear.weight.T + (self.linear.bias - inhibition_prev)
                y = act(raw)
            else:
                y_raw = act(self.linear(x))
                if self.mode == "output_diff":
                    y = y_raw - inhibition_prev
                    if self.scheme == "relu_sigmoid_gate":
                        y = torch.relu(y)  # keep nonnegative after subtraction
                else:  # mult_gate
                    y = y_raw * (1 - inhibition_prev)
            inhibition_prev = y if self.scheme == "sigmoid" else torch.tanh(y / 2)
        return y
```

## Metrics: `metrics.py`

Standard CL measures, Avalanche-style (the project's own reference framework):

- Per-stage accuracy matrix (row = stage, col = digit)
- Average accuracy after each stage
- Backward transfer / forgetting: mean(acc at final stage − acc right after that
  digit was first learned), over digits 0–4

## Files

```sh
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

Flat modules under `src/` (no `__init__.py`, no `[build-system]` in
`pyproject.toml` — still no installable package). Sibling imports between them
(e.g. `from data import Stage`) resolve because `pyproject.toml` sets
`[tool.pytest.ini_options] pythonpath = ["src"]` for tests, and running a
script directly (`uv run python src/run_experiments.py`) auto-adds its own
directory to `sys.path`. `tests/` holds only actual test modules, not
implementation code.

## Verification

- `tests/test_autapse.py`: assert `AutapticLayer(..., T=1)` output equals a plain
  `sigmoid(Linear(x))` / `relu(Linear(x))` for all 3 modes × both schemes (degeneracy
  check); assert gradients flow through `T>1` unrolls (no detached state); assert
  `mult_gate` under `relu_sigmoid_gate` never sign-flips (gate strictly in `(0,1)`).
- `tests/test_data.py`: assert stage `k`'s train set is the superset of stage
  `k-1`'s labels plus exactly one new label; assert test set always contains all 10
  digits.
- `uv run pytest -q` for the above.
- `ruff check` / `ty check` clean on new modules.
- Smoke-run `run_experiments.py` for 1 epoch/stage on all 9 configs, confirm no
  NaNs/crashes and that the accuracy matrix + BWT numbers are produced before
  committing to full-length runs.
