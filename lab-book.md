# Lab book: autapsic-inhibition MNIST tests

## Context

`Mini_project_meta_continual_learning.md` (2026-07-14, "Current Plan") lists MNIST
experiments run/planned but doesn't fully specify them: two baselines (CNN, FCNN)
and "*some* autapsic NN implementation" with three candidate formulas, which the doc
itself flags as unresolved ("who knows what this means as of now"). This lab book
turns that prose into an executable, falsifiable spec: what the tests are and how
to implement them in PyTorch.

## Considerations (resolved ambiguities)

1. **What counts as a "test"?** Counted the "Have run MNIST experiments for"
   bullets: CNN, FCNN, and 3 autapsic formulas — 2 architecture controls + 3
   autapsic formulas, all under the incremental-digit protocol from 2026-07-06.

2. **What does `t` vs `t-1` mean for a static-image classifier?** No natural "time"
   exists in one MNIST feedforward pass. Resolved: unroll each forward pass over a
   fixed `T` internal steps, feeding the same input every step (settling dynamics),
   matching the "performs slightly better than simple RNNs" comparison in the 07-06
   lit review. `T=1` must degenerate to a plain layer — the built-in sanity check.

3. **What does the bias-difference variant (`b_i(t) = b_i(t-1) - b_i(t-1)`) mean?**
   A learned bias is a single constant, so a literal self-difference is degenerate.
   Resolved: `b_i` stays a fixed parameter; the *effective* bias at step `t` is
   `b_i` minus the inhibition signal from `t-1` (spike-frequency-adaptation-style
   self-inhibition through the bias) — the "implicit contribution to bias" in the
   notes.

4. **Sigmoid vs. ReLU — why run both.** Sigmoid is bounded, matching the 07-14
   entropy motivation ("firing rates 0 and 1 waste information") and keeping
   `mult_gate`'s `(1 - y_prev)` a sensible gate, but vanishes over a `T`-step unroll
   and is inconsistent with the ReLU-based CNN control and rest of the project.
   Plain ReLU fixes that but breaks `mult_gate` (unbounded `y_prev` can push
   `1 - y_prev` negative, sign-flipping rather than inhibiting) and can permanently
   kill a unit (exact 0, exact zero gradient, no recovery) — the opposite of the
   project's goal of keeping the network flexible.

   Resolution — run both as parallel axes:
   - **`sigmoid`**: as originally designed, `y(t)` combines `sigmoid(Wx+b)` with
     `y(t-1)` directly (already bounded).
   - **`relu_sigmoid_gate`**: main path `y_raw(t) = relu(Wx+b)` stays unbounded and
     gradient-friendly; the recurrent feedback term is `tanh(y(t-1)/2)` — a
     bounded "activity level" that modulates the next step without being the
     output itself. Fixes `mult_gate`'s sign-flip (gate strictly in `(0,1)`) and
     bounds inhibition strength so a unit with `y_raw ≥ 1` can't be fully killed.
     Cost: the notes' single `y_i` becomes two numbers — an explicit
     reinterpretation, not a literal transcription.

   This forces one implementation detail: the state feeding the *first* unroll
   step must be an explicit `inhibition_prev = 0`, tracked separately from
   `y_prev`, not derived from it via the scheme's squashing function — otherwise
   `sigmoid(0) = 0.5 ≠ 0` would break the `T=1` degeneracy check for
   `relu_sigmoid_gate` specifically.

   **Why `tanh(y/2)`, not `sigmoid(y)`, for the feedback term.** The first
   implementation used `sigmoid(y_prev)`, matching the earlier design
   discussion. `tests/test_autapse.py::test_gradients_flow_through_unroll`
   caught that this is broken beyond the first step: ReLU output is always
   `≥ 0`, so `sigmoid(y_prev) ≥ sigmoid(0) = 0.5` unconditionally — a unit with
   *zero* prior activity still gets a 0.5 inhibition floor, contradicting
   "neurons inhibiting their own activation *more*, the more active they are."
   In the failing test this floor was enough to clip every unit to exactly 0 by
   the final step, zeroing the gradient everywhere — the permanent-dead-neuron
   failure mode the scheme was meant to avoid, one step later than checked for.
   `tanh(y/2) = 2·sigmoid(y) - 1` has the same boundedness and no-sign-flip
   properties but maps `0 ↦ 0`, so zero activity now means zero inhibition at
   every step, not just the artificially-forced first one.

Further defaults, chosen unilaterally as low-risk and reversible:
- Autapse applies to **hidden layers only** — output logits stay a plain `Linear`.
- `T` (unroll depth) isn't given anywhere in the notes — a constructor arg with a
  small default (e.g. 3), not a hardcoded constant.

## The 9 tests

3 controls (no self-inhibition) + 3 formula variants × 2 activation schemes. Within
each scheme, the 3 autapsic variants share one FCNN skeleton with its matching
control, so the *only* varying factor is the formula — a clean ablation. The CNN
control is the separate architecture reference point from the 07-06 notes.

| # | Model | Scheme | Notes |
|---|-------|--------|-------|
| 1 | CNN control | ReLU | small conv net (2×conv+pool+FC), no recurrence |
| 2 | FCNN control | ReLU | 784→256→128→10, plain ReLU (= `relu_sigmoid_gate` autapse layer at `T=1`) |
| 3 | FCNN control | sigmoid | same skeleton, plain sigmoid (= `sigmoid` autapse layer at `T=1`) |
| 4 | FCNN + `output_diff` | sigmoid | `y(t) = y_raw(t) - y(t-1)` |
| 5 | FCNN + `bias_diff` | sigmoid | effective bias `b - y(t-1)` |
| 6 | FCNN + `mult_gate` | sigmoid | `y(t) = y_raw(t) * (1 - y(t-1))` |
| 7 | FCNN + `output_diff` | relu_sigmoid_gate | `y(t) = relu(y_raw(t) - tanh(y(t-1)/2))` |
| 8 | FCNN + `bias_diff` | relu_sigmoid_gate | `y(t) = relu(Wx + b - tanh(y(t-1)/2))` |
| 9 | FCNN + `mult_gate` | relu_sigmoid_gate | `y(t) = relu(y_raw(t)) * (1 - tanh(y(t-1)/2))` |

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

One module, 3 modes × 2 schemes, `T`-step unroll per forward call. `inhibition_prev`
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

```
src/data.py, src/models.py, src/autapse.py, src/train.py, src/metrics.py, src/run_experiments.py
tests/test_autapse.py   # T=1 ≡ plain layer (both schemes), shape/grad checks
tests/test_data.py      # stage splits cumulative & correct
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
