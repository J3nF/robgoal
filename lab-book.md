# Lab book: autapsic-inhibition MNIST tests

Lab book recording experiments with *autapsic inhibition AI (ai2 )* networks, as sketched out in `Mini_project_meta_continual_learning.md`.

## General contents

### Wishlist

- If ai2 implemented *adaption*, it would behave like a leaky integral of activity.
- We wish ai2 to:
  - Diffuse information across neurons (laterally and longitudinally).
  - Have a neuron self-regulate to, more or less, switch on and off between inputs.
  - Intervene by raising and lowering outputs of neurons (i.e., not only reducing/increasing them).
  - Have rigorous formal underbuilding.

#### Terrible(?) brainstorms

- When using a sigmoid-shaped activation function, set the new midpoint to the last output.

### Notation

- $i =$ neuron of concern;
- $\tilde\bullet =$ autapses-related parameter;
- $y_i =$ $i$'s unadapted output;
- $b_i =$ $i$'s bias.

### Possible implementations

$$\begin{align}
  \tilde b_i(t)
    &= b_i(t) - \tilde{b}_i(t-1)
  \tilde y_i(t)
    &= y_i(t) - \tilde{y}_i(t-1)
  \\
  \tilde y_i(t)
    &= y_i(t)\cdot\left(
        1 - \tilde{y}_i(t-1)
      \right) \nonumber\\
    &= y_i(t) - \tilde{y}_i(t-1)y_i(t)
\end{align}$$

### Experiment design

- Learning is to take place between "timesteps" $t, (t-1)$.
      - Option A: Show the same input $T$ times to give the network time to have time to converge, given ai2 adaptions.
      - Option B: Implement memory (leaky integrate-fire/integrate-inhibit).
- (Future control group: Randomly switching subsets of neurons off?)
- Learn subsets of all numbers, adding more and more after having learnt old set.
- Note: "*Having learnt*" is epoch-delimited instead of cost-function delimited.
  - Learning is epoch-delimited because cost function conditions may restrict allowed states too much (i.e., to a sphere in the loss landscape).
    - Checking whether the worry is true is a potential future TODO.
- Note: These ai2 implementations will reduce activity for the vast majority of scenarioes.

### Chosen experiments

- Control groups:
  - CNN
  - Fully connected NN
- Our ai2 implementations:
  $$\begin{align}
    \tilde y_i(t)
      &= y_i(t) - \tilde{y}_i(t-1)
    \\
    \tilde y_i(t)
      &= y_i(t)\cdot\left(
          1 - \tilde{y}_i(t-1)
        \right) \nonumber\\
      &= y_i(t) - \tilde{y}_i(t-1)y_i(t)
  \end{align}$$

## Thought processes on above choices and resolved ambiguities

### The meaning of `t` for static-image classifiers

Unroll each forward pass over `T` fixed internal steps, feeding the same input every step.
This is done to:

- Allow dynamics to settle,
- Enable autapsic inhibition to take place in the first place.
Note how, for `T=1`, ai2 networks reduce to a plain layer.

### Activation functions: ReLU and Sigmoids

A priori, activation functions bound to $[0,1]$ match intuitions of thinking about relative inhibition, and seem closer to biological spiking neural networks (as in, there is a maximum signal (density)).
Further, some ai2 schemes need boundedness to keep the adapted ouput in the activation function's domain (e.g., keeping $1-y(t-1)$ positive).

Yet, contrasting bound activation functions with ReLU activations used by our controls result in an apples-to-oranges comparison, and, more troublingly, makes some ai2 schemes converge to 0 during rollout (again, consider $1-y(t-1)$, which can only decrease $\tilde{y}_(t)$ for positive-only outputs).

Especially the latter incompatibility's problematicness may be interpreted two ways: Either the ai2 formulations, or activation functions, are a bad choice, given the other.

I propose and run two different mitigation strategies:

First, instead of ReLU, networks use a sigmoid activation functions.
While this can keep ouputs in the same domain, inhibition may still make neural outputs converge to 0.

The second approach combines ReLU activation with sigmoid-gated inhibition, i.e.
$$
\tilde y_i(t) = s(a_\text{ReLU}(x))
$$
with a sigmoid function $s:A\to \tilde Y$ yielding ai2 adaptions $\tilde y_i(t)$.

This should ease comparability to ReLU structures while keeping ai2 implementation outputs in the expected domain ($[0,1)$).

#### Implemented activation schemes

TODO

Run both activation schemes in parallel:

- Keyword `sigmoid` implies usage of `y(Wx+b)` with
     `y(t-1)` directly (already bounded).
- **`relu_sigmoid_gate`**: main path `y(t) = relu(Wx+b)` stays unbounded and
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
| --- | ------- | -------- | ------- |
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
