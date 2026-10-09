# Red-team: `main`

Methodology: full read of `src/`, `tests/`, `docs/`, `pyproject.toml`, `README.md`,
`notebooks/`; `uv sync --extra cpu`, `uv run pytest -q` (47 passed), `uv run ruff check .`
(clean), `uv run ty check` (1 error, see #5) run in a disposable worktree; several claims
confirmed by direct computation (seeded model construction/forward/backward). Adversarially
re-verified by an independent subagent against the live worktree — verdicts below include its
findings.

## 1. [CRITICAL] `ai2_diff` + `sigmoid` degenerates to either a duplicate or a dead, constant network

`AutapticLayer._step`/`_get_inhibited_output` (`src/autapse.py:103-121`) feeds each unroll
step's own *output* back as the next step's inhibition signal. For `ai2_variant="ai2_diff"`,
`activation_scheme="sigmoid"`, with `s = sigmoid(linear(x))` constant across the unroll (same
`x` every step, see docstring at `src/autapse.py:20-24`):

```
y1 = s - 0 = s
y2 = s - y1 = 0
y3 = s - y2 = s
...
```

This is exact, not a numerical approximation — confirmed by building `T=1` vs `T=3` models
with identical seeds: `torch.equal(T1_output, T3_output) == True`. For even `T`, the layer's
output is a hard zero, algebraically independent of its own weights — confirmed `T=2`'s
gradient on every hidden-layer parameter is exactly `0.0`.

Because `FCNN_AI2` stacks two such layers with no skip connection (`src/models.py:36-40`), a
dead first hidden layer forces a dead second hidden layer, so **the entire network's output
collapses to a constant — exactly `self.output.bias`, independent of the input image** —
confirmed across unrelated random inputs (`torch.allclose(y(x_a), y(x_b))`, and both equal
`model.output.bias`).

Consequence: `sweep.py`'s base grid (`src/sweep.py:71-88`) only ever runs this combination at
`T=BASE_T=3` (odd), so the live sweep never hits the dead-network case — but nothing guards
against it (`AutapticLayer.__init__` only checks `T >= 1`), and the odd-`T` case that *is* run
(`fcnn_ai2_diff_sigmoid`, lab-book.md's test #4) is mathematically the same function, and
under matching seeds the same trained model, as `fcnn_control_sigmoid` (test #3). Any
reported difference between those two rows is pure seed noise, not a model-structure effect.
`tests/test_sweep.py`'s name-uniqueness check and `tests/test_models.py`'s `T=2` shape test
(which only checks output *shape*, not value) both pass without ever catching this.

## 2. [CRITICAL, academic] The data protocol structurally suppresses the forgetting it's meant to measure

`data.py`'s `STAGE_DIGITS`/`build_stages` (`src/data.py`) is cumulative replay: each stage's
training set is the union of every digit introduced so far (`digits_seen |= set(new_digits)`
in `build_stages`), never a replacement. For any architecture that trains adequately, old
digits keep getting rehearsed every stage, so `backward_transfer` ("the forgetting measure,"
`docs/lab-book.md`, "Evaluation metrics") is expected to hover near/above zero regardless of
whether `ai2` helps. A quick confirmatory run (SGD, `T=1`, deliberately short) gave
`BWT=+0.62` — net improvement, not forgetting, consistent with this.

`docs/lab-book.md` discloses the cumulative design choice itself ("Class-incremental data
protocol") but never connects it to this consequence; `docs/Mini_project_meta_continual_learning.md`'s
"Line of argument" frames the entire project around catastrophic forgetting ("local information
overload is the root of all our troubles (i.e. catastrophic forgetting)"). As implemented, the
protocol can't exhibit much catastrophic forgetting in the first place, for any architecture —
which limits what the sweep's BWT numbers can actually say about the project's central
hypothesis, independent of whether `ai2` works.

## 3. [MEDIUM, academic] Adam→SGD switch reasons from the desired result to the method

`docs/lab-book.md`, "Optimizers: Adam vs SGD": Adam made `ai2`-vs-control results "practically
indistinguishable"; the stated reason for switching to SGD is "to make sure ai2 is able to
meaningfully change the loss landscape." This is circular: it selects the optimizer more
sensitive to *any* change in the compute graph specifically because it's more likely to show a
difference, without a control for the alternative explanation that SGD's general sensitivity
— not a genuine representational effect of `ai2` — produces whatever difference then appears.
The lab book is partially self-aware of this (line 177's TODO to log gradient/weight norms "to
check the ... reasoning above directly rather than inferring it from accuracy"), which tempers
but doesn't resolve the gap — no such control currently exists.

## 4. [MEDIUM, academic — already a disclosed tradeoff, but inconsistently described elsewhere] No state persists across examples

`AutapticLayer.forward` resets `y_self` to zero at the start of every call
(`src/autapse.py:71`), so nothing survives across training examples, batches, or epochs — each
forward pass is an independent fixed-point iteration from a fresh zero start, not a
dynamical/recurrent mechanism with memory. `docs/lab-book.md`'s "Experiment design" section
*does* name this as a deliberate choice ("Option A" — repeat the same input `T` times — over
"Option B: Implement memory"), so this isn't a hidden gap. It is, however, in tension with how
the project's own framing documents (`Mini_project_meta_continual_learning.md`'s "self-recurrent,
self-inhibiting connections," the wishlist's "leaky integral of activity") keep invoking
persistent, biological-autapse-style dynamics — language that doesn't match what Option A
actually implements. Worth tightening the terminology, not fixing a bug.

## 5. [LOW-MEDIUM, verified] Drifted duplicate: `notebooks/mnist-download.ipynb`

Duplicates `src/data.py`'s SSL-monkeypatch workaround (identical comment, copy-pasted) with a
different, narrower lambda signature (`lambda: ...` vs. `data.py`'s `lambda *a, **kw: ...`) and
no `# ty: ignore` annotation. `uv run ty check` currently fails here:

```
error[invalid-assignment]: ... notebooks/mnist-download.ipynb:cell 1:10:1
```

The notebook also uses a bare `root="../data"` (cwd-dependent) — the exact pattern
`run_experiments.py`/`logged_training.py` deliberately moved away from via `REPO_ROOT`
anchoring, for the same brittleness reason.

## 6. [LOW, verified] Doc references a non-existent method

`docs/implementation_doc.md`'s `autapse.py` section says `_step` "delegates to
`_get_main_output`"; no such method exists in `src/autapse.py` (the real helper is
`_get_y_activation`, feeding `_step`). Same class of issue as the lab-book/code mismatch in the
`relu_sigmoid`+`ai2_diff` formula (BUGS.md #3 on `FCNN-debugging`, see that branch's report) —
docs drifting out of sync with the code they describe.

## 7. [LOW, verified] Bare `uv sync` leaves an environment with no `torch`

`torch`/`torchvision` are declared only under `[project.optional-dependencies]` (`cpu`,
`cu130`), not `[project.dependencies]` (`pyproject.toml:7-21`). Reproduced: a bare `uv sync`
installs everything else, then `import torch` raises `ModuleNotFoundError`. README.md does
document `--extra=cpu`/`--extra=cu130`, but it's an easy first-run trap, and
`implementation_doc.md`'s own "Verification" checklist's `uv run pytest -q` silently assumes
the right extra was already chosen.

## Additional, lower-priority observations

- `tests/test_models.py` exercises `T=2` (the exact degenerate case in #1) but only asserts
  output *shape* — it provides false confidence that `T=2` is covered, when its actual
  numerical behavior (constant-bias collapse) is untested.
- `metrics.backward_transfer`'s documented precondition ("every digit in `first_stage_digits`
  must have been introduced at stage 0 — not checked") is enforced nowhere in code; currently
  harmless only because every caller happens to pass `stages[0].digits_seen`.
