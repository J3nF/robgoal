# Red-team: `FCNN-debugging`

Scope: incremental diff from `serious-scan` at `43c53dd` (this branch's base; `serious-scan`
has since gained `7f11486`, docs and style files only): default SGD `lr` raised `1e-3` → `0.1`,
new `--lr` flag, plus this branch's own `BUGS.md`/`IDEAS.md` — a prior review's bug list and
refactor/academic analysis — and test additions. See `red-team-serious-scan.md` and
`red-team-main.md` for inherited issues, both still present and unaddressed here.

Method: findings 1-3 below were derived independently (own reading and own empirical checks)
before reading `BUGS.md`/`IDEAS.md`, so the "Cross-check" section at the end is an independent
comparison and not a restatement.

Verified directly in a disposable worktree: `uv sync --extra cpu`, `uv run pytest -q` → all
passed (including this branch's new `tests/test_train.py` regression test), `uv run ty check`
→ same 4 diagnostics as `serious-scan` (1 notebook, 3 `test_size_scan.py` — neither fixed on
this branch).

## 1. [Confirmed fix, no new issue] `lr=1e-3` → `0.1` genuinely fixes the sigmoid control

Reproduced directly: the sigmoid FCNN control trained at `lr=1e-3` reaches ~chance accuracy
(0.197 on a 5-class toy setup); at `lr=0.1` it reaches 1.0 on the same setup. This branch's own
`tests/test_train.py::test_sigmoid_fcnn_control_learns_within_default_schedule` locks the fix
in as a regression test, and it passes for the right reason (not an artifact of easy data).

## 2. [MEDIUM — re-scoped after verification] The lr increase's interaction with `ai2_gate`'s instability is narrower than it first looks, but still an open question over full runs

Raising `lr` 100× reopens a question neither this branch nor its `BUGS.md`/`IDEAS.md` answer:
whether it pushes `relu_sigmoid`+`ai2_gate` — whose inhibition iteration has no guaranteed
contraction for large pre-activations — into practice during real training. Tested via 300 SGD
steps on a fixed batch at each base config's own `lr`: no NaNs, all `relu_sigmoid` variants
converge smoothly; tracking pre-activation magnitude for `ai2_gate` specifically, it reached
roughly double the instability threshold IDEAS.md's analysis cites (`BUGS.md` #4: slope
magnitude reaches 1 at roughly $a \approx 3$), with no divergence.

This is weaker evidence of a real problem than "may diverge" suggests, though: `ai2_gate`'s
output $a\,(1 - \tanh(\cdot))$ is bounded by construction, since $\tanh$ is bounded in
$(-1, 1)$, so true numerical blow-up (NaN/Inf) is structurally close to impossible regardless
of `lr`. The legitimate open question is narrower — possible *oscillation that degrades
training quality* over a full multi-epoch, multi-stage run with real data drift, not numerical
divergence. One fixed batch cannot show that, and nothing in `BUGS.md`/`IDEAS.md` tests it.
TODO: log pre-activation magnitude / `y_self` curves (`dynamics_insights.get_y_self_absmeans`
already exists) across a full real sweep run and look for non-monotonic or oscillating
patterns, not only NaNs.

## 3. [Sharpens `BUGS.md` #2] The even-`T` degenerate case kills the whole network, not just one layer

`BUGS.md` #2 calls `ai2_diff`+`sigmoid` at even `T` a "dead layer." Verified: it's a dead
*network*. `FCNN_AI2.forward` (`src/models.py`) has no skip/residual path, so the first hidden
layer's exact-zero output (cancellation derived in `red-team-main.md` #1) zeroes the second
hidden layer's input, which by the same cancellation zeroes *its* output. The final output is
then exactly `self.output.bias`, identical across unrelated random inputs, with zero gradient
on every upstream parameter. "Dead layer" understates how total the collapse is.

## 4. [Not in `BUGS.md`/`IDEAS.md`] `ty check` failures are undocumented

Neither file mentions that `uv run ty check` currently fails (grepped both for "ty check" and
variants — no hits). Confirmed still failing on this branch: 1 error inherited from `main`
(notebook SSL-lambda signature) + 3 inherited from `serious-scan`
(`tests/test_size_scan.py`'s unnarrowed `Optional` access). See `red-team-main.md` #5 and
`red-team-serious-scan.md` #1 for detail.

## 5. [Not in `BUGS.md`/`IDEAS.md`] The size-scan amplification of the duplicate-config bug is unaddressed

`IDEAS.md` §3's follow-up list (not `BUGS.md`, which has no numbered follow-up section) item 4
says: "Drop `fcnn_ai2_diff_sigmoid` from the sweep (duplicate of the control at odd `T`) or
guard against even `T`." That only addresses the single instance in `sweep.py`'s base grid.
`size_scan.py` (inherited unchanged from `serious-scan`) repeats the identical pair once per
each of its 5 size steps — see `red-team-serious-scan.md` #2 — and this is mentioned nowhere in
either file.

## 6. [Needs testing — noticed during verification, in neither list] `lr=0.1` was never checked at the largest config

The new global `lr=0.1` default was validated by this branch's own lab-book entry only against
the base sweep's default-sized models and against short smoke batches here — it was never
specifically checked against the sweep's largest-capacity config,
`fcnn_ai2_diff_relu_sigmoid_hidden512x256` (the `hidden_sizes_sweep=[(512,256)]` entry,
`src/sweep.py`). `IDEAS.md` §2.4 lists "a per-family tuned `lr`" as a missing control in
general, but doesn't flag that the one `lr` value actually shipped was never validated against
that specific, largest config.
TODO: a short real run of that config at `lr=0.1`, checking whether `lr` is too high for the
larger model.

## Cross-check against this branch's own `BUGS.md`/`IDEAS.md`

Independently re-derived, and confirmed accurate (no disagreements found), 9 of `BUGS.md`'s 11
"Fixed"/"Open" items: the `lr` bug (#1), the `ai2_diff`+`sigmoid` identity/dead-layer behavior
(#2), the lab-book/code formula mismatch (#3: book says
$\mathrm{relu}(y_\text{raw} - \tanh(\cdot))$, code computes
$\mathrm{relu}(Wx + b) - \tanh(\cdot)$ with no outer $\mathrm{relu}$ — confirmed by reading
`_get_inhibited_output`), the cumulative-replay/BWT caveat (#5), partial seeding (#6), silent
result overwrites (#7), gitignored `results/` (#8), the global SSL monkeypatch (#9), and the
setup pitfalls (#10: bare `uv sync`, `requires-python>=3.14`, `main.py` stub). The other two
are not in this list: #4 (`ai2_gate` contraction) is the subject of finding 2 above, and #11
(single-run results) is not covered by this report.

Two further matches with `IDEAS.md`: its §2 contraction analysis (2.1-2.2) independently
matches this report's own derivation of the no-temporal-memory point (`red-team-main.md` #4),
and its §3 follow-up #1 (re-run the sigmoid rows at a working `lr`) matches the
optimizer-circularity point (`red-team-main.md` #3). No factual error was found in either file.
