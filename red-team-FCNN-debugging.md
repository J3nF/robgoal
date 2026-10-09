# Red-team: `FCNN-debugging`

Scope: incremental diff from `serious-scan` (default SGD `lr` raised `1e-3` → `0.1`, new
`--lr` flag, plus this branch's own `BUGS.md`/`IDEAS.md` — a prior review's bug list and
refactor/academic analysis — and test additions). See `red-team-serious-scan.md` and
`red-team-main.md` for inherited issues, both still present and unaddressed here.

Methodology note: findings 1-3 below were derived independently (own reading + own empirical
checks) before reading `BUGS.md`/`IDEAS.md`, per the agreed review plan. Section "Cross-check"
at the end compares against those two files directly, as instructed.

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
roughly double the instability threshold IDEAS.md's analysis cites, with no divergence.

This is weaker evidence of a real problem than "may diverge" suggests, though: `ai2_gate`'s
output is bounded by construction (`a·(1−tanh(·))`, with `tanh` bounded in `(−1,1)`), so true
numerical blow-up (NaN/Inf) is structurally close to impossible regardless of `lr`. The
legitimate open question is narrower — possible *oscillation that degrades training quality*
over a full multi-epoch, multi-stage run with real data drift, not numerical divergence — and
that narrower question is still untested by the one-fixed-batch smoke test here or by anything
in `BUGS.md`/`IDEAS.md`.

## 3. [Sharpens `BUGS.md` #2] The even-`T` degenerate case kills the whole network, not just one layer

`BUGS.md` #2 calls `ai2_diff`+`sigmoid` at even `T` a "dead layer." Verified: it's a dead
*network*. `FCNN_AI2.forward` (`src/models.py`) has no skip/residual path, so a first hidden
layer that's an exact-zero function of its input (per the algebraic cancellation in
`red-team-main.md` #1) forces the second hidden layer's input to zero too, which — by the same
cancellation — forces *its* output to zero as well. The final output is then exactly
`self.output.bias`, identical across unrelated random inputs, with zero gradient on every
upstream parameter. `BUGS.md`'s phrasing undersells how total the collapse is.

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

## Cross-check against this branch's own `BUGS.md`/`IDEAS.md`

Independently re-derived, and confirmed accurate (no disagreements found), 9 of `BUGS.md`'s 11
"Fixed"/"Open" items: the `lr` bug, the `ai2_diff`+`sigmoid` identity/dead-layer behavior, the
lab-book/code formula mismatch (#3: book says `relu(y_raw − tanh(...))`, code computes
`relu(Wx+b) − tanh(...)`, no outer `relu` — confirmed by reading `_get_inhibited_output`), the
cumulative-replay/BWT caveat, partial seeding, silent result overwrites, gitignored `results/`,
the global SSL monkeypatch, and the setup pitfalls (bare `uv sync`, `requires-python>=3.14`,
`main.py` stub). `IDEAS.md` §2's contraction analysis (2.1-2.2) independently matches this
report's own derivation of the no-temporal-memory point (`red-team-main.md` #4) and the
optimizer-circularity point (`red-team-main.md` #3, matched by `IDEAS.md`'s §3 follow-up #1
flagging the sigmoid rows need re-running at a working `lr`). No factual error was found in
either file.

## Additional issue (noticed during verification, in neither list)

The new global `lr=0.1` default was validated by this branch's own lab-book entry only against
the base sweep's default-sized models and against short smoke batches here — it was never
specifically checked against the sweep's largest-capacity config,
`fcnn_ai2_diff_relu_sigmoid_hidden512x256` (the `hidden_sizes_sweep=[(512,256)]` entry,
`src/sweep.py`). `IDEAS.md` §2.4 lists "a per-family tuned `lr`" as a missing control in
general, but doesn't flag that the one `lr` value actually shipped was never validated against
that specific, largest config.
