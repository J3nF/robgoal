# Triage: all findings across `red-team-{main,serious-scan,FCNN-debugging,python-modularisation}.md`

Consolidates and deduplicates every finding from the four branch red-team reports into one
prioritized backlog. Branches are stacked (`main` → `serious-scan` → `FCNN-debugging` /
`python-modularisation`), so most issues originate once and are inherited; each row below
lists where it *originates* and every branch it *affects*. `python-modularisation` has no code
of its own (identical commit to `serious-scan`), so its findings are about a *proposed future
refactor* (`IDEAS.md` §1) rather than current code — those are kept in their own section at the
end, since "fixing" them means writing the refactor correctly when it happens, not patching
code that exists today.

Severity scale: **Critical** (undermines the project's core claims or produces silently wrong
results) → **High** (violates an explicit project policy, cheap to fix, currently violated) →
**Medium** → **Low/Low-Medium** (cosmetic, doc drift, DX). A separate **Needs further
testing** bucket holds items that were investigated but not confirmed either way — don't
triage these as "fix," triage them as "run the missing experiment."

## Summary table

| # | Title | Severity | Category | Origin | Affects |
|---|---|---|---|---|---|
| 1 | `ai2_diff`+`sigmoid` degenerate (duplicate/dead network) | Critical | implementation+academic | main | all 3 code branches |
| 2 | Cumulative replay suppresses the BWT forgetting signal | Critical | academic | main | all 3 code branches |
| 3 | `ty check` fails: `test_size_scan.py` Optional-narrowing | High | implementation | serious-scan | serious-scan, FCNN-debugging |
| 4 | Adam→SGD optimizer choice is circular reasoning | Medium | academic | main | all 3 code branches |
| 5 | Temporal-persistence framing doesn't match the implementation | Medium | academic | main | all 3 code branches |
| 6 | `CLAUDE.md`/`WRITING_STYLE.md` untracked from git | Medium | technical debt | serious-scan | serious-scan, FCNN-debugging |
| 7 | Silent result overwrites, duplicate MLflow runs on rerun | Medium | technical debt | main* | all 3 code branches |
| 8 | `results/` gitignored — numbers not reproducible from repo | Medium | technical debt | main* | all 3 code branches |
| 9 | `lr=0.1` × `ai2_gate` instability over full runs | *Needs testing* | academic | FCNN-debugging | FCNN-debugging |
| 10 | `lr=0.1` never validated at the largest (512,256) config | *Needs testing* | implementation | FCNN-debugging | FCNN-debugging |
| 11 | Notebook SSL-monkeypatch duplicate, drifted, fails `ty check` | Low-Medium | implementation | main | all 3 code branches |
| 12 | Shape-only tests mask #1's real (value-level) behavior | Low-Medium | test coverage | main | all 3 code branches |
| 13 | Global SSL monkeypatch is a process-wide side effect | Low-Medium | implementation | main* | all 3 code branches |
| 14 | Partial seeding / cross-config RNG sharing (self-disclosed) | Low-Medium | academic | main | all 3 code branches |
| 15 | Doc drift: nonexistent method name + stale formula in docs | Low | technical debt | main | all 3 code branches |
| 16 | Setup/DX pitfalls (no-torch-by-default, py3.14, `main.py` stub) | Low | technical debt | main | all 3 code branches |
| 17 | Lab-book checkbox granularity mismatch | Low | technical debt | serious-scan | serious-scan, FCNN-debugging |
| 18 | `backward_transfer` unchecked precondition | Low | implementation | main | all 3 code branches |
| 19 | Multi-seed `SystemRandom` draws aren't reproducible ahead of time | By design | — | serious-scan | serious-scan, FCNN-debugging |
| 20 | `lr` fix landed on `FCNN-debugging` only, not merged elsewhere | Status | — | FCNN-debugging | — |
| P1 | Plan: Option C breaks `REPO_ROOT` anchoring | High-if-executed | plan risk | IDEAS.md §1 (`python-modularisation` report) | — |
| P2 | Plan: Option G reopens the Adam/SGD risk (cross-ref #4) | Medium-if-executed | plan risk | ″ | — |
| P3 | Plan: Option H undersold, dropped from suggested order | Medium (planning gap + missed opportunity) | plan risk | ″ | — |
| P4 | Plan: CLI/doc migration only partly addressed | Low-Medium-if-executed | plan risk | ″ | — |
| P5 | Plan: "independent" vs. "C is a prerequisite" self-tension | Low | plan risk | ″ | — |
| P6 | Plan: existing `results/`/MLflow schema compatibility unaddressed | Medium-if-executed | plan risk | ″ | — |

\* Rows #7, #8, #13, and #16(b)/(c) are **not** written up as standalone findings in any of
the four `red-team-*.md` reports — they were verified directly against the code during the
original review session but not carried into the final report text. They surface here only
because `red-team-FCNN-debugging.md`'s "Cross-check" section name-checks them in passing while
confirming `FCNN-debugging/BUGS.md`'s list is accurate. Flagging this so the provenance is
honest: don't expect to find these four written up in the source reports themselves; the
detail below was independently re-verified for this triage, not copied from them.

---

## Critical

### 1. `ai2_diff` + `sigmoid` degenerates to a duplicate run or a dead, constant network

**Where:** `src/autapse.py:103-121` (`_step`/`_get_inhibited_output`), `src/models.py:36-40`
(`FCNN_AI2.forward`, no skip connection), `src/sweep.py:71-88`, amplified ×5 by
`src/size_scan.py`'s `_build_fcnn_ai2_grid_step` (one extra identical pair per size-scan
step). **Affects:** `main`, `serious-scan` (5× amplified), `FCNN-debugging` (BUGS.md #2,
confirmed understated there — it's a dead *network*, not just a dead *layer*).

At odd `T` the layer's output is bit-identical to the `T=1` control; at even `T` the whole
`FCNN_AI2` network's output collapses to a constant (`self.output.bias`), independent of
input, zero gradient everywhere upstream. Currently "safe" only because the one sweep config
that runs this combination uses `T=3` (odd) — nothing guards against someone changing that.

**Recommended fix:** (a) guard `AutapticLayer.__init__` or `build_configs`/`build_size_scan_configs`
against `ai2_variant="ai2_diff"` + `activation_scheme="sigmoid"` at even `T`; (b) drop
`fcnn_ai2_diff_sigmoid` from `sweep.py`'s base grid (and the 5 size-scan equivalents) since
it's a guaranteed duplicate of the control at the only `T` it actually runs at, per `IDEAS.md`
§3's own follow-up #4 — that follow-up only mentions the single `sweep.py` instance; extend it
to `size_scan.py`. **Effort:** small (a guard clause + deleting redundant configs).

### 2. The data protocol structurally suppresses the forgetting signal it's meant to measure

**Where:** `src/data.py` (`STAGE_DIGITS`/`build_stages`, cumulative union, never a
replacement). **Affects:** all three code branches (protocol unchanged throughout).

Old digits are rehearsed every stage, so `backward_transfer` is expected to hover near/above
zero for any adequately-trained architecture, `ai2` or not (confirmed: a quick run gave
`BWT=+0.62`). This directly limits what the sweep can say about the project's stated central
hypothesis (catastrophic forgetting, per `Mini_project_meta_continual_learning.md`), independent
of whether `ai2` works. `IDEAS.md` §2.4 ("Caveats to keep in view") already names this and
proposes a replay-free protocol as follow-up #3 — that's the recommended fix, just not yet
done. **Effort:** medium (a new, replay-free stage-building mode in `data.py`, re-running the
sweep under it).

## High

### 3. `ty check` fails — `tests/test_size_scan.py` Optional-narrowing

**Where:** `tests/test_size_scan.py:59,68,69`, indexing `RunConfig.hidden_sizes`/`.channels`
(typed `tuple[...] | None`) without narrowing. **Affects:** `serious-scan`,
`FCNN-debugging` (same, unfixed). Violates `implementation_doc.md`'s own "Verification"
checklist (`ty check` clean). Per `red-team-FCNN-debugging.md` #4: neither that branch's own
`BUGS.md` nor `IDEAS.md` mentions this failure, despite both being written specifically to
catalog the repo's issues — a gap in their own coverage, not just an open bug. **Recommended
fix:** add `assert config.hidden_sizes is not None` / `cast(...)` at the top of the two
affected test functions. **Effort:** trivial (~3 lines).

## Medium

### 4. Adam→SGD optimizer choice is circular reasoning

**Where:** `docs/lab-book.md`, "Optimizers: Adam vs SGD." Switches to SGD specifically because
it's more likely to show *any* difference, with no control for SGD's general sensitivity vs. a
genuine `ai2` effect. **Recommended fix:** add the already-TODO'd gradient/weight-norm logging
(`lab-book.md:177`) or a non-`ai2` architectural-change control run at the same `lr`, to check
whether SGD-sensitivity alone produces a comparable "effect." **Effort:** medium (new control
run + logging). **Cross-ref:** P2 (same landmine, in the future optimizer-as-config plan).

### 5. Temporal-persistence framing doesn't match the implementation

**Where:** `src/autapse.py:71` (`y_self` reset to zero every `forward()` call) vs.
`Mini_project_meta_continual_learning.md`'s "self-recurrent, self-inhibiting connections" and
the wishlist's "leaky integral of activity." The lab book itself discloses the tradeoff
("Option A" over "Option B: Implement memory") — this isn't a hidden bug, but the project's own
framing docs keep using persistence-implying language for a mechanism that has none.
**Recommended fix:** tighten language in `Mini_project_meta_continual_learning.md` and the
wishlist to describe a per-example fixed-point iteration, not a temporal/dynamical mechanism;
or implement Option B/D (lab-book/IDEAS.md) if real memory is actually wanted. **Effort:**
small (doc edit) or large (if implementing real state).

### 6. `CLAUDE.md`/`WRITING_STYLE.md` untracked from git

**Where:** commits `d1586fc`/`57ef9df`, `.gitignore`'s "Personal additions." **Affects:**
`serious-scan`, `FCNN-debugging`. Self-aware (commit message says "ironically"), but a fresh
clone from `serious-scan` onward has neither file unless recreated locally, and
`WRITING_STYLE.md`'s rules (which the lab book's prose follows) are no longer part of the
shared, reviewable repo history. **Recommended action:** confirm with the repo owner this is
intentional; if so, document the decision explicitly (e.g. a line in `README.md`) rather than
leaving it discoverable only via `.gitignore` archaeology. **Effort:** trivial.

### 7. Silent result overwrites, duplicate MLflow runs on rerun

*(Not detailed in any of the four reports — see the `*` note above. Re-verified directly for
this triage.)* **Where:** `src/run_experiments.py:_write_results` — unconditionally overwrites
`results/{name}.json`, and reruns add a new MLflow run under the same name with no warning.
**Affects:** all three code branches. **Recommended fix:** warn (or refuse) on an existing
results file unless an explicit `--force`/overwrite flag is passed; same for re-running a
config name already present in MLflow. **Effort:** small.

### 8. `results/` gitignored — reported numbers aren't reproducible from the repo alone

*(Not detailed in any of the four reports — see the `*` note above. Re-verified directly for
this triage: `git show main:.gitignore` has both `/results/` at line 86 and `results/*` at line
612, a harmless duplicate entry.)* **Affects:** all three code branches. Numbers quoted in the
lab book / `FCNN-debugging/BUGS.md` can't be checked against the repo without re-running the
sweep. **Recommended action:** either commit a representative results snapshot, or make the
lab book explicit that quoted numbers are not re-derivable from the repo as-is. **Effort:**
small (process decision, not code).

## Needs further testing (investigated, not confirmed either way)

### 9. `lr=0.1` × `ai2_gate`'s non-guaranteed contraction, over a full run

A 300-step fixed-batch smoke test at `lr=0.1` showed no NaNs/divergence for any
`relu_sigmoid` variant, and `ai2_gate`'s output is bounded by construction (can't blow up
numerically). The real open question — oscillation that degrades training *quality* over a
full multi-epoch, multi-stage run with real data drift — is narrower than "may diverge" and
still untested. **Recommended next step:** log pre-activation magnitude / `y_self` curves
across a full real sweep run (the logging already exists —
`dynamics_insights.get_y_self_absmeans`) and check for non-monotonic/oscillating patterns, not
just NaNs.

### 10. `lr=0.1` never validated against the largest (512, 256) hidden-size config

The new global default was checked against base-sized models and short smoke batches only, not
against `sweep.py`'s `hidden_sizes_sweep=[(512, 256)]` entry — the sweep's largest-capacity
FCNN. **Recommended next step:** a short real run of that specific config at `lr=0.1`, checking
for the same failure mode (`lr` too high for a larger model) that motivated this branch's own
investigation in the first place.

## Low / Low-Medium

### 11. Notebook SSL-monkeypatch duplicate, drifted, fails `ty check`

`notebooks/mnist-download.ipynb` copies `src/data.py`'s SSL workaround with a narrower lambda
signature (`lambda:` vs. `lambda *a, **kw:`) and no `# ty: ignore`, plus a cwd-dependent
`root="../data"` (the exact pattern `REPO_ROOT` anchoring elsewhere exists to avoid). Causes 1
of the `ty check` failures on every branch. **Fix:** import the workaround from `data.py`
instead of duplicating it, or add the same `# ty: ignore` + anchor the path. **Effort:**
trivial.

### 12. Shape-only tests mask finding #1's real (value-level) behavior

`tests/test_models.py` exercises `T=2` (the exact degenerate case from #1) but only asserts
output *shape*; `tests/test_sweep.py`'s uniqueness check only catches *name* collisions. Both
pass while giving false confidence that the degenerate case is covered. **Fix:** add a value-
level regression test asserting `AutapticLayer`/`FCNN_AI2` output actually varies with input at
even `T` (or, once #1 is fixed, that it raises/warns). **Effort:** small, bundle with #1's fix.

### 13. Global SSL monkeypatch is a process-wide side effect

*(Not detailed in any of the four reports — see the `*` note above; distinct from row #11,
which is about the notebook's *drifted duplicate* of this same patch. Re-verified directly for
this triage.)* `src/data.py:get_mnist_data` reassigns `ssl._create_default_https_context` process-wide as a
side effect of a data-loading call — affects any other HTTPS context creation for the rest of
the process lifetime (e.g. MLflow's own network calls). **Fix:** scope it (context manager
that restores the previous value afterward) instead of a permanent global reassignment.
**Effort:** small.

### 14. Partial seeding / cross-config RNG sharing (self-disclosed)

Only `torch.manual_seed` is set; `DataLoader` shuffling shares the global RNG, so two configs'
"all else equal" comparison isn't exact beyond weight init + first epoch. Already disclosed in
`docs/lab-book.md` ("Logging: MLflow"), independently flagged "high confidence" in
`BUGS.md` #6 on `FCNN-debugging` (bundled there with finding #19 below, which is a distinct,
later-introduced issue — keep them separate when fixing). **Fix (if pursued):** pass an
explicit `torch.Generator` to each `DataLoader` instead of relying on global RNG state.
**Effort:** small.

### 15. Doc drift: nonexistent method name + stale formula

`docs/implementation_doc.md` references a method `_get_main_output` that doesn't exist (real
helper: `_get_y_activation`); separately, `docs/lab-book.md`'s table row 6 says
`relu(y_raw − tanh(...))` but the code computes `relu(Wx+b) − tanh(...)` (no outer `relu`,
behaviorally equivalent only because the output stays ≥0 — confirmed matches `BUGS.md` #3).
**Fix:** correct both doc passages to match the code. **Effort:** trivial.

### 16. Setup/DX pitfalls

Bundle of three small, independent items: (a) a bare `uv sync` installs everything except
`torch`/`torchvision` (only under optional `cpu`/`cu130` extras) — `import torch` then fails;
this one *is* detailed with full repro in `red-team-main.md` #7. (b) `requires-python =
">=3.14"` differs from this org's standard 3.13; (c) `main.py` is an unused stub
(`print("Hello from robgoal!")`). (b)/(c) are, like rows #7/#8/#13, *not* written up as
standalone findings in any of the four reports — see the `*` note above; re-verified directly
(`pyproject.toml:6`, `main.py`) for this triage. **Fix:** (a) make the README's `--extra=cpu`
requirement louder (e.g. a `uv.lock`-checked CI step, or a clearer error message), (b)/(c)
low-priority cleanup. **Effort:** trivial-small.

### 17. Lab-book checkbox granularity mismatch

`docs/lab-book.md`'s "Focused experiments" marks the parent TODO done while its own child
bullet ("Epoch size") stays unchecked. **Fix:** split into two checkboxes or uncheck the
parent. **Effort:** trivial.

### 18. `backward_transfer` unchecked precondition

`metrics.backward_transfer`'s documented precondition (digits passed in must be stage-0's) is
enforced nowhere in code; currently harmless because every caller happens to pass
`stages[0].digits_seen`. **Fix:** assert the precondition. **Effort:** trivial.

## By design (no action needed unless priorities change)

### 19. Multi-seed `SystemRandom` draws aren't reproducible ahead of time

`run_experiments._draw_seeds` draws non-reproducible seeds via `random.SystemRandom()` for
`num_seeds > 1` (fixed `[0]` for `num_seeds == 1`). Deliberate, per its own docstring, and
recorded after the fact (MLflow param / results filename / stdout print). No action
recommended unless exact pre-run reproducibility of multi-seed sweeps becomes a requirement.

## Status: fix landed, not yet everywhere

### 20. `lr` default fix (`1e-3` → `0.1`) is on `FCNN-debugging` only

Confirmed genuinely fixes the sigmoid control's under-training (reproduced: 0.197 vs. 1.0
accuracy on a toy setup) and is locked in by a regression test on that branch. `main`,
`serious-scan`, and `python-modularisation` (= `serious-scan`) still carry the old `1e-3`
default. **Recommended action:** merge/backport this fix (and ideally findings #3 and #11-18,
which are also only fixed nowhere yet) before running any further sweeps on those branches, to
avoid collecting more under-trained-sigmoid results under the old default.

---

## Future-plan risks — `IDEAS.md` §1 (the `python-modularisation` report's subject)

These describe a *proposed* refactor (options A-H) that hasn't been written yet. They're risks
to address *when* that plan is executed, not bugs in current code.

### P1. Option C would break `REPO_ROOT` anchoring

Moving `src/*.py` to `src/robgoal/` makes `REPO_ROOT = Path(__file__).resolve().parent.parent`
(`src/run_experiments.py:20`, `src/logged_training.py:20`) resolve to `src/`, not the repo
root — silently relocating `results/`, `mlflow.db`/`mlartifacts/`, and the MNIST download dir.
**Action when executing C:** update both to `.parent.parent.parent` in the same commit as the
move.

### P2. Option G would reopen the Adam/SGD risk (cross-ref #4)

Making `optimizer` a free config field risks silently reintroducing the already-diagnosed
"Adam makes ai2 indistinguishable from control" problem (#4) with nothing to warn against it.
**Action when executing G:** carry a comment/warning on the config field pointing back to the
lab-book finding.

### P3. Option H is undersold and missing from the suggested execution order

Benchmarked directly: ~250× per-epoch speedup (2.0s DataLoader+transform vs. 0.008s preloaded-
tensor indexing, after a ~3.15s one-time cost) — much larger than `IDEAS.md`'s "probably...
profile first" hedge suggests, and the only option left out of the document's own "Suggested
order" paragraph with no stated reason. **Action:** move H earlier in the execution order; the
profiling IDEAS.md asks for has effectively already been done.

### P4. CLI/doc migration only partly addressed

Option D explicitly replaces `--scan {sweep,size}` with subcommands, but doesn't mention
`--num-seeds`/`--dataset`/`--initial-classes`/`--classes-per-stage`/`--lr`, or the docs
(`scan-list.md`, README's "Adjusting the sweep") that describe them. **Action when executing
D:** commit to migrating/deleting all of these in the same change, per this repo's
replace-don't-deprecate convention — not just the one switch the plan names.

### P5. Plan's own "roughly independent" vs. "C is a prerequisite" tension

`IDEAS.md` §1 calls the options "roughly independent," then its "Suggested order" makes C a
prerequisite for A/B. Likely just means priority order, not a hard dependency — worth a
one-line clarification in the doc, not a blocker.

### P6. Backward-compat of existing `results/`/MLflow data isn't addressed

Options D (config files), E (`model: str`+`model_args`), and G (`stages=[[...]]`) all change
`RunConfig`'s schema, with no mention of whether historical `results/*.json` / MLflow runs
stay loadable/comparable afterward — and the lab book's own practice depends on that
continuity. **Action when executing any of D/E/G:** either keep the schema change backward-
readable, or explicitly accept/document that historical results become incomparable.
