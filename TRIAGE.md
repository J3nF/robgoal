# Triage: all findings across `red-team-{main,serious-scan,FCNN-debugging,python-modularisation}.md`

Consolidates and deduplicates every finding from the four branch red-team reports into one
prioritized backlog. Evidence stays in the source reports; each entry here gives the decision
(severity, fix, effort) plus any evidence that appears nowhere else. The summary table is at the
end.

Branches are stacked (`main` → `serious-scan` → `FCNN-debugging` / `python-modularisation`), so
most issues originate once and are inherited; each entry lists where it *originates* and every
branch it *affects*. "All 3 code branches" means `main`, `serious-scan` and `FCNN-debugging`.
`python-modularisation` has no code of its own: it sits at `43c53dd`, the commit `serious-scan`
had when the reports were written (`serious-scan` has since gained `7f11486`, docs and style
files only). Its report therefore covers a *proposed future refactor* (`IDEAS.md` §1) rather
than current code. Those findings are kept in their own section at the end, since "fixing" them
means writing the refactor correctly when it happens, not patching code that exists today.

Severity scale: **Critical** (undermines the project's core claims or produces silently wrong
results) → **High** (violates an explicit project policy, cheap to fix, currently violated) →
**Medium** → **Low/Low-Medium** (cosmetic, doc drift, DX). A separate **Needs further
testing** bucket holds items that were investigated but not confirmed either way — triage
these as "run the missing experiment", not as "fix". Row numbers are cited from other
documents; they stay fixed even where an entry's status changes.

Source of rows #7, #8, #13 and #16(b)/(c): they were not in the four original reports. They
are now written up in `red-team-main.md` (#8, #9, #10 and #7), were listed earlier in
`FCNN-debugging`'s `BUGS.md` (#7-#10), and are confirmed in the Cross-check of
`red-team-FCNN-debugging.md`. Line numbers below are given per branch where they differ.

---

## Critical

### 1. `ai2_diff` + `sigmoid` degenerates to a duplicate run or a dead, constant network

**Evidence:** `red-team-main.md` #1 (derivation), `red-team-FCNN-debugging.md` #3 (dead
*network*, not just a dead layer as `BUGS.md` #2 says), `red-team-serious-scan.md` #2.
**Where:** `AutapticLayer._step`/`_get_inhibited_output` (`src/autapse.py:103-121`),
`FCNN_AI2.forward` (`src/models.py:36-40`, no skip connection), base grid `src/sweep.py:71-88`,
amplified ×5 by `src/size_scan.py`'s `_build_fcnn_ai2_grid_step` (one extra identical pair per
size-scan step). **Affects:** `main`; `serious-scan` (5× amplified); `FCNN-debugging`.

At odd `T` the layer's output is bit-identical to the `T=1` control; at even `T` the whole
`FCNN_AI2` network's output collapses to a constant (`self.output.bias`), independent of
input, with zero gradient upstream. Currently "safe" only because the one sweep config that
runs this combination uses `T=3` (odd) — nothing guards against someone changing that.

**Recommended fix:** (a) guard `AutapticLayer.__init__` or `build_configs`/`build_size_scan_configs`
against `ai2_variant="ai2_diff"` + `activation_scheme="sigmoid"` at even `T`; (b) drop
`fcnn_ai2_diff_sigmoid` from `sweep.py`'s base grid (and the 5 size-scan equivalents) since
it's a guaranteed duplicate of the control at the only `T` it actually runs at, per `IDEAS.md`
§3's own follow-up #4 — that follow-up only mentions the single `sweep.py` instance; extend it
to `size_scan.py`. **Effort:** small (a guard clause + deleting redundant configs).

### 2. The data protocol structurally suppresses the forgetting signal it's meant to measure

**Evidence:** `red-team-main.md` #2. **Where:** `src/data.py` (`STAGE_DIGITS`/`build_stages`,
cumulative union, never a replacement). **Affects:** all 3 code branches (protocol unchanged
throughout).

Old digits are rehearsed every stage, so `backward_transfer` is expected to hover near/above
zero for any adequately-trained architecture, `ai2` or not (the one confirming run, `BWT=+0.62`,
is n=1). This limits what the sweep can say about the project's stated central hypothesis
(catastrophic forgetting, per `Mini_project_meta_continual_learning.md`), independent of
whether `ai2` works. `IDEAS.md` §2.4 ("Caveats to keep in view") already names this and
proposes a replay-free protocol as follow-up #3 — that's the recommended fix, just not yet
done. **Effort:** medium (a new, replay-free stage-building mode in `data.py`, re-running the
sweep under it).

## High

### 3. `ty check` fails — `tests/test_size_scan.py` Optional-narrowing

**Evidence:** `red-team-serious-scan.md` #1, `red-team-FCNN-debugging.md` #4.
**Where:** `tests/test_size_scan.py:59,68,69`, indexing `RunConfig.hidden_sizes`/`.channels`
(typed `tuple[...] | None`) without narrowing. **Affects:** `serious-scan`, `FCNN-debugging`
(same, unfixed). Violates `implementation_doc.md`'s own "Verification" checklist (`ty check`
clean). Neither `FCNN-debugging`'s `BUGS.md` nor `IDEAS.md` mentions this failure, despite both
being written to catalog the repo's issues — a gap in their coverage, not just an open bug.
**Recommended fix:** add `assert config.hidden_sizes is not None` / `cast(...)` at the top of
the two affected test functions. **Effort:** trivial (~3 lines).

## Medium

### 4. Adam→SGD optimizer choice is circular reasoning

**Evidence:** `red-team-main.md` #3. **Where:** `docs/old-lab-book.md`, "Optimizers: Adam vs SGD."
**Recommended fix:** add the already-TODO'd gradient/weight-norm logging (`old-lab-book.md:177`) or
a non-`ai2` architectural-change control run at the same `lr`, to check whether
SGD-sensitivity alone produces a comparable "effect." **Effort:** medium (new control run +
logging). **Cross-ref:** P2 (same landmine, in the future optimizer-as-config plan).

### 5. Temporal-persistence framing doesn't match the implementation

**Evidence:** `red-team-main.md` #4. **Where:** `src/autapse.py:71` (`y_self` reset to zero
every `forward()` call) vs. `Mini_project_meta_continual_learning.md`'s "self-recurrent,
self-inhibiting connections" and the wishlist's "leaky integral of activity." The lab book
discloses the tradeoff ("Option A" over "Option B: Implement memory"), so this isn't a hidden
bug; the framing docs just keep using persistence-implying language for a mechanism that has
none. **Recommended fix:** tighten language in `Mini_project_meta_continual_learning.md` and
the wishlist to describe a per-example fixed-point iteration, not a temporal/dynamical
mechanism; or implement Option B/D (old-lab-book.md/`IDEAS.md`) if real memory is wanted.
**Effort:** small (doc edit) or large (if implementing real state).

### 6. `CLAUDE.md`/`WRITING_STYLE.md` untracked from git — fixed on `serious-scan`, open elsewhere

**Evidence:** `red-team-serious-scan.md` #3. **Where:** commits `d1586fc`/`57ef9df`,
`.gitignore`'s "Personal additions." **Status:** `7f11486` re-tracked `CLAUDE.md`,
`WRITING_STYLE.md` and `CODING_STYLE.md` and removed the ignore block on `serious-scan`.
`FCNN-debugging` and `python-modularisation` (base `43c53dd`) still ignore and do not track
them (`.gitignore:623-626`). **Recommended action:** carry `7f11486` to those branches by
merge. **Effort:** trivial.

### 7. Silent result overwrites, duplicate MLflow runs on rerun

**Status (branch `pipeline-infrastructure`): the overwrite is solved by construction; duplicate MLflow
runs are not.** Each launch writes to its own `results/{launch_id}/` and `run_store.write_run_file`
raises on an existing file, so a rerun can never overwrite a result (nothing writes flat
`results/*.json` any more). The same code attributes every file to a commit, launch and class
split. **Still open:** every launch (and every retry of a failed job) creates new MLflow runs under
the same run name; MLflow is a viewer here, so that is by design, but the viewer will show
duplicates (tag `launch_id`/`run_id` tell them apart).

**Evidence:** `red-team-main.md` #8; `BUGS.md` #7. **Where (original):** `_write_results`
(`src/run_experiments.py`: line 81 on `main`, 195 on `serious-scan`, 210 on `FCNN-debugging`)
unconditionally overwrote `results/{name}.json`; `mlflow.start_run` (`src/logged_training.py:53,253`)
adds a new MLflow run under the same name on each rerun, with no warning. **Affects:** all 3 code
branches. **Original recommended fix:** warn (or refuse) on an existing results file; same for
MLflow. **Effort:** small.

### 8. `results/` gitignored — reported numbers aren't reproducible from the repo alone

**Status (branch `pipeline-infrastructure`): partly mitigated.** `results/` is still gitignored,
so the numbers cannot be re-derived from the repo alone. What changed: every run file now records
the commit (and branch, config, class split, seed, versions) that produced it, refuses to be
written from a dirty tree, and the launch's `docs/lab-book.md` entry says why it was run, so a
number is attributable to a code state and can be reproduced by checking that commit out and
re-running. The 574 pre-pipeline files were imported with `legacy: true`; their commit is
unknown for the 560 size-scan files (the scan ran on a dirty tree). **Still open:** committing a
results snapshot (or at least `scan_summary.csv`), and the per-epoch `train_loss`/`y_self`
histories of the legacy runs exist only in the gitignored `mlflow.db`/`mlartifacts/`.

**Evidence:** `red-team-main.md` #9; `BUGS.md` #8. **Where:** `.gitignore:86` (`/results/`)
and `:612` (`results/*`), a harmless duplicate, identical on all 3 code branches. Numbers
quoted in the lab book / `FCNN-debugging/BUGS.md` can't be checked against the repo without
re-running the sweep. **Recommended action:** either commit a representative results snapshot,
or make the lab book explicit that quoted numbers are not re-derivable from the repo as-is.
**Effort:** small (process decision, not code).

## Needs further testing (investigated, not confirmed either way)

### 9. `lr=0.1` × `ai2_gate`'s non-guaranteed contraction, over a full run

**Evidence:** `red-team-FCNN-debugging.md` #2. A 300-step fixed-batch smoke test at `lr=0.1`
showed no NaNs/divergence for any `relu_sigmoid` variant, and `ai2_gate`'s output is bounded by
construction. The open question is oscillation that degrades training *quality* over a full
multi-epoch, multi-stage run. **Recommended next step:** log pre-activation magnitude / `y_self`
curves across a full real sweep run (the logging already exists —
`dynamics_insights.get_y_self_absmeans`) and check for non-monotonic/oscillating patterns, not
just NaNs.

### 10. `lr=0.1` never validated against the largest (512, 256) config

**Evidence:** `red-team-FCNN-debugging.md` #6. The new global default was checked against
base-sized models and short smoke batches only, not against `sweep.py`'s
`hidden_sizes_sweep=[(512, 256)]` entry, the sweep's largest-capacity FCNN. **Recommended next
step:** a short real run of that specific config at `lr=0.1`, checking for the failure mode
(`lr` too high for a larger model) that motivated this branch's own investigation.

## Low / Low-Medium

### 11. Notebook SSL-monkeypatch duplicate, drifted, fails `ty check`

**Evidence:** `red-team-main.md` #5. `notebooks/mnist-download.ipynb` copies `src/data.py`'s
SSL workaround with a narrower lambda signature (`lambda:` vs. `lambda *a, **kw:`), no
`# ty: ignore`, and a cwd-dependent `root="../data"` (the pattern `REPO_ROOT` anchoring
elsewhere exists to avoid). Causes 1 of the `ty check` failures on every branch. **Fix:**
import the workaround from `data.py` instead of duplicating it, or add the same `# ty: ignore`
and anchor the path. **Effort:** trivial.

### 12. Shape-only tests mask finding #1's real (value-level) behavior

**Evidence:** `red-team-main.md` #1 (the uniqueness check) and its "Additional, lower-priority
observations" (the `T=2` test). `tests/test_models.py`
exercises `T=2` (the degenerate case from #1) but only asserts output *shape*;
`tests/test_sweep.py`'s uniqueness check only catches *name* collisions. **Fix:** add a
value-level regression test asserting `AutapticLayer`/`FCNN_AI2` output actually varies with
input at even `T` (or, once #1 is fixed, that it raises/warns). **Effort:** small, bundle with
#1's fix.

### 13. Global SSL monkeypatch is a process-wide side effect

**Evidence:** `red-team-main.md` #10; `BUGS.md` #9. Distinct from #11, which is the notebook's
*drifted duplicate* of this same patch. **Where:** `get_mnist_data`
(`src/data.py:56` on `main`, `:94` on `serious-scan`/`FCNN-debugging`) reassigns
`ssl._create_default_https_context` for the rest of the process lifetime. **Fix:** scope it
(a context manager that restores the previous value afterward) instead of a permanent global
reassignment. **Effort:** small.

### 14. Partial seeding / cross-config RNG sharing (self-disclosed)

Only `torch.manual_seed` is set; `DataLoader` shuffling shares the global RNG, so two configs'
"all else equal" comparison isn't exact beyond weight init + first epoch. Disclosed in
`docs/old-lab-book.md` ("Logging: MLflow"), and flagged "high confidence" in `BUGS.md` #6 on
`FCNN-debugging`, which bundles it with finding #19 below — a distinct, later-introduced issue;
keep them separate when fixing. **Fix (if pursued):** pass an explicit `torch.Generator` to each
`DataLoader` instead of relying on global RNG state. **Effort:** small.

### 15. Doc drift: nonexistent method name + stale formula

**Evidence:** `red-team-main.md` #6; `red-team-FCNN-debugging.md` Cross-check (formula).
`docs/implementation_doc.md` references a method `_get_main_output` that doesn't exist (real
helper: `_get_y_activation`). Separately, `docs/old-lab-book.md`'s table row 6 says
$\mathrm{relu}(y_\text{raw} - \tanh(\cdot))$ but the code computes
$\mathrm{relu}(Wx + b) - \tanh(\cdot)$ (no outer $\mathrm{relu}$), behaviorally equivalent only
because the output stays $\ge 0$ (`BUGS.md` #3). **Fix:** correct both doc passages to match the
code. **Effort:** trivial.

### 16. Setup/DX pitfalls

**Evidence:** `red-team-main.md` #7. Three small, independent items: (a) a bare `uv sync`
installs everything except `torch`/`torchvision` (only under optional `cpu`/`cu130` extras) —
`import torch` then fails; (b) `pyproject.toml:6` sets `requires-python = ">=3.14"`, versus the
3.13 in the global standards; (c) `main.py` is an unused stub (`print("Hello from robgoal!")`).
**Fix:** (a) make the README's `--extra=cpu` requirement louder (e.g. a `uv.lock`-checked CI
step, or a clearer error message); (b)/(c) low-priority cleanup. **Effort:** trivial-small.

### 17. Lab-book checkbox granularity mismatch

**Evidence:** `red-team-serious-scan.md` #5. `docs/old-lab-book.md`'s "Focused experiments" marks
the parent TODO done while its own child bullet ("Epoch size") stays unchecked. **Fix:** split
into two checkboxes or uncheck the parent. **Effort:** trivial.

### 18. `backward_transfer` unchecked precondition

**Evidence:** `red-team-main.md`, "Additional, lower-priority observations."
`metrics.backward_transfer`'s documented precondition (digits passed in must be stage-0's) is
enforced nowhere in code; currently harmless because every caller happens to pass
`stages[0].digits_seen`. **Fix:** assert the precondition. **Effort:** trivial.

## By design (no action needed unless priorities change)

### 19. Multi-seed `SystemRandom` draws aren't reproducible ahead of time

**Evidence:** `red-team-serious-scan.md` #4. `run_experiments._draw_seeds` draws
non-reproducible seeds via `random.SystemRandom()` for `num_seeds > 1` (fixed `[0]` for
`num_seeds == 1`). Deliberate, per its own docstring, and recorded after the fact (MLflow param
/ results filename / stdout print). No action recommended unless exact pre-run reproducibility
of multi-seed sweeps becomes a requirement. Since `pipeline-infrastructure` the drawn seeds are
also stored in every run file (`meta["config"]["seed"]`) and are drawn once, in the launching
process, so a retried job reruns the same seeds.

## Status: fix landed, not yet everywhere

### 20. `lr` default fix (`1e-3` → `0.1`) is on `FCNN-debugging` only

**Evidence:** `red-team-FCNN-debugging.md` #1. Genuinely fixes the sigmoid control's
under-training (reproduced: 0.197 vs. 1.0 accuracy on a toy setup) and is locked in by a
regression test on that branch. `main`, `serious-scan`, and `python-modularisation` still carry
the old `1e-3` default. **Recommended action:** merge/backport this fix before running any
further sweeps on those branches, to avoid collecting more under-trained-sigmoid results under
the old default. Findings #3 and #11-#18 are not fixed on any branch either; backport them
alongside.

---

## Future-plan risks — `IDEAS.md` §1 (the `python-modularisation` report's subject)

These describe a *proposed* refactor (options A-H) that hasn't been written yet. They're risks
to address *when* that plan is executed, not bugs in current code. Evidence for all of them is
in `red-team-python-modularisation.md`, under the same numbers (P1 = #1, … P6 = #6).

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

Benchmarked: ~250× per-epoch speedup of the data path (2.0s DataLoader+transform vs. 0.008s
preloaded-tensor indexing, after a ~3.15s one-time cost) — much larger than `IDEAS.md`'s
"probably... profile first" hedge suggests, and H is the only option left out of the
document's own "Suggested order" paragraph, with no stated reason. The timing covers the data
path only, so the end-to-end gain is smaller than 250× and unmeasured. **Action:** move H
earlier in the execution order; the data-path profiling `IDEAS.md` asks for has been done, a
full-run timing has not.

### P4. CLI/doc migration only partly addressed

Option D explicitly replaces `--scan {sweep,size}` with subcommands, but doesn't mention
`--num-seeds`/`--dataset`/`--initial-classes`/`--classes-per-stage`/`--lr`, or the docs
(`scan-list.md`, README's "Adjusting the sweep") that describe them. **Action when executing
D:** commit to migrating/deleting all of these in the same change, per this repo's
replace-don't-deprecate convention — not just the one switch the plan names.

### P5. Plan's own "roughly independent" vs. "C, then A and B" tension

`IDEAS.md` §1 calls the options "roughly independent," then its "Suggested order" reads "C,
then A and B (share one `RunConfig`)." Read as a hard dependency that conflicts with
"independent"; read as priority order it doesn't. The report calls this a fair but not airtight
reading. Worth a one-line clarification in the doc, not a blocker.

### P6. Backward-compat of existing `results/`/MLflow data isn't addressed

Options D (config files), E (`model: str`+`model_args`), and G (`stages=[[...]]`) all change
`RunConfig`'s schema, with no mention of whether historical `results/*.json` / MLflow runs
stay loadable/comparable afterward — and the lab book's own practice depends on that
continuity. The source report added this after its verification pass. **Action when executing
any of D/E/G:** either keep the schema change backward-readable, or explicitly accept/document
that historical results become incomparable.

---

## Summary table

| # | Title | Severity | Category | Origin | Affects |
|---|---|---|---|---|---|
| 1 | `ai2_diff`+`sigmoid` degenerate (duplicate/dead network) | Critical | implementation+academic | main | all 3 code branches |
| 2 | Cumulative replay suppresses the BWT forgetting signal | Critical | academic | main | all 3 code branches |
| 3 | `ty check` fails: `test_size_scan.py` Optional-narrowing | High | implementation | serious-scan | serious-scan, FCNN-debugging |
| 4 | Adam→SGD optimizer choice is circular reasoning | Medium | academic | main | all 3 code branches |
| 5 | Temporal-persistence framing doesn't match the implementation | Medium | academic | main | all 3 code branches |
| 6 | `CLAUDE.md`/`WRITING_STYLE.md` untracked from git | Medium, fixed on serious-scan (`7f11486`) | technical debt | serious-scan | FCNN-debugging, python-modularisation |
| 7 | Silent result overwrites, duplicate MLflow runs on rerun | Medium | technical debt | main | all 3 code branches |
| 8 | `results/` gitignored — numbers not reproducible from repo | Medium | technical debt | main | all 3 code branches |
| 9 | `lr=0.1` × `ai2_gate` instability over full runs | *Needs testing* | academic | FCNN-debugging | FCNN-debugging |
| 10 | `lr=0.1` never validated at the largest (512,256) config | *Needs testing* | implementation | FCNN-debugging | FCNN-debugging |
| 11 | Notebook SSL-monkeypatch duplicate, drifted, fails `ty check` | Low-Medium | implementation | main | all 3 code branches |
| 12 | Shape-only tests mask #1's real (value-level) behavior | Low-Medium | test coverage | main | all 3 code branches |
| 13 | Global SSL monkeypatch is a process-wide side effect | Low-Medium | implementation | main | all 3 code branches |
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
| P5 | Plan: "roughly independent" vs. "C, then A and B" tension | Low | plan risk | ″ | — |
| P6 | Plan: existing `results/`/MLflow schema compatibility unaddressed | Medium-if-executed | plan risk | ″ | — |
