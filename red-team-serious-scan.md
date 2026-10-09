# Red-team: `serious-scan`

Scope: incremental diff from `main` only (`src/size_scan.py` new; `src/data.py`,
`src/run_experiments.py`, `src/run_config.py`, `src/models.py`, `src/sweep.py` extended for
seed/size/dataset/class-split scanning; new tests; `scan-list.md`). See `red-team-main.md` for
baseline issues inherited unchanged from `main` (notably: the `ai2_diff`+`sigmoid`
identity/collapse bug, and the cumulative-replay/BWT concern — both still present and, per
below, amplified by this branch's new code).

Verified directly in this checkout: `uv sync --extra cpu` (torch installs fine),
`uv run pytest -q` → **78 passed**, `uv run ruff check .` → **clean**. Findings below were
independently re-run by an adversarial subagent, which confirmed all five and reproduced the
same tool output.

## 1. [HIGH, verified] `ty check` is not clean — new Optional-narrowing errors

`uv run ty check` reports 3 errors introduced by `tests/test_size_scan.py` (plus the
pre-existing `main`-branch notebook error, see `red-team-main.md` #5):

```
error[invalid-argument-type]  tests/test_size_scan.py:59  list(config.hidden_sizes)
error[not-subscriptable]      tests/test_size_scan.py:68  config.channels[0]
error[not-subscriptable]      tests/test_size_scan.py:69  config.channels[1]
```

`RunConfig.hidden_sizes`/`.channels` are typed `tuple[...] | None`
(`src/run_config.py`). The test's own list-comprehension filters (`if c.model_kind == "fcnn"`
/ `"cnn"`) guarantee non-`None` at runtime, but nothing narrows the type for `ty`, which can't
see that guarantee. `docs/implementation_doc.md`'s own "Verification" checklist lists `ty
check` clean as a completion criterion — this branch doesn't meet it. Trivial fix (an
`assert ... is not None` or `cast` per test), currently unaddressed.

## 2. [MEDIUM, academic — amplifies a `main`-branch bug] Size scan repeats the identical-config pair 5×

`size_scan.py`'s `_build_fcnn_ai2_grid_step` (called once per size step inside
`build_size_scan_configs`'s loop) builds the full `activation_scheme`×`ai2_variant` grid at
`T=sweep.BASE_T` (3, odd) for *every* one of the 5 size steps — including
`ai2_variant="ai2_diff"` + `activation_scheme="sigmoid"` each time. Per `red-team-main.md` #1,
that combination at odd `T` is mathematically identical to the corresponding `T=1` control.
So this branch doesn't just inherit the one pre-existing duplicate-run instance from
`sweep.py` — it adds 5 more (one per size step), all guaranteed-identical pairs dressed up as
35 distinct configs.

## 3. [MEDIUM, housekeeping — self-aware tradeoff, flagged for completeness] `CLAUDE.md`/`WRITING_STYLE.md` untracked

Commits `d1586fc` ("Add LLM contexts to repo and gitignore, ironically") and `57ef9df`
("Untrack ignored files") — both ancestors of this branch — moved `CLAUDE.md`/`WRITING_STYLE.md`
(and `CODING_STYLE.md`) into `.gitignore` under a `# Personal additions` comment and removed
them from tracking. Confirmed still untracked at this branch's HEAD
(`git ls-tree serious-scan -r` has no hits for either file). Not a hidden bug — the commit
message names the irony itself — but worth recording: a fresh clone from `serious-scan`
onward has neither file unless recreated locally, and `WRITING_STYLE.md`'s rules (which the
lab book's own prose follows) are no longer part of the shared, reviewable repo history.

## 4. [LOW, verified] Multi-seed replicate seeds aren't reproducible ahead of time

`run_experiments._draw_seeds` (introduced on this branch, `0398235`) returns the fixed `[0]`
for `num_seeds == 1`, but draws from `random.SystemRandom()` for `num_seeds > 1` — so a
multi-seed sweep's actual seeds can't be reproduced by rerunning the same command; they're
only recoverable after the fact (`seed` MLflow param / results filename / stdout print). A
documented, deliberate tradeoff (per the function's own docstring), not a hidden defect —
noting it here since this is where it originates (not, as its downstream appearance in
`FCNN-debugging/BUGS.md` might suggest, a `FCNN-debugging`-branch issue).

## 5. [LOW, verified] Lab-book checkbox granularity mismatch

`docs/lab-book.md`'s "Focused experiments" section marks the parent TODO `- [x] Compare
networks of increasingly smaller...` done, while its own child bullet `- [ ] Epoch size (10, 5,
2) -- not part of the new size scan, see above.` stays unchecked. Not a code bug — just liable
to mislead someone skimming the lab book's status for what's actually finished.
