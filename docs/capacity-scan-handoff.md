# Capacity-overwhelming size scan -- handoff

Status as of 2026-10-14, branch `serious-scan`. For a future agent/session picking this up.

> **Partly superseded (2026-10-14, branch `pipeline-infrastructure`).** The parts marked
> *Superseded* below describe the pre-pipeline state: results now live in
> `results/{launch_id}/*.h5` (see `implementation_doc.md`, "Run provenance and storage"), launches
> are logged in `docs/lab-book.md`, `--workers N` replaces the throwaway driver scripts, and
> `plot_size_scan.py` reads the `.h5` files. The 574 old `results/*.json` files were imported
> into `results/legacy-import/`.

## What this was

Goal: plot accuracy (and backward_transfer) vs. network size, small enough to overwhelm a
10-class classifier's capacity, across 5 seeds, both datasets, class-incremental starting from
1 class. See `scan-list.md` §5 for the exact commands.

## Code changes (committed)

- `src/run_config.py`, `src/train.py`: `lr` default `1e-3` -> `0.1` (TRIAGE.md #20 -- avoids an
  under-trained-sigmoid-control confound).
- `src/size_scan.py`: `FCNN_HIDDEN_SIZES_SCAN`/`CNN_CHANNELS_SCAN` extended 5 -> 8 steps, down to
  FCNN `(2, 1)` / CNN `(1, 1)`.
- `src/plot_size_scan.py` (new; *superseded*: it now reads the `.h5` run files and is a thin script on `plotting_utils.py`): reads `results/*.json`, aggregates mean/std accuracy and
  backward_transfer per config across seeds, writes `results/plots/{accuracy,backward_transfer}
  _vs_size.png` + `scan_summary.csv`. Run with `uv run python src/plot_size_scan.py` after the
  scan finishes.
- `tests/test_size_scan.py`, `docs/old-lab-book.md`, `docs/implementation_doc.md`, `scan-list.md`:
  updated for the 8-step count / new recipe.
- All committed, `ruff check` / `pytest` clean. `ty check` has the same 2 pre-existing failures
  as before this work (TRIAGE.md #3, #11) -- untouched, out of scope.

## Bug found and fixed -- NOT committed, may recur on a fresh clone (*mitigated*: the launcher now calls `logged_training.configure` before starting and fails fast on a bad artifact location, and an MLflow error can no longer lose results because the `.h5` is written first)

The local (gitignored) `mlflow.db` had a stale `ai2-mnist` experiment with
`artifact_location='file:/home/jens/Documents/Repos/robgoal/mlartifacts'` -- a path from a
different machine. Every run trained to completion and then crashed in the final
`mlflow.log_text` call, silently writing **zero** results. Fixed in this environment's
`mlflow.db` via direct SQL (`UPDATE experiments SET artifact_location = 'file:/workspace/
mlartifacts' WHERE name = 'ai2-mnist'`); backup at `mlflow.db.bak`.

**This fix lives only in this environment's local `mlflow.db`/`mlflow.db.bak`, both gitignored.**
A fresh clone (or deleting/regenerating `mlflow.db`) won't carry it. If a scan run silently
produces zero result files (this no longer loses `.h5` results, but MLflow errors are reported), check for this first: `sqlite3 mlflow.db "SELECT name,
artifact_location FROM experiments"` (or the Python `sqlite3` module -- the `sqlite3` CLI isn't
installed in this environment) and confirm the path exists and is writable from the current
machine/user. Not written up as a TRIAGE.md entry since it's local state, not a code bug --
worth doing if this keeps recurring across environments.

## How the actual 560-run scan was executed -- NOT committed, ephemeral (*superseded*: `run_experiments.py --workers N` does this, with spawn, one torch thread per worker, one retry and a failure report; MLflow's SQLite busy timeout is 20 s hard-coded, only the connection timeout is widened)

`run_experiments.py --scan size --num-seeds 5` already does this correctly, but sequentially --
measured ~8.5 min for the largest config's single seed, model-size-dependent (the smallest config
took ~5.5 min for all 5 seeds). Running it as designed (two sequential `--dataset` invocations)
would take a long time on this box's 16 CPU cores, so it was fanned out across processes instead,
via two scratch scripts that were deliberately **not** added to the repo (one-off execution
tooling, not a feature):

- `run_scan_worker.py`: runs one `(dataset, config_index)`'s 5 seed replicates (reuses
  `run_experiments._run_config` directly), pinned to 1 torch thread, with the sqlite busy-timeout
  widened to 60s.
- `dispatch_scan.py`: fans 112 `(dataset, config_index)` jobs out across 12 concurrent processes,
  2 attempts each, logs each job to its own file.

Both lived under this session's scratchpad (ephemeral, not in `/workspace`). If this scan needs
to be re-run at scale again, recreate an equivalent driver rather than looking for these files --
they're gone. The underlying reusable logic is all in the committed `src/` modules; only the
parallel-fan-out wrapper was throwaway.

## Results (local only, gitignored -- not reproducible from git alone; *superseded layout*: the 560 files were imported to `results/legacy-import/*.h5` with commit unknown, since the scan ran on a dirty tree)

- `results/sizescan_step{0-7}_*_seed*.json` -- 560 files (280 mnist + 280 fashion_mnist), 5 seeds
  each, all from this one scan run.
- `results/plots/accuracy_vs_size.png`, `backward_transfer_vs_size.png`, `scan_summary.csv` --
  generated by `plot_size_scan.py` from the above.
- TRIAGE.md #8 already flags that `results/` isn't git-reproducible; same applies here -- these
  560 runs aren't re-derivable from the repo without re-running the scan (~50 min parallelized,
  ~79 CPU-hours sequential, per the committed 8-step scan lists).

## Findings (see the plots/CSV for the real numbers)

- CNN control degrades far more gracefully than any FCNN family: still ~70-74% accurate at 520
  params, where every FCNN family (ai2 and control alike) is down to ~25-37%.
- Within FCNN, ai2 variants and controls track each other closely at every size on both
  datasets -- no ai2 variant shows a capacity-robustness advantage over control in this scan.
  At the smallest FCNN size, `fcnn_control_sigmoid` (a control) is actually the best performer.
- Caveat (TRIAGE.md #2, unresolved): the training protocol is cumulative (old classes replayed
  every stage), which is why `backward_transfer` stays near 0 for well-trained configs -- this
  scan speaks to final accuracy under shrinking capacity, not to forgetting specifically.

## Open follow-ups, not done here

- Committing a results snapshot (or at least `scan_summary.csv`) so these numbers are
  git-reproducible -- currently they aren't (TRIAGE.md #8).
- If ai2's capacity-robustness hypothesis is to be tested properly, the replay-free protocol
  TRIAGE.md #2 recommends would need to exist first -- this scan's cumulative protocol can't
  cleanly separate "ai2 forgets less" from "ai2 and control both just rehearse."
