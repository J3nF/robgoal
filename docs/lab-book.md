# Lab book

Run log: one entry per launch of `src/run_experiments.py`, appended automatically by
`run_store.begin_launch` (append-only; never edit past entries). Each entry is the launch's
`launch_id`, its provenance, and the `--goal`, `--method` and `--code-changes` text given at
launch. The research reasoning lives in `old-lab-book.md`.

## legacy-import

- **Started:** 2026-10-14T13:36:12+00:00
- **Branch:** pipeline-infrastructure
- **Commit:** cc6136233bbc83646019fd268a326bc6c6b54da6
- **Scan:** legacy
- **Planned runs:** 574
- **Dataset:** mnist + fashion_mnist
- **Seeds:** as recorded in file names
- **Results:** `results/legacy-import/`

Command:

```sh
legacy import script (throwaway, scratchpad), PYTHONPATH=src
```

### Goal

import pre-pipeline results

### Method

One-off script outside the repo (not committed): rebuilt each results/*.json file's config from its name with sweep.build_configs() / size_scan.build_size_scan_configs(), converted the string-digit accuracy dicts to the (stage, digit) `accuracy` array, computed metrics with metrics.py, and wrote one HDF5 file per JSON into results/legacy-import/ via run_store. Originals in results/*.json are untouched. Every file's meta_json has `legacy: true` and an `inferred_fields` list.

### Code changes

n/a

### Notes

Lost in the import: per-epoch `train_loss` and `y_self` histories of these runs exist only in the gitignored `mlflow.db` / `mlartifacts/`, which are the only copy and must not be deleted. The commit and `initial_classes` of legacy runs are inferred or unknown: the 14 flat files (2026-08-25) use commit b5b169174cd004feafab0149cf756321c2896a39 (from MLflow tags) and lr=0.001; the 560 size-scan files (2026-10-13) have commit null (the scan ran on a dirty tree) and lr=0.1; initial_classes was inferred from the stage count (10 stages -> 1, 6 -> 5) with classes_per_stage=1 assumed. Environment fields (hostname, versions, device) and train_seconds are unknown.
