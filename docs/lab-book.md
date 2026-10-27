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

Import pre-pipeline results.

### Method

One-off script outside the repo (not committed): rebuilt each results/\*.json file's config from its name with sweep.build_configs() / size_scan.build_size_scan_configs(), converted the string-digit accuracy dicts to the (stage, digit) `accuracy` array, computed metrics with metrics.py, and wrote one HDF5 file per JSON into results/legacy-import/ via run_store.
Originals in results/*.json are untouched.
Every file's meta_json has `legacy: true` and an `inferred_fields` list.

### Code changes

n/a

### Notes

Lost in the import: per-epoch `train_loss` and `y_self` histories of these runs exist only in the gitignored `mlflow.db` / `mlartifacts/`, which are the only copy and must not be deleted.
The commit and `initial_classes` of legacy runs are inferred or unknown: the 14 flat files (2026-08-25) use commit b5b169174cd004feafab0149cf756321c2896a39 (from MLflow tags) and lr=0.001; the 560 size-scan files (2026-10-13) have commit null (the scan ran on a dirty tree) and lr=0.1; initial_classes was inferred from the stage count (10 stages -> 1, 6 -> 5) with classes_per_stage=1 assumed.
Environment fields (hostname, versions, device) and train_seconds are unknown.

## 2026-10-27-03-54_5254e48

- **Started:** 2026-10-27T03:54:51+00:00
- **Branch:** alternation-studies
- **Commit:** 5254e48232edb837264cc58e70339efb724a1e51
- **Scan:** weights
- **Planned runs:** 140
- **Dataset:** mnist
- **Seeds:** 5 random seeds per config
- **Initial classes / classes per stage:** 1 / 1
- **Workers:** 12
- **MLflow:** off
- **Results:** `results/2026-10-27-03-54_5254e48/`

Command:

```sh
python src/run_experiments.py --scan weights --dataset mnist --initial-classes 1 --classes-per-stage 1 --num-seeds 5 --workers 12 --no-mlflow --goal 'Check whether ai2 inhibition and output switch each other on and off across unroll steps (Q1), and whether activity or input-dependent signal shrinks with depth because of ai2 versus the w=0 control (Q2).' --method '28 configs (2 variants x 2 schemes x 7 ai2_weight values), T=5, four hidden layers of width 32, 5 epochs per stage, 1 initial class and 1 class per stage, 5 random seeds, MNIST.' --code-changes 'ai2_weight in AutapticLayer/FCNN_AI2/RunConfig, per-epoch accuracy array (epoch, digit), unroll trace arrays stored per run, --scan weights, weight-scan plotting.'
```

### Goal

Check whether ai2 inhibition and output switch each other on and off across unroll steps (Q1), and whether activity or input-dependent signal shrinks with depth because of ai2 versus the w=0 control (Q2).

### Method

28 configs (2 variants x 2 schemes x 7 ai2_weight values), T=5, four hidden layers of width 32, 5 epochs per stage, 1 initial class and 1 class per stage, 5 random seeds, MNIST.

### Code changes

ai2_weight in AutapticLayer/FCNN_AI2/RunConfig, per-epoch accuracy array (epoch, digit), unroll trace arrays stored per run, --scan weights, weight-scan plotting.

## 2026-10-27-04-51_5254e48

- **Started:** 2026-10-27T04:51:13+00:00
- **Branch:** alternation-studies
- **Commit:** 5254e48232edb837264cc58e70339efb724a1e51
- **Scan:** weights
- **Planned runs:** 140
- **Dataset:** fashion_mnist
- **Seeds:** 5 random seeds per config
- **Initial classes / classes per stage:** 1 / 1
- **Workers:** 12
- **MLflow:** off
- **Results:** `results/2026-10-27-04-51_5254e48/`

Command:

```sh
python src/run_experiments.py --scan weights --dataset fashion_mnist --initial-classes 1 --classes-per-stage 1 --num-seeds 5 --workers 12 --no-mlflow --goal 'Check whether ai2 inhibition and output switch each other on and off across unroll steps (Q1), and whether activity or input-dependent signal shrinks with depth because of ai2 versus the w=0 control (Q2).' --method '28 configs (2 variants x 2 schemes x 7 ai2_weight values), T=5, four hidden layers of width 32, 5 epochs per stage, 1 initial class and 1 class per stage, 5 random seeds, FashionMNIST.' --code-changes 'Same code as the MNIST launch of the same scan (commit unchanged).'
```

### Goal

Check whether ai2 inhibition and output switch each other on and off across unroll steps (Q1), and whether activity or input-dependent signal shrinks with depth because of ai2 versus the w=0 control (Q2).

### Method

28 configs (2 variants x 2 schemes x 7 ai2_weight values), T=5, four hidden layers of width 32, 5 epochs per stage, 1 initial class and 1 class per stage, 5 random seeds, FashionMNIST.

### Code changes

Same code as the MNIST launch of the same scan (commit unchanged).
