# Scan list

How to run each scan via `src/run_experiments.py`'s CLI, or the matching
function call directly (`PYTHONPATH=src uv run python -c "import run_experiments; ..."`,
or from a script under `src/`; the Python snippets below assume `import run_experiments`).
All flags compose.

Every launch **requires** `--goal`, `--method` and `--code-changes` (free text,
copied verbatim into `docs/lab-book.md`; `--code-changes` is how big source
changes get recorded, it is never auto-generated) and **refuses to run on a
dirty git tree** (commit or stash first; the lab book itself is exempt). A
launch writes `results/{launch_id}/{run_name}.h5`, one file per run, where
`launch_id = YYYY-MM-DD-HH-mm_{git7}`. Two launches in the same minute on the
same commit are refused.

Further flags on every command below:

- `--workers N` (default 1: sequential, in-process): N spawned worker
  processes, one config (with all its seeds) per job. A job that fails is
  retried once; at the end, still-failed jobs are listed and the exit code is
  1 (finished runs are kept). Exit code 2 = refused before starting (bad arguments, dirty
  tree, existing launch directory).
- `--no-mlflow`: do not import or configure MLflow (it is only a live viewer;
  the `.h5` files are the record). An MLflow error never loses results but is
  reported at the end with exit code 1.

In the commands below `"..."` stands for that text; type it out.

## 1. Seed scan

5 random seeds per config, base sweep:

```sh
uv run python src/run_experiments.py --num-seeds 5 --goal "..." --method "..." --code-changes "..."
```

```python
run_experiments.main(goal="...", method="...", code_changes="...", num_seeds=5)
```

## 2. Shrinking-network scan

ai2 + FCNN controls + CNN control, 8 progressively smaller steps (down to
FCNN hidden_sizes `(2, 1)` / CNN channels `(1, 1)`):

```sh
uv run python src/run_experiments.py --scan size --goal "..." --method "..." --code-changes "..."
```

```python
run_experiments.run_size_scan(goal="...", method="...", code_changes="...")
```

## 3. FashionMNIST

```sh
uv run python src/run_experiments.py --dataset fashion_mnist --goal "..." --method "..." --code-changes "..."
```

```python
run_experiments.main(goal="...", method="...", code_changes="...", dataset="fashion_mnist")
```

## 4. One initial class, one at a time

`classes_per_stage` already defaults to `1`, so only `--initial-classes`
needs setting:

```sh
uv run python src/run_experiments.py --initial-classes 1 --goal "..." --method "..." --code-changes "..."
```

```python
run_experiments.main(goal="...", method="...", code_changes="...", initial_classes=1)
```

## Combined example

Size scan on FashionMNIST, 5 seeds, starting from 1 class, 12 workers:

```sh
uv run python src/run_experiments.py --scan size --num-seeds 5 --dataset fashion_mnist --initial-classes 1 --workers 12 --goal "..." --method "..." --code-changes "..."
```

## 5. Capacity-overwhelming scan: size x seeds x both datasets x 1-class start

Every axis above at once, on both datasets. `--dataset` is single-valued, so
this needs two invocations (flags still compose within each); then plot. The
lab-book entry the first launch wrote does not block the second (the lab book
is exempt from the clean-tree check, also if you commit it in between):

```sh
uv run python src/run_experiments.py --scan size --num-seeds 5 --dataset mnist --initial-classes 1 --workers 12 --goal "..." --method "..." --code-changes "..."
uv run python src/run_experiments.py --scan size --num-seeds 5 --dataset fashion_mnist --initial-classes 1 --workers 12 --goal "..." --method "..." --code-changes "..."
uv run python src/plot_size_scan.py results/<launch_id_mnist> results/<launch_id_fashion>
```

`plot_size_scan.py` reads the `.h5` run files under the paths given (default:
all of `results/`, which includes `results/legacy-import/`), aggregates mean/std
accuracy and backward_transfer per config across seeds (population std), and
writes `results/plots/accuracy_vs_size.png`,
`results/plots/backward_transfer_vs_size.png` and
`results/plots/scan_summary.csv` (`--out DIR` to change). It pools seeds across
every launch it is given, so pass only launches that should be combined.

## 6. ai2 weight scan

28 configs (2 variants x 2 schemes x 7 `ai2_weight` values; weight 0 is the no-inhibition
control), T=5, hidden sizes `(32, 32, 32, 32)`, 5 epochs per stage. One launch per dataset:

```sh
uv run python src/run_experiments.py --scan weights --num-seeds 5 --dataset mnist --initial-classes 1 --classes-per-stage 1 --workers 12 --goal "..." --method "..." --code-changes "..."
uv run python src/run_experiments.py --scan weights --num-seeds 5 --dataset fashion_mnist --initial-classes 1 --classes-per-stage 1 --workers 12 --goal "..." --method "..." --code-changes "..."
uv run python src/plot_weight_scan.py results/<launch_id_mnist> results/<launch_id_fashion>
```

`plot_weight_scan.py` writes `results/plots/unroll/*.png`, `dieoff_{dataset}.png`,
`accuracy_over_time_{dataset}.png`, `bwt_over_time_{dataset}.png`, `unroll_summary.csv`,
`weight_scan_summary.csv` and `q2_tests.csv` (`--out DIR` to change). Runs written before
per-epoch evaluation (`accuracy` with dims `(stage, digit)`) are refused.

## Looking at runs

```python
import plotting_utils as pu

runs = pu.load_runs("results/<launch_id>")           # or a results root
fcnn = pu.select_runs(runs, dataset="mnist", model_kind="fcnn")
loss = pu.stack_arrays(fcnn, "train_loss")           # (seeds, epochs)
pu.plot_setup()
pu.save_figure(pu.plot_epoch_curves(loss), "results/plots/loss.png")
```

Each run file's `meta` (`run.meta`) has the config, commit, start time, versions
and final metrics.

## Scale/results note

`--num-seeds 5` x the base sweep's 14 configs = 70 training runs; combined
with `--scan size` (8 steps x 7 configs = 56 configs) x 5 seeds = 280 runs per
dataset, 560 across both datasets (§5). Results land in
`results/{launch_id}/*.h5` and, unless `--no-mlflow`, in MLflow (`uv run mlflow
ui --backend-store-uri sqlite:///mlflow.db`). `results/` is gitignored, so
the numbers are not reproducible from the repo alone, but every file records
the commit that produced it and the launch's lab-book entry says why.
