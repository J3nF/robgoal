# Scan list

How to run each scan via `src/run_experiments.py`'s CLI, or the matching
function call directly (`PYTHONPATH=src`, e.g. `uv run python -c "..."`, or
from a script under `src/`). All flags compose.

## 1. Seed scan

5 random seeds per config, base sweep:

```sh
uv run python src/run_experiments.py --num-seeds 5
```

```python
run_experiments.main(num_seeds=5)
```

## 2. Shrinking-network scan

ai2 + FCNN controls + CNN control, 8 progressively smaller steps (down to
FCNN hidden_sizes `(2, 1)` / CNN channels `(1, 1)`):

```sh
uv run python src/run_experiments.py --scan size
```

```python
run_experiments.run_size_scan()
```

## 3. FashionMNIST

```sh
uv run python src/run_experiments.py --dataset fashion_mnist
```

```python
run_experiments.main(dataset="fashion_mnist")
```

## 4. One initial class, one at a time

`classes_per_stage` already defaults to `1`, so only `--initial-classes`
needs setting:

```sh
uv run python src/run_experiments.py --initial-classes 1
```

```python
run_experiments.main(initial_classes=1)
```

## Combined example

Size scan on FashionMNIST, 5 seeds, starting from 1 class:

```sh
uv run python src/run_experiments.py --scan size --num-seeds 5 --dataset fashion_mnist --initial-classes 1
```

## 5. Capacity-overwhelming scan: size x seeds x both datasets x 1-class start

Every axis above at once, on both datasets. `--dataset` is single-valued, so
this needs two invocations (flags still compose within each); then plot:

```sh
uv run python src/run_experiments.py --scan size --num-seeds 5 --dataset mnist --initial-classes 1
uv run python src/run_experiments.py --scan size --num-seeds 5 --dataset fashion_mnist --initial-classes 1
uv run python src/plot_size_scan.py
```

`plot_size_scan.py` reads `results/*.json`, aggregates mean/std accuracy and
backward_transfer per config across seeds, and writes
`results/plots/accuracy_vs_size.png`, `results/plots/backward_transfer_vs_size.png`,
and `results/plots/scan_summary.csv`.

## Scale/results note

`--num-seeds 5` x the base sweep's 14 configs = 70 training runs; combined
with `--scan size` (8 steps x 7 configs = 56 configs) x 5 seeds = 280 runs per
dataset, 560 across both datasets (§5). Results land in `results/*.json`
(named to reflect non-default dataset/seed) and in MLflow (`uv run mlflow ui
--backend-store-uri sqlite:///mlflow.db`).
