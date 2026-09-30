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

ai2 + FCNN controls + CNN control, 5 progressively smaller steps:

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

## Scale/results note

`--num-seeds 5` x the base sweep's 14 configs = 70 training runs; combined
with `--scan size` (35 configs) x 5 seeds = 175 runs. Results land in
`results/*.json` (named to reflect non-default dataset/seed) and in MLflow
(`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`).
