# robgoal

A multi-week project on implementing neural networks capable of robust goal-keeping; specifically tackling catastrophic forgetting of features.

More detailed notes can be found in  the `docs` folder, particularly in `docs/Mini_project_meta_continual_learning.md`.

## Architecture

How the pieces in `src/` interact when running the sweep (see
`docs/implementation_doc.md` for the full module-by-module breakdown):

```mermaid
flowchart LR
    data[data.py] -->|Stages, dataset choice| orchestrator[run_experiments.py]
    config[sweep.py / size_scan.py<br/>+ run_config.py] -->|RunConfig, model| orchestrator
    models[autapse.py<br/>+ models.py] -->|constructs CNN/FCNN_AI2| config
    orchestrator -->|config, model, stages| logging[logged_training.py]
    logging -->|callbacks| train[train.py]
    train -->|loss, per-digit accuracy| logging
    train --> metrics[metrics.py]
    orchestrator --> metrics
    logging --> metrics
    logging --> dynamics[dynamics_insights.py]
    models -.->|last_y_self| dynamics
    logging -->|params/metrics/artifacts| mlflow[(MLflow:<br/>mlflow.db, mlartifacts/)]
    orchestrator -->|per-run JSON| results[(results/*.json)]
```

## Set-up

Use [`uv`](https://docs.astral.sh/uv/), together with the preferred Pytorch version (CPU-only or CUDA), to reproduce our Python environment:

```sh
uv sync --extra=cpu

# or, if you want to use CUDA:
uv sync --extra=cu130
```

## Running experiments

Run the full sweep (autapsic-inhibition variants/schemes, plus a `T` /
`epochs_per_stage` / hidden-layer-size sweep — see `docs/old-lab-book.md`,
"Implemented tests") from the repo root:

```sh
uv run python src/run_experiments.py
```

This downloads MNIST on first run (to `data/`), trains every config in
`src/sweep.py`'s sweep, and:

- writes each run's per-stage accuracy to `results/{name}.json`,
- prints an accuracy matrix + summary line per run,
- live-logs params/metrics to a local [MLflow](https://mlflow.org/) tracking
  database (`mlflow.db`, with artifacts in `mlartifacts/`).

To watch runs live while the sweep is running (or afterward), start the
MLflow UI in another terminal from the repo root:

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

then open `http://localhost:5000`.

### Adjusting the sweep

Both `main()` (the base sweep) and `run_size_scan()` (below) take the same
`--num-seeds`/`--dataset`/`--initial-classes`/`--classes-per-stage` CLI flags,
also usable as keyword args when calling either function directly:

```sh
uv run python src/run_experiments.py --num-seeds 5 --dataset fashion_mnist --initial-classes 1
uv run python src/run_experiments.py --scan size --num-seeds 5
```

- **Which configs run**: edit the constants at the top of `src/sweep.py`
  (`VARIANTS`, `SCHEMES`, `BASE_T`, `T_SWEEP`, `EPOCHS_SWEEP`,
  `HIDDEN_SIZES_SWEEP`, `ANCHOR_VARIANT`, `ANCHOR_SCHEME`) or extend
  `build_configs()` directly with more `RunConfig`s.
- **Seed replicates**: `--num-seeds`/`num_seeds` (both default `1`: fast,
  single-seed, deterministic seed `0`). Set it higher (e.g. `3`) to run each
  config as that many replicates instead, each with a randomly drawn seed
  (recorded as the `seed` MLflow param and in each replicate's results
  filename, for reproducibility) — grouped under one MLflow parent run per
  config with mean/stdev summary metrics. Multiplies total training runs, so
  it's opt-in.
- **Dataset**: `--dataset`/`dataset`, `"mnist"` (default) or
  `"fashion_mnist"`. Runs on a non-default dataset get their result
  filename/MLflow run name suffixed with the dataset name, so switching back
  and forth never overwrites the other dataset's results.
- **Initial vs. sequential class split**: `--initial-classes`/
  `initial_classes` (default `5`) and `--classes-per-stage`/
  `classes_per_stage` (default `1`) control `data.build_stage_digits`'s
  split — e.g. `initial_classes=1, classes_per_stage=1` starts from a single
  class and adds one more per stage all the way through.
- **Shrinking-network scan**: `uv run python src/run_experiments.py --scan
  size`, or `run_experiments.run_size_scan()` — runs `src/size_scan.py`'s
  scan (ai2 variants, FCNN controls, and the CNN control, all at
  progressively smaller widths/channel counts) instead of the base sweep, to
  see how ai2 vs. controls degrade as capacity shrinks. Override the widths
  via `hidden_sizes_scan`/`channels_scan`.
- **A single config, without the rest of the sweep**: build a `RunConfig`
  (or take one from `sweep.build_configs()`) and call `sweep.build_model` +
  `logged_training.run_and_log` directly. The sibling imports below need
  `src/` importable (see `docs/implementation_doc.md`, "Files") — save this
  as a script under `src/` and run it from the repo root, or run
  `PYTHONPATH=src uv run python -c "..."`:

  ```python
  import torch
  from data import build_stages, get_mnist_data
  from logged_training import run_and_log
  from sweep import build_configs, build_model

  train_data, test_data = get_mnist_data(root="data")  # from the repo root
  stages = build_stages(train_data, test_data)
  config = build_configs()[0]  # or construct your own RunConfig
  torch.manual_seed(config.seed)
  run_and_log(config, build_model(config), stages, torch.device("cpu"))
  ```

See `docs/implementation_doc.md` for what each module does, and
`docs/old-lab-book.md` for the research reasoning behind the sweep design.

## Tests

```sh
uv run pytest -q
```
