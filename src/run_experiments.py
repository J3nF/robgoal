"""Runs the autapsic-inhibition sweep defined in sweep.py, live-logging to
MLflow. See lab-book.md, "Implemented tests"."""

import contextlib
import dataclasses
import json
from pathlib import Path

import torch

import logged_training
import sweep
from data import Stage, build_stages, get_mnist_data
from metrics import average_accuracy, backward_transfer, format_accuracy_matrix
from run_config import RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
# Seed replicates per config, for variance estimates -- e.g. 3 turns the 14
# configs in sweep.py into 42 runs. 1 (default) reproduces the single-seed
# sweep exactly: no nesting, no aggregation, same run names.
NUM_SEEDS = 1
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    """Runs every config in sweep.build_configs(), live-logging each to MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = get_mnist_data(root=str(REPO_ROOT / "data"))
    stages = build_stages(train_data, test_data)
    RESULTS_DIR.mkdir(exist_ok=True)
    for config in sweep.build_configs():
        _run_config(config, stages, device)


def _run_config(config: RunConfig, stages: list[Stage], device: torch.device) -> None:
    """Runs `config` (as `NUM_SEEDS` replicates if >1), writing each seed's results."""
    replicate_context = _get_replicate_context(config)
    with replicate_context as collector:
        for seed in range(NUM_SEEDS):
            avg_acc, bwt = _run_seed(config, seed, stages, device)
            if collector is not None:
                collector.append((avg_acc, bwt))


def _get_replicate_context(
    config: RunConfig,
) -> contextlib.AbstractContextManager[list[tuple[float, float]] | None]:
    """Picks the per-config MLflow context: grouped replicates if `NUM_SEEDS > 1`, else a no-op."""
    if NUM_SEEDS > 1:
        return logged_training.replicate_group(config, NUM_SEEDS)
    return contextlib.nullcontext()


def _run_seed(
    config: RunConfig, seed: int, stages: list[Stage], device: torch.device
) -> tuple[float, float]:
    """Runs one seed replicate of `config` end to end: build, train, log, write results.

    Returns:
        `(avg_accuracy, backward_transfer)` for this seed.
    """
    seed_config = _build_seed_variant(config, seed)
    torch.manual_seed(seed_config.seed)
    model = sweep.build_model(seed_config)
    accuracy_by_stage = logged_training.run_and_log(seed_config, model, stages, device)
    avg_acc = average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen)
    bwt = backward_transfer(accuracy_by_stage, stages[0].digits_seen)
    _write_results(seed_config.name, accuracy_by_stage, avg_acc, bwt)
    return avg_acc, bwt


def _build_seed_variant(config: RunConfig, seed: int) -> RunConfig:
    """Per-seed copy of `config`: sets `seed`, suffixes `name` only when replicating."""
    if NUM_SEEDS == 1:
        return dataclasses.replace(config, seed=seed)
    return dataclasses.replace(config, seed=seed, name=f"{config.name}_seed{seed}")


def _write_results(
    name: str, accuracy_by_stage: list[dict[int, float]], avg_acc: float, bwt: float
) -> None:
    """Dumps `accuracy_by_stage` to `RESULTS_DIR/{name}.json` and prints a summary."""
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(accuracy_by_stage, indent=2))
    print(f"{name}: avg_accuracy={avg_acc:.4f} backward_transfer={bwt:+.4f}")
    print(format_accuracy_matrix(accuracy_by_stage))


if __name__ == "__main__":
    main()
