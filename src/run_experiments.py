"""Runs the autapsic-inhibition sweep defined in sweep.py, live-logging to
MLflow. See lab-book.md, "Implemented tests"."""

import argparse
import contextlib
import dataclasses
import json
import random
from pathlib import Path

import torch

import logged_training
import size_scan
import sweep
from data import DatasetName, Stage, build_stage_digits, build_stages, get_mnist_data
from metrics import average_accuracy, backward_transfer, format_accuracy_matrix
from run_config import RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
# Set number of seed replicates per config for variance estimates to default 1.
NUM_SEEDS = 1
RESULTS_DIR = REPO_ROOT / "results"
# Draws span [0, _SEED_UPPER_BOUND) -- generous enough that collisions between
# distinct draws are practically impossible without needing dedup logic.
_SEED_UPPER_BOUND = 2**31


def main(
    num_seeds: int = NUM_SEEDS,
    dataset: DatasetName = "mnist",
    initial_classes: int = 5,
    classes_per_stage: int = 1,
) -> None:
    """Runs every config in sweep.build_configs(), live-logging each to MLflow.

    Args:
        num_seeds: Seed replicates per config, see `_draw_seeds`.
        dataset: "mnist" (default) or "fashion_mnist".
        initial_classes: How many classes stage 0 introduces, see
            `data.build_stage_digits`.
        classes_per_stage: How many new classes each later stage introduces.
    """
    device, stages = _prepare_run(dataset, initial_classes, classes_per_stage)
    _run_configs(sweep.build_configs(), stages, device, num_seeds, dataset)


def run_size_scan(
    num_seeds: int = 1,
    dataset: DatasetName = "mnist",
    initial_classes: int = 5,
    classes_per_stage: int = 1,
    hidden_sizes_scan: list[tuple[int, ...]] | None = None,
    channels_scan: list[tuple[int, int]] | None = None,
) -> None:
    """Runs the shrinking-network scan from size_scan.py, live-logging each to MLflow.

    Same args as `main`, plus `hidden_sizes_scan`/`channels_scan` (see
    `size_scan.build_size_scan_configs`) to override the scan's widths;
    `None` (default) uses `size_scan`'s own defaults for both.
    """
    device, stages = _prepare_run(dataset, initial_classes, classes_per_stage)
    configs = size_scan.build_size_scan_configs(
        hidden_sizes_scan if hidden_sizes_scan is not None else size_scan.FCNN_HIDDEN_SIZES_SCAN,
        channels_scan if channels_scan is not None else size_scan.CNN_CHANNELS_SCAN,
    )
    _run_configs(configs, stages, device, num_seeds, dataset)


def _prepare_run(
    dataset: DatasetName, initial_classes: int, classes_per_stage: int
) -> tuple[torch.device, list[Stage]]:
    """Shared setup for `main`/`run_size_scan`: device, data download, staging, results dir."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = get_mnist_data(root=str(REPO_ROOT / "data"), dataset=dataset)
    stage_digits = build_stage_digits(
        initial_classes=initial_classes, classes_per_stage=classes_per_stage
    )
    stages = build_stages(train_data, test_data, stage_digits=stage_digits)
    RESULTS_DIR.mkdir(exist_ok=True)
    return device, stages


def _run_configs(
    configs: list[RunConfig],
    stages: list[Stage],
    device: torch.device,
    num_seeds: int,
    dataset: DatasetName,
) -> None:
    """Runs every entry in `configs` (as `num_seeds` replicates if >1)."""
    for config in configs:
        _run_config(config, stages, device, num_seeds, dataset)


def _run_config(
    config: RunConfig,
    stages: list[Stage],
    device: torch.device,
    num_seeds: int,
    dataset: DatasetName,
) -> None:
    """Runs `config` (as `num_seeds` replicates if >1), writing each seed's results."""
    seeds = _draw_seeds(num_seeds)
    print(f"{config.name}: seeds={seeds}")
    replicate_context = _get_replicate_context(config, num_seeds)
    with replicate_context as collector:
        for seed in seeds:
            avg_acc, bwt = _run_seed(config, seed, stages, device, num_seeds, dataset)
            if collector is not None:
                collector.append((avg_acc, bwt))


def _draw_seeds(num_seeds: int) -> list[int]:
    """Draws `num_seeds` distinct random seeds for reproducible multi-seed runs.

    `num_seeds == 1` stays deterministic (`[0]`, matching this repo's
    single-seed behavior before random seeds existed) -- only opting into
    multiple replicates opts into randomized seeds. Recorded for
    reproducibility via the usual `RunConfig.seed` -> MLflow param / results
    filename path (see `_build_seed_variant`), plus the print above.

    Raises:
        ValueError: if `num_seeds < 1` -- fails fast rather than silently
            running zero seeds (an empty seed list would otherwise make
            `_run_config` loop zero times: no training, no results, no
            error, just a misleadingly "successful" no-op run).
    """
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    if num_seeds == 1:
        return [0]
    rng = random.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < num_seeds:
        seeds.add(rng.randrange(_SEED_UPPER_BOUND))
    return sorted(seeds)


def _get_replicate_context(
    config: RunConfig, num_seeds: int
) -> contextlib.AbstractContextManager[list[tuple[float, float]] | None]:
    """Picks the per-config MLflow context: grouped replicates if `num_seeds > 1`, else a no-op."""
    if num_seeds > 1:
        return logged_training.replicate_group(config, num_seeds)
    return contextlib.nullcontext()


def _run_seed(
    config: RunConfig,
    seed: int,
    stages: list[Stage],
    device: torch.device,
    num_seeds: int,
    dataset: DatasetName,
) -> tuple[float, float]:
    """Runs one seed replicate of `config` end to end: build, train, log, write results.

    Returns:
        `(avg_accuracy, backward_transfer)` for this seed.
    """
    seed_config = _build_seed_variant(config, seed, num_seeds, dataset)
    torch.manual_seed(seed_config.seed)
    model = sweep.build_model(seed_config)
    accuracy_by_stage = logged_training.run_and_log(seed_config, model, stages, device)
    avg_acc = average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen)
    bwt = backward_transfer(accuracy_by_stage, stages[0].digits_seen)
    _write_results(seed_config.name, accuracy_by_stage, avg_acc, bwt)
    return avg_acc, bwt


def _build_seed_variant(
    config: RunConfig, seed: int, num_seeds: int, dataset: DatasetName
) -> RunConfig:
    """Per-seed, per-dataset copy of `config`.

    Sets `seed`/`dataset`, and suffixes `name` with the dataset (whenever
    it's not the default "mnist") and the seed (only when replicating) --
    keeps default-dataset, single-seed run names exactly what they were
    before either feature existed, while still guaranteeing distinct
    filenames/MLflow run names across datasets and across replicates.
    """
    name = config.name
    if dataset != "mnist":
        name = f"{name}_{dataset}"
    if num_seeds > 1:
        name = f"{name}_seed{seed}"
    return dataclasses.replace(config, seed=seed, dataset=dataset, name=name)


def _write_results(
    name: str, accuracy_by_stage: list[dict[int, float]], avg_acc: float, bwt: float
) -> None:
    """Dumps `accuracy_by_stage` to `RESULTS_DIR/{name}.json` and prints a summary."""
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(accuracy_by_stage, indent=2))
    print(f"{name}: avg_accuracy={avg_acc:.4f} backward_transfer={bwt:+.4f}")
    print(format_accuracy_matrix(accuracy_by_stage))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        choices=["sweep", "size"],
        default="sweep",
        help="'sweep' (default): sweep.py's 14 configs. 'size': size_scan.py's "
        "shrinking-network scan.",
    )
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--initial-classes", type=int, default=5)
    parser.add_argument("--classes-per-stage", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.scan == "sweep":
        main(
            num_seeds=args.num_seeds,
            dataset=args.dataset,
            initial_classes=args.initial_classes,
            classes_per_stage=args.classes_per_stage,
        )
    else:
        run_size_scan(
            num_seeds=args.num_seeds,
            dataset=args.dataset,
            initial_classes=args.initial_classes,
            classes_per_stage=args.classes_per_stage,
        )
