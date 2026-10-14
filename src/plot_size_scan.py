"""Plots the capacity-overwhelming size scan: accuracy / backward_transfer vs. network size
(total parameter count), ai2 vs. control, both datasets. See scan-list.md, "Capacity-overwhelming
scan".

Thin script on `plotting_utils` and `run_store`: reads the run files of the given launches
(`run_experiments.py --scan size --num-seeds 5 --initial-classes 1`, on both `mnist` and
`fashion_mnist`) -- no MLflow, no training. Seed replicates are pooled per config and dataset
across all given launches, so pass only launches that should be combined.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import plotting_utils
import size_scan
import sweep
from data import DatasetName
from run_config import RunConfig
from run_store import RunFile

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATASETS: tuple[DatasetName, ...] = ("mnist", "fashion_mnist")
CHANCE_LEVEL = 0.1  # 1 / num_classes.
FIGSIZE = (4.5, 3.0)  # larger than the house default: 7 families x 2 datasets need room.
FAMILIES: tuple[str, ...] = (
    "cnn_control",
    "fcnn_control_relu",
    "fcnn_control_sigmoid",
    "fcnn_ai2_diff_sigmoid",
    "fcnn_ai2_diff_relu_sigmoid",
    "fcnn_ai2_gate_sigmoid",
    "fcnn_ai2_gate_relu_sigmoid",
)
DATASET_LINESTYLES: dict[DatasetName, Literal["-", "--"]] = {"mnist": "-", "fashion_mnist": "--"}
CSV_COLUMNS = (
    "config_name",
    "family",
    "dataset",
    "param_count",
    "num_seeds",
    "accuracy_mean",
    "accuracy_std",
    "bwt_mean",
    "bwt_std",
)


@dataclass(frozen=True)
class ScanPoint:
    """Mean +/- std (population, across seeds) of the final metrics of one config on one dataset."""

    config_name: str
    family: str
    dataset: DatasetName
    param_count: int
    num_seeds: int
    accuracy_mean: float
    accuracy_std: float
    bwt_mean: float
    bwt_std: float


def main(paths: list[Path], plots_dir: Path) -> None:
    """Writes the two figures and `scan_summary.csv` for the scan runs found under `paths`."""
    runs = plotting_utils.load_runs(*paths)
    points = collect_scan_points(runs)
    _warn_if_mixed(runs)
    plotting_utils.plot_setup()
    write_summary_csv(points, plots_dir / "scan_summary.csv")
    for metric, ylabel, reference in (
        ("accuracy", "Final average accuracy", (CHANCE_LEVEL, "chance (10%)")),
        ("bwt", "Backward transfer", (0.0, "no forgetting/gain")),
    ):
        ax = _plot_metric(points, metric, ylabel, reference)
        name = "accuracy_vs_size.png" if metric == "accuracy" else "backward_transfer_vs_size.png"
        print(f"Wrote {plotting_utils.save_figure(ax, plots_dir / name)}")
    print(f"Wrote {plots_dir / 'scan_summary.csv'}")


def collect_scan_points(runs: list[RunFile]) -> list[ScanPoint]:
    """Aggregates the 1-class-at-a-time size-scan runs per canonical config and dataset.

    Follows `size_scan.build_size_scan_configs` order; config/dataset pairs without runs are
    skipped (e.g. a scan still in progress).

    Raises:
        ValueError: if no size-scan run with `initial_classes=1, classes_per_stage=1` exists.
    """
    scan_runs = plotting_utils.select_runs(
        runs, sweep_axis="size_scan", initial_classes=1, classes_per_stage=1
    )
    groups = plotting_utils.group_runs(scan_runs, "config_name", "dataset")
    points = []
    for config in size_scan.build_size_scan_configs():
        size = param_count(config)
        for dataset in DATASETS:
            group = groups.get((config.name, dataset))
            if group is None:
                continue
            accuracy = np.array([run.meta["metrics"]["final_avg_accuracy"] for run in group])
            bwt = np.array([run.meta["metrics"]["final_backward_transfer"] for run in group])
            points.append(
                ScanPoint(
                    config_name=config.name,
                    family=family_of(config),
                    dataset=dataset,
                    param_count=size,
                    num_seeds=len(group),
                    accuracy_mean=float(accuracy.mean()),
                    accuracy_std=float(accuracy.std()),
                    bwt_mean=float(bwt.mean()),
                    bwt_std=float(bwt.std()),
                )
            )
    if not points:
        raise ValueError(
            "Size-scan runs exist but none matches a canonical config of "
            "size_scan.build_size_scan_configs() (custom widths?); nothing to plot."
        )
    return points


def family_of(config: RunConfig) -> str:
    """Classifies a config into one of the 7 fixed family labels (see FAMILIES)."""
    if config.model_kind == "cnn":
        return "cnn_control"
    if config.T == 1:
        suffix = "relu" if config.activation_scheme == "relu_sigmoid" else "sigmoid"
        return f"fcnn_control_{suffix}"
    return f"fcnn_{config.ai2_variant}_{config.activation_scheme}"


def param_count(config: RunConfig) -> int:
    """Total trainable parameter count -- the one size axis comparable across
    FCNN (hidden_sizes) and CNN (channels) families."""
    return sum(p.numel() for p in sweep.build_model(config).parameters())


def write_summary_csv(points: list[ScanPoint], path: Path) -> None:
    """Writes one row per `ScanPoint` to `path`, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for point in points:
            writer.writerow([getattr(point, column) for column in CSV_COLUMNS])


def _plot_metric(points: list[ScanPoint], metric: str, ylabel: str, reference: tuple[float, str]):
    """One line per (family, dataset): `metric` mean +/- std against parameter count."""
    colors = dict(zip(FAMILIES, plotting_utils.get_colorcycle(len(FAMILIES))))
    ax = None
    for family in FAMILIES:
        for dataset in DATASETS:
            series = sorted(
                (p for p in points if p.family == family and p.dataset == dataset),
                key=lambda p: p.param_count,
            )
            if not series:
                continue
            ax = plotting_utils.plot_line_band(
                np.array([p.param_count for p in series]),
                np.array([getattr(p, f"{metric}_mean") for p in series]),
                np.array([getattr(p, f"{metric}_std") for p in series]),
                ax,
                logx=True,
                figsize=FIGSIZE,
                color=colors[family],
                linestyle=DATASET_LINESTYLES[dataset],
                marker="o",
                markersize=2,
                linewidth=1,
                label=f"{family} ({dataset})",
            )
    assert ax is not None  # collect_scan_points raises if there are no points
    value, label = reference
    ax.axhline(value, color="gray", linestyle=":", linewidth=0.5)
    ax.annotate(label, (0, value), xycoords=("axes fraction", "data"), fontsize="x-small")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize="xx-small")
    return ax


def _warn_if_mixed(runs: list[RunFile]) -> None:
    """Warns on stderr when the pooled scan runs come from different commits or learning rates."""
    scan_runs = plotting_utils.select_runs(
        runs, sweep_axis="size_scan", initial_classes=1, classes_per_stage=1
    )
    commits = {run.meta["git"]["commit"] for run in scan_runs}
    learning_rates = {plotting_utils.run_field(run, "lr") for run in scan_runs}
    if len(commits) > 1 or len(learning_rates) > 1:
        print(
            f"WARNING: pooling runs from {len(commits)} commit(s) {sorted(map(str, commits))} and "
            f"lr values {sorted(learning_rates)}; pass only launches that belong together.",
            file=sys.stderr,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[RESULTS_DIR],
        help="Launch directories and/or results roots to read (default: results/).",
    )
    parser.add_argument("--out", type=Path, default=PLOTS_DIR, help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.paths, args.out)
