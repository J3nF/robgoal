"""Plots the capacity-overwhelming size scan: accuracy / backward_transfer vs.
network size (total parameter count), ai2 vs. control, both datasets. See
scan-list.md, "Capacity-overwhelming scan".

Reads `results/*.json` written by `run_experiments.py --scan size`
(`--num-seeds 5 --initial-classes 1`, on both `mnist` and `fashion_mnist`) --
no MLflow dependency, no torch training. Run after both dataset scans finish.
"""

import csv
import json
import statistics
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import size_scan
import sweep
from data import DatasetName
from run_config import RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATASETS: tuple[DatasetName, ...] = ("mnist", "fashion_mnist")
INITIAL_DIGIT = "0"  # stage_digits[0] under initial_classes=1 is always [0].
CHANCE_LEVEL = 0.1  # 1 / num_classes.

# Fixed categorical order (palette.md slots 1-7) -- assigned by family
# identity, consistent across both plots, never cycled/reassigned.
FAMILY_COLORS: dict[str, str] = {
    "cnn_control": "#2a78d6",
    "fcnn_control_relu": "#eb6834",
    "fcnn_control_sigmoid": "#1baf7a",
    "fcnn_ai2_diff_sigmoid": "#eda100",
    "fcnn_ai2_diff_relu_sigmoid": "#e87ba4",
    "fcnn_ai2_gate_sigmoid": "#008300",
    "fcnn_ai2_gate_relu_sigmoid": "#4a3aa7",
}
DATASET_LINESTYLES: dict[DatasetName, Literal["-", "--"]] = {
    "mnist": "-",
    "fashion_mnist": "--",
}
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
PRIMARY_INK = "#0b0b0b"
SURFACE = "#fcfcfb"


def family_of(config: RunConfig) -> str:
    """Classifies a config into one of the 7 fixed family labels (see FAMILY_COLORS)."""
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


def result_files(config_name: str, dataset: DatasetName) -> list[Path]:
    """Finds a config's seed-replicate result files for `dataset`.

    Matches `run_experiments._build_seed_variant`'s naming: non-mnist runs get
    a `_{dataset}` suffix, and every run here uses `--num-seeds 5` so every
    filename carries a `_seed{N}` suffix too.
    """
    suffix = "" if dataset == "mnist" else f"_{dataset}"
    return sorted(RESULTS_DIR.glob(f"{config_name}{suffix}_seed*.json"))


def load_metrics(path: Path) -> tuple[float, float]:
    """Loads one result file, returning `(final_accuracy, backward_transfer)`.

    `final_accuracy` is the mean over all 10 digits at the last stage -- valid
    because `initial_classes=1, classes_per_stage=1` means every digit has
    been introduced by then, and every stage's dict is already dense over all
    10 digits (see data.py, "Evaluation metrics").
    """
    accuracy_by_stage = json.loads(path.read_text())
    final_accuracy = statistics.mean(accuracy_by_stage[-1].values())
    bwt = accuracy_by_stage[-1][INITIAL_DIGIT] - accuracy_by_stage[0][INITIAL_DIGIT]
    return final_accuracy, bwt


class ScanPoint:
    """Aggregated (mean +/- std across seeds) metrics for one config/dataset pair."""

    def __init__(
        self,
        config: RunConfig,
        dataset: DatasetName,
        size: int,
        accuracies: list[float],
        bwts: list[float],
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.param_count = size
        self.num_seeds = len(accuracies)
        self.accuracy_mean = statistics.mean(accuracies)
        self.accuracy_std = statistics.pstdev(accuracies)
        self.bwt_mean = statistics.mean(bwts)
        self.bwt_std = statistics.pstdev(bwts)


def collect_scan_points() -> list[ScanPoint]:
    """Rebuilds the canonical scan configs and aggregates each one's results
    (across seeds, per dataset) from `results/*.json`. Skips config/dataset
    pairs with no result files yet (e.g. a scan still in progress)."""
    configs = size_scan.build_size_scan_configs()
    points = []
    for config in configs:
        size = param_count(config)
        for dataset in DATASETS:
            files = result_files(config.name, dataset)
            if not files:
                continue
            accuracies, bwts = [], []
            for path in files:
                accuracy, bwt = load_metrics(path)
                accuracies.append(accuracy)
                bwts.append(bwt)
            points.append(ScanPoint(config, dataset, size, accuracies, bwts))
    return points


def write_summary_csv(points: list[ScanPoint], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config_name",
                "family",
                "dataset",
                "param_count",
                "num_seeds",
                "accuracy_mean",
                "accuracy_std",
                "bwt_mean",
                "bwt_std",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.config.name,
                    family_of(point.config),
                    point.dataset,
                    point.param_count,
                    point.num_seeds,
                    point.accuracy_mean,
                    point.accuracy_std,
                    point.bwt_mean,
                    point.bwt_std,
                ]
            )


def _family_dataset_series(
    points: list[ScanPoint], metric: str
) -> dict[tuple[str, DatasetName], list[tuple[int, float, float]]]:
    """Groups points by (family, dataset) into sorted `(param_count, mean, std)` lists."""
    series: dict[tuple[str, DatasetName], list[tuple[int, float, float]]] = {}
    for point in points:
        key = (family_of(point.config), point.dataset)
        mean, std = (
            (point.accuracy_mean, point.accuracy_std)
            if metric == "accuracy"
            else (point.bwt_mean, point.bwt_std)
        )
        series.setdefault(key, []).append((point.param_count, mean, std))
    for values in series.values():
        values.sort(key=lambda v: v[0])
    return series


def _plot_metric(
    points: list[ScanPoint],
    metric: str,
    ylabel: str,
    reference_line: tuple[float, str] | None,
    out_path: Path,
) -> None:
    series = _family_dataset_series(points, metric)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (family, dataset), values in series.items():
        sizes = [v[0] for v in values]
        means = [v[1] for v in values]
        stds = [v[2] for v in values]
        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            color=FAMILY_COLORS[family],
            linestyle=DATASET_LINESTYLES[dataset],
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
        )

    ax.set_xscale("log")

    if reference_line is not None:
        value, label = reference_line
        ax.axhline(value, color=MUTED_INK, linestyle=":", linewidth=1)
        ax.text(
            ax.get_xlim()[0],
            value,
            f"{label}  ",
            color=MUTED_INK,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    ax.set_xlabel("Parameter count (log scale)", color=PRIMARY_INK)
    ax.set_ylabel(ylabel, color=PRIMARY_INK)
    ax.set_title(
        "Class-incremental (1 class at a time) -- ai2 vs. control under shrinking capacity",
        color=PRIMARY_INK,
    )
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=MUTED_INK)

    family_handles = [
        Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=6, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    dataset_handles = [
        Line2D([0], [0], color=PRIMARY_INK, linewidth=2, linestyle=style, label=dataset)
        for dataset, style in DATASET_LINESTYLES.items()
    ]
    ax.legend(
        handles=family_handles + dataset_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    points = collect_scan_points()
    if not points:
        raise RuntimeError(
            f"No scan results found under {RESULTS_DIR} -- run the size scan "
            "(see scan-list.md, 'Capacity-overwhelming scan') before plotting."
        )
    PLOTS_DIR.mkdir(exist_ok=True)
    write_summary_csv(points, PLOTS_DIR / "scan_summary.csv")
    _plot_metric(
        points,
        metric="accuracy",
        ylabel="Final average accuracy",
        reference_line=(CHANCE_LEVEL, "chance (10%)"),
        out_path=PLOTS_DIR / "accuracy_vs_size.png",
    )
    _plot_metric(
        points,
        metric="bwt",
        ylabel="Backward transfer",
        reference_line=(0.0, "no forgetting/gain"),
        out_path=PLOTS_DIR / "backward_transfer_vs_size.png",
    )
    print(f"Wrote {PLOTS_DIR / 'accuracy_vs_size.png'}")
    print(f"Wrote {PLOTS_DIR / 'backward_transfer_vs_size.png'}")
    print(f"Wrote {PLOTS_DIR / 'scan_summary.csv'}")


if __name__ == "__main__":
    main()
