"""Plots the ai2 weight scan: unroll traces, die-off across depth, accuracy and BWT over time.

Thin script on `weight_scan_data` (loading, aggregation, CSVs) and `plotting_utils`: reads the
run files of the given launches (`run_experiments.py --scan weights --initial-classes 1
--num-seeds 5`, once per dataset) -- no MLflow, no training. Seeds are aggregated as mean with a
+/-1 std band (population std). Diverged seeds are left out and counted in the CSVs.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import plotting_utils
from data import build_stage_digits
from metrics import average_accuracy, backward_transfer_over_time
from sweep import weight_label
from weight_scan_data import (
    DATASETS,
    NUM_CLASSES,
    NUM_LAYERS,
    WEIGHTS,
    Cell,
    CellKey,
    get_variant_scheme_columns,
    load_weight_scan,
    write_q2_tests_csv,
    write_scan_summary_csv,
    write_unroll_summary_csv,
)

__all__ = [
    "main",
    "plot_accuracy_over_time",
    "plot_bwt_over_time",
    "plot_dieoff",
    "plot_unroll_allclasses",
    "plot_unroll_classes3",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "results" / "plots"
SAVE_DPI = 300
THREE_CLASSES = {"mnist": (0, 1, 8), "fashion_mnist": (0, 1, 9)}
LOG_FLOOR = 1e-6
# Ordered weights, negative to positive: blues (inhibition reversed), black (none), then
# oranges to dark red (increasing inhibition). Weight 1 is drawn thicker.
WEIGHT_COLORS = dict(
    zip(
        WEIGHTS,
        ("#08519c", "#6baed6", "#000000", "#fdae6b", "#f16913", "#cb181d", "#67000d"),
        strict=True,
    )
)
WEIGHT_TEXT = dict(zip(WEIGHTS, ("-2/3", "-1/3", "0", "1/3", "2/3", "1", "4/3"), strict=True))
Y_SELF_COLOR, Y_TILDE_COLOR, Y_COLOR = "#d95f02", "#1b4f9c", "#555555"
BWT_CAPTION = (
    "Stages replay all previously seen digits (TRIAGE.md #2), so forgetting is structurally "
    "suppressed and flat curves near 0 do not show that ai2 is irrelevant.\n"
    "A value near 0 is also what a chance-level network produces, so read this together with "
    "the accuracy figure."
)


def main(paths: list[Path], plots_dir: Path) -> None:
    """Writes every figure and CSV of the weight scan for the runs found under `paths`."""
    cells = load_weight_scan(*paths)
    _warn_about_gaps(cells)
    plotting_utils.plot_setup(save_dpi=SAVE_DPI)
    write_unroll_summary_csv(cells, plots_dir / "unroll_summary.csv")
    write_scan_summary_csv(cells, plots_dir / "weight_scan_summary.csv")
    write_q2_tests_csv(cells, plots_dir / "q2_tests.csv")
    for dataset in DATASETS:
        if not any(key[0] == dataset for key in cells):
            continue
        _save_unroll_figures(cells, dataset, plots_dir)
        _save(plot_dieoff(cells, dataset), plots_dir / f"dieoff_{dataset}.png")
        _save(
            plot_accuracy_over_time(cells, dataset), plots_dir / f"accuracy_over_time_{dataset}.png"
        )
        _save(plot_bwt_over_time(cells, dataset), plots_dir / f"bwt_over_time_{dataset}.png")
    print(f"Wrote figures and CSVs to {plots_dir}")


def _save_unroll_figures(cells: dict[CellKey, Cell], dataset: str, plots_dir: Path) -> None:
    for variant, scheme in get_variant_scheme_columns():
        for weight in WEIGHTS:
            cell = cells.get((dataset, variant, scheme, weight))
            if cell is None or cell.num_finite == 0:
                print(
                    f"WARNING: no finite run for {dataset} {variant} {scheme} w={weight:.3f}",
                    file=sys.stderr,
                )
                continue
            stem = f"{dataset}_{variant}_{scheme}_w{weight_label(weight)}"
            fig = plot_unroll_classes3(cells, dataset, variant, scheme, weight)
            _save(fig, plots_dir / "unroll" / f"{stem}_classes3.png")
            fig = plot_unroll_allclasses(cells, dataset, variant, scheme, weight)
            _save(fig, plots_dir / "unroll" / f"{stem}_allclasses.png")


def _save(fig: Figure, path: Path) -> None:
    plotting_utils.save_figure(fig, path, transparent=False)
    plt.close(fig)


# --- F1: unroll traces ----------------------------------------------------------------------


def plot_unroll_classes3(
    cells: dict[CellKey, Cell], dataset: str, variant: str, scheme: str, weight: float
) -> Figure:
    """F1a: rows = three classes, columns = hidden layers; y_self and y_tilde over the unroll.

    Solid lines with a seed band; dashed line is the constant uninhibited y. Each column shares
    its y-axis across classes, with a range spanning all weights.
    """
    classes = THREE_CLASSES[dataset]
    cell = cells[(dataset, variant, scheme, weight)]
    fig, axes = plt.subplots(len(classes), NUM_LAYERS, figsize=(11, 6.5), squeeze=False)
    steps = np.arange(cell.arrays["unroll/mean_y_tilde"].shape[-1])
    for layer in range(NUM_LAYERS):
        limits = _unroll_limits(cells, (dataset, variant, scheme), layer, list(classes), True)
        for row, digit in enumerate(classes):
            ax = axes[row, layer]
            for name, color, label in (
                ("unroll/mean_y_self_used", Y_SELF_COLOR, r"$y_{self}$ (weighted)"),
                ("unroll/mean_y_tilde", Y_TILDE_COLOR, r"$\tilde y$"),
            ):
                mean, std = plotting_utils.mean_std(cell.arrays[name][:, layer, digit])
                plotting_utils.plot_line_band(
                    steps, mean, std, ax, color=color, label=label, linewidth=1.5
                )
            y_mean = cell.arrays["unroll/mean_y"][:, layer, digit].mean()
            ax.axhline(y_mean, color=Y_COLOR, linestyle="--", linewidth=1, label=r"$y$")
            ax.set_ylim(*limits)
            ax.set_xticks(steps)
            _label_grid_axis(ax, row, layer, len(classes), f"class {digit}")
    axes[0, 0].legend(frameon=False, fontsize="x-small")
    fig.suptitle(_unroll_title(dataset, variant, scheme, weight, cell))
    return fig


def plot_unroll_allclasses(
    cells: dict[CellKey, Cell], dataset: str, variant: str, scheme: str, weight: float
) -> Figure:
    """F1b: rows = hidden layers, columns = (y_self, y_tilde); all 10 classes overlaid.

    Seed means only; the dashed lines in the y_tilde column are each class's uninhibited y.
    Each row shares one y-axis with a range spanning all weights.
    """
    cell = cells[(dataset, variant, scheme, weight)]
    colors = plt.get_cmap("tab10").colors  # ty: ignore[unresolved-attribute]
    fig, axes = plt.subplots(NUM_LAYERS, 2, figsize=(8, 9), squeeze=False)
    steps = np.arange(cell.arrays["unroll/mean_y_tilde"].shape[-1])
    for layer in range(NUM_LAYERS):
        limits = _unroll_limits(
            cells, (dataset, variant, scheme), layer, list(range(NUM_CLASSES)), False
        )
        for digit in range(NUM_CLASSES):
            y_self = cell.arrays["unroll/mean_y_self_used"][:, layer, digit].mean(axis=0)
            y_tilde = cell.arrays["unroll/mean_y_tilde"][:, layer, digit].mean(axis=0)
            y = cell.arrays["unroll/mean_y"][:, layer, digit].mean()
            axes[layer, 0].plot(steps, y_self, color=colors[digit], linewidth=0.8, label=str(digit))
            axes[layer, 1].plot(steps, y_tilde, color=colors[digit], linewidth=0.8)
            axes[layer, 1].axhline(y, color=colors[digit], linestyle="--", linewidth=0.6)
        for column, title in enumerate((r"$y_{self}$ (weighted)", r"$\tilde y$ (dashed: $y$)")):
            ax = axes[layer, column]
            ax.set_ylim(*limits)
            ax.set_xticks(steps)
            if layer == 0:
                ax.set_title(title)
            if layer == NUM_LAYERS - 1:
                ax.set_xlabel("unroll step t")
        axes[layer, 0].set_ylabel(f"hidden layer {layer + 1}")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        title="class",
        loc="outside right center",
        frameon=False,
    )
    fig.suptitle(_unroll_title(dataset, variant, scheme, weight, cell))
    return fig


def _unroll_limits(
    cells: dict[CellKey, Cell],
    group: tuple[str, str, str],
    layer: int,
    classes: list[int],
    with_band: bool,
) -> tuple[float, float]:
    """y-limits of one layer spanning y, y_self and y_tilde over all weights and `classes`."""
    lows, highs = [], []
    for weight in WEIGHTS:
        cell = cells.get((*group, weight))
        if cell is None or cell.num_finite == 0:
            continue
        for name in ("unroll/mean_y", "unroll/mean_y_self_used", "unroll/mean_y_tilde"):
            values = cell.arrays[name][:, layer][:, classes]
            mean, std = plotting_utils.mean_std(values)
            band = std if with_band else 0.0
            lows.append(float((mean - band).min()))
            highs.append(float((mean + band).max()))
    low, high = min(lows), max(highs)
    pad = 0.05 * (high - low) if high > low else 0.05
    return low - pad, high + pad


def _label_grid_axis(ax: Axes, row: int, layer: int, num_rows: int, row_label: str) -> None:
    if row == 0:
        ax.set_title(f"hidden layer {layer + 1}")
    if layer == 0:
        ax.set_ylabel(row_label)
    if row == num_rows - 1:
        ax.set_xlabel("unroll step t")


def _unroll_title(dataset: str, variant: str, scheme: str, weight: float, cell: Cell) -> str:
    return (
        f"{dataset}, {variant}, {scheme}, w={WEIGHT_TEXT[weight]} "
        f"({cell.num_finite}/{cell.num_runs} seeds finite; band = +/-1 std)"
    )


# --- F2: die-off across depth ---------------------------------------------------------------


def plot_dieoff(cells: dict[CellKey, Cell], dataset: str) -> Figure:
    """F2: mean |y_tilde| (row 1) and mean per-neuron std of y_tilde (row 2) at the last step.

    Columns are the four variant x scheme combinations, x is the hidden layer, y is log-scaled,
    one line per weight (black = no-inhibition control). Values below `LOG_FLOOR` are drawn at it.
    """
    columns = get_variant_scheme_columns()
    fig, axes = plt.subplots(2, len(columns), figsize=(13, 6), squeeze=False, sharey="row")
    rows = (
        ("unroll/mean_abs_y_tilde", r"mean $|\tilde y|$ at last step"),
        ("unroll/neuron_std_y_tilde", r"mean neuron std of $\tilde y$ at last step"),
    )
    layers = np.arange(1, NUM_LAYERS + 1)
    for column, (variant, scheme) in enumerate(columns):
        for row, (name, ylabel) in enumerate(rows):
            ax = axes[row, column]
            for weight in WEIGHTS:
                cell = cells.get((dataset, variant, scheme, weight))
                if cell is None or cell.num_finite == 0:
                    continue
                mean, std = plotting_utils.mean_std(cell.arrays[name][:, :, -1])
                mean = np.maximum(mean, LOG_FLOOR)
                std = np.minimum(std, mean - LOG_FLOOR)
                plotting_utils.plot_line_band(
                    layers, mean, std, ax, label=f"w={WEIGHT_TEXT[weight]}", **_weight_style(weight)
                )
            ax.set_yscale("log")
            ax.set_xticks(layers)
            if row == 0:
                ax.set_title(f"{variant}, {scheme}")
            if row == 1:
                ax.set_xlabel("hidden layer")
            if column == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False)
    fig.suptitle(f"{dataset}: die-off across depth (band = +/-1 std over finite seeds)")
    return fig


def _weight_style(weight: float) -> dict:
    return {"color": WEIGHT_COLORS[weight], "linewidth": 2.4 if weight == 1.0 else 1.1}


# --- F3, F4: over training time -------------------------------------------------------------


def plot_accuracy_over_time(cells: dict[CellKey, Cell], dataset: str) -> Figure:
    """F3: average accuracy over the digits seen so far, after every epoch; one line per weight."""
    return _plot_over_time(
        cells,
        dataset,
        _get_accuracy_curves,
        ylabel="average accuracy (seen digits)",
        symmetric=False,
    )


def plot_bwt_over_time(cells: dict[CellKey, Cell], dataset: str) -> Figure:
    """F4: backward transfer over earlier-introduced digits after every epoch, with a caption.

    The y-axis is symmetric around 0 and shared by all panels; the curve starts in stage 1.
    """
    fig = _plot_over_time(
        cells, dataset, _get_bwt_curves, ylabel="backward transfer", symmetric=True
    )
    fig.text(0.01, -0.02, BWT_CAPTION, fontsize="x-small", va="top", ha="left", wrap=True)
    return fig


def _get_accuracy_curves(cell: Cell) -> np.ndarray:
    """Per-seed average accuracy over the digits seen at each epoch: (seeds, epochs)."""
    stage_digits = build_stage_digits(NUM_CLASSES, 1, 1)
    curves = np.zeros(cell.arrays["accuracy"].shape[:2])
    for seed, epoch in np.ndindex(*curves.shape):
        stage = epoch // cell.epochs_per_stage
        seen = frozenset(digit for digits in stage_digits[: stage + 1] for digit in digits)
        row = cell.arrays["accuracy"][seed, epoch]
        curves[seed, epoch] = average_accuracy(dict(enumerate(row)), seen)
    return curves


def _get_bwt_curves(cell: Cell) -> np.ndarray:
    """Per-seed time-resolved backward transfer: (seeds, epochs), NaN during stage 0."""
    stage_digits = build_stage_digits(NUM_CLASSES, 1, 1)
    return np.stack(
        [
            backward_transfer_over_time(accuracy, cell.epochs_per_stage, stage_digits)
            for accuracy in cell.arrays["accuracy"]
        ]
    )


def _plot_over_time(
    cells: dict[CellKey, Cell], dataset: str, get_curves, ylabel: str, symmetric: bool
) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), squeeze=False, sharex=True, sharey=True)
    extent = 0.0
    stage_digits = build_stage_digits(NUM_CLASSES, 1, 1)
    for (variant, scheme), ax in zip(get_variant_scheme_columns(), axes.flat):
        epochs_per_stage = 0
        for weight in WEIGHTS:
            cell = cells.get((dataset, variant, scheme, weight))
            if cell is None or cell.num_finite == 0:
                continue
            epochs_per_stage = cell.epochs_per_stage
            mean, std = plotting_utils.mean_std(get_curves(cell))
            epochs = np.arange(1, len(mean) + 1)
            plotting_utils.plot_line_band(
                epochs, mean, std, ax, label=f"w={WEIGHT_TEXT[weight]}", **_weight_style(weight)
            )
            extent = max(extent, float(np.nanmax(np.abs(np.concatenate([mean - std, mean + std])))))
        _mark_stage_boundaries(ax, stage_digits, epochs_per_stage)
        ax.set_title(f"{variant}, {scheme}")
    if symmetric:
        extent = extent if extent > 0 else 1.0
        axes[0, 0].set_ylim(-1.05 * extent, 1.05 * extent)
        for ax in axes.flat:
            ax.axhline(0.0, color="gray", linewidth=0.6)
    else:
        axes[0, 0].set_ylim(0.0, 1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("cumulative epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False)
    fig.suptitle(f"{dataset} (band = +/-1 std over finite seeds)")
    return fig


def _mark_stage_boundaries(ax: Axes, stage_digits: list[list[int]], epochs_per_stage: int) -> None:
    """Dotted vertical line before each stage 1.., labelled with the class entering."""
    for stage, digits in enumerate(stage_digits[1:], start=1):
        boundary = stage * epochs_per_stage + 0.5
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.7)
        ax.annotate(
            ",".join(str(d) for d in digits),
            (boundary, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -8),
            textcoords="offset points",
            fontsize="xx-small",
            color="gray",
        )


def _warn_about_gaps(cells: dict[CellKey, Cell]) -> None:
    """Prints runs lost to divergence and cells missing from the expected grid."""
    for key, cell in sorted(cells.items()):
        if cell.num_finite < cell.num_runs:
            print(
                f"WARNING: {key}: {cell.num_runs - cell.num_finite} of {cell.num_runs} runs diverged (excluded)",
                file=sys.stderr,
            )
    expected = {
        (dataset, variant, scheme, weight)
        for dataset in {key[0] for key in cells}
        for variant, scheme in get_variant_scheme_columns()
        for weight in WEIGHTS
    }
    for key in sorted(expected - set(cells)):
        print(f"WARNING: no runs for {key}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "results"],
        help="Launch directories and/or results roots to read (default: results/).",
    )
    parser.add_argument("--out", type=Path, default=PLOTS_DIR, help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.paths, args.out)
