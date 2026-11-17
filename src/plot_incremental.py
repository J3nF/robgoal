"""Figures of the incremental-inhibition launches: heatmaps, newest-class curves, inhibition grids.

Thin script on `plotting_utils`: reads the run files of the given `--scan incremental` launches
(one dataset each) -- no MLflow, no training. Seeds are aggregated as mean with a +/-1 std band
(population std, `plotting_utils.mean_std`). Controls have no inhibition signal, so they appear in
the heatmaps and curves but not in the inhibition grids.

Class order: the base protocol introduces classes 0, 1, ..., so heatmaps are drawn by class
label. The mixed protocol draws a random order per seed (`meta["stage_digits"]`), which would
blur a label-averaged heatmap's introduction diagonal, so it also gets heatmaps by
introduction rank.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

import plotting_utils as pu
from data import MIXED_FASHION_OFFSET
from metrics import backward_transfer_over_time
from run_store import RunFile

__all__ = [
    "Cell",
    "build_cells",
    "get_model_label",
    "get_newest_class_curves",
    "main",
    "plot_heatmaps",
    "plot_inhibition_eval_comparison",
    "plot_inhibition_grid",
    "plot_newest_class_curves",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
SAVE_DPI = 300
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">")
_LAYER_KEY = re.compile(r"^inhibition/train_mean/(.+)$")


@dataclass(frozen=True)
class Cell:
    """All seeds of one model on one dataset.

    Attributes:
        label: Legend label, see `get_model_label`.
        scheme: "relu_sigmoid", "sigmoid", or "" for the CNN.
        runs: The seed replicates.
    """

    label: str
    scheme: str
    runs: list[RunFile]

    @property
    def dataset(self) -> str:
        return pu.run_field(self.runs[0], "dataset")

    @property
    def epochs_per_stage(self) -> int:
        return pu.run_field(self.runs[0], "epochs_per_stage")

    def accuracy(self) -> np.ndarray:
        """(seeds, epochs, classes) accuracy."""
        return pu.stack_arrays(self.runs, "accuracy")

    def stage_digits(self) -> list[list[list[int]]]:
        """Per seed, the classes each stage introduces."""
        return [run.meta["stage_digits"] for run in self.runs]


def get_model_label(run: RunFile) -> str:
    """Legend label of the model a run trained, e.g. `leaky lam=0.9 (relu_sigmoid)`."""
    config = run.meta["config"]
    if config["model_kind"] == "cnn":
        return "CNN control"
    scheme = config["activation_scheme"]
    if config.get("layer_kind", "unrolled") == "leaky":
        return f"leaky lam={config['decay']:g} ({scheme})"
    if config["T"] == 1:
        return f"FCNN control ({scheme})"
    return f"unrolled gate T={config['T']} ({scheme})"


def build_cells(runs: list[RunFile], dataset: str) -> list[Cell]:
    """One `Cell` per model of `dataset`, controls first, in a fixed order.

    Raises:
        ValueError: if no run is of `dataset`.
    """
    chosen = pu.select_runs(runs, dataset=dataset)
    by_label: dict[str, list[RunFile]] = {}
    for run in chosen:
        by_label.setdefault(get_model_label(run), []).append(run)

    for label, group in by_label.items():
        if len({run.meta["config_name"] for run in group}) > 1:
            raise ValueError(f"label {label!r} covers several configs; labels must be unique")

    def sort_key(label: str) -> tuple[int, str]:
        rank = 0 if "control" in label else 1 if "unrolled" in label else 2
        return rank, label

    return [
        Cell(label, "" if label == "CNN control" else label.rsplit("(", 1)[1].rstrip(")"), runs_)
        for label, runs_ in sorted(by_label.items(), key=lambda item: sort_key(item[0]))
    ]


def get_styles(cells: list[Cell]) -> dict[str, dict]:
    """Colour per model (fixed order), line style per scheme, marker per model."""
    colors = pu.get_colorcycle(len(cells))
    return {
        cell.label: {
            "color": colors[i],
            "marker": MARKERS[i % len(MARKERS)],
            "linestyle": "--" if cell.scheme == "sigmoid" else "-",
        }
        for i, cell in enumerate(cells)
    }


def get_width_text(cells: list[Cell]) -> str:
    """ "width W" of the FCNN hidden layers in `cells` (the CNN's channels are matched to it)."""
    for cell in cells:
        sizes = pu.run_field(cell.runs[0], "hidden_sizes")
        if sizes:
            return f"width {sizes[0]}"
    return "width n/a"


def _class_name(label: int) -> str:
    return str(label) if label < MIXED_FASHION_OFFSET else f"F{label - MIXED_FASHION_OFFSET}"


# --- heatmaps (2.0) ---------------------------------------------------------------------------


def plot_heatmaps(cells: list[Cell], dataset: str, tag: str, by_rank: bool = False) -> Figure:
    """One heatmap per model: classes on x, epochs on y, seed-mean accuracy as colour (0-1).

    Args:
        by_rank: Order each seed's classes by introduction before averaging (the mixed
            protocol's random order); a strip below shows the share of seeds in which each
            rank is a MNIST class.
    """
    num_seeds = len(cells[0].runs)
    columns = min(4, len(cells))
    rows = -(-len(cells) // columns)
    mixed = dataset == "mixed"
    fig, axes = plt.subplots(
        rows + (1 if by_rank else 0),
        columns,
        figsize=(2.7 * columns + 0.8, 2.3 * rows + (0.9 if by_rank else 0.6)),
        squeeze=False,
        gridspec_kw={"height_ratios": [1] * rows + ([0.12] if by_rank else [])},
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for cell, ax in zip(cells, axes.flat):
        ax.set_visible(True)
        _plot_cell_heatmap(cell, ax, mixed, by_rank)
        ax.set_title(cell.label, fontsize="x-small")
    if by_rank and mixed:
        share = np.mean(
            [
                [d < MIXED_FASHION_OFFSET for digits in sd for d in digits]
                for sd in cells[0].stage_digits()
            ],
            axis=0,
        )
        strip = axes[-1, 0]
        strip.set_visible(True)
        strip.imshow(share[None, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        strip.set_yticks([])
        strip.set_xlabel(
            "introduction rank; strip = share of seeds with a MNIST class (1) vs FashionMNIST (0)",
            fontsize="x-small",
        )
    sm = ScalarMappable(Normalize(0, 1), cmap="viridis")
    fig.colorbar(sm, ax=axes[:rows].ravel().tolist(), label="test accuracy (0-1)", shrink=0.8)
    order = "by introduction rank" if by_rank else "by class label"
    fig.suptitle(
        f"{tag}: per-class test accuracy over epochs, {dataset}, {get_width_text(cells)}, {order}, "
        f"seed mean ({num_seeds} seeds)",
        fontsize="small",
    )
    return fig


def _plot_cell_heatmap(cell: Cell, ax: Axes, mixed: bool, by_rank: bool) -> None:
    accuracy = cell.accuracy()
    eps = cell.epochs_per_stage
    if by_rank:
        orders = [[d for digits in sd for d in digits] for sd in cell.stage_digits()]
        accuracy = np.stack([acc[:, order] for acc, order in zip(accuracy, orders)])
        num_classes = accuracy.shape[2]
        stage_digits = cell.stage_digits()[0]
        introduced = [s for s, digits in enumerate(stage_digits) for _ in digits]
        labels: list[str] | None = [str(i) for i in range(num_classes)]
        spans = None
    else:
        sd = cell.stage_digits()[0]
        introduced_map = {d: s for s, digits in enumerate(sd) for d in digits}
        num_classes = accuracy.shape[2]
        # A label's introduction stage is seed-independent only for a fixed order.
        introduced = (
            None if mixed else [introduced_map.get(c, float("nan")) for c in range(num_classes)]
        )
        labels = [_class_name(c) for c in range(num_classes)] if mixed else None
        spans = [(0, 9, "MNIST"), (10, 19, "FashionMNIST")] if mixed else None
    pu.plot_class_time_heatmap(
        accuracy,
        ax,
        epochs_per_stage=eps,
        introduced_in=introduced,
        class_labels=labels,
        group_spans=spans,
        colorbar=False,
    )
    ax.tick_params(labelsize="xx-small")
    if num_classes > 12:
        for label in ax.get_xticklabels()[1::2]:
            label.set_visible(False)


# --- newest-class curves (2.1) ----------------------------------------------------------------


def get_newest_class_curves(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    """Per seed and stage: accuracy on the class(es) introduced in that stage, and backward transfer.

    Both are read after the last epoch of the stage. Backward transfer is
    `metrics.backward_transfer_over_time` (mean over all earlier-introduced classes), NaN in
    stage 0 where nothing is earlier.

    Returns:
        `(accuracy, bwt)`, each of shape (seeds, stages).
    """
    eps = cell.epochs_per_stage
    accuracies, bwts = [], []
    for accuracy, stage_digits in zip(cell.accuracy(), cell.stage_digits()):
        last_rows = [(s + 1) * eps - 1 for s in range(len(stage_digits))]
        accuracies.append(
            [accuracy[row, digits].mean() for row, digits in zip(last_rows, stage_digits)]
        )
        bwts.append(backward_transfer_over_time(accuracy, eps, stage_digits)[last_rows])
    return np.array(accuracies), np.array(bwts)


def plot_newest_class_curves(cells: list[Cell], dataset: str, tag: str) -> Figure:
    """Newest-class accuracy and backward transfer per stage, all models in the same axes."""
    num_seeds = len(cells[0].runs)
    styles = get_styles(cells)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
    low = 0.0
    for cell in cells:
        accuracy, bwt = get_newest_class_curves(cell)
        stages = np.arange(accuracy.shape[1])
        for ax, values in zip(axes, (accuracy, bwt)):
            mean, std = pu.mean_std(values)
            pu.plot_line_band(
                stages,
                mean,
                std,
                ax,
                label=cell.label,
                markersize=3,
                linewidth=1.1,
                **styles[cell.label],
            )
        low = min(low, float(np.nanmin(bwt.mean(axis=0) - bwt.std(axis=0))))
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("accuracy, newest class (0-1)")
    axes[1].set_ylim(low * 1.05, max(0.0, 0.05))
    axes[1].axhline(0, color="gray", linewidth=0.6)
    axes[1].set_ylabel("backward transfer (accuracy change)")
    for ax in axes:
        ax.set_xlabel("stage (one new class per stage)")
        ax.tick_params(labelsize="x-small")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False, fontsize="x-small")
    fig.suptitle(
        f"{tag}: newest class per stage, {dataset}, {get_width_text(cells)}, "
        f"mean +/- std over {num_seeds} seeds; "
        "BWT axis linear, spans the data",
        fontsize="small",
    )
    return fig


# --- inhibition per layer (2.4) ---------------------------------------------------------------


def get_layers(cell: Cell) -> list[str]:
    """Names of the layers whose inhibition the runs recorded, in layer order."""
    names = [m.group(1) for p in cell.runs[0].dataset_paths() if (m := _LAYER_KEY.match(p))]
    return sorted(names, key=lambda n: int(n.rsplit(".", 1)[1]))


def _train_traces(cell: Cell, layer: str, statistic: str) -> list[np.ndarray]:
    """Per stage, a (seeds, batches) array; seeds truncated to the shortest stage trace."""
    per_seed = []
    for run in cell.runs:
        stage = run.array("inhibition/train_batch_stage").read()
        values = run.array(f"inhibition/train_{statistic}/{layer}").read()
        per_seed.append([values[stage == s] for s in range(int(stage.max()) + 1)])
    traces = []
    for s in range(len(per_seed[0])):
        length = min(len(seed[s]) for seed in per_seed)
        traces.append(np.stack([seed[s][:length] for seed in per_seed]))
    return traces


def _eval_series(cell: Cell, layer: str, statistic: str) -> np.ndarray:
    return pu.stack_arrays(cell.runs, f"inhibition/test_{statistic}/{layer}")


def get_row_limits(cells: list[Cell], statistic: str) -> dict[str, float]:
    """Largest value (mean + std) per layer over all models, train and eval, for shared axes."""
    limits: dict[str, float] = {}
    for cell in cells:
        for layer in get_layers(cell):
            top = limits.get(layer, 0.0)
            for trace in _train_traces(cell, layer, statistic):
                top = max(top, float((trace.mean(0) + trace.std(0)).max()))
            series = _eval_series(cell, layer, statistic)
            top = max(top, float((series.mean(0) + series.std(0)).max()))
            limits[layer] = top
    return limits


def plot_inhibition_grid(
    cell: Cell, dataset: str, tag: str, statistic: str, limits: dict[str, float]
) -> Figure:
    """One row per layer: training traces (one per stage) left, stage-end test values right.

    Both columns of a row share the y axis, which spans 0 to the largest value over all
    models (`limits`), so models are comparable.
    """
    num_seeds = len(cell.runs)
    layers = get_layers(cell)
    fig, axes = plt.subplots(
        len(layers), 2, figsize=(6.4, 1.5 * len(layers) + 0.9), squeeze=False, sharey="row"
    )
    num_stages = len(_eval_series(cell, layers[0], statistic)[0])
    colors = pu.get_colorcycle(num_stages)
    for row, layer in enumerate(layers):
        for stage, trace in enumerate(_train_traces(cell, layer, statistic)):
            mean, std = pu.mean_std(trace)
            pu.plot_line_band(
                np.arange(len(mean)), mean, std, axes[row, 0], color=colors[stage], linewidth=0.7
            )
        mean, std = pu.mean_std(_eval_series(cell, layer, statistic))
        pu.plot_line_band(
            np.arange(num_stages),
            mean,
            std,
            axes[row, 1],
            color="black",
            marker="o",
            markersize=3,
            linewidth=1.0,
        )
        axes[row, 0].set_ylim(0, limits[layer] * 1.05)
        axes[row, 0].set_ylabel(f"{layer}\n{statistic}", fontsize="x-small")
        axes[row, 0].tick_params(labelsize="xx-small")
        axes[row, 1].tick_params(labelsize="xx-small")
    axes[-1, 0].set_xlabel("training batch within the stage (all epochs)", fontsize="x-small")
    axes[-1, 1].set_xlabel("stage (test set, after the stage)", fontsize="x-small")
    axes[0, 0].set_title("training", fontsize="small")
    axes[0, 1].set_title("test", fontsize="small")
    fig.colorbar(
        ScalarMappable(Normalize(0, num_stages - 1), cmap="viridis"),
        ax=axes.ravel().tolist(),
        label="stage",
        shrink=0.8,
    )
    fig.suptitle(
        f"{tag}: inhibition {statistic} per layer, {cell.label}, {dataset}, "
        f"{get_width_text([cell])}, mean +/- std over {num_seeds} seeds; "
        "controls have no inhibition and are not shown",
        fontsize="x-small",
    )
    return fig


def plot_inhibition_eval_comparison(cells: list[Cell], dataset: str, tag: str) -> Figure:
    """Test-set inhibition mean (left) and variance (right) per layer, one line per model."""
    inhibiting = [c for c in cells if get_layers(c)]
    num_seeds = len(inhibiting[0].runs)
    styles = get_styles(inhibiting)
    layers = get_layers(inhibiting[0])
    fig, axes = plt.subplots(
        len(layers), 2, figsize=(7.0, 1.5 * len(layers) + 0.9), squeeze=False, sharex=True
    )
    for col, statistic in enumerate(("mean", "var")):
        limits = get_row_limits(inhibiting, statistic)
        for row, layer in enumerate(layers):
            for cell in inhibiting:
                mean, std = pu.mean_std(_eval_series(cell, layer, statistic))
                pu.plot_line_band(
                    np.arange(len(mean)),
                    mean,
                    std,
                    axes[row, col],
                    label=cell.label,
                    markersize=3,
                    linewidth=1.0,
                    **styles[cell.label],
                )
            axes[row, col].set_ylim(0, limits[layer] * 1.05)
            axes[row, col].tick_params(labelsize="xx-small")
        axes[0, col].set_title(f"test-set {statistic}", fontsize="small")
        axes[-1, col].set_xlabel("stage", fontsize="x-small")
    for row, layer in enumerate(layers):
        axes[row, 0].set_ylabel(layer, fontsize="x-small")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False, fontsize="x-small")
    fig.suptitle(
        f"{tag}: inhibition after each stage, {dataset}, {get_width_text(inhibiting)}, "
        f"mean +/- std over {num_seeds} seeds",
        fontsize="small",
    )
    return fig


# --- driver -----------------------------------------------------------------------------------


def _save(fig: Figure, directory: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        pu.save_figure(fig, directory / f"{name}.{suffix}", dpi=SAVE_DPI, transparent=False)
    plt.close(fig)
    print(f"saved {directory / name}.png")


def main(paths: list[Path], out_dir: Path, tag: str) -> None:
    """Writes every figure for every dataset found under `paths` into `out_dir`."""
    pu.plot_setup()
    runs = pu.load_runs(*paths)
    for dataset in sorted({pu.run_field(run, "dataset") for run in runs}):
        cells = build_cells(runs, dataset)
        _save(plot_heatmaps(cells, dataset, tag), out_dir, f"heatmap_{dataset}")
        if dataset == "mixed":
            _save(
                plot_heatmaps(cells, dataset, tag, by_rank=True), out_dir, "heatmap_mixed_by_rank"
            )
        _save(plot_newest_class_curves(cells, dataset, tag), out_dir, f"newest_class_{dataset}")
        inhibiting = [c for c in cells if get_layers(c)]
        if not inhibiting:
            continue
        _save(
            plot_inhibition_eval_comparison(cells, dataset, tag),
            out_dir,
            f"inhibition_eval_{dataset}",
        )
        for statistic in ("mean", "var"):
            limits = get_row_limits(inhibiting, statistic)
            for cell in inhibiting:
                slug = re.sub(r"[^a-z0-9]+", "_", cell.label.lower()).strip("_")
                _save(
                    plot_inhibition_grid(cell, dataset, tag, statistic, limits),
                    out_dir,
                    f"inhibition_{statistic}_{dataset}_{slug}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="launch directories")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", default="coarse", help="'coarse' or 'exhaustive', shown in titles")
    arguments = parser.parse_args()
    main(arguments.paths, arguments.out_dir, arguments.tag)
