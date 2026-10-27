"""Loads the ai2 weight-scan runs, aggregates them over seeds, and writes the summary CSVs.

Entry points: `load_weight_scan` (run files to `Cell`s), `write_unroll_summary_csv`,
`write_scan_summary_csv`, `write_q2_tests_csv`. The figures live in `plot_weight_scan.py`.
A "cell" is one (dataset, variant, scheme, weight) with all its seeds.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import plotting_utils
from data import DatasetName
from run_store import RunFile
from sweep import SCHEMES, VARIANTS, WEIGHT_SCAN_WEIGHTS
from weight_scan_stats import holm_adjust, welch_test

__all__ = [
    "DATASETS",
    "WEIGHTS",
    "Cell",
    "CellKey",
    "get_depth_log_ratios",
    "get_variant_scheme_columns",
    "load_weight_scan",
    "write_q2_tests_csv",
    "write_scan_summary_csv",
    "write_unroll_summary_csv",
]

DATASETS: tuple[DatasetName, ...] = ("mnist", "fashion_mnist")
WEIGHTS: tuple[float, ...] = tuple(round(weight, 6) for weight in WEIGHT_SCAN_WEIGHTS)
NUM_LAYERS = 4
NUM_CLASSES = 10
UNROLL_ARRAYS = (
    "unroll/mean_y",
    "unroll/mean_y_self_used",
    "unroll/mean_y_tilde",
    "unroll/mean_abs_y_tilde_class",
    "unroll/mean_abs_y_tilde",
    "unroll/neuron_std_y_tilde",
)
FINITE_CHECKED = ("train_loss", *UNROLL_ARRAYS)
SCALAR_METRICS = ("final_avg_accuracy", "final_backward_transfer")

CellKey = tuple[str, str, str, float]  # dataset, ai2_variant, activation_scheme, ai2_weight


@dataclass(frozen=True)
class Cell:
    """All seeds of one (dataset, variant, scheme, weight) config.

    Attributes:
        num_runs: Runs found, including diverged ones.
        arrays: Name -> array stacked over the *finite* seeds (axis 0). Diverged runs (a
            non-finite training loss or trace) are left out of every aggregate and only counted.
            Holds the datasets of the run files plus the `SCALAR_METRICS`, which are (seeds,).
        epochs_per_stage: Epochs each stage trained for.
    """

    num_runs: int
    arrays: dict[str, np.ndarray]
    epochs_per_stage: int

    @property
    def num_finite(self) -> int:
        """Number of seeds that did not diverge."""
        return len(self.arrays["train_loss"]) if self.arrays else 0


def load_weight_scan(*paths: Path) -> dict[CellKey, Cell]:
    """The weight-scan cells found under `paths` (launch directories and/or results roots).

    Raises:
        ValueError: if a scan run stores `accuracy` with other dims than (epoch, digit), i.e. it
            predates per-epoch evaluation, or if no weight-scan run with one initial class exists.
    """
    runs = plotting_utils.load_runs(*paths)
    scan_runs = plotting_utils.select_runs(
        runs, sweep_axis="ai2_weight", initial_classes=1, classes_per_stage=1
    )
    groups = plotting_utils.group_runs(
        scan_runs, "dataset", "ai2_variant", "activation_scheme", "ai2_weight"
    )
    return {
        (dataset, variant, scheme, round(weight, 6)): _build_cell(group)
        for (dataset, variant, scheme, weight), group in groups.items()
    }


def _build_cell(group: list[RunFile]) -> Cell:
    for run in group:
        dims = run.array("accuracy").dims
        if dims != ("epoch", "digit"):
            raise ValueError(
                f"{run.path}: accuracy has dims {dims}, expected ('epoch', 'digit'); the run "
                "predates per-epoch evaluation and cannot be plotted."
            )
    finite = [run for run in group if _is_finite(run)]
    arrays: dict[str, np.ndarray] = {}
    if finite:
        for name in ("accuracy", *FINITE_CHECKED):
            arrays[name] = plotting_utils.stack_arrays(finite, name)
        for name in SCALAR_METRICS:
            arrays[name] = np.array([run.meta["metrics"][name] for run in finite])
    return Cell(len(group), arrays, plotting_utils.run_field(group[0], "epochs_per_stage"))


def _is_finite(run: RunFile) -> bool:
    return all(np.isfinite(run.array(name).read()).all() for name in FINITE_CHECKED)


def get_depth_log_ratios(cell: Cell) -> dict[str, np.ndarray]:
    """Per-seed log ratio of the deepest to the first layer at the last unroll step.

    Keys: `abs` (mean |y_tilde|) and `std` (mean per-neuron std of y_tilde over samples).
    A negative value means the signal shrinks with depth. Empty if `cell` has no finite seed.
    """
    if cell.num_finite == 0:
        return {}
    ratios = {}
    for key, name in (("abs", "unroll/mean_abs_y_tilde"), ("std", "unroll/neuron_std_y_tilde")):
        last_step = cell.arrays[name][:, :, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[key] = np.log(last_step[:, -1] / last_step[:, 0])
    return ratios


def write_unroll_summary_csv(cells: dict[CellKey, Cell], path: Path) -> None:
    """One row per (dataset, variant, scheme, weight, layer, class, step): seed means and stds.

    `y` is constant over steps and repeated for every step.
    """
    columns = ["dataset", "variant", "scheme", "weight", "layer", "class", "step", "num_finite"]
    quantities = ("y", "y_self_used", "y_tilde", "abs_y_tilde")
    for quantity in quantities:
        columns += [f"{quantity}_mean", f"{quantity}_std"]
    sources = {
        "y": "unroll/mean_y",
        "y_self_used": "unroll/mean_y_self_used",
        "y_tilde": "unroll/mean_y_tilde",
        "abs_y_tilde": "unroll/mean_abs_y_tilde_class",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for key, cell in sorted(cells.items()):
            if cell.num_finite == 0:
                continue
            stats = {q: plotting_utils.mean_std(cell.arrays[sources[q]]) for q in quantities}
            num_steps = cell.arrays["unroll/mean_y_tilde"].shape[-1]
            for layer, digit, step in np.ndindex(NUM_LAYERS, NUM_CLASSES, num_steps):
                row = [*key, layer + 1, digit, step, cell.num_finite]
                for quantity in quantities:
                    mean, std = stats[quantity]
                    index = (layer, digit) if quantity == "y" else (layer, digit, step)
                    row += [mean[index], std[index]]
                writer.writerow(row)


def write_scan_summary_csv(cells: dict[CellKey, Cell], path: Path) -> None:
    """One row per cell: final metrics and per-layer die-off statistics (seed mean and std).

    `backward_transfer` is the existing scalar (digit 0 only, final stage).
    `ratio_last_over_first_layerL` is mean |y_tilde| at the last step over step 0 (exactly 1
    for weight 0). `abs_last`, `abs_prelast`, `abs_stepmean` are mean |y_tilde| at the last
    step, the step before, and averaged over steps 1..T-1; they show the unroll-parity effect.
    `neuron_std_last` is the input-dependent part at the last step.
    """
    metrics = {
        "ratio_last_over_first": lambda a: a[:, :, -1] / a[:, :, 0],
        "abs_last": lambda a: a[:, :, -1],
        "abs_prelast": lambda a: a[:, :, -2],
        "abs_stepmean": lambda a: a[:, :, 1:].mean(axis=-1),
    }
    columns = ["dataset", "variant", "scheme", "weight", "num_runs", "num_finite"]
    columns += ["accuracy_mean", "accuracy_std", "backward_transfer_mean", "backward_transfer_std"]
    for name in (*metrics, "neuron_std_last"):
        columns += [
            f"{name}_layer{layer}_{stat}" for layer in range(1, 5) for stat in ("mean", "std")
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for key, cell in sorted(cells.items()):
            writer.writerow([*key, cell.num_runs, cell.num_finite, *_summary_values(cell, metrics)])


def _summary_values(cell: Cell, metrics: dict) -> list[float]:
    """The metric columns of `write_scan_summary_csv` for one cell (NaN without finite seeds)."""
    if cell.num_finite == 0:
        return [math.nan] * (4 + 5 * 2 * NUM_LAYERS)
    values: list[float] = []
    for name in SCALAR_METRICS:
        values += [float(cell.arrays[name].mean()), float(cell.arrays[name].std())]
    abs_y_tilde = cell.arrays["unroll/mean_abs_y_tilde"]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_metric = [function(abs_y_tilde) for function in metrics.values()]
    per_metric.append(cell.arrays["unroll/neuron_std_y_tilde"][:, :, -1])
    for per_seed in per_metric:
        mean, std = plotting_utils.mean_std(per_seed)
        for layer in range(NUM_LAYERS):
            values += [float(mean[layer]), float(std[layer])]
    return values


def write_q2_tests_csv(cells: dict[CellKey, Cell], path: Path) -> None:
    """Weight-minus-control tests of the depth log ratios (the Q2 evidence), Holm-adjusted.

    Each row compares one cell with the weight-0 cell of the same dataset, variant and scheme
    on the per-seed log of (layer 4 / layer 1) at the last step, for `abs` and `std`.
    Welch test (sample variances, integer-floored df); Holm over all rows with a p-value, the
    two statistics pooled. Seeds are not paired across cells.
    """
    rows = []
    for (dataset, variant, scheme, weight), cell in sorted(cells.items()):
        control = cells.get((dataset, variant, scheme, 0.0))
        if weight == 0.0 or control is None or control.num_finite == 0:
            continue
        for statistic, values in get_depth_log_ratios(cell).items():
            result = welch_test(values, get_depth_log_ratios(control)[statistic])
            rows.append((dataset, variant, scheme, weight, statistic, cell.num_finite, result))
    adjusted = holm_adjust([row[-1].p for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "variant", "scheme", "weight", "statistic", "num_finite", "diff_log_ratio"]
            + ["se", "df", "p", "p_holm", "ci_low", "ci_high"]
        )
        for row, p_holm in zip(rows, adjusted):
            *head, result = row
            writer.writerow(
                [
                    *head,
                    result.diff,
                    result.se,
                    result.df,
                    result.p,
                    p_holm,
                    result.ci_low,
                    result.ci_high,
                ]
            )


def get_variant_scheme_columns() -> list[tuple[str, str]]:
    """The four (variant, scheme) panels in plotting order: variants outer, schemes inner."""
    return [(variant, scheme) for variant in VARIANTS for scheme in SCHEMES]
