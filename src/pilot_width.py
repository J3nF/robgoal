"""Size pilot analysis: where do the controls show catastrophic interference? Picks the width `W`.

Reads the run files of a `--scan width` launch. Definition, fixed before the first launch:

- `D` = last value of `metrics.backward_transfer_over_time`: the mean over classes introduced
  before the last stage of (final accuracy - accuracy at the end of the stage that introduced it).
- 95 % t-interval of `D` over seeds, per (family, width).
- Interference PRESENT at a width: upper bound of the interval < `PRESENT_BELOW` (-0.05).
- Interference ABSENT at a width: lower bound > `ABSENT_ABOVE` (-0.02) and the seed-mean final
  average accuracy >= `MIN_FINAL_ACCURACY` (0.8), so a network that never learned does not count.
- The decision uses the FCNN controls: a width counts only if both activation schemes satisfy it.
  The CNN control is reported, not decisive.
- `W_low` = largest width with interference present; `W_high` = smallest larger width with it
  absent; `W` = their geometric mean rounded to a multiple of 4 (of 2 below 8).
"""

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plotting_utils
from metrics import backward_transfer_over_time
from run_store import RunFile
from weight_scan_stats import t_quantile

__all__ = [
    "ABSENT_ABOVE",
    "MIN_FINAL_ACCURACY",
    "PRESENT_BELOW",
    "WidthChoice",
    "WidthRow",
    "build_width_table",
    "choose_width",
    "round_width",
]

PRESENT_BELOW = -0.05
ABSENT_ABOVE = -0.02
MIN_FINAL_ACCURACY = 0.8
_WIDTH_PATTERN = re.compile(r"^width(\d+)_")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class WidthRow:
    """`D` and final accuracy of one family at one width, over seeds.

    Attributes:
        family: "fcnn relu_sigmoid", "fcnn sigmoid" or "cnn".
        width: FCNN hidden width the config was built for (the CNN's channels are matched to it).
        num_seeds: Number of seeds.
        d_mean, d_low, d_high: Mean and 95 % t-interval of `D`.
        accuracy_mean: Seed-mean final average accuracy over all classes.
    """

    family: str
    width: int
    num_seeds: int
    d_mean: float
    d_low: float
    d_high: float
    accuracy_mean: float

    @property
    def present(self) -> bool:
        """Whole interval below `PRESENT_BELOW`."""
        return self.d_high < PRESENT_BELOW

    @property
    def absent(self) -> bool:
        """Whole interval above `ABSENT_ABOVE` and the network learned (accuracy floor)."""
        return self.d_low > ABSENT_ABOVE and self.accuracy_mean >= MIN_FINAL_ACCURACY


@dataclass(frozen=True)
class WidthChoice:
    """The chosen width and how it was reached.

    Attributes:
        width: The chosen `W`.
        w_low: Largest width with interference present, or `None`.
        w_high: Smallest larger width with interference absent, or `None`.
        note: What was done when an end was missing, empty otherwise.
    """

    width: int
    w_low: int | None
    w_high: int | None
    note: str


def build_width_table(runs: list[RunFile]) -> list[WidthRow]:
    """One `WidthRow` per (family, width), sorted by family then width.

    Raises:
        ValueError: if a run's config name does not start with `width<W>_`, or a group has
            fewer than 2 seeds.
    """
    per_cell: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for run in runs:
        width = _get_width(run)
        family = _get_family(run)
        accuracy = run.array("accuracy").read()
        epochs_per_stage = run.meta["config"]["epochs_per_stage"]
        d = backward_transfer_over_time(accuracy, epochs_per_stage, run.meta["stage_digits"])[-1]
        per_cell.setdefault((family, width), []).append((float(d), float(accuracy[-1].mean())))
    rows = []
    for (family, width), values in sorted(per_cell.items()):
        if len(values) < 2:
            raise ValueError(f"{family} width {width} has {len(values)} seed(s); need >= 2")
        d_values = np.array([d for d, _ in values])
        mean = float(d_values.mean())
        half = (
            t_quantile(0.975, len(values) - 1)
            * float(d_values.std(ddof=1))
            / math.sqrt(len(values))
        )
        accuracy_mean = float(np.mean([a for _, a in values]))
        rows.append(
            WidthRow(family, width, len(values), mean, mean - half, mean + half, accuracy_mean)
        )
    return rows


def choose_width(rows: list[WidthRow]) -> WidthChoice:
    """Applies the module's rule to the FCNN rows; falls back to the closest width, with a note.

    Raises:
        ValueError: if `rows` has no FCNN rows.
    """
    fcnn = [row for row in rows if row.family.startswith("fcnn")]
    if not fcnn:
        raise ValueError("no FCNN rows to choose a width from")
    widths = sorted({row.width for row in fcnn})
    schemes = len({row.family for row in fcnn})

    def holds(width: int, check: str) -> bool:
        cell = [row for row in fcnn if row.width == width]
        return len(cell) == schemes and all(getattr(row, check) for row in cell)

    present = [w for w in widths if holds(w, "present")]
    w_low = max(present) if present else None
    absent = [w for w in widths if holds(w, "absent") and (w_low is None or w > w_low)]
    w_high = min(absent) if absent else None
    if w_low is not None and w_high is not None:
        return WidthChoice(round_width(math.sqrt(w_low * w_high)), w_low, w_high, "")
    if w_low is None and w_high is None:
        closest = min(fcnn, key=lambda row: abs(row.d_mean - (PRESENT_BELOW + ABSENT_ABOVE) / 2))
        return WidthChoice(
            closest.width, None, None, "neither end found; closest mean D to the band's middle"
        )
    if w_low is None:
        assert w_high is not None
        return WidthChoice(
            w_high, None, w_high, "no width shows interference; smallest absent width"
        )
    return WidthChoice(w_low, w_low, None, "no width without interference; largest present width")


def round_width(width: float) -> int:
    """Rounds to a multiple of 4 (of 2 below 8), at least 2."""
    step = 4 if width >= 8 else 2
    return max(2, round(width / step) * step)


def _get_width(run: RunFile) -> int:
    match = _WIDTH_PATTERN.match(run.meta["config_name"])
    if match is None:
        raise ValueError(f"{run.run_id}: config name {run.meta['config_name']!r} is not width<W>_*")
    return int(match.group(1))


def _get_family(run: RunFile) -> str:
    config = run.meta["config"]
    if config["model_kind"] == "cnn":
        return "cnn"
    return f"fcnn {config['activation_scheme']}"


def write_outputs(rows: list[WidthRow], choice: WidthChoice, plots_dir: Path) -> None:
    """Writes `width_table.csv`, `width_choice.txt` and the `D` versus width figure."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    with (plots_dir / "width_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family",
                "width",
                "seeds",
                "D_mean",
                "D_ci_low",
                "D_ci_high",
                "final_accuracy",
                "present",
                "absent",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.family,
                    row.width,
                    row.num_seeds,
                    f"{row.d_mean:.4f}",
                    f"{row.d_low:.4f}",
                    f"{row.d_high:.4f}",
                    f"{row.accuracy_mean:.4f}",
                    row.present,
                    row.absent,
                ]
            )
    (plots_dir / "width_choice.txt").write_text(
        f"W={choice.width} W_low={choice.w_low} W_high={choice.w_high} {choice.note}\n"
    )
    _plot_width_table(rows, choice, plots_dir)


def _plot_width_table(rows: list[WidthRow], choice: WidthChoice, plots_dir: Path) -> None:
    plotting_utils.plot_setup()
    families = sorted({row.family for row in rows})
    colors = plotting_utils.get_colorcycle(len(families))
    markers = ["o", "s", "^"]
    styles = ["-", "--", ":"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True)
    for family, color, marker, style in zip(families, colors, markers, styles):
        own = [row for row in rows if row.family == family]
        widths = np.array([row.width for row in own])
        d = np.array([row.d_mean for row in own])
        low = np.array([row.d_low for row in own])
        high = np.array([row.d_high for row in own])
        axes[0].errorbar(
            widths,
            d,
            yerr=[d - low, high - d],
            color=color,
            marker=marker,
            linestyle=style,
            capsize=2,
            label=family,
        )
        axes[1].plot(
            widths,
            [row.accuracy_mean for row in own],
            color=color,
            marker=marker,
            linestyle=style,
        )
    axes[0].axhspan(-1, PRESENT_BELOW, color="gray", alpha=0.15, linewidth=0)
    axes[0].axhline(ABSENT_ABOVE, color="gray", linestyle=":", linewidth=0.8)
    axes[1].axhline(MIN_FINAL_ACCURACY, color="gray", linestyle=":", linewidth=0.8)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("hidden width W (log2 scale)")
        if choice.width:
            ax.axvline(choice.width, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("final backward transfer D")
    axes[1].set_ylabel("final avg accuracy (all classes)")
    axes[0].set_ylim(min(-0.05, min(row.d_low for row in rows)) * 1.05, 0.05)
    axes[1].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", frameon=False)
    fig.suptitle(
        f"Size pilot (coarse: 1 epoch/stage), MNIST, {rows[0].num_seeds} seeds; mean and 95% t-interval; "
        f"chosen W={choice.width}"
    )
    for suffix in ("png", "pdf"):
        plotting_utils.save_figure(fig, plots_dir / f"width_pilot.{suffix}", dpi=300)
    plt.close(fig)


def main(paths: list[Path], plots_dir: Path) -> None:
    """Loads the runs, prints the table and the choice, writes the outputs."""
    rows = build_width_table(plotting_utils.load_runs(*paths))
    choice = choose_width(rows)
    for row in rows:
        print(
            f"{row.family:18s} W={row.width:4d} D={row.d_mean:+.3f} "
            f"[{row.d_low:+.3f}, {row.d_high:+.3f}] acc={row.accuracy_mean:.3f} "
            f"present={row.present} absent={row.absent}"
        )
    print(f"chosen W={choice.width} (W_low={choice.w_low}, W_high={choice.w_high}) {choice.note}")
    write_outputs(rows, choice, plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="launch directories")
    parser.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "results/plots/size_pilot")
    arguments = parser.parse_args()
    main(arguments.paths, arguments.plots_dir)
    sys.exit(0)
