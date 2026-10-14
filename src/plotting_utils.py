"""Loading, grouping and plotting of run files (numpy + matplotlib only).

The pipeline stays plain: `load_runs` -> `select_runs` / `group_runs` -> `stack_arrays` (seeds
x ...) -> `mean_std` -> one of the plot functions. Every plot function takes arrays plus an
optional `ax` (one is created if `None`; `figsize` overrides the rcParams default only then)
and returns the `Axes`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from run_store import RunFile, list_runs

__all__ = [
    "get_colorcycle",
    "group_runs",
    "load_runs",
    "mean_std",
    "plot_accuracy_heatmap",
    "plot_activity_heatmap",
    "plot_epoch_curves",
    "plot_line_band",
    "plot_setup",
    "run_field",
    "save_figure",
    "select_runs",
    "stack_arrays",
]


def plot_setup(fig_dpi=100, save_dpi=1200):
    """Sets the house style: small constrained-layout figures saved at high dpi, transparent.

    Small figures at high dpi keep text large relative to the figure; this is deliberate.
    """
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.dpi"] = fig_dpi
    plt.rcParams["figure.figsize"] = (3, 1.5)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.dpi"] = save_dpi
    plt.rcParams["savefig.transparent"] = True


def get_colorcycle(N_colors, cmap="viridis"):
    """`N_colors` evenly spaced colors of `cmap`, avoiding its extreme ends."""
    colormap = plt.get_cmap(cmap)
    return [colormap(i) for i in np.linspace(1 / N_colors, 1 - 1 / N_colors, N_colors)]


# --- loading, selecting, grouping -------------------------------------------------------------


def load_runs(*paths: Path) -> list[RunFile]:
    """All runs under the given launch directories and/or results roots, sorted by `run_id`.

    Raises:
        ValueError: if no run file is found, naming the paths searched.
    """
    runs = [run for path in paths for run in list_runs(Path(path))]
    if not runs:
        raise ValueError(f"No run files (*.h5) found under {[str(path) for path in paths]}.")
    return sorted(runs, key=lambda run: run.run_id)


def run_field(run: RunFile, key: str) -> Any:
    """`key` looked up in the run's config first, then in the top-level meta.

    So `"dataset"` and `"hidden_sizes"` come from the config, `"config_name"` and `"launch_id"`
    from the meta. Lists (JSON's tuples) become tuples, so values are comparable and hashable.

    Raises:
        KeyError: if neither has `key`; the message lists the valid keys.
    """
    for source in (run.meta["config"], run.meta):
        if key in source:
            value = source[key]
            return tuple(value) if isinstance(value, list) else value
    valid = sorted({*run.meta["config"], *run.meta})
    raise KeyError(f"Unknown run field {key!r}. Valid fields: {valid}")


def select_runs(runs: Sequence[RunFile], **fields: Any) -> list[RunFile]:
    """Runs whose fields equal all of `fields` (e.g. `dataset="mnist", model_kind="fcnn"`).

    Raises:
        ValueError: if nothing matches, naming the filter and what values exist.
        KeyError: if a field name is unknown.
    """
    wanted = {
        key: tuple(value) if isinstance(value, list) else value for key, value in fields.items()
    }
    selected = [
        run for run in runs if all(run_field(run, key) == value for key, value in wanted.items())
    ]
    if not selected:
        raise ValueError(f"No run matches {wanted} among {len(runs)} runs.")
    return selected


def group_runs(runs: Sequence[RunFile], *keys: str) -> dict[tuple, list[RunFile]]:
    """Groups runs by the tuple of their `keys` fields (e.g. `"config_name", "dataset"`).

    Groups may have different sizes (different seed counts). Group keys come back sorted when
    their values are comparable, in first-seen order otherwise.
    """
    groups: dict[tuple, list[RunFile]] = {}
    for run in runs:
        groups.setdefault(tuple(run_field(run, key) for key in keys), []).append(run)
    try:
        return dict(sorted(groups.items(), key=lambda item: item[0]))
    except TypeError:
        return groups


def stack_arrays(runs: Sequence[RunFile], dataset: str) -> np.ndarray:
    """Stacks `dataset` over `runs` into an array of shape (seeds, ...).

    Raises:
        ValueError: if `runs` is empty or the arrays differ in shape (e.g. runs with different
            epoch counts); the message names the shapes.
    """
    if not runs:
        raise ValueError("Cannot stack arrays of zero runs.")
    arrays = [run.array(dataset).read() for run in runs]
    shapes = {array.shape for array in arrays}
    if len(shapes) > 1:
        detail = ", ".join(f"{run.run_id}: {array.shape}" for run, array in zip(runs, arrays))
        raise ValueError(f"Cannot stack {dataset!r}: shapes differ ({detail}).")
    return np.stack(arrays)


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation (ddof=0) over `axis`, normally the seed axis."""
    return stack.mean(axis=axis), stack.std(axis=axis)


# --- plots ----------------------------------------------------------------------------------------


def plot_line_band(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None = None,
    ax: Axes | None = None,
    *,
    logx: bool = False,
    figsize: tuple[float, float] | None = None,
    **line_kwargs: Any,
) -> Axes:
    """Line of `mean` against `x` with a `mean +/- std` band, e.g. accuracy vs. network size.

    Args:
        std: Half-width of the band; no band if `None`.
        logx: Log-scale x axis.
        line_kwargs: Passed to `Axes.plot` (color, label, linestyle, ...); the band reuses
            the line's color.
    """
    ax = _get_axes(ax, figsize)
    (line,) = ax.plot(x, mean, **line_kwargs)
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, linewidth=0)
    if logx:
        ax.set_xscale("log")
    return ax


def plot_epoch_curves(
    curves: np.ndarray,
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    **line_kwargs: Any,
) -> Axes:
    """Per-epoch curve, e.g. `train_loss` or `y_self_absmean/*`; (seeds, epochs) adds a band.

    Args:
        curves: Shape (epochs,) for one run, or (seeds, epochs) for mean +/- std over seeds.
    """
    curves = np.asarray(curves)
    if curves.ndim == 1:
        mean, std = curves, None
    elif curves.ndim == 2:
        mean, std = mean_std(curves)
    else:
        raise ValueError(f"curves must be (epochs,) or (seeds, epochs), got shape {curves.shape}")
    ax = plot_line_band(np.arange(len(mean)), mean, std, ax, figsize=figsize, **line_kwargs)
    ax.set_xlabel("epoch")
    return ax


def plot_accuracy_heatmap(
    accuracy: np.ndarray,
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Axes:
    """Heatmap of the (stage, digit) accuracy matrix of one run, or the seed mean.

    Args:
        accuracy: Shape (stages, digits), or (seeds, stages, digits) which is averaged over
            seeds first. Colors span [0, 1].
    """
    accuracy = np.asarray(accuracy)
    if accuracy.ndim == 3:
        accuracy = accuracy.mean(axis=0)
    if accuracy.ndim != 2:
        raise ValueError(f"accuracy must be (stages, digits) or (seeds, ...), got {accuracy.shape}")
    ax = _get_axes(ax, figsize)
    image = ax.imshow(accuracy, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(image, ax=ax, label="accuracy")
    ax.set_xlabel("digit")
    ax.set_ylabel("stage")
    ax.set_xticks(range(accuracy.shape[1]))
    ax.set_yticks(range(accuracy.shape[0]))
    return ax


def plot_activity_heatmap(
    activity: np.ndarray,
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    xlabel: str = "time step",
) -> Axes:
    """Raster of a (time, neuron) array, e.g. `y_self_per_neuron_batch/*`; time runs along x.

    Raises:
        ValueError: if `activity` is not 2-dimensional.
    """
    activity = np.asarray(activity)
    if activity.ndim != 2:
        raise ValueError(f"activity must be (time, neuron), got shape {activity.shape}")
    ax = _get_axes(ax, figsize)
    image = ax.imshow(activity.T, aspect="auto", cmap=cmap, origin="lower")
    ax.figure.colorbar(image, ax=ax, label="activity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("neuron")
    return ax


def save_figure(target: Figure | Axes, path: Path, **savefig_kwargs: Any) -> Path:
    """Saves the figure of `target` to `path`, creating the directory; rcParams `savefig.*` apply."""
    figure = target.get_figure(root=True) if isinstance(target, Axes) else target
    assert figure is not None, "Axes is not attached to a figure"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, **savefig_kwargs)
    return path


def _get_axes(ax: Axes | None, figsize: tuple[float, float] | None) -> Axes:
    if ax is not None:
        return ax
    _, new_ax = plt.subplots(figsize=figsize)
    return new_ax
