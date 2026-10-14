"""Tests for plotting_utils.py: loading, selecting, grouping, stacking and the plot functions."""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from fakes import make_config
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import plotting_utils as pu
import run_store
from run_store import Launch, RunArray


def _figure(ax: Axes) -> Figure:
    figure = ax.get_figure(root=True)
    assert figure is not None
    return figure


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _write_run(launch: Launch, name: str, config_name: str, accuracy=None, loss=None, **config):
    cfg = dataclasses.replace(make_config(name), **config)
    arrays = {
        "accuracy": RunArray(
            np.full((2, 10), 0.5) if accuracy is None else accuracy, ("stage", "digit")
        ),
        "train_loss": RunArray(np.array([1.0, 0.5, 0.25]) if loss is None else loss, ("epoch",)),
    }
    return run_store.write_run(
        launch,
        cfg,
        arrays,
        config_name=config_name,
        metrics={"final_avg_accuracy": 0.5, "final_backward_transfer": 0.0, "train_seconds": 1.0},
        started_at="2026-10-14T12:00:00+02:00",
        device="cpu",
    )


@pytest.fixture
def runs(launch: Launch):
    """A: 3 seeds fcnn mnist; B: 2 seeds fcnn fashion_mnist; C: 1 seed cnn mnist."""
    for seed in (1, 2, 3):
        _write_run(launch, f"A_seed{seed}", "A", loss=np.array([seed, 0.5, 0.25]), seed=seed)
    for seed in (4, 5):
        _write_run(launch, f"B_seed{seed}", "B", dataset="fashion_mnist", seed=seed)
    _write_run(launch, "C", "C", model_kind="cnn", ai2_variant=None, hidden_sizes=None, T=None)
    return pu.load_runs(launch.directory)


def test_load_runs_reads_a_launch_directory_and_a_root(launch: Launch, runs) -> None:
    assert len(runs) == 6
    assert [run.run_id for run in runs] == sorted(run.run_id for run in runs)
    assert len(pu.load_runs(launch.directory.parent)) == 6
    assert len(pu.load_runs(launch.directory, launch.directory.parent)) == 12


def test_load_runs_with_nothing_found_raises_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="No run files"):
        pu.load_runs(tmp_path / "empty")


def test_run_field_reads_config_then_meta_and_tuples(runs) -> None:
    run = runs[0]

    assert pu.run_field(run, "dataset") == "mnist"
    assert pu.run_field(run, "config_name") == "A"
    assert pu.run_field(run, "hidden_sizes") == (4, 2)
    with pytest.raises(KeyError, match="Valid fields"):
        pu.run_field(run, "nope")


def test_select_runs_filters_by_config_fields_including_tuples(runs) -> None:
    assert len(pu.select_runs(runs, dataset="mnist")) == 4
    assert len(pu.select_runs(runs, dataset="mnist", model_kind="fcnn")) == 3
    assert len(pu.select_runs(runs, hidden_sizes=[4, 2])) == 5
    assert len(pu.select_runs(runs, config_name="B")) == 2


def test_empty_selection_raises_a_clear_error(runs) -> None:
    with pytest.raises(ValueError, match="No run matches.*'cifar'"):
        pu.select_runs(runs, dataset="cifar")
    with pytest.raises(KeyError):
        pu.select_runs(runs, not_a_field=1)


def test_group_runs_keeps_unequal_seed_counts_and_sorts_keys(runs) -> None:
    groups = pu.group_runs(runs, "config_name", "dataset")

    assert list(groups) == [("A", "mnist"), ("B", "fashion_mnist"), ("C", "mnist")]
    assert [len(group) for group in groups.values()] == [3, 2, 1]


def test_group_runs_with_incomparable_keys_falls_back_to_first_seen_order(runs) -> None:
    groups = pu.group_runs(runs, "T")  # A/B have T=2, C has T=None: int vs None is unorderable

    assert sorted(len(group) for group in groups.values()) == [1, 5]


def test_stack_arrays_stacks_over_seeds_per_group(runs) -> None:
    groups = pu.group_runs(runs, "config_name")

    loss_a = pu.stack_arrays(groups[("A",)], "train_loss")
    loss_b = pu.stack_arrays(groups[("B",)], "train_loss")

    assert loss_a.shape == (3, 3) and loss_b.shape == (2, 3)
    np.testing.assert_array_equal(loss_a[:, 0], [1, 2, 3])
    assert pu.stack_arrays(groups[("A",)], "accuracy").shape == (3, 2, 10)


def test_stack_arrays_rejects_ragged_shapes_and_empty_groups(launch: Launch) -> None:
    _write_run(launch, "short", "X", loss=np.zeros(2))
    _write_run(launch, "long", "X", loss=np.zeros(5))
    both = pu.load_runs(launch.directory)

    with pytest.raises(ValueError, match="shapes differ") as excinfo:
        pu.stack_arrays(both, "train_loss")
    assert "short: (2,)" in str(excinfo.value) and "long: (5,)" in str(excinfo.value)
    with pytest.raises(ValueError, match="zero runs"):
        pu.stack_arrays([], "train_loss")


def test_mean_std_uses_population_std_over_the_seed_axis() -> None:
    stack = np.array([[1.0, 10.0], [3.0, 14.0]])

    mean, std = pu.mean_std(stack)

    np.testing.assert_allclose(mean, [2.0, 12.0])
    np.testing.assert_allclose(std, [1.0, 2.0])


def test_plot_setup_and_colorcycle_defaults() -> None:
    saved = dict(plt.rcParams)
    try:
        pu.plot_setup()
        assert plt.rcParams["figure.figsize"] == [3, 1.5]
        assert plt.rcParams["savefig.dpi"] == 1200
        assert plt.rcParams["figure.dpi"] == 100
        assert plt.rcParams["savefig.transparent"] is True
        assert plt.rcParams["figure.constrained_layout.use"] is True
        pu.plot_setup(fig_dpi=50, save_dpi=75)
        assert (plt.rcParams["figure.dpi"], plt.rcParams["savefig.dpi"]) == (50, 75)
    finally:
        plt.rcParams.update(saved)
    colors = pu.get_colorcycle(4)
    assert len(colors) == 4
    assert colors[0] != colors[-1]


def test_line_band_returns_axes_with_band_and_log_axis() -> None:
    x = np.array([1.0, 10.0, 100.0])

    ax = pu.plot_line_band(x, np.array([0.1, 0.5, 0.9]), np.array([0.05, 0.1, 0.05]), logx=True)

    assert isinstance(ax, Axes)
    assert ax.get_xscale() == "log"
    assert len(ax.collections) == 1  # the std band


def test_line_band_without_std_draws_no_band_and_honours_ax_and_style_kwargs() -> None:
    _, existing = plt.subplots()

    ax = pu.plot_line_band(
        np.arange(3), np.arange(3.0), None, existing, color="red", label="lbl", marker="o"
    )

    assert ax is existing
    assert len(ax.collections) == 0
    assert ax.lines[0].get_color() == "red"
    assert ax.lines[0].get_label() == "lbl"


def test_figsize_applies_only_when_the_function_creates_the_axes() -> None:
    ax = pu.plot_line_band(np.arange(3), np.arange(3.0), figsize=(5, 2))

    assert tuple(_figure(ax).get_size_inches()) == (5.0, 2.0)


def test_epoch_curves_single_run_and_seed_band() -> None:
    single = pu.plot_epoch_curves(np.array([1.0, 0.5, 0.25]))
    banded = pu.plot_epoch_curves(np.array([[1.0, 0.5], [0.8, 0.4], [1.2, 0.6]]))

    assert isinstance(single, Axes) and len(single.collections) == 0
    assert isinstance(banded, Axes) and len(banded.collections) == 1
    assert banded.get_xlabel() == "epoch"
    with pytest.raises(ValueError, match="epochs"):
        pu.plot_epoch_curves(np.zeros((2, 2, 2)))


def test_accuracy_heatmap_for_one_run_and_the_seed_mean() -> None:
    one = pu.plot_accuracy_heatmap(np.random.default_rng(0).random((6, 10)))
    mean_over_seeds = pu.plot_accuracy_heatmap(np.random.default_rng(0).random((3, 6, 10)))

    for ax in (one, mean_over_seeds):
        assert isinstance(ax, Axes)
        assert np.asarray(ax.images[0].get_array()).shape == (6, 10)
        assert ax.images[0].get_clim() == (0.0, 1.0)
    with pytest.raises(ValueError, match="stages, digits"):
        pu.plot_accuracy_heatmap(np.zeros(5))


def test_activity_heatmap_puts_time_on_x_and_neurons_on_y() -> None:
    ax = pu.plot_activity_heatmap(np.random.default_rng(0).standard_normal((50, 8)))

    assert isinstance(ax, Axes)
    assert np.asarray(ax.images[0].get_array()).shape == (8, 50)
    assert ax.get_xlabel() == "time step" and ax.get_ylabel() == "neuron"
    with pytest.raises(ValueError, match="time, neuron"):
        pu.plot_activity_heatmap(np.zeros((2, 3, 4)))


def test_plots_work_straight_from_run_files(runs) -> None:
    group = pu.select_runs(runs, config_name="A")

    loss = pu.stack_arrays(group, "train_loss")
    accuracy = pu.stack_arrays(group, "accuracy")

    assert isinstance(pu.plot_epoch_curves(loss), Axes)
    assert isinstance(pu.plot_accuracy_heatmap(accuracy), Axes)


def test_save_figure_creates_directories_and_accepts_axes_or_figure(tmp_path: Path) -> None:
    ax = pu.plot_line_band(np.arange(3), np.arange(3.0))

    from_axes = pu.save_figure(ax, tmp_path / "deep" / "dir" / "a.png", dpi=50)
    from_figure = pu.save_figure(_figure(ax), tmp_path / "b.png", dpi=50)

    assert from_axes.read_bytes().startswith(b"\x89PNG")
    assert from_figure.exists()


def test_save_figure_respects_savefig_rcparams(tmp_path: Path) -> None:
    ax = pu.plot_line_band(np.arange(3), np.arange(3.0), figsize=(2, 1))
    with plt.rc_context({"savefig.dpi": 40, "savefig.bbox": "standard"}):
        small = pu.save_figure(ax, tmp_path / "small.png")
    with plt.rc_context({"savefig.dpi": 80, "savefig.bbox": "standard"}):
        large = pu.save_figure(ax, tmp_path / "large.png")

    assert plt.imread(large).shape[0] == 2 * plt.imread(small).shape[0]
