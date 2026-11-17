"""Tests for plot_incremental.py and pilot_width.py on tiny real runs written to a temp launch."""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from fakes import make_cnn_config, make_config, make_stages

import pilot_width
import plot_incremental as pi
import plotting_utils as pu
from data import get_stage_digits
from recorded_training import build_arrays, train_and_record
from run_config import RunConfig
from run_store import Launch, write_run
from sweep import build_model


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _write(launch: Launch, config: RunConfig, seed: int, num_stages: int = 3) -> None:
    stages = make_stages(num_stages)
    seeded = dataclasses.replace(config, seed=seed, name=f"{config.name}_seed{seed}")
    torch.manual_seed(seed)
    record = train_and_record(seeded, build_model(seeded), stages, torch.device("cpu"))
    write_run(
        launch,
        seeded,
        build_arrays(record),
        config_name=config.name,
        metrics={},
        started_at="now",
        device="cpu",
        extra_meta={"stage_digits": get_stage_digits(stages)},
    )


@pytest.fixture
def runs(launch: Launch):
    configs = [
        make_cnn_config("cnn"),
        make_config("ctl", T=1, hidden_sizes=(4, 4)),
        make_config("gate", ai2_variant="ai2_gate", T=5, hidden_sizes=(4, 4)),
        make_config("leaky", layer_kind="leaky", T=1, hidden_sizes=(4, 4), decay=0.9),
    ]
    for config in configs:
        for seed in (1, 2):
            _write(launch, config, seed)
    return pu.load_runs(launch.directory)


def test_model_labels_and_cell_order(runs) -> None:
    cells = pi.build_cells(runs, "mnist")
    labels = [cell.label for cell in cells]
    assert labels[0] == "CNN control" or "control" in labels[0]
    assert labels[-1].startswith("leaky")
    assert any(label.startswith("unrolled gate T=5") for label in labels)
    assert all(len(cell.runs) == 2 for cell in cells)


def test_unknown_dataset_raises(runs) -> None:
    with pytest.raises(ValueError):
        pi.build_cells(runs, "mixed")


def test_newest_class_curves_have_one_value_per_stage_and_nan_bwt_in_stage_zero(runs) -> None:
    cell = pi.build_cells(runs, "mnist")[0]
    accuracy, bwt = pi.get_newest_class_curves(cell)
    assert accuracy.shape == bwt.shape == (2, 3)
    assert np.isnan(bwt[:, 0]).all() and np.isfinite(bwt[:, 1:]).all()
    assert ((accuracy >= 0) & (accuracy <= 1)).all()


def test_newest_class_accuracy_reads_the_stage_end_row(runs) -> None:
    cell = pi.build_cells(runs, "mnist")[0]
    accuracy, _ = pi.get_newest_class_curves(cell)
    stored = cell.accuracy()
    assert accuracy[0, 2] == stored[0, 2, 2]  # one epoch per stage; stage 2 introduces class 2


def test_all_figures_render(runs, tmp_path: Path) -> None:
    pu.plot_setup()
    cells = pi.build_cells(runs, "mnist")
    figures = [
        pi.plot_heatmaps(cells, "mnist", "coarse"),
        pi.plot_heatmaps(cells, "mnist", "coarse", by_rank=True),
        pi.plot_newest_class_curves(cells, "mnist", "coarse"),
        pi.plot_inhibition_eval_comparison(cells, "mnist", "coarse"),
    ]
    inhibiting = [c for c in cells if pi.get_layers(c)]
    assert len(inhibiting) == 2  # controls and CNN record nothing
    limits = pi.get_row_limits(inhibiting, "mean")
    figures.append(pi.plot_inhibition_grid(inhibiting[0], "mnist", "coarse", "mean", limits))
    assert all(fig.axes for fig in figures)


def test_inhibition_grid_shares_one_y_range_across_all_subplots(runs) -> None:
    pu.plot_setup()
    inhibiting = [c for c in pi.build_cells(runs, "mnist") if pi.get_layers(c)]
    limits = pi.get_row_limits(inhibiting, "mean")
    fig = pi.plot_inhibition_grid(inhibiting[0], "mnist", "coarse", "mean", limits)
    grid = fig.axes[: 2 * len(pi.get_layers(inhibiting[0]))]  # the colourbar axes come last
    assert len(grid) > 2
    assert len({ax.get_ylim() for ax in grid}) == 1
    assert grid[0].get_ylim()[0] == 0
    assert len(set(limits.values())) == 1


def test_main_writes_png_and_pdf(runs, launch: Launch, tmp_path: Path) -> None:
    pi.main([launch.directory], tmp_path / "out", "coarse")
    names = {p.name for p in (tmp_path / "out").iterdir()}
    assert "heatmap_mnist.png" in names and "heatmap_mnist.pdf" in names
    assert "newest_class_mnist.png" in names and "inhibition_eval_mnist.png" in names
    assert any(n.startswith("inhibition_var_mnist_leaky") for n in names)


# --- pilot_width ------------------------------------------------------------------------------


def _row(family: str, width: int, low: float, high: float, accuracy: float = 0.9):
    return pilot_width.WidthRow(family, width, 5, (low + high) / 2, low, high, accuracy)


def test_round_width() -> None:
    assert [pilot_width.round_width(w) for w in (1.0, 3.2, 5.7, 11.3, 22.6, 45)] == [
        2,
        4,
        6,
        12,
        24,
        44,
    ]


def test_choose_width_takes_the_geometric_mean_between_the_ends() -> None:
    rows = []
    for scheme in ("fcnn relu_sigmoid", "fcnn sigmoid"):
        rows += [
            _row(scheme, 2, -0.5, -0.3),
            _row(scheme, 4, -0.2, -0.1),
            _row(scheme, 8, -0.1, -0.03),  # neither
            _row(scheme, 32, -0.01, 0.01),
        ]
    choice = pilot_width.choose_width(rows)
    assert (choice.w_low, choice.w_high, choice.width) == (4, 32, 12)
    assert choice.note == ""


def test_absent_requires_the_accuracy_floor_and_both_schemes() -> None:
    floored = [_row("fcnn sigmoid", 2, -0.01, 0.0, accuracy=0.3)]
    assert not floored[0].absent
    rows = [
        _row("fcnn relu_sigmoid", 4, -0.3, -0.1),
        _row("fcnn sigmoid", 4, -0.3, -0.1),
        _row("fcnn relu_sigmoid", 16, -0.01, 0.0),
        _row("fcnn sigmoid", 16, -0.05, 0.0),  # one scheme not absent
    ]
    choice = pilot_width.choose_width(rows)
    assert choice.w_high is None and choice.w_low == 4
    assert "largest present" in choice.note


def test_choose_width_without_either_end_and_with_only_one() -> None:
    middle = [_row("fcnn sigmoid", 8, -0.1, -0.03), _row("fcnn sigmoid", 16, -0.04, 0.0)]
    choice = pilot_width.choose_width(middle)
    assert choice.w_low is None and choice.w_high is None and "closest" in choice.note
    only_absent = [_row("fcnn sigmoid", 8, -0.01, 0.0)]
    assert pilot_width.choose_width(only_absent).note.startswith("no width shows")


def test_choose_width_ignores_cnn_and_rejects_empty() -> None:
    with pytest.raises(ValueError, match="FCNN"):
        pilot_width.choose_width([_row("cnn", 4, -0.3, -0.2)])


def test_width_table_from_runs_and_bad_names(launch: Launch, tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        _write(launch, make_config("width4_fcnn_control_relu", T=1, hidden_sizes=(4,) * 4), seed)
    rows = pilot_width.build_width_table(pu.load_runs(launch.directory))
    (row,) = rows
    assert (row.family, row.width, row.num_seeds) == ("fcnn relu_sigmoid", 4, 3)
    assert row.d_low <= row.d_mean <= row.d_high
    pilot_width.write_outputs(rows, pilot_width.WidthChoice(4, None, None, "x"), tmp_path / "p")
    assert (tmp_path / "p" / "width_pilot.png").exists()
    assert "D_mean" in (tmp_path / "p" / "width_table.csv").read_text()


def test_width_table_rejects_other_config_names_and_single_seeds(launch: Launch) -> None:
    _write(launch, make_config("other_name", T=1, hidden_sizes=(4,)), 1)
    with pytest.raises(ValueError, match="width<W>"):
        pilot_width.build_width_table(pu.load_runs(launch.directory))


def test_width_table_needs_two_seeds(launch: Launch) -> None:
    _write(launch, make_config("width4_fcnn_control_relu", T=1, hidden_sizes=(4,)), 1)
    with pytest.raises(ValueError, match="need >= 2"):
        pilot_width.build_width_table(pu.load_runs(launch.directory))


def test_choose_width_falls_back_to_the_trainable_scheme() -> None:
    rows = [
        _row("fcnn sigmoid", 4, -0.3, -0.1, accuracy=0.1),
        _row("fcnn sigmoid", 16, -0.3, -0.1, accuracy=0.1),
        _row("fcnn relu_sigmoid", 4, -0.3, -0.1),
        _row("fcnn relu_sigmoid", 16, -0.01, 0.01),
    ]
    choice = pilot_width.choose_width(rows)
    assert (choice.w_low, choice.w_high, choice.width) == (4, 16, 8)
    assert "fcnn sigmoid" in choice.note


def test_t_interval_matches_a_hand_computation() -> None:
    mean, low, high = pilot_width.get_t_interval(np.array([1.0, 2.0, 3.0]))
    half = 4.302653 / np.sqrt(3)  # t(0.975, df=2) * s / sqrt(n), s = 1
    assert mean == 2.0
    assert high - mean == pytest.approx(half, rel=1e-5)
    assert mean - low == pytest.approx(half, rel=1e-5)


def test_t_interval_needs_two_values() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        pilot_width.get_t_interval(np.array([1.0]))


def test_chosen_width_stays_strictly_between_adjacent_ends() -> None:
    rows = [_row("fcnn sigmoid", 2, -0.4, -0.2), _row("fcnn sigmoid", 4, -0.01, 0.01)]
    choice = pilot_width.choose_width(rows)
    assert (choice.w_low, choice.w_high, choice.width) == (2, 4, 3)


def test_width_text_and_rank_marks_follow_the_stage_digits(runs) -> None:
    cells = pi.build_cells(runs, "mnist")
    assert pi.get_width_text(cells) == "width 4"
    fig = pi.plot_heatmaps(cells, "mnist", "coarse", by_rank=True)
    marks = np.asarray(fig.axes[0].collections[0].get_offsets())[:, 1]
    assert marks[:3].tolist() == [0, 1, 2]  # rank r enters in stage r with one epoch per stage
    assert "width 4" in fig.get_suptitle()


def test_row_limits_of_no_cells_are_empty() -> None:
    assert pi.get_row_limits([], "mean") == {}
