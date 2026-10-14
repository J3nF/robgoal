"""Tests for plot_size_scan.py: aggregation over run files, the summary CSV and the figures."""

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

import plot_size_scan
import plotting_utils
import run_store
import size_scan
from run_store import Launch, RunArray


def _write_scan_run(
    launch: Launch, config, dataset: str, seed: int, accuracy: float, bwt: float, **overrides
) -> None:
    name = (
        f"{config.name}_seed{seed}" if dataset == "mnist" else f"{config.name}_{dataset}_seed{seed}"
    )
    fields = {
        "name": name,
        "dataset": dataset,
        "seed": seed,
        "initial_classes": 1,
        "classes_per_stage": 1,
        **overrides,
    }
    run_store.write_run(
        launch,
        dataclasses.replace(config, **fields),
        {"accuracy": RunArray(np.zeros((2, 10)), ("stage", "digit"))},
        config_name=config.name,
        metrics={
            "final_avg_accuracy": accuracy,
            "final_backward_transfer": bwt,
            "train_seconds": 1.0,
        },
        started_at="2026-10-14T12:00:00+02:00",
        device="cpu",
    )


@pytest.fixture
def scan_launch(launch: Launch) -> Launch:
    for config in size_scan.build_size_scan_configs():
        if not config.name.startswith(("sizescan_step0_", "sizescan_step3_")):
            continue  # the other steps stay empty, as in a scan still in progress
        _write_scan_run(launch, config, "mnist", 1, 0.8, -0.1)
        _write_scan_run(launch, config, "mnist", 2, 0.6, -0.3)
        _write_scan_run(launch, config, "fashion_mnist", 3, 0.5, 0.0)
    return launch


def test_points_aggregate_mean_and_population_std_across_seeds(scan_launch: Launch) -> None:
    points = plot_size_scan.collect_scan_points(plotting_utils.load_runs(scan_launch.directory))

    mnist = [p for p in points if p.dataset == "mnist"]
    fashion = [p for p in points if p.dataset == "fashion_mnist"]
    assert len(mnist) == len(fashion) == 14  # seven configs at each of two steps
    assert all(p.num_seeds == 2 for p in mnist) and all(p.num_seeds == 1 for p in fashion)
    first = mnist[0]
    assert first.accuracy_mean == pytest.approx(0.7)
    assert first.accuracy_std == pytest.approx(0.1)  # population std of (0.8, 0.6)
    assert first.bwt_mean == pytest.approx(-0.2)
    assert first.bwt_std == pytest.approx(0.1)
    assert fashion[0].accuracy_std == 0.0


def test_points_follow_the_canonical_config_order_and_have_sizes_and_families(
    scan_launch: Launch,
) -> None:
    points = plot_size_scan.collect_scan_points(plotting_utils.load_runs(scan_launch.directory))

    assert points[0].config_name == "sizescan_step0_cnn_control_c16x32"
    assert [p.dataset for p in points[:2]] == ["mnist", "fashion_mnist"]
    assert {p.family for p in points} == set(plot_size_scan.FAMILIES)
    assert all(p.param_count > 0 for p in points)


def test_runs_outside_the_one_class_at_a_time_scan_are_ignored(launch: Launch) -> None:
    config = size_scan.build_size_scan_configs()[0]
    _write_scan_run(launch, config, "mnist", 1, 0.8, -0.1)
    _write_scan_run(launch, config, "mnist", 2, 0.1, 0.9, initial_classes=5)
    _write_scan_run(launch, config, "mnist", 3, 0.1, 0.9, sweep_axis="base")

    [point] = plot_size_scan.collect_scan_points(plotting_utils.load_runs(launch.directory))

    assert point.num_seeds == 1 and point.accuracy_mean == pytest.approx(0.8)


def test_no_scan_runs_raises_a_clear_error(launch: Launch) -> None:
    config = size_scan.build_size_scan_configs()[0]
    _write_scan_run(launch, config, "mnist", 1, 0.8, 0.0, initial_classes=5)

    with pytest.raises(ValueError, match="No run matches"):
        plot_size_scan.collect_scan_points(plotting_utils.load_runs(launch.directory))


def test_main_writes_csv_and_both_figures(scan_launch: Launch, tmp_path: Path) -> None:
    out = tmp_path / "plots"

    plot_size_scan.main([scan_launch.directory], out)

    assert (out / "accuracy_vs_size.png").read_bytes().startswith(b"\x89PNG")
    assert (out / "backward_transfer_vs_size.png").exists()
    with (out / "scan_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(plot_size_scan.CSV_COLUMNS)
    assert len(rows) == 28
    assert float(rows[0]["accuracy_mean"]) == pytest.approx(0.7)
    assert rows[0]["num_seeds"] == "2"


def test_family_of_classifies_controls_and_ai2_variants() -> None:
    configs = {c.name: c for c in size_scan.build_size_scan_configs()}

    families = {
        name.removeprefix("sizescan_step0_"): plot_size_scan.family_of(c)
        for name, c in configs.items()
    }

    assert families["cnn_control_c16x32"] == "cnn_control"
    assert families["fcnn_control_relu_sigmoid_hidden256x128"] == "fcnn_control_relu"
    assert families["fcnn_control_sigmoid_hidden256x128"] == "fcnn_control_sigmoid"
    assert families["fcnn_ai2_gate_relu_sigmoid_hidden256x128"] == "fcnn_ai2_gate_relu_sigmoid"


def test_size_scan_runs_without_a_canonical_config_raise_instead_of_plotting_nothing(
    launch: Launch,
) -> None:
    custom = size_scan.build_size_scan_configs([(4, 2)], [(2, 4)])[1]  # not a default-width name
    _write_scan_run(launch, custom, "mnist", 1, 0.8, 0.0)

    with pytest.raises(ValueError, match="none matches a canonical config"):
        plot_size_scan.collect_scan_points(plotting_utils.load_runs(launch.directory))


def test_pooling_runs_from_different_commits_or_learning_rates_warns(
    scan_launch: Launch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = size_scan.build_size_scan_configs()[0]
    _write_scan_run(scan_launch, config, "mnist", 99, 0.5, 0.0, lr=0.001)

    plotting_runs = plotting_utils.load_runs(scan_launch.directory)
    plot_size_scan._warn_if_mixed(plotting_runs)

    assert "WARNING: pooling runs" in capsys.readouterr().err


def test_runs_from_one_commit_and_lr_do_not_warn(
    scan_launch: Launch, capsys: pytest.CaptureFixture[str]
) -> None:
    plot_size_scan._warn_if_mixed(plotting_utils.load_runs(scan_launch.directory))

    assert capsys.readouterr().err == ""
