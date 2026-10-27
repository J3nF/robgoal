"""Tests for the weight-scan loading, CSVs and figures on synthetic run files."""

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

import plot_weight_scan
import weight_scan_data
from run_store import RunArray, write_run_file
from sweep import build_weight_scan_configs
from weight_scan_data import WEIGHTS, load_weight_scan

STEPS = 5
LAYERS = 4
EPOCHS = 10  # one epoch per stage, 10 stages


def _arrays(rng: np.random.Generator, weight: float, diverged: bool) -> dict[str, RunArray]:
    scale = 1.0 - 0.1 * np.arange(LAYERS)[:, None]  # signal shrinks with depth
    abs_y = np.abs(rng.normal(1.0, 0.05, (LAYERS, STEPS))) * scale
    abs_y[:, 0] = 1.0 * scale[:, 0]
    arrays = {
        "accuracy": RunArray(rng.uniform(0.3, 1.0, (EPOCHS, 10)), ("epoch", "digit")),
        "train_loss": RunArray(np.full(EPOCHS, np.nan if diverged else 0.5), ("epoch",)),
        "unroll/mean_y": RunArray(rng.uniform(0, 1, (LAYERS, 10)), ("layer", "class")),
        "unroll/mean_y_self_used": RunArray(
            weight * rng.uniform(0, 1, (LAYERS, 10, STEPS)), ("layer", "class", "step")
        ),
        "unroll/mean_y_tilde": RunArray(
            rng.uniform(0, 1, (LAYERS, 10, STEPS)), ("layer", "class", "step")
        ),
        "unroll/mean_abs_y_tilde_class": RunArray(
            rng.uniform(0, 1, (LAYERS, 10, STEPS)), ("layer", "class", "step")
        ),
        "unroll/mean_abs_y_tilde": RunArray(abs_y, ("layer", "step")),
        "unroll/neuron_std_y_tilde": RunArray(0.5 * abs_y, ("layer", "step")),
    }
    return arrays


def _write_runs(
    directory: Path,
    seeds: tuple[int, ...] = (1, 2),
    datasets: tuple[str, ...] = ("mnist",),
    diverged: set[str] | None = None,
    accuracy_dims: tuple[str, str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for dataset in datasets:
        for config in build_weight_scan_configs():
            for seed in seeds:
                full = dataclasses.replace(
                    config, dataset=dataset, initial_classes=1, classes_per_stage=1, seed=seed
                )
                name = f"{config.name}_{dataset}_seed{seed}"
                is_bad = diverged is not None and config.name in diverged and seed == seeds[0]
                arrays = _arrays(rng, config.ai2_weight, is_bad)
                if accuracy_dims is not None:
                    arrays["accuracy"] = RunArray(arrays["accuracy"].data, accuracy_dims)
                meta = {
                    "run_id": f"launch/{name}",
                    "run_name": name,
                    "launch_id": "launch",
                    "config_name": config.name,
                    "config": dataclasses.asdict(full),
                    "git": {"commit": "abc", "branch": "x", "dirty": False},
                    "started_at": "now",
                    "metrics": {"final_avg_accuracy": 0.9, "final_backward_transfer": 0.1},
                }
                write_run_file(directory / f"{name}.h5", meta, arrays)
    return directory


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_runs(tmp_path_factory.mktemp("runs"), diverged={"fcnn_ai2_gate_sigmoid_w1"})


def test_loader_groups_seeds_and_excludes_diverged_runs(runs_dir: Path) -> None:
    cells = load_weight_scan(runs_dir)

    assert len(cells) == 28
    healthy = cells[("mnist", "ai2_diff", "sigmoid", 1.0)]
    assert (healthy.num_runs, healthy.num_finite) == (2, 2)
    assert healthy.arrays["accuracy"].shape == (2, EPOCHS, 10)
    bad = cells[("mnist", "ai2_gate", "sigmoid", 1.0)]
    assert (bad.num_runs, bad.num_finite) == (2, 1)
    assert bad.arrays["unroll/mean_y_tilde"].shape[0] == 1
    assert set(cells) == {
        ("mnist", v, s, w)
        for v in ("ai2_diff", "ai2_gate")
        for s in ("sigmoid", "relu_sigmoid")
        for w in WEIGHTS
    }


def test_loader_rejects_runs_that_predate_per_epoch_accuracy(tmp_path: Path) -> None:
    directory = _write_runs(tmp_path / "legacy", seeds=(1,), accuracy_dims=("stage", "digit"))

    with pytest.raises(ValueError, match="predates per-epoch evaluation"):
        load_weight_scan(directory)


def test_unroll_summary_has_one_row_per_layer_class_step(runs_dir: Path, tmp_path: Path) -> None:
    cells = load_weight_scan(runs_dir)

    weight_scan_data.write_unroll_summary_csv(cells, tmp_path / "unroll.csv")

    rows = list(csv.DictReader((tmp_path / "unroll.csv").open()))
    assert len(rows) == 28 * LAYERS * 10 * STEPS
    assert {"y_mean", "y_self_used_std", "abs_y_tilde_mean"} <= set(rows[0])


def test_scan_summary_ratio_is_one_for_the_no_inhibition_control(
    runs_dir: Path, tmp_path: Path
) -> None:
    cells = load_weight_scan(runs_dir)
    for cell in cells.values():  # make step 0 equal to the last step, as for weight 0
        cell.arrays["unroll/mean_abs_y_tilde"][:, :, -1] = cell.arrays["unroll/mean_abs_y_tilde"][
            :, :, 0
        ]

    weight_scan_data.write_scan_summary_csv(cells, tmp_path / "scan.csv")

    rows = list(csv.DictReader((tmp_path / "scan.csv").open()))
    assert len(rows) == 28
    assert all(float(r["ratio_last_over_first_layer3_mean"]) == pytest.approx(1.0) for r in rows)
    bad = next(
        r
        for r in rows
        if (r["variant"], r["scheme"], r["weight"]) == ("ai2_gate", "sigmoid", "1.0")
    )
    assert (bad["num_runs"], bad["num_finite"]) == ("2", "1")


def test_q2_tests_compare_every_nonzero_weight_with_the_control(
    runs_dir: Path, tmp_path: Path
) -> None:
    cells = load_weight_scan(runs_dir)

    weight_scan_data.write_q2_tests_csv(cells, tmp_path / "q2.csv")

    rows = list(csv.DictReader((tmp_path / "q2.csv").open()))
    assert len(rows) == 4 * 6 * 2  # 4 columns x 6 nonzero weights x 2 statistics (1 dataset)
    assert {r["statistic"] for r in rows} == {"abs", "std"}
    valid = [float(r["p_holm"]) for r in rows if not math.isnan(float(r["p"]))]
    assert all(0.0 <= p <= 1.0 for p in valid)


def test_depth_log_ratio_is_negative_when_the_signal_shrinks(runs_dir: Path) -> None:
    cell = load_weight_scan(runs_dir)[("mnist", "ai2_diff", "sigmoid", 1.0)]

    ratios = weight_scan_data.get_depth_log_ratios(cell)

    assert (ratios["abs"] < 0).all() and (ratios["std"] < 0).all()


def test_main_writes_the_named_figures_and_csvs(
    runs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plot_weight_scan, "SAVE_DPI", 30)
    monkeypatch.setattr(plot_weight_scan, "_save_unroll_figures", lambda *args: None)

    plot_weight_scan.main([runs_dir], tmp_path / "plots")

    names = {path.name for path in (tmp_path / "plots").iterdir()}
    assert {
        "dieoff_mnist.png",
        "accuracy_over_time_mnist.png",
        "bwt_over_time_mnist.png",
        "unroll_summary.csv",
        "weight_scan_summary.csv",
        "q2_tests.csv",
    } <= names
    assert "scan_summary.csv" not in names


def test_unroll_figures_use_the_documented_file_names(
    runs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plot_weight_scan, "SAVE_DPI", 30)
    cells = load_weight_scan(runs_dir)
    plot_weight_scan.plotting_utils.plot_setup(save_dpi=30)

    plot_weight_scan._save_unroll_figures(cells, "mnist", tmp_path)

    files = sorted(p.name for p in (tmp_path / "unroll").iterdir())
    assert len(files) == 56
    assert "mnist_ai2_diff_sigmoid_w-0p667_classes3.png" in files
    assert "mnist_ai2_gate_relu_sigmoid_w1p333_allclasses.png" in files


def test_bwt_figure_carries_the_caption_and_a_symmetric_axis(runs_dir: Path) -> None:
    cells = load_weight_scan(runs_dir)

    fig = plot_weight_scan.plot_bwt_over_time(cells, "mnist")

    assert any("TRIAGE.md #2" in text.get_text() for text in fig.texts)
    low, high = fig.axes[0].get_ylim()
    assert low == pytest.approx(-high)


def test_accuracy_figure_spans_zero_to_one(runs_dir: Path) -> None:
    cells = load_weight_scan(runs_dir)

    fig = plot_weight_scan.plot_accuracy_over_time(cells, "mnist")

    assert fig.axes[0].get_ylim() == (0.0, 1.0)
