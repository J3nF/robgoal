"""Tests for logged_training.py against a real local MLflow (sqlite) backend.

Uses the real backend pointed at `tmp_path`, not a mock -- it's fast and
deterministic (no network), so this isn't the "slow or non-deterministic"
case STYLE.md's "mock boundaries, not logic" guidance is about.
"""

import dataclasses
import statistics
from typing import cast

import mlflow
import pytest
import torch
from mlflow.entities import Run
from torch.utils.data import Dataset, Subset

import logged_training
import sweep
from data import Stage
from metrics import average_accuracy, backward_transfer
from models import CNN
from run_config import RunConfig


class _FakeDataset:
    """Tiny stand-in dataset for exercising the MLflow adapter without real data.

    Exactly one example per digit 0-9: `average_accuracy`/`backward_transfer`
    (used by `run_and_log`) require every digit to have an entry in the
    per-digit accuracy dict, which `evaluate_per_digit_accuracy` only
    populates for digits actually present in the dataset -- a random sample
    smaller than 10 examples isn't guaranteed to cover all ten.
    """

    def __init__(self) -> None:
        self.x = torch.randn(10, 1, 28, 28)
        self.targets = torch.arange(10)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.x[index], int(self.targets[index].item())


def _make_stages(num_stages: int = 2) -> list[Stage]:
    stages = []
    for index in range(num_stages):
        dataset = _FakeDataset()
        train_subset = Subset(
            cast(Dataset[tuple[torch.Tensor, int]], dataset), list(range(len(dataset)))
        )
        stages.append(
            Stage(
                index=index,
                digits_seen=frozenset(range(index + 1)),
                train_subset=train_subset,
                test_data=dataset,
            )
        )
    return stages


def _make_config(name: str) -> RunConfig:
    return RunConfig(
        name=name,
        sweep_axis="base",
        model_kind="fcnn",
        epochs_per_stage=1,
        ai2_variant="ai2_diff",
        activation_scheme="relu_sigmoid",
        T=2,
        hidden_sizes=(4, 2),
    )


def _get_run_by_name(name: str) -> Run:
    runs = mlflow.search_runs(
        experiment_names=[logged_training.EXPERIMENT_NAME], output_format="list"
    )
    matches = [run for run in runs if run.info.run_name == name]
    assert len(matches) == 1, f"expected exactly one run named {name!r}, found {len(matches)}"
    return matches[0]


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    monkeypatch.setattr(logged_training, "TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(logged_training, "ARTIFACT_LOCATION", f"file:{tmp_path / 'artifacts'}")
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_run_and_log_logs_params_and_metrics() -> None:
    config = _make_config("test_run")
    model = sweep.build_model(config)
    stages = _make_stages()

    logged_training.run_and_log(config, model, stages, torch.device("cpu"))

    run = _get_run_by_name("test_run")
    assert run.data.params["ai2_variant"] == "ai2_diff"
    assert run.data.params["T"] == "2"
    assert "final_avg_accuracy" in run.data.metrics
    assert "final_backward_transfer" in run.data.metrics


def test_y_self_magnitude_present_for_fcnn_absent_for_cnn() -> None:
    stages = _make_stages(num_stages=1)

    fcnn_config = _make_config("test_fcnn_yself")
    logged_training.run_and_log(
        fcnn_config, sweep.build_model(fcnn_config), stages, torch.device("cpu")
    )
    fcnn_run = _get_run_by_name("test_fcnn_yself")
    assert any(key.startswith("y_self_magnitude_") for key in fcnn_run.data.metrics)

    cnn_config = dataclasses.replace(
        fcnn_config,
        name="test_cnn_yself",
        model_kind="cnn",
        ai2_variant=None,
        activation_scheme=None,
        T=None,
        hidden_sizes=None,
    )
    logged_training.run_and_log(cnn_config, CNN(), stages, torch.device("cpu"))
    cnn_run = _get_run_by_name("test_cnn_yself")
    assert not any(key.startswith("y_self_magnitude_") for key in cnn_run.data.metrics)


def test_replicate_group_rejects_fewer_than_two_seeds() -> None:
    config = _make_config("test_too_few_seeds")

    with pytest.raises(ValueError, match="num_seeds"), logged_training.replicate_group(config, 1):
        pass

    assert mlflow.active_run() is None


def _run_replicates(
    config: RunConfig, stages: list[Stage], num_seeds: int
) -> list[tuple[float, float]]:
    with logged_training.replicate_group(config, num_seeds) as collector:
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            seed_config = dataclasses.replace(config, seed=seed, name=f"{config.name}_seed{seed}")
            accuracy_by_stage = logged_training.run_and_log(
                seed_config, sweep.build_model(seed_config), stages, torch.device("cpu")
            )
            avg_acc = average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen)
            bwt = backward_transfer(accuracy_by_stage, stages[0].digits_seen)
            collector.append((avg_acc, bwt))
    return collector


def test_replicate_group_aggregates_correctly() -> None:
    config = _make_config("test_group")
    stages = _make_stages()

    collector = _run_replicates(config, stages, num_seeds=3)

    parent_run = _get_run_by_name("test_group")
    expected_mean_acc = statistics.mean(avg_acc for avg_acc, _ in collector)
    expected_std_acc = statistics.stdev(avg_acc for avg_acc, _ in collector)
    assert parent_run.data.metrics["mean_final_avg_accuracy"] == pytest.approx(expected_mean_acc)
    assert parent_run.data.metrics["std_final_avg_accuracy"] == pytest.approx(expected_std_acc)


def test_replicate_group_partial_failure_skips_aggregation() -> None:
    config = _make_config("test_failure")
    stages = _make_stages()

    with (
        pytest.raises(RuntimeError, match="simulated"),
        logged_training.replicate_group(config, 3) as collector,
    ):
        for seed in range(3):
            if seed == 1:
                raise RuntimeError("simulated training failure")
            torch.manual_seed(seed)
            seed_config = dataclasses.replace(config, seed=seed, name=f"{config.name}_seed{seed}")
            accuracy_by_stage = logged_training.run_and_log(
                seed_config, sweep.build_model(seed_config), stages, torch.device("cpu")
            )
            avg_acc = average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen)
            bwt = backward_transfer(accuracy_by_stage, stages[0].digits_seen)
            collector.append((avg_acc, bwt))

    parent_run = _get_run_by_name("test_failure")
    assert "mean_final_avg_accuracy" not in parent_run.data.metrics
    assert parent_run.info.status == "FAILED"
