"""Tests for run_experiments.py's per-config/per-seed orchestration.

Against a real local MLflow (sqlite) backend, same isolation pattern as
tests/test_logged_training.py.
"""

import contextlib
import json
from typing import cast

import mlflow
import pytest
import torch
from mlflow.entities import Run
from torch.utils.data import Dataset, Subset

import logged_training
import run_experiments
from data import Stage
from metrics import average_accuracy, backward_transfer
from run_config import RunConfig


class _FakeDataset:
    """Tiny stand-in dataset for exercising the orchestration layer without real data.

    Exactly one example per digit 0-9, see test_logged_training.py's
    equivalent for why: average_accuracy/backward_transfer need every digit
    to have an entry in the per-digit accuracy dict.
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
def _isolated_run_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(logged_training, "TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(logged_training, "ARTIFACT_LOCATION", f"file:{tmp_path / 'artifacts'}")
    monkeypatch.setattr(run_experiments, "RESULTS_DIR", tmp_path / "results")
    run_experiments.RESULTS_DIR.mkdir()
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_run_config_single_seed_writes_one_result(monkeypatch) -> None:
    monkeypatch.setattr(run_experiments, "NUM_SEEDS", 1)
    config = _make_config("test_single")
    stages = _make_stages()

    run_experiments._run_config(config, stages, torch.device("cpu"))

    assert (run_experiments.RESULTS_DIR / "test_single.json").exists()
    run = _get_run_by_name("test_single")
    assert "mlflow.parentRunId" not in run.data.tags


def test_run_config_replicates_writes_one_result_per_seed_and_aggregates(monkeypatch) -> None:
    monkeypatch.setattr(run_experiments, "NUM_SEEDS", 2)
    config = _make_config("test_group")
    stages = _make_stages()

    run_experiments._run_config(config, stages, torch.device("cpu"))

    assert (run_experiments.RESULTS_DIR / "test_group_seed0.json").exists()
    assert (run_experiments.RESULTS_DIR / "test_group_seed1.json").exists()
    parent_run = _get_run_by_name("test_group")
    assert "mean_final_avg_accuracy" in parent_run.data.metrics
    assert "std_final_avg_accuracy" in parent_run.data.metrics
    child_run = _get_run_by_name("test_group_seed0")
    assert child_run.data.tags["mlflow.parentRunId"] == parent_run.info.run_id


def test_run_seed_return_value_matches_its_own_written_json(monkeypatch) -> None:
    monkeypatch.setattr(run_experiments, "NUM_SEEDS", 1)
    config = _make_config("test_run_seed")
    stages = _make_stages()

    avg_acc, bwt = run_experiments._run_seed(config, 0, stages, torch.device("cpu"))

    written = json.loads((run_experiments.RESULTS_DIR / "test_run_seed.json").read_text())
    accuracy_by_stage = [{int(digit): acc for digit, acc in stage.items()} for stage in written]
    assert avg_acc == pytest.approx(average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen))
    assert bwt == pytest.approx(backward_transfer(accuracy_by_stage, stages[0].digits_seen))


def test_get_replicate_context_dispatches_on_num_seeds(monkeypatch) -> None:
    config = _make_config("test_dispatch")

    monkeypatch.setattr(run_experiments, "NUM_SEEDS", 1)
    single_seed_context = run_experiments._get_replicate_context(config)
    assert isinstance(single_seed_context, contextlib.nullcontext)

    monkeypatch.setattr(run_experiments, "NUM_SEEDS", 2)
    replicated_context = run_experiments._get_replicate_context(config)
    assert not isinstance(replicated_context, contextlib.nullcontext)
    assert hasattr(replicated_context, "__enter__")
    assert hasattr(replicated_context, "__exit__")
