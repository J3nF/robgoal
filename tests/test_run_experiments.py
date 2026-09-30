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


def test_run_config_single_seed_writes_one_result() -> None:
    config = _make_config("test_single")
    stages = _make_stages()

    run_experiments._run_config(config, stages, torch.device("cpu"), num_seeds=1, dataset="mnist")

    assert (run_experiments.RESULTS_DIR / "test_single.json").exists()
    run = _get_run_by_name("test_single")
    assert "mlflow.parentRunId" not in run.data.tags


def test_run_config_single_seed_is_deterministic_seed_zero() -> None:
    """num_seeds=1 must stay deterministic (seed 0) -- only opting into
    multiple replicates opts into randomized seeds, see run_experiments.py's
    `_draw_seeds`."""
    config = _make_config("test_single_seed_value")
    stages = _make_stages()

    avg_acc, _ = run_experiments._run_seed(
        config, 0, stages, torch.device("cpu"), num_seeds=1, dataset="mnist"
    )

    run = _get_run_by_name("test_single_seed_value")
    assert run.data.params["seed"] == "0"
    assert avg_acc == pytest.approx(avg_acc)  # sanity: run completed


def test_run_config_replicates_writes_one_result_per_seed_and_aggregates() -> None:
    config = _make_config("test_group")
    stages = _make_stages()

    run_experiments._run_config(config, stages, torch.device("cpu"), num_seeds=2, dataset="mnist")

    results_files = sorted(run_experiments.RESULTS_DIR.glob("test_group_seed*.json"))
    assert len(results_files) == 2
    parent_run = _get_run_by_name("test_group")
    assert "mean_final_avg_accuracy" in parent_run.data.metrics
    assert "std_final_avg_accuracy" in parent_run.data.metrics
    child_runs = [
        run
        for run in mlflow.search_runs(
            experiment_names=[logged_training.EXPERIMENT_NAME], output_format="list"
        )
        if run.data.tags.get("mlflow.parentRunId") == parent_run.info.run_id
    ]
    assert len(child_runs) == 2


def test_run_config_replicates_draws_distinct_seeds() -> None:
    """Replicate seeds are randomly drawn (not sequential 0, 1, ...) but
    still distinct and recorded on each child run's `seed` param."""
    config = _make_config("test_random_seeds")
    stages = _make_stages()

    run_experiments._run_config(config, stages, torch.device("cpu"), num_seeds=3, dataset="mnist")

    child_runs = [
        run
        for run in mlflow.search_runs(
            experiment_names=[logged_training.EXPERIMENT_NAME], output_format="list"
        )
        if run.data.tags.get("mlflow.parentRunId") is not None
        and run.info.run_name.startswith("test_random_seeds_seed")
    ]
    seeds = {int(run.data.params["seed"]) for run in child_runs}
    assert len(seeds) == 3


def test_run_seed_return_value_matches_its_own_written_json() -> None:
    config = _make_config("test_run_seed")
    stages = _make_stages()

    avg_acc, bwt = run_experiments._run_seed(
        config, 0, stages, torch.device("cpu"), num_seeds=1, dataset="mnist"
    )

    written = json.loads((run_experiments.RESULTS_DIR / "test_run_seed.json").read_text())
    accuracy_by_stage = [{int(digit): acc for digit, acc in stage.items()} for stage in written]
    assert avg_acc == pytest.approx(average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen))
    assert bwt == pytest.approx(backward_transfer(accuracy_by_stage, stages[0].digits_seen))


def test_run_seed_suffixes_name_by_dataset_when_not_mnist() -> None:
    """Switching datasets must not silently collide with the mnist result of
    the same config name."""
    config = _make_config("test_dataset_suffix")
    stages = _make_stages()

    run_experiments._run_seed(
        config, 0, stages, torch.device("cpu"), num_seeds=1, dataset="fashion_mnist"
    )

    assert (run_experiments.RESULTS_DIR / "test_dataset_suffix_fashion_mnist.json").exists()
    run = _get_run_by_name("test_dataset_suffix_fashion_mnist")
    assert run.data.params["dataset"] == "fashion_mnist"


def test_get_replicate_context_dispatches_on_num_seeds() -> None:
    config = _make_config("test_dispatch")

    single_seed_context = run_experiments._get_replicate_context(config, num_seeds=1)
    assert isinstance(single_seed_context, contextlib.nullcontext)

    replicated_context = run_experiments._get_replicate_context(config, num_seeds=2)
    assert not isinstance(replicated_context, contextlib.nullcontext)
    assert hasattr(replicated_context, "__enter__")
    assert hasattr(replicated_context, "__exit__")


def test_draw_seeds_is_deterministic_at_one_and_random_above() -> None:
    assert run_experiments._draw_seeds(1) == [0]

    seeds_a = run_experiments._draw_seeds(5)
    seeds_b = run_experiments._draw_seeds(5)
    assert len(seeds_a) == len(set(seeds_a)) == 5
    assert seeds_a != seeds_b  # random draws essentially never collide


@pytest.mark.parametrize("num_seeds", [0, -1])
def test_draw_seeds_rejects_non_positive_num_seeds(num_seeds: int) -> None:
    with pytest.raises(ValueError, match="num_seeds"):
        run_experiments._draw_seeds(num_seeds)


def test_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_experiments.py"])

    args = run_experiments._parse_args()

    assert args.scan == "sweep"
    assert args.num_seeds == run_experiments.NUM_SEEDS
    assert args.dataset == "mnist"
    assert args.initial_classes == 5
    assert args.classes_per_stage == 1


def test_parse_args_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiments.py",
            "--scan",
            "size",
            "--num-seeds",
            "5",
            "--dataset",
            "fashion_mnist",
            "--initial-classes",
            "1",
            "--classes-per-stage",
            "2",
        ],
    )

    args = run_experiments._parse_args()

    assert args.scan == "size"
    assert args.num_seeds == 5
    assert args.dataset == "fashion_mnist"
    assert args.initial_classes == 1
    assert args.classes_per_stage == 2
