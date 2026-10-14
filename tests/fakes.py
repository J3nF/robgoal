"""Shared test doubles: tiny datasets/stages/configs and failure-injecting stage builders.

Importable by spawned worker processes (pytest puts `tests/` on `sys.path` and `spawn` forwards
it), which is why the stage builders are module-level functions steered by environment
variables: module-level monkeypatching in the test process cannot reach a worker.
"""

import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset, Subset

from data import DatasetName, Stage
from run_config import RunConfig

FAIL_MARKER_ENV = "FAKES_FAIL_MARKER"
SKIP_PID_ENV = "FAKES_SKIP_PID"
MLFLOW_PROBE_ENV = "FAKES_MLFLOW_PROBE"


class FakeDataset:
    """Tiny stand-in dataset: exactly one example per digit 0-9.

    `average_accuracy`/`backward_transfer` need every digit to have an entry in the per-digit
    accuracy dict, which `evaluate_per_digit_accuracy` only fills for digits present in the
    dataset -- a random sample smaller than 10 examples would not cover all ten.
    """

    def __init__(self) -> None:
        self.x = torch.randn(10, 1, 28, 28)
        self.targets = torch.arange(10)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.x[index], int(self.targets[index].item())


def make_stages(num_stages: int = 2) -> list[Stage]:
    """`num_stages` stages over `FakeDataset`; stage i has seen digits 0..i."""
    stages = []
    for index in range(num_stages):
        dataset = FakeDataset()
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


def make_config(name: str, **overrides) -> RunConfig:
    """A tiny FCNN config (one epoch per stage), fast enough for process-spawning tests."""
    fields: dict[str, Any] = {
        "name": name,
        "sweep_axis": "base",
        "model_kind": "fcnn",
        "epochs_per_stage": 1,
        "ai2_variant": "ai2_diff",
        "activation_scheme": "relu_sigmoid",
        "T": 2,
        "hidden_sizes": (4, 2),
    }
    fields.update(overrides)
    return RunConfig(**fields)


def make_cnn_config(name: str) -> RunConfig:
    """A tiny CNN config, which has no AutapticLayer and so no y_self data."""
    return RunConfig(
        name=name, sweep_axis="base", model_kind="cnn", epochs_per_stage=1, channels=(1, 1)
    )


def make_broken_config(name: str) -> RunConfig:
    """An FCNN config missing `T`: `sweep.build_model` asserts, so this job always fails."""
    return make_config(name, T=None)


def tiny_stage_builder(
    dataset: DatasetName, initial_classes: int, classes_per_stage: int
) -> list[Stage]:
    """Stage builder returning `make_stages()` regardless of the arguments."""
    return make_stages()


def fail_once_stage_builder(
    dataset: DatasetName, initial_classes: int, classes_per_stage: int
) -> list[Stage]:
    """Raises on its first call (marker file in `$FAKES_FAIL_MARKER`), except in `$FAKES_SKIP_PID`.

    The skip pid is the launching process: its fail-fast stage build must not use up the one
    failure meant for a job.
    """
    marker = Path(os.environ[FAIL_MARKER_ENV])
    if os.getpid() != int(os.environ.get(SKIP_PID_ENV, "-1")) and not marker.exists():
        marker.write_text("failed once")
        raise RuntimeError("injected one-time failure")
    return make_stages()


def die_once_stage_builder(
    dataset: DatasetName, initial_classes: int, classes_per_stage: int
) -> list[Stage]:
    """Kills its (worker) process on the first call, to exercise `BrokenProcessPool`."""
    marker = Path(os.environ[FAIL_MARKER_ENV])
    if os.getpid() != int(os.environ.get(SKIP_PID_ENV, "-1")) and not marker.exists():
        marker.write_text("died once")
        os._exit(1)
    return make_stages()


def mlflow_probe_stage_builder(
    dataset: DatasetName, initial_classes: int, classes_per_stage: int
) -> list[Stage]:
    """Appends whether `mlflow` is imported in this process to `$FAKES_MLFLOW_PROBE`."""
    with Path(os.environ[MLFLOW_PROBE_ENV]).open("a") as f:
        f.write(f"{os.getpid()} {'mlflow' in sys.modules}\n")
    return make_stages()
