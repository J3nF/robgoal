"""Tests for run_incremental_training's callback hooks. See lab-book.md."""

from typing import cast

import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from data import Stage
from train import run_incremental_training


class _FakeDataset:
    """Tiny stand-in dataset for exercising the training loop without real data."""

    def __init__(self, num_examples: int = 8) -> None:
        self.x = torch.randn(num_examples, 4)
        self.targets = torch.randint(0, 2, (num_examples,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.x[index], int(self.targets[index].item())


def _make_stage(index: int) -> Stage:
    dataset = _FakeDataset()
    train_subset = Subset(
        cast(Dataset[tuple[torch.Tensor, int]], dataset), list(range(len(dataset)))
    )
    return Stage(
        index=index, digits_seen=frozenset({0, 1}), train_subset=train_subset, test_data=dataset
    )


def test_callbacks_invoked_once_per_epoch_and_stage() -> None:
    model = nn.Linear(4, 2)
    stages = [_make_stage(0), _make_stage(1), _make_stage(2)]
    device = torch.device("cpu")
    epochs_per_stage = 3

    epoch_calls: list[tuple[int, int, float]] = []
    stage_calls: list[tuple[int, dict[int, float]]] = []

    run_incremental_training(
        model,
        stages,
        device,
        epochs_per_stage=epochs_per_stage,
        batch_size=4,
        on_epoch_end=lambda stage_index, epoch_index, avg_loss: epoch_calls.append(
            (stage_index, epoch_index, avg_loss)
        ),
        on_stage_end=lambda stage_index, per_digit_accuracy: stage_calls.append(
            (stage_index, per_digit_accuracy)
        ),
    )

    assert len(epoch_calls) == len(stages) * epochs_per_stage
    assert [stage_index for stage_index, _, _ in epoch_calls] == [
        stage.index for stage in stages for _ in range(epochs_per_stage)
    ]
    assert len(stage_calls) == len(stages)
    assert [stage_index for stage_index, _ in stage_calls] == [stage.index for stage in stages]


def test_callbacks_are_optional() -> None:
    model = nn.Linear(4, 2)
    stages = [_make_stage(0)]
    device = torch.device("cpu")

    accuracy_by_stage = run_incremental_training(
        model, stages, device, epochs_per_stage=1, batch_size=4
    )

    assert len(accuracy_by_stage) == 1
