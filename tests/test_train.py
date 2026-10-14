"""Tests for run_incremental_training's callback hooks. See old-lab-book.md."""

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


def _make_stage(index: int, num_examples: int = 8) -> Stage:
    dataset = _FakeDataset(num_examples)
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


def test_on_batch_end_fires_correct_indices_across_varying_stage_sizes() -> None:
    """Stage sizes differ (mirroring the real cumulative protocol, where later
    stages have more training data) so each stage has a different batch
    count per epoch -- a fixed-batches-per-epoch bug would surface here.
    """
    model = nn.Linear(4, 2)
    stages = [_make_stage(0, num_examples=8), _make_stage(1, num_examples=16)]
    device = torch.device("cpu")
    epochs_per_stage = 2
    batch_size = 4

    batch_calls: list[tuple[int, int, int]] = []

    run_incremental_training(
        model,
        stages,
        device,
        epochs_per_stage=epochs_per_stage,
        batch_size=batch_size,
        on_batch_end=lambda stage_index, epoch_index, batch_index: batch_calls.append(
            (stage_index, epoch_index, batch_index)
        ),
    )

    expected = [
        (stage.index, epoch_index, batch_index)
        for stage, num_examples in zip(stages, [8, 16])
        for epoch_index in range(epochs_per_stage)
        for batch_index in range(num_examples // batch_size)
    ]
    assert batch_calls == expected


def test_on_batch_end_fires_for_partial_last_batch() -> None:
    model = nn.Linear(4, 2)
    stages = [_make_stage(0, num_examples=10)]
    device = torch.device("cpu")

    batch_calls: list[int] = []

    run_incremental_training(
        model,
        stages,
        device,
        epochs_per_stage=1,
        batch_size=4,
        on_batch_end=lambda stage_index, epoch_index, batch_index: batch_calls.append(batch_index),
    )

    # 10 examples / batch_size 4 -> batches of 4, 4, 2 (partial last batch).
    assert batch_calls == [0, 1, 2]


def test_on_batch_end_is_optional() -> None:
    model = nn.Linear(4, 2)
    stages = [_make_stage(0)]
    device = torch.device("cpu")

    accuracy_by_stage = run_incremental_training(
        model, stages, device, epochs_per_stage=1, batch_size=4, on_batch_end=None
    )

    assert len(accuracy_by_stage) == 1
