"""Stage-wise, epoch-delimited training loop. See old-lab-book.md, "Experiment design"."""

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import Stage
from metrics import evaluate_per_digit_accuracy


def run_incremental_training(
    model: nn.Module,
    stages: list[Stage],
    device: torch.device,
    *,
    epochs_per_stage: int = 5,
    lr: float = 0.1,
    batch_size: int = 128,
    on_epoch_end: Callable[[int, int, float], None] | None = None,
    on_stage_end: Callable[[int, dict[int, float]], None] | None = None,
    on_batch_end: Callable[[int, int, int], None] | None = None,
) -> list[dict[int, float]]:
    """Trains `model` through the cumulative incremental-digit protocol.

    "Learnt" is epoch-delimited, not loss-delimited, see old-lab-book.md,
    "Experiment design".

    Args:
        on_epoch_end: Called after each epoch as
            `(stage_index, epoch_index, avg_train_loss)`, e.g. for live logging.
        on_stage_end: Called after each stage's eval as
            `(stage_index, per_digit_accuracy)`, e.g. for live logging.
        on_batch_end: Called after each training batch as
            `(stage_index, epoch_index, batch_index)`, e.g. for live logging.

    Returns:
        One per-digit accuracy dict (see `metrics.evaluate_per_digit_accuracy`)
        per stage, evaluated on the full test set right after that stage.
    """
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # Note default momentum = 0
    criterion = nn.CrossEntropyLoss()

    accuracy_by_stage = []
    for stage in stages:
        loader = DataLoader(stage.train_subset, batch_size=batch_size, shuffle=True)
        for epoch_index in range(epochs_per_stage):
            batch_callback = (
                _bind_batch_index(on_batch_end, stage.index, epoch_index)
                if on_batch_end is not None
                else None
            )
            avg_loss = train_one_epoch(
                model, loader, optimizer, criterion, device, on_batch_end=batch_callback
            )
            if on_epoch_end is not None:
                on_epoch_end(stage.index, epoch_index, avg_loss)
        per_digit_accuracy = evaluate_per_digit_accuracy(model, stage.test_data, device)
        accuracy_by_stage.append(per_digit_accuracy)
        if on_stage_end is not None:
            on_stage_end(stage.index, per_digit_accuracy)
    return accuracy_by_stage


def _bind_batch_index(
    on_batch_end: Callable[[int, int, int], None], stage_index: int, epoch_index: int
) -> Callable[[int], None]:
    """Adapts a `(stage_index, epoch_index, batch_index)` callback to `train_one_epoch`'s
    per-batch `(batch_index)` signature by binding the stage/epoch indices.

    Called fresh each epoch and invoked synchronously within that same epoch's
    `train_one_epoch` call, so plain closure capture is safe -- no late-binding
    hazard, since the closure never outlives the loop iteration it's built in.
    """
    return lambda batch_index: on_batch_end(stage_index, epoch_index, batch_index)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    on_batch_end: Callable[[int], None] | None = None,
) -> float:
    """Runs one training pass over `loader`, updating `model` in place.

    Args:
        on_batch_end: Called after each batch as `(batch_index)`, e.g. for
            live logging.

    Returns:
        Mean training loss over the epoch, weighted by each batch's actual size.
    """
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
        total_examples += images.shape[0]
        if on_batch_end is not None:
            on_batch_end(batch_index)
    return total_loss / total_examples
