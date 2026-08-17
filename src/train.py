"""Stage-wise, epoch-delimited training loop. See lab-book.md, "Experiment design"."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import Stage
from metrics import evaluate_per_digit_accuracy


def run_incremental_training(
    model: nn.Module,
    stages: list[Stage],
    device: torch.device,
    epochs_per_stage: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> list[dict[int, float]]:
    """Trains `model` through the cumulative incremental-digit protocol.

    "Learnt" is epoch-delimited, not loss-delimited, see lab-book.md,
    "Experiment design".

    Returns:
        One per-digit accuracy dict (see `metrics.evaluate_per_digit_accuracy`)
        per stage, evaluated on the full test set right after that stage.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    accuracy_by_stage = []
    for stage in stages:
        loader = DataLoader(stage.train_subset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs_per_stage):
            train_one_epoch(model, loader, optimizer, criterion, device)
        accuracy_by_stage.append(evaluate_per_digit_accuracy(model, stage.test_data, device))
    return accuracy_by_stage


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """Runs one training pass over `loader`, updating `model` in place."""
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
