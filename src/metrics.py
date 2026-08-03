"""Continual-learning evaluation metrics. See lab-book.md, Metrics section."""

from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import LabeledDataset


@torch.no_grad()
def evaluate_per_digit_accuracy(
    model: nn.Module, test_data: LabeledDataset, device: torch.device, batch_size: int = 256
) -> dict[int, float]:
    """Computes test accuracy for each of the 10 digits.

    Covers digits already introduced (forgetting) and not-yet-introduced ones
    (generalization), per the 07-06 notes' "see how the network generalises".
    """
    model.eval()
    correct: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    # Same nominal-vs-structural mismatch as data.py's Subset call.
    loader = DataLoader(test_data, batch_size=batch_size)  # ty: ignore[invalid-argument-type]
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for digit in labels.unique().tolist():
            digit_mask = labels == digit
            correct[digit] += (preds[digit_mask] == digit).sum().item()
            total[digit] += int(digit_mask.sum().item())
    return {digit: correct[digit] / total[digit] for digit in sorted(total)}


def average_accuracy(per_digit_accuracy: dict[int, float], digits: frozenset[int]) -> float:
    """Mean accuracy over the given digits (typically: all digits seen so far)."""
    return sum(per_digit_accuracy[d] for d in digits) / len(digits)


def backward_transfer(accuracy_by_stage: list[dict[int, float]], digits: frozenset[int]) -> float:
    """Mean accuracy drop on `digits` between when they were first learned and the
    final stage — the forgetting measure from lab-book.md.
    """
    first_stage_accuracy = accuracy_by_stage[0]
    final_stage_accuracy = accuracy_by_stage[-1]
    return sum(final_stage_accuracy[d] - first_stage_accuracy[d] for d in digits) / len(digits)
