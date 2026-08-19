"""Continual-learning evaluation metrics. See lab-book.md, "Evaluation metrics"."""

from collections import defaultdict
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import LabeledDataset


@torch.no_grad()
def evaluate_per_digit_accuracy(
    model: nn.Module, test_data: LabeledDataset, device: torch.device, batch_size: int = 256
) -> dict[int, float]:
    """Computes test accuracy for each of the 10 digits.

    Covers both forgetting (already-introduced digits) and generalization
    (not-yet-introduced ones), see lab-book.md, "Evaluation metrics".
    """
    model.eval()
    correct: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    loader = DataLoader(cast(Dataset[tuple[torch.Tensor, int]], test_data), batch_size=batch_size)
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


def backward_transfer(
    accuracy_by_stage: list[dict[int, float]], first_stage_digits: frozenset[int]
) -> float:
    """Mean accuracy drop on `first_stage_digits` between stage 0 and the final stage.

    The forgetting measure, see lab-book.md, "Evaluation metrics".
    Precondition: every digit in `first_stage_digits` must have been
    introduced at stage 0 — not checked.
    """
    first_stage_accuracy = accuracy_by_stage[0]
    final_stage_accuracy = accuracy_by_stage[-1]
    return sum(
        final_stage_accuracy[d] - first_stage_accuracy[d] for d in first_stage_digits
    ) / len(first_stage_digits)


def format_accuracy_matrix(accuracy_by_stage: list[dict[int, float]]) -> str:
    """Renders per-stage, per-digit accuracy as a table (row = stage, col = digit).

    A blank cell means a caller passed a partial per-stage accuracy dict;
    see lab-book.md, "Evaluation metrics".
    """
    digits = sorted({digit for stage_accuracy in accuracy_by_stage for digit in stage_accuracy})
    header = "stage | " + " ".join(f"{digit:>5}" for digit in digits)
    lines = [header, "-" * len(header)]
    for stage_idx, stage_accuracy in enumerate(accuracy_by_stage):
        row = " ".join(
            f"{stage_accuracy[digit]:5.3f}" if digit in stage_accuracy else "    -"
            for digit in digits
        )
        lines.append(f"{stage_idx:>5} | {row}")
    return "\n".join(lines)

# TODO: Add "dynamics_insights.py" (or so)-named file recording y_self dynamics.