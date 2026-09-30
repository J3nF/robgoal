"""Cumulative MNIST/FashionMNIST protocol. See lab-book.md, "Class-incremental
data protocol"."""

import ssl
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import certifi
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms

DatasetName = Literal["mnist", "fashion_mnist"]

# torchvision dataset class + per-channel normalization (mean, std), computed
# over each dataset's own training set.
_DATASETS: dict[DatasetName, tuple[type, tuple[float, ...], tuple[float, ...]]] = {
    "mnist": (torchvision.datasets.MNIST, (0.1307,), (0.3081,)),
    "fashion_mnist": (torchvision.datasets.FashionMNIST, (0.2860,), (0.3530,)),
}


def build_stage_digits(
    num_classes: int = 10, initial_classes: int = 5, classes_per_stage: int = 1
) -> list[list[int]]:
    """Splits `range(num_classes)` into an initial stage plus one-or-more-classes-at-a-time
    follow-up stages, see lab-book.md, "Class-incremental data protocol".

    Defaults reproduce the original hardcoded split (5 initial digits, then
    1 at a time): `[[0,1,2,3,4],[5],[6],[7],[8],[9]]`.

    Args:
        initial_classes: How many classes stage 0 introduces.
        classes_per_stage: How many new classes each later stage introduces.
            Doesn't need to evenly divide the remaining classes -- the last
            stage takes whatever's left over, so every class is always
            introduced exactly once, never dropped.
    """
    if not 1 <= initial_classes <= num_classes:
        raise ValueError(f"initial_classes must be in [1, {num_classes}], got {initial_classes}")
    if classes_per_stage < 1:
        raise ValueError(f"classes_per_stage must be >= 1, got {classes_per_stage}")
    remaining = list(range(initial_classes, num_classes))
    stage_digits = [list(range(initial_classes))]
    for start in range(0, len(remaining), classes_per_stage):
        stage_digits.append(remaining[start : start + classes_per_stage])
    return stage_digits


# Default split: one digit per stage after the initial 5, see
# `build_stage_digits`'s docstring and lab-book.md, "Class-incremental data
# protocol".
STAGE_DIGITS: list[list[int]] = build_stage_digits()


class LabeledDataset(Protocol):
    """Structural type for datasets exposing a `targets` tensor (e.g. MNIST)."""

    targets: torch.Tensor

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]: ...


@dataclass
class Stage:
    """Data for one step of the incremental-digit protocol: what's been
    introduced so far, and what to train/evaluate on.

    Attributes:
        index: Stage number, 0-5.
        digits_seen: All digit labels introduced by this stage (cumulative).
        train_subset: Training examples for every digit in `digits_seen`.
        test_data: The full MNIST test set (all 10 digits), used both to
            measure forgetting on seen digits and generalization to unseen ones.
    """

    index: int
    digits_seen: frozenset[int]
    train_subset: Subset
    test_data: LabeledDataset


def get_mnist_data(
    root: str = "data", dataset: DatasetName = "mnist"
) -> tuple[LabeledDataset, LabeledDataset]:
    """Downloads (if needed) and returns a train/test dataset pair.

    Args:
        dataset: "mnist" (default) or "fashion_mnist".
    """
    # certifi's bundle covers CAs missing from some systems' trust stores.
    ssl._create_default_https_context = (  # ty: ignore[invalid-assignment]
        lambda *a, **kw: ssl.create_default_context(cafile=certifi.where())
    )
    dataset_class, mean, std = _DATASETS[dataset]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_data = dataset_class(root=root, train=True, download=True, transform=transform)
    test_data = dataset_class(root=root, train=False, download=True, transform=transform)
    return train_data, test_data


def build_stages(
    train_data: LabeledDataset,
    test_data: LabeledDataset,
    stage_digits: list[list[int]] | None = None,
) -> list[Stage]:
    """Split class-grouped data into stages which NNs train on in one after the other.

    Args:
        stage_digits: Which classes each stage introduces, cumulative; see
            `build_stage_digits`. Defaults to `STAGE_DIGITS` (5 initial
            digits, then 1 per stage) when `None`.
    """
    if stage_digits is None:
        stage_digits = STAGE_DIGITS
    stages = []
    digits_seen: set[int] = set()
    for stage_idx, new_digits in enumerate(stage_digits):
        digits_seen |= set(new_digits)  # Combine to-be-seen digits in-place
        stages.append(_build_stage(stage_idx, frozenset(digits_seen), train_data, test_data))
    return stages


def _build_stage(
    stage_idx: int,
    digits_seen: frozenset[int],
    train_data: LabeledDataset,
    test_data: LabeledDataset,
) -> Stage:
    """Constructs one Stage: `digits_seen`'s train subset plus the full test set."""
    train_subset = Subset(
        cast(Dataset[tuple[torch.Tensor, int]], train_data),
        _get_digit_indices(train_data, digits_seen),
    )
    stage = Stage(stage_idx, digits_seen, train_subset, test_data)
    return stage


def _get_digit_indices(dataset: LabeledDataset, digits: frozenset[int]) -> list[int]:
    mask = torch.isin(dataset.targets, torch.tensor(sorted(digits)))
    digit_indices = torch.nonzero(mask).squeeze(1).tolist()
    return digit_indices
