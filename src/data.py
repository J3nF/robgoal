"""Cumulative MNIST protocol. See lab-book.md, "Class-incremental data protocol"."""

import ssl
from dataclasses import dataclass
from typing import Protocol, cast

import certifi
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms

# One digit per stage after the initial 5, see lab-book.md, "Class-incremental
# data protocol".
STAGE_DIGITS: list[list[int]] = [[0, 1, 2, 3, 4], [5], [6], [7], [8], [9]]


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


def get_mnist_data(root: str = "data") -> tuple[LabeledDataset, LabeledDataset]:
    """Downloads (if needed) and returns the MNIST train/test datasets."""
    # certifi's bundle covers CAs missing from some systems' trust stores.
    ssl._create_default_https_context = (  # ty: ignore[invalid-assignment]
        lambda *a, **kw: ssl.create_default_context(cafile=certifi.where())
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )
    return train_data, test_data


def build_stages(train_data: LabeledDataset, test_data: LabeledDataset) -> list[Stage]:
    """Builds the 6-stage cumulative class-incremental protocol."""
    stages = []
    digits_seen: set[int] = set()
    for stage_idx, new_digits in enumerate(STAGE_DIGITS):
        digits_seen |= set(new_digits)
        frozen_digits = frozenset(digits_seen)
        train_subset = Subset(
            cast(Dataset[tuple[torch.Tensor, int]], train_data),
            _indices_for_digits(train_data, frozen_digits),
        )
        stages.append(Stage(stage_idx, frozen_digits, train_subset, test_data))
    return stages


def _indices_for_digits(dataset: LabeledDataset, digits: frozenset[int]) -> list[int]:
    mask = torch.isin(dataset.targets, torch.tensor(sorted(digits)))
    return torch.nonzero(mask).squeeze(1).tolist()
