"""Class-incremental cumulative MNIST protocol. See lab-book.md, Data protocol."""

import ssl
from dataclasses import dataclass
from typing import Protocol

import certifi
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms

# Stage 0 introduces the first 5 digits at once, stages 1-4 add one digit each,
# per the 07-06 meeting notes ("one by one enables clearer results with regard
# to novelty of new data").
STAGE_DIGITS: list[list[int]] = [[0, 1, 2, 3, 4], [5], [6], [7], [8], [9]]


class LabeledDataset(Protocol):
    """Structural type for datasets exposing a `targets` tensor (e.g. MNIST)."""

    targets: torch.Tensor

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]: ...


@dataclass
class Stage:
    """One step of the incremental-digit protocol.

    Attributes:
        index: Stage number, 0-4.
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
    # Some systems' OpenSSL trust store is missing the CA that signs the MNIST
    # mirror's cert; use certifi's bundle instead. The stdlib's stub for this
    # attribute is stricter than what's actually assignable at runtime.
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


def _indices_for_digits(dataset: LabeledDataset, digits: frozenset[int]) -> list[int]:
    mask = torch.isin(dataset.targets, torch.tensor(sorted(digits)))
    return torch.nonzero(mask).squeeze(1).tolist()


def build_stages(train_data: LabeledDataset, test_data: LabeledDataset) -> list[Stage]:
    """Builds the 5-stage cumulative class-incremental protocol."""
    stages = []
    digits_seen: set[int] = set()
    for stage_idx, new_digits in enumerate(STAGE_DIGITS):
        digits_seen |= set(new_digits)
        frozen_digits = frozenset(digits_seen)
        # torch's Subset stub wants a nominal Dataset; LabeledDataset only requires the
        # same __len__/__getitem__ methods structurally, which is all Subset actually uses.
        train_subset = Subset(
            train_data,  # ty: ignore[invalid-argument-type]
            _indices_for_digits(train_data, frozen_digits),
        )
        stages.append(Stage(stage_idx, frozen_digits, train_subset, test_data))
    return stages
