"""Cumulative MNIST/FashionMNIST protocol. See old-lab-book.md, "Class-incremental
data protocol"."""

import random
import ssl
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import certifi
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms

DatasetName = Literal["mnist", "fashion_mnist", "mixed"]

# Label count per dataset; "mixed" is MNIST (labels 0-9) plus FashionMNIST (labels 10-19).
NUM_CLASSES: dict[DatasetName, int] = {"mnist": 10, "fashion_mnist": 10, "mixed": 20}
MIXED_FASHION_OFFSET = 10
MIXED_TEST_ORDER_SEED = 0

# torchvision dataset class + per-channel normalization (mean, std), computed
# over each dataset's own training set.
_DATASETS: dict[str, tuple[type, tuple[float, ...], tuple[float, ...]]] = {
    "mnist": (torchvision.datasets.MNIST, (0.1307,), (0.3081,)),
    "fashion_mnist": (torchvision.datasets.FashionMNIST, (0.2860,), (0.3530,)),
}


def build_stage_digits(
    num_classes: int = 10, initial_classes: int = 5, classes_per_stage: int = 1
) -> list[list[int]]:
    """Splits `range(num_classes)` into an initial stage plus one-or-more-classes-at-a-time
    follow-up stages, see old-lab-book.md, "Class-incremental data protocol".

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
# `build_stage_digits`'s docstring and old-lab-book.md, "Class-incremental data
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
        test_data: The full test set (all classes of the dataset), used both to
            measure forgetting on seen digits and generalization to unseen ones.
    """

    index: int
    digits_seen: frozenset[int]
    train_subset: Subset
    test_data: LabeledDataset


class MixedDataset:
    """MNIST followed by FashionMNIST as one 20-class dataset (FashionMNIST labels shifted by 10).

    Each part keeps its own transform, so each dataset is normalised with its own mean and
    standard deviation. A network can then in principle tell the datasets apart by intensity
    statistics alone, a shortcut this protocol does not remove.

    Args:
        order_seed: If given, the examples come in one fixed random order drawn from this seed
            instead of MNIST first. The test set uses this so a leaky layer's stream mixes both
            datasets, as the shuffled training batches do, and the order is the same for every run.

    Attributes:
        targets: Labels of all examples in the dataset's order.
    """

    def __init__(
        self, mnist: LabeledDataset, fashion: LabeledDataset, order_seed: int | None = None
    ) -> None:
        self._mnist = mnist
        self._fashion = fashion
        targets = torch.cat([mnist.targets, fashion.targets + MIXED_FASHION_OFFSET])
        self._order = (
            None
            if order_seed is None
            else torch.randperm(len(targets), generator=torch.Generator().manual_seed(order_seed))
        )
        self.targets = targets if self._order is None else targets[self._order]

    def __len__(self) -> int:
        return len(self._mnist) + len(self._fashion)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} examples")
        if self._order is not None:
            index = int(self._order[index].item())
        if index < len(self._mnist):
            return self._mnist[index]
        image, label = self._fashion[index - len(self._mnist)]
        return image, label + MIXED_FASHION_OFFSET


def get_mnist_data(
    root: str = "data", dataset: DatasetName = "mnist"
) -> tuple[LabeledDataset, LabeledDataset]:
    """Downloads (if needed) and returns a train/test dataset pair.

    Args:
        dataset: "mnist" (default), "fashion_mnist", or "mixed" (see `MixedDataset`; train and
            test set are the MNIST plus FashionMNIST union, the test set in one fixed random
            order).
    """
    if dataset == "mixed":
        mnist_train, mnist_test = get_mnist_data(root, "mnist")
        fashion_train, fashion_test = get_mnist_data(root, "fashion_mnist")
        return (
            MixedDataset(mnist_train, fashion_train),
            MixedDataset(mnist_test, fashion_test, order_seed=MIXED_TEST_ORDER_SEED),
        )
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


def permute_stage_digits(stage_digits: list[list[int]], seed: int) -> list[list[int]]:
    """Relabels the classes of `stage_digits` by a random permutation drawn from `seed`.

    The stage sizes stay, only which classes enter in which stage changes; so with one class per
    stage this is a random introduction order. The same seed always gives the same order.
    """
    classes = sorted(digit for digits in stage_digits for digit in digits)
    shuffled = classes.copy()
    random.Random(seed).shuffle(shuffled)
    relabel = dict(zip(classes, shuffled, strict=True))
    return [[relabel[digit] for digit in digits] for digits in stage_digits]


def reorder_stages(stages: list[Stage], stage_digits: list[list[int]]) -> list[Stage]:
    """Rebuilds `stages` for another class-to-stage assignment on the same underlying data.

    `stage_digits` must cover the classes the stages cover; the datasets are taken from the
    last stage's training subset and test data, so a `Stage` list from `build_stages` is needed.
    """
    train_data = cast(LabeledDataset, stages[-1].train_subset.dataset)
    return build_stages(train_data, stages[-1].test_data, stage_digits=stage_digits)


def subsample_stages(stages: list[Stage], fraction: float, seed: int) -> list[Stage]:
    """Keeps a random `fraction` of the training examples (the same ones in every stage).

    Used for coarse runs. The draw depends only on `seed`, and a class keeps `fraction` of its
    examples in expectation, so the cumulative stages stay nested.

    Raises:
        ValueError: if `fraction` is not in (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return stages
    base = stages[-1].train_subset.dataset
    keep = torch.rand(len(base), generator=torch.Generator().manual_seed(seed)) < fraction  # ty: ignore[invalid-argument-type]
    return [
        Stage(
            stage.index,
            stage.digits_seen,
            Subset(base, [i for i in stage.train_subset.indices if keep[i]]),
            stage.test_data,
        )
        for stage in stages
    ]


def get_stage_digits(stages: list[Stage]) -> list[list[int]]:
    """The classes each stage introduces (sorted within a stage), recovered from `digits_seen`."""
    stage_digits = []
    seen: frozenset[int] = frozenset()
    for stage in stages:
        stage_digits.append(sorted(stage.digits_seen - seen))
        seen = stage.digits_seen
    return stage_digits
