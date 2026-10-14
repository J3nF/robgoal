"""Tests for the incremental-digit MNIST protocol. See old-lab-book.md."""

import pytest
import torch
import torchvision

from data import _DATASETS, STAGE_DIGITS, build_stage_digits, build_stages, get_mnist_data


class FakeMNIST:
    """Tiny stand-in for torchvision's MNIST: 2 examples per digit, no download."""

    def __init__(self) -> None:
        self.targets = torch.tensor([digit for digit in range(10) for _ in range(2)])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(1), int(self.targets[index].item())


def test_stages_are_cumulative_and_add_one_digit_at_a_time() -> None:
    stages = build_stages(FakeMNIST(), FakeMNIST())

    assert stages[0].digits_seen == frozenset(STAGE_DIGITS[0])
    for i in range(1, len(stages)):
        assert stages[i - 1].digits_seen < stages[i].digits_seen  # strict superset
        assert stages[i].digits_seen - stages[i - 1].digits_seen == frozenset(STAGE_DIGITS[i])


def test_train_subset_only_contains_digits_seen_so_far() -> None:
    train_data = FakeMNIST()
    stages = build_stages(train_data, FakeMNIST())

    for stage in stages:
        labels_in_subset = {train_data.targets[i].item() for i in stage.train_subset.indices}
        assert labels_in_subset == set(stage.digits_seen)


def test_test_data_always_contains_all_ten_digits() -> None:
    stages = build_stages(FakeMNIST(), FakeMNIST())

    for stage in stages:
        labels = {stage.test_data.targets[i].item() for i in range(len(stage.test_data))}
        assert labels == set(range(10))


def test_build_stage_digits_default_reproduces_stage_digits() -> None:
    assert build_stage_digits() == STAGE_DIGITS == [[0, 1, 2, 3, 4], [5], [6], [7], [8], [9]]


def test_build_stage_digits_covers_every_class_exactly_once() -> None:
    for initial_classes, classes_per_stage in [(1, 1), (5, 2), (1, 3), (9, 1), (10, 1)]:
        stage_digits = build_stage_digits(
            num_classes=10, initial_classes=initial_classes, classes_per_stage=classes_per_stage
        )
        all_digits = [digit for stage in stage_digits for digit in stage]
        assert sorted(all_digits) == list(range(10))  # no digit dropped or duplicated
        assert len(stage_digits[0]) == initial_classes


def test_build_stage_digits_last_stage_takes_uneven_remainder() -> None:
    # 10 - 5 initial = 5 remaining; classes_per_stage=3 -> chunks of 3, then 2.
    stage_digits = build_stage_digits(num_classes=10, initial_classes=5, classes_per_stage=3)

    assert stage_digits == [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]]


def test_build_stage_digits_single_initial_class_one_at_a_time() -> None:
    stage_digits = build_stage_digits(num_classes=10, initial_classes=1, classes_per_stage=1)

    assert stage_digits == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]


@pytest.mark.parametrize("initial_classes", [0, 11, -1])
def test_build_stage_digits_rejects_out_of_range_initial_classes(initial_classes: int) -> None:
    with pytest.raises(ValueError, match="initial_classes"):
        build_stage_digits(num_classes=10, initial_classes=initial_classes)


@pytest.mark.parametrize("classes_per_stage", [0, -1])
def test_build_stage_digits_rejects_non_positive_classes_per_stage(classes_per_stage: int) -> None:
    with pytest.raises(ValueError, match="classes_per_stage"):
        build_stage_digits(num_classes=10, classes_per_stage=classes_per_stage)


def test_build_stages_accepts_custom_stage_digits() -> None:
    custom = [[0, 1], [2], [3]]
    stages = build_stages(FakeMNIST(), FakeMNIST(), stage_digits=custom)

    assert [sorted(stage.digits_seen) for stage in stages] == [[0, 1], [0, 1, 2], [0, 1, 2, 3]]


def test_get_mnist_data_dataset_param_selects_dataset_class(monkeypatch, tmp_path) -> None:
    """Doesn't hit the network: monkeypatches the torchvision dataset classes
    with recording stand-ins so this only checks `get_mnist_data` picks the
    right class/normalization per `dataset`, not the download itself."""
    calls: list[str] = []

    class _RecordingDataset:
        def __init__(self, root, train, download, transform) -> None:
            del root, train, download, transform
            calls.append(self.__class__.__name__)

    monkeypatch.setitem(
        _DATASETS, "mnist", (type("FakeMNIST", (_RecordingDataset,), {}), (0.1307,), (0.3081,))
    )
    monkeypatch.setitem(
        _DATASETS,
        "fashion_mnist",
        (type("FakeFashionMNIST", (_RecordingDataset,), {}), (0.2860,), (0.3530,)),
    )

    get_mnist_data(root=str(tmp_path), dataset="mnist")
    get_mnist_data(root=str(tmp_path), dataset="fashion_mnist")

    assert calls == ["FakeMNIST", "FakeMNIST", "FakeFashionMNIST", "FakeFashionMNIST"]


def test_dataset_registry_covers_mnist_and_fashion_mnist() -> None:
    assert _DATASETS["mnist"][0] is torchvision.datasets.MNIST
    assert _DATASETS["fashion_mnist"][0] is torchvision.datasets.FashionMNIST
