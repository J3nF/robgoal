"""Tests for the incremental-digit MNIST protocol. See lab-book.md, Verification."""

import torch

from data import STAGE_DIGITS, build_stages


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
