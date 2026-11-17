"""Tests for the 20-class mixed dataset, class-order permutation and training subsampling."""

import itertools

import pytest
import torch
from torch.utils.data import Subset

from data import (
    MixedDataset,
    build_stage_digits,
    build_stages,
    get_stage_digits,
    permute_stage_digits,
    reorder_stages,
    subsample_stages,
)


class Fake:
    """`per_class` examples of each of 10 classes; the image encodes (offset, index)."""

    def __init__(self, per_class: int, value: float) -> None:
        self.targets = torch.tensor([c for c in range(10) for _ in range(per_class)])
        self.value = value

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.full((1,), self.value), int(self.targets[index].item())


def _mixed(per_class: int = 4) -> MixedDataset:
    return MixedDataset(Fake(per_class, 1.0), Fake(per_class, 2.0))


def test_mixed_labels_offset_fashion_by_ten_and_keep_each_datasets_images() -> None:
    mixed = _mixed()
    assert len(mixed) == 80
    assert sorted(set(mixed.targets.tolist())) == list(range(20))
    image, label = mixed[0]
    assert (image.item(), label) == (1.0, 0)
    image, label = mixed[40]
    assert (image.item(), label) == (2.0, 10)
    image, label = mixed[79]
    assert (image.item(), label) == (2.0, 19)


@pytest.mark.parametrize("index", [-1, 80])
def test_mixed_index_out_of_range_raises(index: int) -> None:
    with pytest.raises(IndexError):
        _mixed()[index]


def test_twenty_one_class_stages_cover_all_twenty_labels() -> None:
    stages = build_stages(_mixed(), _mixed(), build_stage_digits(20, 1, 1))
    assert len(stages) == 20
    assert stages[-1].digits_seen == frozenset(range(20))
    assert {int(t) for t in stages[-1].test_data.targets} == set(range(20))


def test_permute_stage_digits_is_a_seeded_permutation() -> None:
    canonical = build_stage_digits(20, 1, 1)
    first = permute_stage_digits(canonical, seed=7)
    assert first == permute_stage_digits(canonical, seed=7)
    assert first != permute_stage_digits(canonical, seed=8)
    assert sorted(digit for digits in first for digit in digits) == list(range(20))
    assert [len(digits) for digits in first] == [1] * 20


def test_permute_keeps_stage_sizes_for_multi_class_stages() -> None:
    canonical = build_stage_digits(10, 5, 2)
    permuted = permute_stage_digits(canonical, seed=1)
    assert [len(d) for d in permuted] == [len(d) for d in canonical]


def test_reorder_stages_changes_which_class_enters_when() -> None:
    mixed = _mixed()
    canonical = build_stages(mixed, mixed, build_stage_digits(20, 1, 1))
    order = permute_stage_digits(get_stage_digits(canonical), seed=3)
    reordered = reorder_stages(canonical, order)
    assert get_stage_digits(reordered) == order
    for stage in reordered:
        labels = {int(mixed.targets[i]) for i in stage.train_subset.indices}
        assert labels == set(stage.digits_seen)
    assert reordered[-1].digits_seen == canonical[-1].digits_seen


def test_get_stage_digits_roundtrips_build_stage_digits() -> None:
    wanted = build_stage_digits(10, 3, 2)
    assert get_stage_digits(build_stages(Fake(2, 0.0), Fake(2, 0.0), wanted)) == wanted


def test_subsample_keeps_nested_stages_and_is_seeded() -> None:
    mixed = _mixed(per_class=50)
    stages = build_stages(mixed, mixed, build_stage_digits(20, 1, 1))
    small = subsample_stages(stages, 0.25, seed=1)
    sizes = [len(stage.train_subset) for stage in small]
    assert sizes == sorted(sizes) and sizes[-1] < len(stages[-1].train_subset)
    assert 0.15 < sizes[-1] / len(stages[-1].train_subset) < 0.35
    again = subsample_stages(stages, 0.25, seed=1)
    assert [s.train_subset.indices for s in small] == [s.train_subset.indices for s in again]
    for earlier, later in itertools.pairwise(small):
        assert set(earlier.train_subset.indices) <= set(later.train_subset.indices)
    assert isinstance(small[0].train_subset, Subset)


def test_subsample_fraction_one_is_the_identity() -> None:
    stages = build_stages(Fake(2, 0.0), Fake(2, 0.0))
    assert subsample_stages(stages, 1.0, seed=0) is stages


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_subsample_rejects_fractions_outside_zero_one(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        subsample_stages(build_stages(Fake(2, 0.0), Fake(2, 0.0)), fraction, seed=0)


def test_ordered_mixed_dataset_is_a_fixed_permutation_with_consistent_labels() -> None:
    plain = _mixed()
    shuffled = MixedDataset(Fake(4, 1.0), Fake(4, 2.0), order_seed=0)
    again = MixedDataset(Fake(4, 1.0), Fake(4, 2.0), order_seed=0)
    assert torch.equal(shuffled.targets, again.targets)
    assert not torch.equal(shuffled.targets, plain.targets)
    assert sorted(shuffled.targets.tolist()) == sorted(plain.targets.tolist())
    for index in range(len(shuffled)):
        image, label = shuffled[index]
        assert label == int(shuffled.targets[index])
        assert image.item() == (1.0 if label < 10 else 2.0)
    first_half = shuffled.targets[: len(shuffled) // 2]
    assert (first_half < 10).any() and (first_half >= 10).any()  # both datasets interleaved
