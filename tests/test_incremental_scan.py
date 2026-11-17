"""Tests for incremental_scan.py's config factories and CNN width matching."""

import argparse

import pytest

import run_experiments
from incremental_scan import (
    build_incremental_configs,
    build_width_pilot_configs,
    matched_channels,
)
from size_scan import CNN_CHANNELS_SCAN, FCNN_HIDDEN_SIZES_SCAN
from sweep import build_model


@pytest.mark.parametrize(
    ("hidden", "channels"), list(zip(FCNN_HIDDEN_SIZES_SCAN, CNN_CHANNELS_SCAN))
)
def test_matched_channels_reproduces_the_size_scan_table(
    hidden: tuple[int, ...], channels: tuple[int, int]
) -> None:
    assert matched_channels(hidden[-1]) == channels


def test_matched_channels_grows_with_width_and_extrapolates() -> None:
    widths = [1, 2, 3, 5, 10, 20, 50, 100, 128, 200, 256, 1024]
    channels = [matched_channels(width) for width in widths]
    assert [c2 for _, c2 in channels] == sorted(c2 for _, c2 in channels)
    assert all(c1 >= 1 and c2 >= c1 for c1, c2 in channels)
    assert matched_channels(256)[1] > 32


@pytest.mark.parametrize("width", [0, -4])
def test_matched_channels_rejects_non_positive_width(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        matched_channels(width)


def test_width_pilot_has_two_fcnn_controls_and_one_cnn_per_width() -> None:
    configs = build_width_pilot_configs((4, 16))
    assert len(configs) == 6
    assert len({config.name for config in configs}) == 6
    fcnn = [config for config in configs if config.model_kind == "fcnn"]
    assert all(config.T == 1 and config.hidden_sizes in {(4,) * 4, (16,) * 4} for config in fcnn)
    assert {config.activation_scheme for config in fcnn} == {"sigmoid", "relu_sigmoid"}
    assert all(config.epochs_per_stage == 1 for config in configs)


def test_incremental_scan_contents_and_buildability() -> None:
    configs = build_incremental_configs(12, decays=(0.5, 0.9))
    assert len(configs) == 3 + 2 * (1 + 2)
    assert len({config.name for config in configs}) == len(configs)
    leaky = [config for config in configs if config.layer_kind == "leaky"]
    assert sorted({config.decay for config in leaky}) == [0.5, 0.9]
    assert all(
        config.hidden_sizes == (12,) * 4 for config in configs if config.model_kind == "fcnn"
    )
    for config in configs:
        build_model(config)


def test_incremental_scan_without_width_is_refused() -> None:
    args = argparse.Namespace(scan="incremental", width=None, widths=[4], leaky_decays=[0.9])
    with pytest.raises(ValueError, match="--width"):
        run_experiments._build_scan_configs(args)


def test_width_scan_is_built_from_the_cli_widths() -> None:
    args = argparse.Namespace(scan="width", width=None, widths=[4, 8], leaky_decays=[0.9])
    assert len(run_experiments._build_scan_configs(args)) == 6
