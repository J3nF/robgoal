"""Tests for size_scan.py's shrinking-network scan. See lab-book.md, "ToDos
2026-08-26 or so"."""

from typing import cast

import pytest

from autapse import AutapticLayer
from models import CNN, FCNN_AI2
from size_scan import build_size_scan_configs
from sweep import build_configs, build_model


def test_configs_have_unique_names() -> None:
    configs = build_size_scan_configs()

    names = [config.name for config in configs]
    assert len(names) == len(set(names))


def test_names_never_collide_with_the_base_sweep() -> None:
    size_scan_names = {config.name for config in build_size_scan_configs()}
    base_sweep_names = {config.name for config in build_configs()}

    assert size_scan_names.isdisjoint(base_sweep_names)


def test_every_config_builds_a_model_without_raising() -> None:
    for config in build_size_scan_configs():
        build_model(config)


def test_hidden_sizes_and_channels_shrink_across_steps() -> None:
    configs = build_size_scan_configs()
    fcnn_hidden_sizes = [config.hidden_sizes for config in configs if config.hidden_sizes]
    cnn_channels = [config.channels for config in configs if config.channels]

    assert list(fcnn_hidden_sizes) == sorted(fcnn_hidden_sizes, key=sum, reverse=True)
    assert list(cnn_channels) == sorted(cnn_channels, key=sum, reverse=True)


def test_every_step_includes_ai2_and_control_families() -> None:
    configs = build_size_scan_configs()
    num_steps = 5
    for step in range(num_steps):
        step_configs = [c for c in configs if c.name.startswith(f"sizescan_step{step}_")]
        assert any(c.model_kind == "cnn" for c in step_configs)
        assert any(c.model_kind == "fcnn" and c.T == 1 for c in step_configs)  # fcnn controls
        assert any(c.model_kind == "fcnn" and c.T != 1 for c in step_configs)  # ai2 grid


def test_built_fcnn_model_matches_configured_hidden_sizes() -> None:
    configs = build_size_scan_configs()
    fcnn_configs = [c for c in configs if c.model_kind == "fcnn" and c.T != 1]

    for config in fcnn_configs:
        model = cast(FCNN_AI2, build_model(config))
        hidden_layers = [cast(AutapticLayer, layer) for layer in model.hidden_layers]
        assert [layer.linear.out_features for layer in hidden_layers] == list(config.hidden_sizes)


def test_built_cnn_model_matches_configured_channels() -> None:
    configs = build_size_scan_configs()
    cnn_configs = [c for c in configs if c.model_kind == "cnn"]

    for config in cnn_configs:
        model = cast(CNN, build_model(config))
        assert model.conv1.out_channels == config.channels[0]
        assert model.conv2.out_channels == config.channels[1]


def test_rejects_mismatched_scan_lengths() -> None:
    with pytest.raises(ValueError, match="same number of steps"):
        build_size_scan_configs(hidden_sizes_scan=[(64, 32), (32, 16)], channels_scan=[(16, 32)])
