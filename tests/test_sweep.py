"""Tests for sweep.py's config/model factory. See old-lab-book.md."""

from typing import cast

from autapse import AutapticLayer
from models import CNN, FCNN_AI2
from run_config import RunConfig
from sweep import (
    WEIGHT_SCAN_WEIGHTS,
    build_configs,
    build_model,
    build_weight_scan_configs,
    weight_label,
)


def test_configs_have_unique_names() -> None:
    configs = build_configs()

    assert len(configs) == 14
    assert len({config.name for config in configs}) == 14


def test_every_config_builds_a_model_without_raising() -> None:
    for config in build_configs():
        build_model(config)


def test_cnn_config_builds_cnn() -> None:
    config = RunConfig(name="x", sweep_axis="base", model_kind="cnn", epochs_per_stage=1)

    model = cast(CNN, build_model(config))
    assert isinstance(model, CNN)
    assert model.conv1.out_channels == 16  # CNN's own default, unaffected by channels=None


def test_cnn_config_with_channels_builds_shrunk_cnn() -> None:
    config = RunConfig(
        name="x", sweep_axis="base", model_kind="cnn", epochs_per_stage=1, channels=(4, 8)
    )

    model = cast(CNN, build_model(config))
    assert model.conv1.out_channels == 4
    assert model.conv2.out_channels == 8


def test_fcnn_config_builds_fcnn_with_matching_hyperparameters() -> None:
    config = RunConfig(
        name="x",
        sweep_axis="base",
        model_kind="fcnn",
        epochs_per_stage=1,
        ai2_variant="ai2_gate",
        activation_scheme="sigmoid",
        T=4,
        hidden_sizes=(32, 16),
    )

    model = build_model(config)

    assert isinstance(model, FCNN_AI2)
    assert len(model.hidden_layers) == 2
    assert model.output.in_features == 16
    for hidden_layer in (cast(AutapticLayer, layer) for layer in model.hidden_layers):
        assert hidden_layer.T == 4
        assert hidden_layer.ai2_variant == "ai2_gate"
        assert hidden_layer.activation_scheme == "sigmoid"


def test_weight_scan_has_28_unique_configs_over_the_seven_weights() -> None:
    configs = build_weight_scan_configs()

    assert len(configs) == 28
    assert len({config.name for config in configs}) == 28
    assert {config.sweep_axis for config in configs} == {"ai2_weight"}
    assert {(config.T, config.hidden_sizes) for config in configs} == {(5, (32, 32, 32, 32))}
    assert sorted({config.ai2_weight for config in configs}) == sorted(WEIGHT_SCAN_WEIGHTS)
    assert all(
        cast(FCNN_AI2, build_model(config)).hidden_layers[0].ai2_weight == config.ai2_weight
        for config in configs
    )


def test_weight_label_is_filename_safe_and_distinct() -> None:
    labels = [weight_label(weight) for weight in WEIGHT_SCAN_WEIGHTS]

    assert labels == ["-0p667", "-0p333", "0", "0p333", "0p667", "1", "1p333"]
