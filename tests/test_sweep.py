"""Tests for sweep.py's config/model factory. See lab-book.md."""

from typing import cast

from autapse import AutapticLayer
from models import CNN, FCNN_AI2
from run_config import RunConfig
from sweep import build_configs, build_model


def test_configs_have_unique_names() -> None:
    configs = build_configs()

    assert len(configs) == 14
    assert len({config.name for config in configs}) == 14


def test_every_config_builds_a_model_without_raising() -> None:
    for config in build_configs():
        build_model(config)


def test_cnn_config_builds_cnn() -> None:
    config = RunConfig(name="x", sweep_axis="base", model_kind="cnn", epochs_per_stage=1)

    assert isinstance(build_model(config), CNN)


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
