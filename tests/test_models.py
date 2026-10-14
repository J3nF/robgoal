"""Tests for FCNN_AI2's configurable hidden_sizes and CNN's configurable channels.

See old-lab-book.md."""

from typing import cast

import torch

from autapse import AutapticLayer
from models import CNN, FCNN_AI2


def test_default_hidden_sizes_produce_correct_output_shape() -> None:
    model = FCNN_AI2(T=2, ai2_variant="ai2_diff", activation_scheme="sigmoid")
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
    assert len(model.hidden_layers) == 2


def test_custom_hidden_sizes_produce_correct_output_shape() -> None:
    model = FCNN_AI2(
        T=2, ai2_variant="ai2_gate", activation_scheme="relu_sigmoid", hidden_sizes=(64, 32, 16)
    )
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
    assert len(model.hidden_layers) == 3
    hidden_layers = [cast(AutapticLayer, layer) for layer in model.hidden_layers]
    assert [layer.linear.out_features for layer in hidden_layers] == [64, 32, 16]
    assert model.output.in_features == 16


def test_default_channels_produce_correct_output_shape() -> None:
    model = CNN()
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
    assert model.conv1.out_channels == 16
    assert model.conv2.out_channels == 32


def test_custom_channels_produce_correct_output_shape() -> None:
    model = CNN(channels=(4, 8))
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
    assert model.conv1.out_channels == 4
    assert model.conv2.out_channels == 8
    assert model.output.in_features == 8 * 7 * 7
