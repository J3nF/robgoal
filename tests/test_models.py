"""Tests for FCNN_AI2's configurable hidden_sizes. See lab-book.md."""

from typing import cast

import torch

from autapse import AutapticLayer
from models import FCNN_AI2


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
