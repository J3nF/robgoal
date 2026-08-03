"""Tests for AutapticLayer. See lab-book.md, Verification."""

import pytest
import torch

from autapse import AutapticLayer, Mode, Scheme

MODES: list[Mode] = ["output_diff", "bias_diff", "mult_gate"]
SCHEMES: list[Scheme] = ["sigmoid", "relu_sigmoid_gate"]


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", MODES)
def test_t1_degenerates_to_plain_layer(mode: Mode, scheme: Scheme) -> None:
    layer = AutapticLayer(4, 3, T=1, mode=mode, scheme=scheme)
    x = torch.randn(2, 4)
    activation = torch.sigmoid if scheme == "sigmoid" else torch.relu
    expected = activation(layer.linear(x))
    assert torch.allclose(layer(x), expected, atol=1e-6)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", MODES)
def test_gradients_flow_through_unroll(mode: Mode, scheme: Scheme) -> None:
    layer = AutapticLayer(4, 3, T=3, mode=mode, scheme=scheme)
    x = torch.randn(2, 4, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer.linear.weight.grad is not None
    assert layer.linear.weight.grad.abs().sum() > 0


def test_mult_gate_relu_sigmoid_gate_never_sign_flips() -> None:
    layer = AutapticLayer(4, 3, T=5, mode="mult_gate", scheme="relu_sigmoid_gate")
    x = torch.randn(8, 4) * 10  # large magnitude to stress the gate
    y = layer(x)
    y_raw = torch.relu(layer.linear(x))
    assert torch.all(y >= 0)
    assert torch.all(y <= y_raw + 1e-6)
