"""Tests for AutapticLayer. See lab-book.md."""

import pytest
import torch

from autapse import AutapticLayer, Mode, Scheme

MODES: list[Mode] = ["ai2_diff", "ai2_gate"]
SCHEMES: list[Scheme] = ["sigmoid", "relu_sigmoid"]


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


BOUNDED_COMBOS: list[tuple[Mode, Scheme]] = [
    ("ai2_gate", "sigmoid"),
    ("ai2_gate", "relu_sigmoid"),
    ("ai2_diff", "relu_sigmoid"),
]


@pytest.mark.parametrize("mode,scheme", BOUNDED_COMBOS)
def test_output_never_exceeds_raw_activation(mode: Mode, scheme: Scheme) -> None:
    """0 <= y <= y_raw, for every combo except `ai2_diff` + `sigmoid`.

    That combo has no such guarantee: `sigmoid` never clips the difference
    back to non-negative, so `y_tilde_prev` can go negative — the
    convergence problem lab-book.md's "Activation functions" section
    discusses, and why `relu_sigmoid` exists.
    """
    layer = AutapticLayer(4, 3, T=5, mode=mode, scheme=scheme)
    x = torch.randn(8, 4) * 10  # large magnitude to stress the gate
    y = layer(x)
    activation = torch.sigmoid if scheme == "sigmoid" else torch.relu
    y_raw = activation(layer.linear(x))
    assert torch.all(y >= 0)
    assert torch.all(y <= y_raw + 1e-6)
