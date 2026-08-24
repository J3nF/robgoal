"""Tests for get_y_self_magnitudes. See lab-book.md."""

import torch

from dynamics_insights import get_y_self_magnitudes
from models import CNN, FCNN_AI2


def test_no_forward_pass_yet_returns_empty() -> None:
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="sigmoid")
    assert get_y_self_magnitudes(model) == {}


def test_t1_degenerates_to_zero_magnitude() -> None:
    model = FCNN_AI2(T=1, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    model(torch.randn(4, 1, 28, 28))

    magnitudes = get_y_self_magnitudes(model)

    assert magnitudes == {"hidden_layers.0": 0.0, "hidden_layers.1": 0.0}


def test_t_greater_than_one_produces_nonzero_magnitude() -> None:
    torch.manual_seed(0)
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    model(torch.randn(4, 1, 28, 28) * 10)

    magnitudes = get_y_self_magnitudes(model)

    assert set(magnitudes) == {"hidden_layers.0", "hidden_layers.1"}
    assert any(magnitude > 0 for magnitude in magnitudes.values())


def test_cnn_has_no_autaptic_layers() -> None:
    model = CNN()
    model(torch.randn(4, 1, 28, 28))

    assert get_y_self_magnitudes(model) == {}
