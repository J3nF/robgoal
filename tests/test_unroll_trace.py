"""Tests for unroll_trace.py: the step-by-step replay of AutapticLayer unrolls."""

from typing import cast

import pytest
import torch
from fakes import FakeDataset

from autapse import Activation_Scheme, AI2_Variant, AutapticLayer
from models import CNN, FCNN_AI2
from unroll_trace import build_trace_arrays, get_layer_trace, get_model_trace

VARIANTS: list[AI2_Variant] = ["ai2_diff", "ai2_gate"]
SCHEMES: list[Activation_Scheme] = ["sigmoid", "relu_sigmoid"]
WEIGHTS = [-2 / 3, 0.0, 1 / 3, 1.0, 4 / 3]


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("activation_scheme", SCHEMES)
@pytest.mark.parametrize("ai2_variant", VARIANTS)
def test_trace_ends_exactly_where_forward_ends(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme, weight: float
) -> None:
    torch.manual_seed(0)
    layer = AutapticLayer(4, 3, 5, ai2_variant, activation_scheme, weight)
    x = torch.randn(7, 4)

    trace = get_layer_trace(layer, x)

    assert trace.y_tilde.shape == (5, 7, 3)
    assert torch.equal(trace.y_tilde[-1], layer(x))
    assert layer.last_y_self is not None
    assert torch.equal(trace.y_self_used[-1], weight * layer.last_y_self)
    assert torch.equal(trace.y_self_used[0], torch.zeros(7, 3))
    assert torch.equal(trace.y_tilde[0], trace.y)


def test_t1_trace_is_the_plain_activation_for_any_weight() -> None:
    layer = AutapticLayer(4, 3, 1, "ai2_gate", "sigmoid", ai2_weight=4 / 3)
    x = torch.randn(5, 4)

    trace = get_layer_trace(layer, x)

    assert torch.equal(trace.y_tilde[0], torch.sigmoid(layer.linear(x)))
    assert torch.equal(trace.y_self_used, torch.zeros(1, 5, 3))


def test_diff_sigmoid_weight_one_alternates_between_y_and_zero() -> None:
    layer = AutapticLayer(4, 3, 5, "ai2_diff", "sigmoid", ai2_weight=1.0)

    trace = get_layer_trace(layer, torch.randn(5, 4))

    zeros = torch.zeros_like(trace.y)
    for step, expected in enumerate([trace.y, zeros, trace.y, zeros, trace.y]):
        assert torch.equal(trace.y_tilde[step], expected)


def test_trace_of_a_diverged_layer_does_not_raise() -> None:
    layer = AutapticLayer(4, 3, 3, "ai2_diff", "relu_sigmoid")
    with torch.no_grad():
        layer.linear.weight.fill_(float("nan"))

    trace = get_layer_trace(layer, torch.randn(5, 4))

    assert torch.isnan(trace.y_tilde).all()


def test_model_trace_feeds_each_layer_the_previous_final_output() -> None:
    model = FCNN_AI2(T=3, ai2_variant="ai2_gate", activation_scheme="sigmoid", hidden_sizes=(5, 4))
    images = torch.randn(6, 1, 28, 28)

    traces = get_model_trace(model, images)

    assert [trace.y.shape for trace in traces] == [(6, 5), (6, 4)]
    second_input = traces[0].y_tilde[-1]
    assert torch.equal(
        traces[1].y, torch.sigmoid(cast(AutapticLayer, model.hidden_layers[1]).linear(second_input))
    )


def test_cnn_has_no_trace() -> None:
    assert get_model_trace(CNN(channels=(1, 1)), torch.randn(2, 1, 28, 28)) == []
    assert build_trace_arrays(CNN(channels=(1, 1)), FakeDataset(), torch.device("cpu")) == {}


def test_trace_arrays_have_documented_shapes_and_dims() -> None:
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="sigmoid", hidden_sizes=(5, 4))

    arrays = build_trace_arrays(model, FakeDataset(), torch.device("cpu"))

    expected = {
        "unroll/mean_y": ((2, 10), ("layer", "class")),
        "unroll/mean_y_self_used": ((2, 10, 3), ("layer", "class", "step")),
        "unroll/mean_y_tilde": ((2, 10, 3), ("layer", "class", "step")),
        "unroll/mean_abs_y_tilde_class": ((2, 10, 3), ("layer", "class", "step")),
        "unroll/mean_abs_y_tilde": ((2, 3), ("layer", "step")),
        "unroll/neuron_std_y_tilde": ((2, 3), ("layer", "step")),
    }
    assert {name: (a.data.shape, a.dims) for name, a in arrays.items()} == expected


def test_trace_arrays_match_a_hand_computation_for_one_class() -> None:
    model = FCNN_AI2(T=2, ai2_variant="ai2_gate", activation_scheme="sigmoid", hidden_sizes=(3,))
    dataset = FakeDataset()
    arrays = build_trace_arrays(model, dataset, torch.device("cpu"))

    image = dataset[7][0].view(1, -1)
    layer = cast(AutapticLayer, model.hidden_layers[0])
    y = torch.sigmoid(layer.linear(image))
    y_tilde_1 = y * (1 - y)

    mean_y_tilde = arrays["unroll/mean_y_tilde"].data
    assert mean_y_tilde[0, 7, 0] == pytest.approx(y.mean().item(), abs=1e-6)
    assert mean_y_tilde[0, 7, 1] == pytest.approx(y_tilde_1.mean().item(), abs=1e-6)
    assert arrays["unroll/mean_y"].data[0, 7] == pytest.approx(y.mean().item(), abs=1e-6)
