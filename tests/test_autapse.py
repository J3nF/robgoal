"""Tests for AutapticLayer. See old-lab-book.md."""

import pytest
import torch

from autapse import Activation_Scheme, AI2_Variant, AutapticLayer

VARIANTS: list[AI2_Variant] = ["ai2_diff", "ai2_gate"]
SCHEMES: list[Activation_Scheme] = ["sigmoid", "relu_sigmoid"]


@pytest.mark.parametrize("activation_scheme", SCHEMES)
@pytest.mark.parametrize("ai2_variant", VARIANTS)
def test_t1_degenerates_to_plain_layer(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme
) -> None:
    layer = AutapticLayer(4, 3, T=1, ai2_variant=ai2_variant, activation_scheme=activation_scheme)
    x = torch.randn(2, 4)
    activation = torch.sigmoid if activation_scheme == "sigmoid" else torch.relu
    expected = activation(layer.linear(x))
    assert torch.allclose(layer(x), expected, atol=1e-6)


@pytest.mark.parametrize("activation_scheme", SCHEMES)
@pytest.mark.parametrize("ai2_variant", VARIANTS)
def test_gradients_flow_through_unroll(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme
) -> None:
    # Fixed seed: an unlucky unseeded draw can put every ReLU pre-activation
    # below 0, zeroing the gradient through no fault of the layer itself.
    torch.manual_seed(0)
    layer = AutapticLayer(4, 3, T=3, ai2_variant=ai2_variant, activation_scheme=activation_scheme)
    x = torch.randn(2, 4, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer.linear.weight.grad is not None
    assert layer.linear.weight.grad.abs().sum() > 0


BOUNDED_COMBOS: list[tuple[AI2_Variant, Activation_Scheme]] = [
    ("ai2_gate", "sigmoid"),
    ("ai2_gate", "relu_sigmoid"),
    ("ai2_diff", "relu_sigmoid"),
]


@pytest.mark.parametrize("ai2_variant,activation_scheme", BOUNDED_COMBOS)
def test_output_never_exceeds_raw_activation(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme
) -> None:
    """0 <= y <= y_raw, for every combo except `ai2_diff` + `sigmoid`.

    That combo has no such guarantee: `sigmoid` never clips the difference
    back to non-negative, so `y_tilde_prev` can go negative — the
    convergence problem old-lab-book.md's "Activation functions" section
    discusses, and why `relu_sigmoid` exists.
    """
    layer = AutapticLayer(4, 3, T=5, ai2_variant=ai2_variant, activation_scheme=activation_scheme)
    x = torch.randn(8, 4) * 10  # large input to stress the gate
    y = layer(x)
    activation = torch.sigmoid if activation_scheme == "sigmoid" else torch.relu
    y_raw = activation(layer.linear(x))
    assert torch.all(y >= 0)
    assert torch.all(y <= y_raw + 1e-6)


def _reference_forward(layer: AutapticLayer, x: torch.Tensor) -> torch.Tensor:
    """The unweighted unroll as implemented before `ai2_weight` existed, written out inline."""
    activation = torch.sigmoid if layer.activation_scheme == "sigmoid" else torch.relu
    y = activation(layer.linear(x))
    y_tilde = y
    for _ in range(layer.T - 1):
        y_self = y_tilde if layer.activation_scheme == "sigmoid" else torch.tanh(y_tilde / 2)
        y_tilde = y - y_self if layer.ai2_variant == "ai2_diff" else y * (1 - y_self)
    return y_tilde


def _make_layer(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme, T: int, **kwargs: float
) -> AutapticLayer:
    torch.manual_seed(0)
    return AutapticLayer(4, 3, T, ai2_variant, activation_scheme, **kwargs)


@pytest.mark.parametrize("T", [1, 3, 5])
@pytest.mark.parametrize("activation_scheme", SCHEMES)
@pytest.mark.parametrize("ai2_variant", VARIANTS)
def test_default_weight_equals_explicit_one_and_the_unweighted_reference(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme, T: int
) -> None:
    x = torch.randn(6, 4)
    default = _make_layer(ai2_variant, activation_scheme, T)
    explicit = _make_layer(ai2_variant, activation_scheme, T, ai2_weight=1.0)

    assert default.ai2_weight == 1.0
    assert torch.equal(default(x), _reference_forward(default, x))
    assert torch.equal(default(x), explicit(x))


@pytest.mark.parametrize("activation_scheme", SCHEMES)
@pytest.mark.parametrize("ai2_variant", VARIANTS)
def test_weight_zero_at_t5_equals_t1_in_output_and_gradients(
    ai2_variant: AI2_Variant, activation_scheme: Activation_Scheme
) -> None:
    x = torch.randn(6, 4)
    unrolled = _make_layer(ai2_variant, activation_scheme, 5, ai2_weight=0.0)
    single = _make_layer(ai2_variant, activation_scheme, 1)

    output = unrolled(x)
    output.sum().backward()
    single(x).sum().backward()

    assert torch.equal(output, single(x))
    assert unrolled.linear.weight.grad is not None and single.linear.weight.grad is not None
    assert torch.equal(unrolled.linear.weight.grad, single.linear.weight.grad)


def test_diff_sigmoid_weight_one_odd_t_is_the_t1_control_and_even_t_is_dead() -> None:
    x = torch.randn(6, 4)
    odd = _make_layer("ai2_diff", "sigmoid", 5)
    control = _make_layer("ai2_diff", "sigmoid", 1)
    odd_out = odd(x)
    odd_out.sum().backward()
    control(x).sum().backward()

    assert torch.equal(odd_out, control(x))
    assert odd.linear.weight.grad is not None and control.linear.weight.grad is not None
    assert torch.equal(odd.linear.weight.grad, control.linear.weight.grad)
    assert torch.equal(_make_layer("ai2_diff", "sigmoid", 4)(x), torch.zeros(6, 3))


@pytest.mark.parametrize("weight", [-2 / 3, -1 / 3, 1 / 3, 2 / 3, 4 / 3])
def test_diff_sigmoid_output_follows_the_closed_form(weight: float) -> None:
    """y_tilde at the last step is y times the geometric sum of (-w)^k, k = 0..T-1."""
    T = 5
    x = torch.randn(6, 4)
    layer = _make_layer("ai2_diff", "sigmoid", T, ai2_weight=weight)

    expected = torch.sigmoid(layer.linear(x)) * sum((-weight) ** k for k in range(T))

    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_negative_weight_excites_instead_of_inhibiting() -> None:
    x = torch.randn(6, 4)
    gate = _make_layer("ai2_gate", "sigmoid", 2, ai2_weight=-1.0)
    y = torch.sigmoid(gate.linear(x))

    assert torch.all(gate(x) >= y)
    assert torch.allclose(gate(x), y * (1 + y), atol=1e-6)
