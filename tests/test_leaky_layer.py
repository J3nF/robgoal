"""Tests for LeakyLayer and reset_inhibition_state."""

import pytest
import torch

from autapse import Activation_Scheme, LeakyLayer, reset_inhibition_state
from models import FCNN_AI2

SCHEMES: list[Activation_Scheme] = ["sigmoid", "relu_sigmoid"]


def _signal(layer: LeakyLayer, x: torch.Tensor) -> torch.Tensor:
    raw = layer.linear(x)
    if layer.activation_scheme == "sigmoid":
        return torch.sigmoid(raw).detach()
    return torch.tanh(torch.relu(raw) / 2).detach()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_matches_a_per_sample_loop(scheme: Activation_Scheme) -> None:
    torch.manual_seed(0)
    layer = LeakyLayer(5, 3, scheme, decay=0.7)
    x = torch.randn(9, 5)
    layer(x)
    signal = _signal(layer, x)
    state = torch.zeros(3)
    expected = []
    for i in range(9):
        expected.append(state)
        state = 0.7 * state + 0.3 * signal[i]
    assert layer.last_inhibition is not None
    torch.testing.assert_close(layer.last_inhibition, torch.stack(expected))


def test_chunked_stream_equals_one_batch() -> None:
    torch.manual_seed(1)
    layer = LeakyLayer(4, 3, "sigmoid", decay=0.8)
    x = torch.randn(11, 4)
    whole = layer(x)
    layer.reset_state()
    parts = torch.cat([layer(x[:3]), layer(x[3:4]), layer(x[4:])])
    torch.testing.assert_close(parts, whole)


def test_inhibition_is_causal() -> None:
    torch.manual_seed(2)
    layer = LeakyLayer(4, 3, "sigmoid", decay=0.9)
    x = torch.randn(8, 4)
    layer(x)
    before = layer.last_inhibition
    assert before is not None
    changed = x.clone()
    changed[5] += 10
    layer.reset_state()
    layer(changed)
    after = layer.last_inhibition
    assert after is not None
    torch.testing.assert_close(after[:6], before[:6])  # sample 5 does not see itself
    assert (after[6:] - before[6:]).abs().max() > 0


def test_first_sample_is_uninhibited_and_inhibition_is_bounded() -> None:
    layer = LeakyLayer(4, 3, "relu_sigmoid", decay=0.9)
    layer(torch.randn(50, 4) * 5)
    assert layer.last_inhibition is not None
    assert torch.all(layer.last_inhibition[0] == 0)
    assert layer.last_inhibition.min() >= 0
    assert layer.last_inhibition.max() < 1


def test_decay_zero_uses_only_the_previous_sample() -> None:
    torch.manual_seed(3)
    layer = LeakyLayer(4, 2, "sigmoid", decay=0.0)
    x = torch.randn(6, 4)
    layer(x)
    signal = _signal(layer, x)
    assert layer.last_inhibition is not None
    torch.testing.assert_close(layer.last_inhibition[1:], signal[:-1])


def test_weight_zero_is_a_plain_layer() -> None:
    torch.manual_seed(4)
    layer = LeakyLayer(4, 3, "sigmoid", ai2_weight=0.0)
    x = torch.randn(5, 4)
    torch.testing.assert_close(layer(x), torch.sigmoid(layer.linear(x)))


def test_state_carries_between_batches_and_reset_clears_it() -> None:
    torch.manual_seed(5)
    layer = LeakyLayer(4, 3, "sigmoid")
    x = torch.randn(4, 4)
    layer(x)
    layer(x)
    assert layer.last_inhibition is not None
    assert layer.last_inhibition[0].abs().sum() > 0  # carried from the first batch
    reset_inhibition_state(torch.nn.Sequential(layer))
    layer(x)
    assert layer.last_inhibition[0].abs().sum() == 0


def test_train_and_eval_states_are_separate() -> None:
    torch.manual_seed(6)
    layer = LeakyLayer(4, 3, "sigmoid")
    x = torch.randn(4, 4)
    layer.train()
    layer(x)
    layer.eval()
    layer(x)
    assert layer.last_inhibition is not None
    assert layer.last_inhibition[0].abs().sum() == 0  # eval did not inherit the train state
    layer.train()
    layer(x)
    assert layer.last_inhibition[0].abs().sum() > 0  # train state survived the eval pass


def test_carried_state_is_detached_but_in_batch_gradients_flow() -> None:
    torch.manual_seed(7)
    layer = LeakyLayer(4, 3, "sigmoid")
    first = torch.randn(3, 4, requires_grad=True)
    layer(first)
    second = torch.randn(3, 4, requires_grad=True)
    layer(second)[1:].sum().backward()
    assert second.grad is not None and second.grad[0].abs().sum() > 0  # earlier sample, same batch
    assert first.grad is None  # previous batch is detached


def test_single_sample_batches_work() -> None:
    layer = LeakyLayer(4, 3, "sigmoid")
    assert layer(torch.randn(1, 4)).shape == (1, 3)
    assert layer(torch.randn(1, 4)).shape == (1, 3)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 2.0])
def test_decay_outside_unit_interval_raises(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        LeakyLayer(4, 3, "sigmoid", decay=decay)


def test_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError, match="activation_scheme"):
        LeakyLayer(4, 3, "tanh")  # ty: ignore[invalid-argument-type]


def test_fcnn_with_leaky_layers_has_the_requested_width_and_classes() -> None:
    model = FCNN_AI2(
        T=1,
        ai2_variant="ai2_gate",
        activation_scheme="sigmoid",
        hidden_sizes=(6, 6),
        num_classes=20,
        layer_kind="leaky",
        decay=0.5,
    )
    assert all(isinstance(layer, LeakyLayer) for layer in model.hidden_layers)
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 20)


def test_resetting_only_eval_state_keeps_the_training_stream() -> None:
    torch.manual_seed(8)
    layer = LeakyLayer(4, 3, "sigmoid")
    x = torch.randn(4, 4)
    layer.train()
    layer(x)
    layer.eval()
    layer(x)
    reset_inhibition_state(torch.nn.Sequential(layer), training=False)
    layer(x)
    assert layer.last_inhibition is not None
    assert layer.last_inhibition[0].abs().sum() == 0  # eval state cleared
    layer.train()
    layer(x)
    assert layer.last_inhibition[0].abs().sum() > 0  # train state kept
