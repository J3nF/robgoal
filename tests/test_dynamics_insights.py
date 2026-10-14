"""Tests for get_y_self_absmeans and get_y_self_per_neuron. See old-lab-book.md."""

import torch

from dynamics_insights import get_y_self_absmeans, get_y_self_per_neuron
from models import CNN, FCNN_AI2


def test_no_forward_pass_yet_returns_empty() -> None:
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="sigmoid")
    assert get_y_self_absmeans(model) == {}


def test_t1_degenerates_to_zero_absmean() -> None:
    model = FCNN_AI2(T=1, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    model(torch.randn(4, 1, 28, 28))

    absmeans = get_y_self_absmeans(model)

    assert absmeans == {"hidden_layers.0": 0.0, "hidden_layers.1": 0.0}


def test_t_greater_than_one_produces_nonzero_absmean() -> None:
    torch.manual_seed(0)
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    model(torch.randn(4, 1, 28, 28) * 10)

    absmeans = get_y_self_absmeans(model)

    assert set(absmeans) == {"hidden_layers.0", "hidden_layers.1"}
    assert any(absmean > 0 for absmean in absmeans.values())


def test_cnn_has_no_autaptic_layers() -> None:
    model = CNN()
    model(torch.randn(4, 1, 28, 28))

    assert get_y_self_absmeans(model) == {}


def test_per_neuron_no_forward_pass_yet_returns_empty() -> None:
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="sigmoid")
    assert get_y_self_per_neuron(model) == {}


def test_per_neuron_t1_degenerates_to_all_zeros() -> None:
    model = FCNN_AI2(T=1, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    model(torch.randn(4, 1, 28, 28))

    per_neuron = get_y_self_per_neuron(model)

    assert set(per_neuron) == {"hidden_layers.0", "hidden_layers.1"}
    assert all(value == 0.0 for values in per_neuron.values() for value in values)


def test_per_neuron_length_matches_out_features() -> None:
    torch.manual_seed(0)
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    model(torch.randn(4, 1, 28, 28) * 10)

    per_neuron = get_y_self_per_neuron(model)

    assert set(per_neuron) == {"hidden_layers.0", "hidden_layers.1"}
    for layer_name, values in per_neuron.items():
        out_features = dict(model.named_modules())[layer_name].linear.out_features
        assert len(values) == out_features


def test_per_neuron_keeps_sign() -> None:
    """A real forward pass can't exercise this: every (ai2_variant,
    activation_scheme) combination reachable in this codebase produces a
    non-negative last_y_self, so a test built on real forward output can't
    distinguish "kept sign" from a silent `.abs()` regression. Sets
    `last_y_self` directly instead.
    """
    model = FCNN_AI2(T=3, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid")
    layer = dict(model.named_modules())["hidden_layers.0"]
    layer.last_y_self = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])

    per_neuron = get_y_self_per_neuron(model)

    assert per_neuron["hidden_layers.0"] == [-2.0, 3.0]


def test_per_neuron_cnn_has_no_autaptic_layers() -> None:
    model = CNN()
    model(torch.randn(4, 1, 28, 28))

    assert get_y_self_per_neuron(model) == {}
