"""Tests for per-layer inhibition recording, and its wiring into training records."""

from typing import cast

import numpy as np
import pytest
import torch
from fakes import make_config, make_stages

from autapse import LeakyLayer, reset_inhibition_state
from inhibition_recording import (
    InhibitionRecord,
    build_inhibition_arrays,
    collect_test_inhibition,
    get_inhibition_layers,
    get_inhibition_signal,
    record_test_stage,
    record_train_batch,
)
from models import CNN, FCNN_AI2
from recorded_training import build_arrays, train_and_record
from sweep import build_model


def _train_state(model: FCNN_AI2) -> torch.Tensor:
    state = cast(LeakyLayer, model.hidden_layers[0])._states[True]
    assert state is not None
    return state


def _fcnn(T: int = 3, layer_kind: str = "unrolled") -> FCNN_AI2:
    return FCNN_AI2(
        T=T,
        ai2_variant="ai2_gate",
        activation_scheme="sigmoid",
        hidden_sizes=(4, 3),
        layer_kind=layer_kind,  # ty: ignore[invalid-argument-type]
    )


def test_controls_and_cnn_have_no_inhibition_layers() -> None:
    assert get_inhibition_layers(_fcnn(T=1)) == {}
    assert get_inhibition_layers(CNN((1, 1))) == {}
    assert len(get_inhibition_layers(_fcnn(T=3))) == 2
    assert len(get_inhibition_layers(_fcnn(layer_kind="leaky"))) == 2


def test_signal_is_none_before_a_forward_pass() -> None:
    layer = next(iter(get_inhibition_layers(_fcnn(layer_kind="leaky")).values()))
    assert get_inhibition_signal(layer) is None


@pytest.mark.parametrize("layer_kind", ["unrolled", "leaky"])
def test_train_batch_stats_match_the_signal(layer_kind: str) -> None:
    model = _fcnn(layer_kind=layer_kind)
    model(torch.randn(6, 1, 28, 28))
    record = InhibitionRecord()
    record_train_batch(model, record, stage_index=2)
    assert record.train_stage == [2]
    for name, layer in get_inhibition_layers(model).items():
        signal = get_inhibition_signal(layer)
        assert signal is not None
        assert record.train_mean[name] == [pytest.approx(signal.mean().item())]
        assert record.train_var[name] == [pytest.approx(signal.var(unbiased=False).item())]


def test_recording_a_control_records_nothing() -> None:
    model = _fcnn(T=1)
    model(torch.randn(2, 1, 28, 28))
    record = InhibitionRecord()
    record_train_batch(model, record, 0)
    assert record.train_stage == [] and build_inhibition_arrays(record) == {}


def test_test_pass_statistics_match_a_manual_computation() -> None:
    torch.manual_seed(0)
    model = _fcnn(layer_kind="leaky")
    data = make_stages()[0].test_data
    stats = collect_test_inhibition(model, data, torch.device("cpu"), num_classes=12)
    first = stats["hidden_layers.0"]
    model.eval()
    reset_inhibition_state(model, training=False)
    images = torch.stack([data[i][0] for i in range(len(data))])
    model(images)
    signal = get_inhibition_signal(model.hidden_layers[0])
    assert signal is not None
    assert first[0] == pytest.approx(signal.mean().item(), abs=1e-6)
    assert first[1] == pytest.approx(signal.var(unbiased=False).item(), abs=1e-6)
    assert first[2].shape == (12,)
    assert np.isnan(first[2][10:]).all() and not np.isnan(first[2][:10]).any()
    assert first[2][3] == pytest.approx(signal[3].mean().item(), abs=1e-6)


def test_test_pass_does_not_touch_the_training_rng_or_state() -> None:
    model = _fcnn(layer_kind="leaky")
    data = make_stages()[0].test_data
    model.train()
    model(torch.randn(3, 1, 28, 28))
    state_before = _train_state(model).clone()
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    record_test_stage(model, InhibitionRecord(), data, torch.device("cpu"), 10)
    assert torch.equal(torch.rand(3), expected)
    assert torch.equal(_train_state(model), state_before)


def test_controls_return_no_test_statistics() -> None:
    data = make_stages()[0].test_data
    assert collect_test_inhibition(_fcnn(T=1), data, torch.device("cpu"), 10) == {}


def test_training_record_holds_one_batch_row_per_batch_and_one_test_row_per_stage() -> None:
    config = make_config("a", T=3, hidden_sizes=(4, 3))
    stages = make_stages()
    record = train_and_record(config, build_model(config), stages, torch.device("cpu"))
    arrays = build_arrays(record)
    batches = len(arrays["inhibition/train_batch_stage"].data)
    assert batches == 2  # one batch per stage with the 10-image fake data
    assert arrays["inhibition/train_batch_stage"].data.tolist() == [0, 1]
    for layer in ("hidden_layers.0", "hidden_layers.1"):
        assert arrays[f"inhibition/train_mean/{layer}"].data.shape == (batches,)
        assert arrays[f"inhibition/train_var/{layer}"].data.shape == (batches,)
        assert arrays[f"inhibition/test_mean/{layer}"].data.shape == (2,)
        assert arrays[f"inhibition/test_var/{layer}"].data.shape == (2,)
        assert arrays[f"inhibition/test_class_mean/{layer}"].dims == ("stage", "class")
        assert arrays[f"inhibition/test_class_mean/{layer}"].data.shape == (2, 10)
        assert np.all(arrays[f"inhibition/test_var/{layer}"].data >= -1e-9)


def test_leaky_training_record_and_control_record() -> None:
    leaky = make_config("l", layer_kind="leaky", T=1, hidden_sizes=(4, 3))
    control = make_config("c", T=1, hidden_sizes=(4, 3))
    stages = make_stages()
    assert "inhibition/test_mean/hidden_layers.0" in build_arrays(
        train_and_record(leaky, build_model(leaky), stages, torch.device("cpu"))
    )
    assert not [
        key
        for key in build_arrays(
            train_and_record(control, build_model(control), stages, torch.device("cpu"))
        )
        if key.startswith("inhibition/")
    ]
