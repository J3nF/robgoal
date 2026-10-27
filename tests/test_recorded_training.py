"""Tests for recorded_training.py: what every run records, independent of MLflow."""

import dataclasses

import numpy as np
import torch
from fakes import make_cnn_config, make_config, make_stages

import sweep
from dynamics_insights import get_y_self_absmeans
from recorded_training import NUM_DIGITS, TrainingHooks, build_arrays, train_and_record
from train import run_incremental_training

CPU = torch.device("cpu")


def _train(config, num_stages: int = 2, hooks=None):
    model = sweep.build_model(config)
    stages = make_stages(num_stages)
    return train_and_record(config, model, stages, CPU, hooks), stages


def test_record_has_accuracy_per_stage_and_loss_per_epoch() -> None:
    config = make_config("rec", epochs_per_stage=3)

    record, _ = _train(config, num_stages=2)

    assert len(record.accuracy_by_stage) == 2
    assert len(record.train_loss) == 2 * 3
    assert record.train_seconds > 0


def test_arrays_have_documented_shapes_dims_and_dtypes() -> None:
    record, _ = _train(make_config("rec", epochs_per_stage=2), num_stages=3)

    arrays = build_arrays(record)

    assert arrays["accuracy"].dims == ("epoch", "digit")
    assert arrays["accuracy"].data.shape == (6, NUM_DIGITS)
    assert arrays["accuracy"].data.dtype == np.float64
    assert arrays["train_loss"].dims == ("epoch",)
    assert arrays["train_loss"].data.shape == (6,)
    layer_paths = [path for path in arrays if path.startswith("y_self_absmean/")]
    assert layer_paths == ["y_self_absmean/hidden_layers.0", "y_self_absmean/hidden_layers.1"]
    for path in layer_paths:
        assert arrays[path].dims == ("epoch",)
        assert arrays[path].data.shape == (6,)
    assert not any(path.startswith("y_self_per_neuron") for path in arrays)


def test_accuracy_matrix_matches_the_per_digit_dicts() -> None:
    record, _ = _train(make_config("rec", epochs_per_stage=2), num_stages=2)

    accuracy = build_arrays(record)["accuracy"].data

    for epoch_index, per_digit in enumerate(record.accuracy_by_epoch):
        for digit, value in per_digit.items():
            assert accuracy[epoch_index, digit] == value


def test_last_epoch_of_each_stage_equals_the_stage_accuracy() -> None:
    record, _ = _train(make_config("rec", epochs_per_stage=3), num_stages=2)

    assert len(record.accuracy_by_epoch) == 6
    for stage_index, per_digit in enumerate(record.accuracy_by_stage):
        assert record.accuracy_by_epoch[(stage_index + 1) * 3 - 1] == per_digit


def test_per_epoch_evaluation_leaves_training_and_recorded_y_self_unchanged() -> None:
    config = make_config("rec", epochs_per_stage=2, T=3)
    stages = make_stages(2)

    torch.manual_seed(0)
    model = sweep.build_model(config)
    losses, y_self = [], []

    def on_epoch_end(stage: int, epoch: int, loss: float) -> None:
        losses.append(loss)
        y_self.append(get_y_self_absmeans(model))

    run_incremental_training(model, stages, CPU, epochs_per_stage=2, on_epoch_end=on_epoch_end)
    torch.manual_seed(0)
    record, _ = _train(config, num_stages=2)

    assert record.train_loss == losses
    assert [
        {name: values[epoch] for name, values in record.y_self_absmean.items()}
        for epoch in range(4)
    ] == y_self


def test_digits_missing_from_an_epoch_are_nan() -> None:
    record, _ = _train(make_config("rec"), num_stages=1)
    record.accuracy_by_epoch[0] = {0: 1.0, 1: 0.5}

    accuracy = build_arrays(record)["accuracy"].data

    assert accuracy[0, 0] == 1.0
    assert np.isnan(accuracy[0, 2:]).all()


def test_per_neuron_epoch_capture_adds_epoch_by_neuron_datasets() -> None:
    config = make_config("rec", epochs_per_stage=2, capture_y_self_per_epoch=True)

    record, _ = _train(config, num_stages=2)
    arrays = build_arrays(record)

    first_layer = arrays["y_self_per_neuron_epoch/hidden_layers.0"]
    assert first_layer.dims == ("epoch", "neuron")
    assert first_layer.data.shape == (4, 4)
    assert arrays["y_self_per_neuron_epoch/hidden_layers.1"].data.shape == (4, 2)
    assert not any(path.startswith("y_self_per_neuron_batch") for path in arrays)


def test_per_neuron_batch_capture_has_one_row_per_training_batch() -> None:
    config = make_config("rec", capture_y_self_per_batch=True)

    record, _ = _train(config, num_stages=2)
    arrays = build_arrays(record)

    batch_layer = arrays["y_self_per_neuron_batch/hidden_layers.0"]
    assert batch_layer.dims == ("batch", "neuron")
    # 10 training examples, batch_size 128 -> 1 batch per epoch; 2 stages x 1 epoch.
    assert batch_layer.data.shape == (2, 4)


def test_cnn_records_no_y_self_datasets() -> None:
    config = dataclasses.replace(
        make_cnn_config("rec"), capture_y_self_per_epoch=True, capture_y_self_per_batch=True
    )

    record, _ = _train(config, num_stages=2)
    arrays = build_arrays(record)

    assert sorted(arrays) == ["accuracy", "train_loss"]


def test_hooks_receive_every_epoch_and_stage_event() -> None:
    epochs, stages = [], []
    hooks = TrainingHooks(
        on_epoch_end=lambda stage, epoch, loss: epochs.append((stage, epoch)),
        on_stage_end=lambda stage, per_digit: stages.append(stage),
    )

    record, _ = _train(make_config("rec", epochs_per_stage=2), num_stages=2, hooks=hooks)

    assert epochs == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert stages == [0, 1]
    assert len(record.train_loss) == 4
    assert len(record.accuracy_by_stage) == 2


def test_batch_hook_alone_is_enough_to_enable_the_batch_callback() -> None:
    calls = []
    hooks = TrainingHooks(on_batch_end=lambda stage, epoch, batch: calls.append((stage, epoch)))

    _train(make_config("rec"), num_stages=2, hooks=hooks)

    assert calls == [(0, 0), (1, 0)]
