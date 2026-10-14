"""Training with always-on recording: what every run keeps, independent of MLflow.

`train_and_record` runs `train.run_incremental_training` and collects accuracy, per-epoch loss
and y_self, and (if the config asks) per-neuron y_self histories. `build_arrays` turns that
record into the datasets stored in the run's `.h5` file (see `run_store`). The optional MLflow
viewer plugs in through `TrainingHooks`; see `logged_training.py`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from data import Stage
from dynamics_insights import get_y_self_absmeans, get_y_self_per_neuron
from run_config import RunConfig
from run_store import RunArray
from train import run_incremental_training

__all__ = ["NUM_DIGITS", "TrainingHooks", "TrainingRecord", "build_arrays", "train_and_record"]

NUM_DIGITS = 10

EpochCallback = Callable[[int, int, float], None]
StageCallback = Callable[[int, dict[int, float]], None]
BatchCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class TrainingHooks:
    """Extra live callbacks (e.g. MLflow logging), called after the recorder's own.

    Same signatures as the `on_*` parameters of `train.run_incremental_training`.
    """

    on_epoch_end: EpochCallback | None = None
    on_stage_end: StageCallback | None = None
    on_batch_end: BatchCallback | None = None


@dataclass
class TrainingRecord:
    """Everything one training run produced.

    Attributes:
        accuracy_by_stage: One per-digit accuracy dict per stage.
        train_loss: Mean loss per epoch, indexed by global epoch
            (`stage_index * epochs_per_stage + epoch_index`).
        y_self_absmean: Per AutapticLayer name, mean |y_self| after each epoch.
        y_self_per_neuron_epoch: Per layer name, one per-neuron row per epoch; empty unless
            `config.capture_y_self_per_epoch`.
        y_self_per_neuron_batch: Per layer name, one per-neuron row per training batch; empty
            unless `config.capture_y_self_per_batch`.
        train_seconds: Wall-clock training time.
    """

    accuracy_by_stage: list[dict[int, float]] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    y_self_absmean: dict[str, list[float]] = field(default_factory=dict)
    y_self_per_neuron_epoch: dict[str, list[list[float]]] = field(default_factory=dict)
    y_self_per_neuron_batch: dict[str, list[list[float]]] = field(default_factory=dict)
    train_seconds: float = 0.0


def train_and_record(
    config: RunConfig,
    model: nn.Module,
    stages: list[Stage],
    device: torch.device,
    hooks: TrainingHooks | None = None,
) -> TrainingRecord:
    """Trains `model` through `stages` and returns what happened; never touches MLflow.

    Args:
        hooks: Optional live callbacks, invoked after the recorder's own for each event.
    """
    hooks = hooks if hooks is not None else TrainingHooks()
    record = TrainingRecord()
    on_batch_end = _chain_batch(
        _make_batch_recorder(model, record) if config.capture_y_self_per_batch else None,
        hooks.on_batch_end,
    )
    start_time = time.perf_counter()
    run_incremental_training(
        model,
        stages,
        device,
        epochs_per_stage=config.epochs_per_stage,
        lr=config.lr,
        batch_size=config.batch_size,
        on_epoch_end=_chain_epoch(
            _make_epoch_recorder(model, record, config.capture_y_self_per_epoch),
            hooks.on_epoch_end,
        ),
        on_stage_end=_chain_stage(_make_stage_recorder(record), hooks.on_stage_end),
        on_batch_end=on_batch_end,
    )
    record.train_seconds = time.perf_counter() - start_time
    return record


def build_arrays(record: TrainingRecord) -> dict[str, RunArray]:
    """The datasets a run file stores for `record`, keyed by dataset path.

    `accuracy` is (stage, digit) with NaN for digits a stage did not evaluate; the per-layer
    datasets exist only for models with AutapticLayers (the CNN has none).
    """
    arrays = {
        "accuracy": RunArray(_to_accuracy_matrix(record.accuracy_by_stage), ("stage", "digit")),
        "train_loss": RunArray(np.asarray(record.train_loss, dtype=np.float64), ("epoch",)),
    }
    for layer, values in record.y_self_absmean.items():
        arrays[f"y_self_absmean/{layer}"] = RunArray(
            np.asarray(values, dtype=np.float64), ("epoch",)
        )
    for prefix, history, axis in (
        ("y_self_per_neuron_epoch", record.y_self_per_neuron_epoch, "epoch"),
        ("y_self_per_neuron_batch", record.y_self_per_neuron_batch, "batch"),
    ):
        for layer, rows in history.items():
            arrays[f"{prefix}/{layer}"] = RunArray(
                np.asarray(rows, dtype=np.float64), (axis, "neuron")
            )
    return arrays


def _to_accuracy_matrix(accuracy_by_stage: list[dict[int, float]]) -> np.ndarray:
    matrix = np.full((len(accuracy_by_stage), NUM_DIGITS), np.nan, dtype=np.float64)
    for stage_index, per_digit in enumerate(accuracy_by_stage):
        for digit, accuracy in per_digit.items():
            matrix[stage_index, digit] = accuracy
    return matrix


def _make_epoch_recorder(
    model: nn.Module, record: TrainingRecord, capture_per_neuron: bool
) -> EpochCallback:
    """Callback appending loss and y_self summaries to `record` (mutates it) after each epoch.

    The y_self values are the last batch's snapshot, not an epoch-weighted average like the
    loss: averaging would need train.py to know AutapticLayer exists.
    """

    def on_epoch_end(stage_index: int, epoch_index: int, avg_loss: float) -> None:
        del stage_index, epoch_index
        record.train_loss.append(avg_loss)
        for layer, absmean in get_y_self_absmeans(model).items():
            record.y_self_absmean.setdefault(layer, []).append(absmean)
        if capture_per_neuron:
            _append_rows(record.y_self_per_neuron_epoch, get_y_self_per_neuron(model))

    return on_epoch_end


def _make_batch_recorder(model: nn.Module, record: TrainingRecord) -> BatchCallback:
    """Callback appending a per-neuron y_self row per training batch to `record` (mutates it)."""

    def on_batch_end(stage_index: int, epoch_index: int, batch_index: int) -> None:
        del stage_index, epoch_index, batch_index
        _append_rows(record.y_self_per_neuron_batch, get_y_self_per_neuron(model))

    return on_batch_end


def _make_stage_recorder(record: TrainingRecord) -> StageCallback:
    """Callback appending each stage's per-digit accuracy to `record` (mutates it)."""

    def on_stage_end(stage_index: int, per_digit_accuracy: dict[int, float]) -> None:
        del stage_index
        record.accuracy_by_stage.append(per_digit_accuracy)

    return on_stage_end


def _append_rows(history: dict[str, list[list[float]]], snapshot: dict[str, list[float]]) -> None:
    """Appends each layer's per-neuron row of `snapshot` to `history` (mutates it)."""
    for layer, row in snapshot.items():
        history.setdefault(layer, []).append(row)


def _chain_epoch(first: EpochCallback, second: EpochCallback | None) -> EpochCallback:
    if second is None:
        return first

    def chained(stage_index: int, epoch_index: int, avg_loss: float) -> None:
        first(stage_index, epoch_index, avg_loss)
        second(stage_index, epoch_index, avg_loss)

    return chained


def _chain_stage(first: StageCallback, second: StageCallback | None) -> StageCallback:
    if second is None:
        return first

    def chained(stage_index: int, per_digit_accuracy: dict[int, float]) -> None:
        first(stage_index, per_digit_accuracy)
        second(stage_index, per_digit_accuracy)

    return chained


def _chain_batch(first: BatchCallback | None, second: BatchCallback | None) -> BatchCallback | None:
    """Combines optional batch callbacks; `None` when neither exists so training skips the hook."""
    if first is None or second is None:
        return first if first is not None else second

    def chained(stage_index: int, epoch_index: int, batch_index: int) -> None:
        first(stage_index, epoch_index, batch_index)
        second(stage_index, epoch_index, batch_index)

    return chained
