"""MLflow adapter for live experiment tracking. See lab-book.md, "Logging: MLflow"."""

import dataclasses
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import torch
from torch import nn

from data import Stage
from dynamics_insights import get_y_self_magnitudes
from metrics import average_accuracy, backward_transfer, format_accuracy_matrix
from run_config import RunConfig
from train import run_incremental_training

REPO_ROOT = Path(__file__).resolve().parent.parent
# Plain filesystem tracking ("file:./mlruns") is in maintenance mode as of
# MLflow 3.15 and raises unless MLFLOW_ALLOW_FILE_STORE is set -- use the
# recommended sqlite backend instead, see lab-book.md, "Logging: MLflow".
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACT_LOCATION = f"file:{REPO_ROOT / 'mlartifacts'}"
EXPERIMENT_NAME = "ai2-mnist"
ALL_DIGITS = frozenset(range(10))


def _configure_mlflow() -> None:
    """Points MLflow at the local tracking backend, creating the experiment
    with a repo-root-anchored artifact location the one time it doesn't yet
    exist (a brand-new experiment's default artifact location is cwd-relative
    -- "./mlruns" -- independent of the tracking URI, which would silently
    reintroduce the cwd-dependence TRACKING_URI was anchored to avoid)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)


def run_and_log(
    config: RunConfig, model: nn.Module, stages: list[Stage], device: torch.device
) -> list[dict[int, float]]:
    """Trains `model` through `stages`, live-logging progress to MLflow.

    Becomes a nested child run if called while another MLflow run is already
    active (e.g. from inside `replicate_group`) -- otherwise a flat top-level
    run, same as calling this standalone.

    Returns:
        One per-digit accuracy dict per stage, see
        `train.run_incremental_training`.
    """
    _configure_mlflow()
    with mlflow.start_run(run_name=config.name, nested=mlflow.active_run() is not None):
        mlflow.set_tag("sweep_axis", config.sweep_axis)
        mlflow.log_params(_get_loggable_params(config))
        accuracy_by_stage, train_seconds = _train_with_logging(config, model, stages, device)
        _log_final_metrics(accuracy_by_stage, stages, train_seconds)
    return accuracy_by_stage


def _get_loggable_params(config: RunConfig) -> dict[str, object]:
    """MLflow-loggable fields of `config`: everything except name/sweep_axis and unset (None) fields."""
    return {
        key: value
        for key, value in dataclasses.asdict(config).items()
        if key not in ("name", "sweep_axis") and value is not None
    }


def _train_with_logging(
    config: RunConfig, model: nn.Module, stages: list[Stage], device: torch.device
) -> tuple[list[dict[int, float]], float]:
    """Runs `run_incremental_training`, live-logging per-epoch/per-stage progress.

    Owns `accuracy_by_stage`'s full lifecycle: creates it, passes it to
    `_make_on_stage_end` to mutate in place, and returns the finished list --
    keeps that mutation contained here rather than exposing it across
    `run_and_log`'s boundary.

    Returns:
        The per-stage accuracy dicts (see `train.run_incremental_training`)
        and wall-clock training time in seconds.
    """
    accuracy_by_stage: list[dict[int, float]] = []
    start_time = time.perf_counter()
    run_incremental_training(
        model,
        stages,
        device,
        epochs_per_stage=config.epochs_per_stage,
        lr=config.lr,
        batch_size=config.batch_size,
        on_epoch_end=_make_on_epoch_end(model, config.epochs_per_stage),
        on_stage_end=_make_on_stage_end(stages, accuracy_by_stage),
    )
    train_seconds = time.perf_counter() - start_time
    return accuracy_by_stage, train_seconds


def _make_on_epoch_end(
    model: nn.Module, epochs_per_stage: int
) -> Callable[[int, int, float], None]:
    """Builds an `on_epoch_end` callback logging `train_loss` and per-layer y_self magnitude.

    epochs_per_stage: needed to compute a stage-spanning step index
        (`stage_index * epochs_per_stage + epoch_index`) so MLflow's x-axis
        is continuous across stages, not reset each stage.
    """

    def on_epoch_end(stage_index: int, epoch_index: int, avg_loss: float) -> None:
        step = stage_index * epochs_per_stage + epoch_index
        mlflow.log_metric("train_loss", avg_loss, step=step)
        # Last-batch-of-the-epoch snapshot, not a proper epoch-weighted
        # average like avg_loss -- averaging would need train.py to know
        # AutapticLayer exists, which its callback-based design avoids.
        for layer_name, magnitude in get_y_self_magnitudes(model).items():
            mlflow.log_metric(f"y_self_magnitude_{layer_name}", magnitude, step=step)

    return on_epoch_end


def _make_on_stage_end(
    stages: list[Stage], accuracy_by_stage: list[dict[int, float]]
) -> Callable[[int, dict[int, float]], None]:
    """Builds an `on_stage_end` callback that records and logs one stage's results.

    The returned closure mutates `accuracy_by_stage` in place (appends each
    stage's result) -- an explicit, documented exception to STYLE.md's
    mutation-avoidance rule: `train.run_incremental_training`'s
    callback-based loop is the only place a stage's result becomes
    available, so accumulating there is unavoidable without restructuring
    that loop.
    """
    first_stage_digits = stages[0].digits_seen

    def on_stage_end(stage_index: int, per_digit_accuracy: dict[int, float]) -> None:
        accuracy_by_stage.append(per_digit_accuracy)
        for digit, accuracy in per_digit_accuracy.items():
            mlflow.log_metric(f"acc_digit_{digit}", accuracy, step=stage_index)
        digits_seen = stages[stage_index].digits_seen
        mlflow.log_metric(
            "avg_accuracy_seen",
            average_accuracy(per_digit_accuracy, digits_seen),
            step=stage_index,
        )
        mlflow.log_metric(
            "avg_accuracy_all",
            average_accuracy(per_digit_accuracy, ALL_DIGITS),
            step=stage_index,
        )
        # 0.0 at stage_index=0 by construction (first stage compared to
        # itself) -- the curve's fixed starting point, not a measurement.
        mlflow.log_metric(
            "backward_transfer_so_far",
            backward_transfer(accuracy_by_stage, first_stage_digits),
            step=stage_index,
        )

    return on_stage_end


def _log_final_metrics(
    accuracy_by_stage: list[dict[int, float]], stages: list[Stage], train_seconds: float
) -> None:
    """Logs end-of-run summary metrics: train_seconds, final accuracy/BWT, and the accuracy matrix artifact."""
    first_stage_digits = stages[0].digits_seen
    mlflow.log_metric("train_seconds", train_seconds)
    mlflow.log_metric(
        "final_avg_accuracy", average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen)
    )
    mlflow.log_metric(
        "final_backward_transfer", backward_transfer(accuracy_by_stage, first_stage_digits)
    )
    mlflow.log_text(format_accuracy_matrix(accuracy_by_stage), "accuracy_matrix.txt")


@contextmanager
def replicate_group(config: RunConfig, num_seeds: int) -> Iterator[list[tuple[float, float]]]:
    """Groups `num_seeds` seed replicates of `config` under one parent MLflow run.

    Yields a list to append each replicate's `(avg_accuracy,
    backward_transfer)` to -- an explicit exception to preferring return
    values over mutation (STYLE.md): a context-manager accumulator the caller
    appends to across its own seed loop is the natural shape here, and
    there's no return-based alternative that doesn't restructure that loop
    just to avoid one `.append()`.

    On clean exit, logs the mean/stdev of the collected tuples onto the
    parent run. If the `with` block raises (e.g. a replicate's training
    fails), the exception propagates before this function's post-`yield`
    code runs at all (no `try`/`finally` around `yield`, deliberately) -- no
    misleading partial aggregate ever gets logged, and MLflow's own
    `start_run` marks the parent run FAILED.
    """
    _check_num_seeds(num_seeds)
    _configure_mlflow()
    with mlflow.start_run(run_name=config.name, nested=mlflow.active_run() is not None):
        mlflow.set_tag("sweep_axis", config.sweep_axis)
        final_metrics: list[tuple[float, float]] = []
        yield final_metrics
        _log_replicate_aggregates(final_metrics, num_seeds)


def _check_num_seeds(num_seeds: int) -> None:
    """Fails fast if `num_seeds < 2`, before any MLflow state is touched.

    statistics.stdev needs >=2 points, and grouping a single replicate isn't
    meaningful anyway. Failing before `_configure_mlflow()` (rather than
    logging a partial mean and then crashing on stdev) means a rejected call
    leaves no trace in MLflow at all.
    """
    if num_seeds < 2:
        raise ValueError(f"replicate_group requires num_seeds >= 2, got {num_seeds}")


def _log_replicate_aggregates(final_metrics: list[tuple[float, float]], num_seeds: int) -> None:
    """Logs mean/stdev of `final_metrics`'s two columns onto the active (parent) MLflow run."""
    assert len(final_metrics) == num_seeds
    avg_accuracies = [avg_acc for avg_acc, _ in final_metrics]
    backward_transfers = [bwt for _, bwt in final_metrics]
    mlflow.log_metric("mean_final_avg_accuracy", statistics.mean(avg_accuracies))
    mlflow.log_metric("std_final_avg_accuracy", statistics.stdev(avg_accuracies))
    mlflow.log_metric("mean_final_backward_transfer", statistics.mean(backward_transfers))
    mlflow.log_metric("std_final_backward_transfer", statistics.stdev(backward_transfers))
