"""Optional MLflow viewer for live run supervision. See old-lab-book.md, "Logging: MLflow".

MLflow is a viewer; the HDF5 run files (`run_store`) are the canonical record. Nothing here may
lose or block results, so `MlflowSession` guards every logging call: on the first MLflow error
it warns on stderr, stops logging for the rest of the job, and records the error for the final
report -- training and the `.h5` write go on regardless. Tearing runs down is never skipped.
"""

import dataclasses
import statistics
import sys
from collections.abc import Callable, Mapping
from typing import Any

import mlflow
from torch import nn

from data import Stage
from dynamics_insights import get_y_self_absmeans
from metrics import average_accuracy, backward_transfer, format_accuracy_matrix
from recorded_training import TrainingHooks, TrainingRecord
from run_config import MlflowSettings, RunConfig
from run_store import Launch

__all__ = ["MlflowSession", "configure", "with_busy_timeout"]

ALL_DIGITS = frozenset(range(10))
# Seconds. MLflow 3.15 hard-codes `PRAGMA busy_timeout = 20000` per session (no setting), so
# this only widens the connection-level timeout, via the SQLAlchemy URL.
SQLITE_CONNECT_TIMEOUT = 60


def configure(settings: MlflowSettings) -> None:
    """Points MLflow at `settings` and creates the experiment if it does not exist yet.

    Call once in the launching process before any worker starts: first use of a fresh database
    runs schema migrations, which fail when several processes race to do it. Also fails fast
    on an unwritable artifact location. A *new* experiment's default artifact location is
    cwd-relative, so it is always created with the explicit repo-anchored one.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)
    if mlflow.get_experiment_by_name(settings.experiment_name) is None:
        mlflow.create_experiment(
            settings.experiment_name, artifact_location=settings.artifact_location
        )
    mlflow.set_experiment(settings.experiment_name)


def with_busy_timeout(settings: MlflowSettings) -> MlflowSettings:
    """`settings` with the SQLite connect timeout widened, for many concurrent writer processes."""
    separator = "&" if "?" in settings.tracking_uri else "?"
    uri = f"{settings.tracking_uri}{separator}timeout={SQLITE_CONNECT_TIMEOUT}"
    return dataclasses.replace(settings, tracking_uri=uri)


class MlflowSession:
    """MLflow state of one job (one config, all its seeds): optional group run plus child runs.

    Every method except teardown is a no-op once an MLflow call has failed.

    Attributes:
        errors: Text of every MLflow error swallowed so far; the launcher reports them.
    """

    def __init__(self, settings: MlflowSettings, launch: Launch) -> None:
        self.errors: list[str] = []
        self._launch = launch
        self._disabled = False
        self._open_runs = 0
        self._guarded("configure", configure, settings)

    def begin_group(self, config: RunConfig) -> None:
        """Opens the parent run under which the config's seed replicates nest."""
        self._guarded("begin_group", self._start_run, config, False)

    def end_group(self, final_metrics: list[tuple[float, float]]) -> None:
        """Logs mean/stdev of the replicates' final `(avg_accuracy, bwt)` and ends the group."""
        self._guarded("end_group", _log_replicate_aggregates, final_metrics)
        self._end_run("FINISHED")

    def begin_run(
        self, config: RunConfig, model: nn.Module, stages: list[Stage]
    ) -> TrainingHooks | None:
        """Opens the run for one seed and returns the live-logging hooks (None once disabled)."""
        self._guarded("begin_run", self._start_run, config, True)
        if self._disabled:
            return None
        return TrainingHooks(
            on_epoch_end=self._guard_callback(
                "on_epoch_end", _make_on_epoch_end(model, config.epochs_per_stage)
            ),
            on_stage_end=self._guard_callback("on_stage_end", _make_on_stage_end(stages)),
        )

    def log_final(self, record: TrainingRecord, metrics: Mapping[str, float | None]) -> None:
        """Logs end-of-run summary metrics and the accuracy matrix, then ends the seed's run."""
        self._guarded("log_final", _log_final_metrics, record, metrics)
        self._end_run("FINISHED")

    def cleanup(self, failed: bool) -> None:
        """Ends every run this session still has open, innermost first. Never raises."""
        while self._open_runs:
            self._end_run("FAILED" if failed else "FINISHED")

    def _start_run(self, config: RunConfig, is_seed_run: bool) -> None:
        """Starts a run (nested if one is open), tags it, and for a seed run logs its params."""
        mlflow.start_run(run_name=config.name, nested=self._open_runs > 0)
        self._open_runs += 1
        tags = {
            "sweep_axis": config.sweep_axis,
            "launch_id": self._launch.launch_id,
            "mlflow.source.git.commit": self._launch.git.commit,
            "mlflow.source.git.branch": self._launch.git.branch,
        }
        if is_seed_run:
            tags["run_id"] = self._launch.run_id(config.name)
        mlflow.set_tags(tags)
        if is_seed_run:
            mlflow.log_params(_get_loggable_params(config))

    def _end_run(self, status: str) -> None:
        """Ends the innermost open run; teardown runs even after logging was disabled."""
        if not self._open_runs:
            return
        self._open_runs -= 1
        try:
            mlflow.end_run(status)
        except Exception as exc:  # noqa: BLE001 -- by design: recorded, reported, never fatal
            self._record_error("end_run", exc)

    def _guarded(self, where: str, function: Callable[..., Any], *args: Any) -> None:
        """Calls `function(*args)` unless disabled; the first exception disables the session."""
        if self._disabled:
            return
        try:
            function(*args)
        except Exception as exc:  # noqa: BLE001 -- by design: recorded, reported, never fatal
            self._record_error(where, exc)

    def _guard_callback(self, where: str, callback: Callable[..., None]) -> Callable[..., None]:
        def guarded(*args: Any) -> None:
            self._guarded(where, callback, *args)

        return guarded

    def _record_error(self, where: str, exc: Exception) -> None:
        self._disabled = True
        message = f"{where}: {type(exc).__name__}: {exc}"
        self.errors.append(message)
        print(
            f"WARNING: MLflow failed ({message}); continuing without MLflow for the rest of "
            "this job. Results are still written to the .h5 files.",
            file=sys.stderr,
        )


def _get_loggable_params(config: RunConfig) -> dict[str, object]:
    """MLflow-loggable fields of `config`: all but name/sweep_axis and unset (None) fields."""
    return {
        key: value
        for key, value in dataclasses.asdict(config).items()
        if key not in ("name", "sweep_axis") and value is not None
    }


def _make_on_epoch_end(
    model: nn.Module, epochs_per_stage: int
) -> Callable[[int, int, float], None]:
    """Callback logging `train_loss` and per-layer y_self absmean at a stage-spanning step.

    `step = stage_index * epochs_per_stage + epoch_index` keeps MLflow's x-axis continuous
    across stages; valid because `epochs_per_stage` is constant across stages.
    """

    def on_epoch_end(stage_index: int, epoch_index: int, avg_loss: float) -> None:
        step = stage_index * epochs_per_stage + epoch_index
        mlflow.log_metric("train_loss", avg_loss, step=step)
        for layer_name, absmean in get_y_self_absmeans(model).items():
            mlflow.log_metric(f"y_self_absmean_{layer_name}", absmean, step=step)

    return on_epoch_end


def _make_on_stage_end(stages: list[Stage]) -> Callable[[int, dict[int, float]], None]:
    """Callback logging per-digit accuracy, average accuracy and running backward transfer.

    Keeps its own list of the stage results seen so far (the running backward transfer needs
    stage 0's accuracy); the recorder owns the canonical copy.
    """
    first_stage_digits = stages[0].digits_seen
    seen_so_far: list[dict[int, float]] = []

    def on_stage_end(stage_index: int, per_digit_accuracy: dict[int, float]) -> None:
        seen_so_far.append(per_digit_accuracy)
        for digit, accuracy in per_digit_accuracy.items():
            mlflow.log_metric(f"acc_digit_{digit}", accuracy, step=stage_index)
        digits_seen = stages[stage_index].digits_seen
        mlflow.log_metric(
            "avg_accuracy_seen", average_accuracy(per_digit_accuracy, digits_seen), step=stage_index
        )
        mlflow.log_metric(
            "avg_accuracy_all", average_accuracy(per_digit_accuracy, ALL_DIGITS), step=stage_index
        )
        # 0.0 at stage_index=0 by construction (first stage compared to itself).
        mlflow.log_metric(
            "backward_transfer_so_far",
            backward_transfer(seen_so_far, first_stage_digits),
            step=stage_index,
        )

    return on_stage_end


def _log_final_metrics(record: TrainingRecord, metrics: Mapping[str, float | None]) -> None:
    """Logs the run's final scalar metrics (those that are set) and the accuracy matrix."""
    for name, value in metrics.items():
        if value is not None:
            mlflow.log_metric(name, value)
    mlflow.log_text(format_accuracy_matrix(record.accuracy_by_stage), "accuracy_matrix.txt")


def _log_replicate_aggregates(final_metrics: list[tuple[float, float]]) -> None:
    """Logs mean/stdev of `final_metrics`'s two columns onto the active (parent) run."""
    avg_accuracies = [avg_acc for avg_acc, _ in final_metrics]
    backward_transfers = [bwt for _, bwt in final_metrics]
    mlflow.log_metric("mean_final_avg_accuracy", statistics.mean(avg_accuracies))
    mlflow.log_metric("std_final_avg_accuracy", statistics.stdev(avg_accuracies))
    mlflow.log_metric("mean_final_backward_transfer", statistics.mean(backward_transfers))
    mlflow.log_metric("std_final_backward_transfer", statistics.stdev(backward_transfers))
