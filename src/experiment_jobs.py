"""Worker-side execution of experiment jobs: one config with all its seeds.

Everything a job needs travels inside the picklable `Job` (a spawned worker shares no module
state with the launcher). A job writes one `.h5` per seed through `run_store`; the optional
MLflow viewer (`logged_training`) is imported only when the job asks for it, so a `--no-mlflow`
launch never imports mlflow.
"""

import dataclasses
import functools
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

import sweep
from data import DatasetName, Stage, build_stage_digits, build_stages, get_mnist_data
from metrics import average_accuracy, backward_transfer, format_accuracy_matrix
from recorded_training import TrainingHooks, TrainingRecord, build_arrays, train_and_record
from run_config import MlflowSettings, RunConfig
from run_store import Launch, RunExistsError, open_run, write_run

__all__ = [
    "Job",
    "JobOutcome",
    "LaunchError",
    "StageBuilder",
    "build_seed_variant",
    "build_stages_for",
    "init_worker",
    "run_job",
    "run_seed",
]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

StageBuilder = Callable[[DatasetName, int, int], list[Stage]]


@functools.lru_cache(maxsize=2)
def build_stages_for(
    dataset: DatasetName, initial_classes: int, classes_per_stage: int
) -> list[Stage]:
    """Loads `dataset` and splits it into stages; cached per process so a worker loads it once.

    Workers call this themselves instead of receiving datasets, which would have to be pickled.
    The cache holds only this default builder, never an injected one.
    """
    train_data, test_data = get_mnist_data(root=str(DATA_ROOT), dataset=dataset)
    stage_digits = build_stage_digits(
        initial_classes=initial_classes, classes_per_stage=classes_per_stage
    )
    return build_stages(train_data, test_data, stage_digits=stage_digits)


@dataclass(frozen=True)
class Job:
    """One config run for all its seeds; the unit of parallel work.

    Attributes:
        config: Base config with dataset/class split set; `name` is unsuffixed. The per-seed
            copies get `seed` and the name suffixes from `build_seed_variant`.
        seeds: Seeds drawn once by the launcher, so a retry reruns the same seeds. More than
            one seed means replicates: suffixed names and an MLflow parent run.
        launch: The launch the run files belong to.
        mlflow: MLflow settings, or `None` for no MLflow.
        stage_builder: Builds the stages inside the worker.
        verbose: Print per-seed summaries and accuracy matrices (sequential launches only).
    """

    config: RunConfig
    seeds: tuple[int, ...]
    launch: Launch
    mlflow: MlflowSettings | None = None
    stage_builder: StageBuilder = build_stages_for
    verbose: bool = True


@dataclass(frozen=True)
class JobOutcome:
    """What running a job returned; always picklable (errors are text).

    Attributes:
        name: The job's base config name.
        error: Traceback text if the job failed, else `None`.
        mlflow_errors: MLflow errors that were swallowed; results are unaffected.
    """

    name: str
    error: str | None
    mlflow_errors: tuple[str, ...] = ()


class LaunchError(RuntimeError):
    """A launch finished with failed jobs and/or swallowed MLflow errors.

    Attributes:
        failed: Job name -> traceback text for jobs that still failed after the retry.
        mlflow_errors: Job name -> swallowed MLflow errors (the results were written).
    """

    def __init__(self, failed: dict[str, str], mlflow_errors: dict[str, list[str]]) -> None:
        self.failed = failed
        self.mlflow_errors = mlflow_errors
        lines = []
        if failed:
            lines.append(f"{len(failed)} job(s) failed after one retry (completed runs kept):")
            lines += [f"--- {name}\n{error}" for name, error in failed.items()]
        if mlflow_errors:
            lines.append("MLflow logging failed (all results were still written to .h5):")
            lines += [f"  {name}: {'; '.join(errors)}" for name, errors in mlflow_errors.items()]
        super().__init__("\n".join(lines))


class TrackingSession(Protocol):
    """What a job needs from a tracking viewer; `NullSession` and `MlflowSession` provide it."""

    errors: list[str]

    def begin_group(self, config: RunConfig) -> None: ...
    def end_group(self, final_metrics: list[tuple[float, float]]) -> None: ...
    def begin_run(
        self, config: RunConfig, model: nn.Module, stages: list[Stage]
    ) -> TrainingHooks | None: ...
    def log_final(self, record: TrainingRecord, metrics: Mapping[str, float | None]) -> None: ...
    def cleanup(self, failed: bool) -> None: ...


class NullSession:
    """The tracking session used without MLflow: does nothing."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def begin_group(self, config: RunConfig) -> None:
        pass

    def end_group(self, final_metrics: list[tuple[float, float]]) -> None:
        pass

    def begin_run(
        self, config: RunConfig, model: nn.Module, stages: list[Stage]
    ) -> TrainingHooks | None:
        return None

    def log_final(self, record: TrainingRecord, metrics: Mapping[str, float | None]) -> None:
        pass

    def cleanup(self, failed: bool) -> None:
        pass


def init_worker() -> None:
    """Worker-process initializer: one torch thread, so N workers do not oversubscribe cores.

    Single-thread reductions sum in a different order than multi-thread ones, so a parallel
    launch can differ from a sequential one in the last digits (up to ~1 pp accuracy on real
    MNIST runs). Every run file therefore records `torch_num_threads`.
    """
    torch.set_num_threads(1)


def run_job(job: Job, skip_existing: bool = False) -> JobOutcome:
    """Runs every seed of `job`, never raising: failures come back as text in the outcome.

    Args:
        skip_existing: On a retry, skip seeds whose `.h5` exists (atomic writes make an
            existing file a complete run). On a first attempt an existing file is an error.
    """
    session = _make_session(job)
    error = None
    interrupted = False
    try:
        _run_job_seeds(job, skip_existing, session)
    except Exception:  # noqa: BLE001 -- any job failure becomes a reported, retried outcome
        error = traceback.format_exc()
    except BaseException:
        interrupted = True  # e.g. Ctrl-C: still close MLflow runs as FAILED, then propagate
        raise
    finally:
        session.cleanup(failed=error is not None or interrupted)
    return JobOutcome(job.config.name, error, tuple(session.errors))


def _make_session(job: Job) -> TrackingSession:
    """The MLflow session for `job`, importing mlflow only here, only if wanted."""
    if job.mlflow is None:
        return NullSession()
    from logged_training import MlflowSession  # deferred: --no-mlflow must not load mlflow

    return MlflowSession(job.mlflow, job.launch)


def _run_job_seeds(job: Job, skip_existing: bool, session: TrackingSession) -> None:
    job.launch.verify_unchanged()
    config = job.config
    stages = job.stage_builder(config.dataset, config.initial_classes, config.classes_per_stage)
    replicates = len(job.seeds) > 1
    if job.verbose:
        print(f"{config.name}: seeds={list(job.seeds)}")
    if replicates:
        session.begin_group(config)
    final_metrics = [run_seed(job, seed, stages, session, skip_existing) for seed in job.seeds]
    if replicates:
        session.end_group(final_metrics)


def run_seed(
    job: Job, seed: int, stages: list[Stage], session: TrackingSession, skip_existing: bool
) -> tuple[float, float]:
    """Trains one seed replicate, writes its `.h5`, then does the optional MLflow logging.

    The file is written before any end-of-run MLflow logging, so MLflow can never lose it.

    Returns:
        `(avg_accuracy, backward_transfer)` of the final stage, from the written file when the
        seed was skipped.
    """
    seed_config = build_seed_variant(job.config, seed, len(job.seeds))
    path = job.launch.run_path(seed_config.name)
    if path.exists():
        return _read_existing_metrics(path, skip_existing)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    torch.manual_seed(seed)
    model = sweep.build_model(seed_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hooks = session.begin_run(seed_config, model, stages)
    record = train_and_record(seed_config, model, stages, device, hooks)
    avg_acc = average_accuracy(record.accuracy_by_stage[-1], stages[-1].digits_seen)
    bwt = backward_transfer(record.accuracy_by_stage, stages[0].digits_seen)
    metrics: Mapping[str, float | None] = {
        "final_avg_accuracy": avg_acc,
        "final_backward_transfer": bwt,
        "train_seconds": record.train_seconds,
    }
    write_run(
        job.launch,
        seed_config,
        build_arrays(record),
        config_name=job.config.name,
        metrics=metrics,
        started_at=started_at,
        device=str(device),
        extra_meta={"torch_num_threads": torch.get_num_threads()},
    )
    session.log_final(record, metrics)
    if job.verbose:
        print(f"{seed_config.name}: avg_accuracy={avg_acc:.4f} backward_transfer={bwt:+.4f}")
        print(format_accuracy_matrix(record.accuracy_by_stage))
    return avg_acc, bwt


def _read_existing_metrics(path: Path, skip_existing: bool) -> tuple[float, float]:
    """Metrics of an already-written run, so a retried replicate group still aggregates fully."""
    if not skip_existing:
        raise RunExistsError(f"{path} already exists; run files are never overwritten.")
    metrics = open_run(path).meta["metrics"]
    return metrics["final_avg_accuracy"], metrics["final_backward_transfer"]


def build_seed_variant(config: RunConfig, seed: int, num_seeds: int) -> RunConfig:
    """Per-seed copy of `config` with `seed` set and `name` suffixed.

    The name gets the dataset (whenever it is not the default "mnist") and the seed (only when
    replicating): default-dataset single-seed names stay what they were before either feature
    existed, while names stay distinct across datasets and replicates.
    """
    name = config.name
    if config.dataset != "mnist":
        name = f"{name}_{config.dataset}"
    if num_seeds > 1:
        name = f"{name}_seed{seed}"
    return dataclasses.replace(config, seed=seed, name=name)
