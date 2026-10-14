"""Tests for logged_training.py (the optional MLflow viewer) against a real local backend.

Uses a real SQLite backend in `tmp_path`, not a mock: it is fast and deterministic (no
network), so this is not the "slow or non-deterministic" case that mocking is for. Failures
are injected by monkeypatching `mlflow` calls, which is the external boundary.
"""

import dataclasses
import statistics
from pathlib import Path
from typing import cast

import mlflow
import pytest
import torch
from fakes import make_cnn_config, make_config, make_stages
from mlflow.entities import Run

import logged_training
import sweep
from recorded_training import TrainingRecord, train_and_record
from run_config import MlflowSettings
from run_store import GitInfo, Launch

COMMIT = "c" * 40


@pytest.fixture
def settings(tmp_path: Path) -> MlflowSettings:
    return MlflowSettings(
        tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        artifact_location=f"file:{tmp_path / 'artifacts'}",
    )


@pytest.fixture
def launch(tmp_path: Path) -> Launch:
    return Launch("2026-10-14-12-30_ccccccc", tmp_path, tmp_path, None, GitInfo(COMMIT, "feature"))


@pytest.fixture(autouse=True)
def _no_leftover_runs():
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _runs(settings: MlflowSettings) -> list[Run]:
    runs = mlflow.search_runs(experiment_names=[settings.experiment_name], output_format="list")
    return cast(list[Run], runs)


def _run_named(settings: MlflowSettings, name: str) -> Run:
    matches = [run for run in _runs(settings) if run.info.run_name == name]
    assert len(matches) == 1, f"expected exactly one run named {name!r}, found {len(matches)}"
    return matches[0]


def _run_one_seed(session: logged_training.MlflowSession, config, num_stages: int = 2):
    """What a job does for one seed: begin, train with hooks, log final. Returns the metrics."""
    model = sweep.build_model(config)
    stages = make_stages(num_stages)
    hooks = session.begin_run(config, model, stages)
    record = train_and_record(config, model, stages, torch.device("cpu"), hooks)
    metrics = {"final_avg_accuracy": 0.5, "final_backward_transfer": -0.25, "train_seconds": 1.0}
    session.log_final(record, metrics)
    return record


def test_configure_creates_experiment_with_the_explicit_artifact_location(
    settings: MlflowSettings,
) -> None:
    logged_training.configure(settings)
    logged_training.configure(settings)

    experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    assert experiment is not None
    assert experiment.artifact_location == settings.artifact_location


def test_with_busy_timeout_widens_only_the_connection_timeout(settings: MlflowSettings) -> None:
    widened = logged_training.with_busy_timeout(settings)
    with_query = logged_training.with_busy_timeout(widened)

    assert widened.tracking_uri == f"{settings.tracking_uri}?timeout=60"
    assert with_query.tracking_uri == f"{settings.tracking_uri}?timeout=60&timeout=60"
    assert widened.artifact_location == settings.artifact_location


def test_seed_run_logs_params_tags_and_final_metrics(
    settings: MlflowSettings, launch: Launch
) -> None:
    session = logged_training.MlflowSession(settings, launch)
    config = make_config("seed_run", initial_classes=1, classes_per_stage=2)

    _run_one_seed(session, config)

    run = _run_named(settings, "seed_run")
    assert run.data.params["initial_classes"] == "1"
    assert run.data.params["classes_per_stage"] == "2"
    assert run.data.params["ai2_variant"] == "ai2_diff"
    assert run.data.tags["launch_id"] == launch.launch_id
    assert run.data.tags["run_id"] == f"{launch.launch_id}/seed_run"
    assert run.data.tags["mlflow.source.git.commit"] == COMMIT
    assert run.data.tags["mlflow.source.git.branch"] == "feature"
    assert run.data.tags["sweep_axis"] == "base"
    assert run.data.metrics["final_avg_accuracy"] == 0.5
    assert run.data.metrics["train_seconds"] == 1.0
    assert "train_loss" in run.data.metrics
    assert "avg_accuracy_seen" in run.data.metrics
    assert run.info.status == "FINISHED"
    assert session.errors == []
    assert mlflow.active_run() is None


def test_y_self_absmean_present_for_fcnn_absent_for_cnn(
    settings: MlflowSettings, launch: Launch
) -> None:
    session = logged_training.MlflowSession(settings, launch)

    _run_one_seed(session, make_config("fcnn_run"))
    _run_one_seed(session, make_cnn_config("cnn_run"))

    fcnn = _run_named(settings, "fcnn_run")
    cnn = _run_named(settings, "cnn_run")
    assert any(key.startswith("y_self_absmean_") for key in fcnn.data.metrics)
    assert not any(key.startswith("y_self_absmean_") for key in cnn.data.metrics)


def test_accuracy_matrix_is_logged_as_an_artifact(settings: MlflowSettings, launch: Launch) -> None:
    session = logged_training.MlflowSession(settings, launch)

    _run_one_seed(session, make_config("with_matrix"))

    run = _run_named(settings, "with_matrix")
    artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
    assert [artifact.path for artifact in artifacts] == ["accuracy_matrix.txt"]


def test_group_run_nests_children_and_logs_replicate_aggregates(
    settings: MlflowSettings, launch: Launch
) -> None:
    session = logged_training.MlflowSession(settings, launch)
    base = make_config("grouped")
    session.begin_group(base)
    for seed in (1, 2):
        _run_one_seed(session, dataclasses.replace(base, name=f"grouped_seed{seed}", seed=seed))
    session.end_group([(0.5, -0.1), (0.7, -0.3)])

    parent = _run_named(settings, "grouped")
    children = [
        run
        for run in _runs(settings)
        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    ]
    assert len(children) == 2
    assert parent.data.metrics["mean_final_avg_accuracy"] == pytest.approx(0.6)
    assert parent.data.metrics["std_final_avg_accuracy"] == pytest.approx(
        statistics.stdev([0.5, 0.7])
    )
    assert "run_id" not in parent.data.tags
    assert parent.data.tags["launch_id"] == launch.launch_id
    assert mlflow.active_run() is None


def test_mlflow_error_during_training_does_not_stop_training_or_lose_the_record(
    settings: MlflowSettings,
    launch: Launch,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = logged_training.MlflowSession(settings, launch)
    calls = []

    def failing_log_metric(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mlflow, "log_metric", failing_log_metric)

    record = _run_one_seed(session, make_config("flaky"), num_stages=3)

    assert len(record.accuracy_by_stage) == 3
    assert len(calls) == 1  # logging stopped after the first failure
    assert len(session.errors) == 1
    assert "database is locked" in session.errors[0]
    assert "WARNING: MLflow failed" in capsys.readouterr().err
    assert mlflow.active_run() is None


def test_start_run_failure_disables_the_session_and_training_hooks(
    settings: MlflowSettings, launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = logged_training.MlflowSession(settings, launch)

    def failing_start_run(*args, **kwargs):
        raise RuntimeError("cannot start")

    monkeypatch.setattr(mlflow, "start_run", failing_start_run)
    config = make_config("no_start")

    hooks = session.begin_run(config, sweep.build_model(config), make_stages())

    assert hooks is None
    assert session.errors and "cannot start" in session.errors[0]


def test_log_final_failure_is_recorded_and_the_run_is_still_ended(
    settings: MlflowSettings, launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = logged_training.MlflowSession(settings, launch)

    def failing_log_text(*args, **kwargs):
        raise OSError("artifact location not writable")

    monkeypatch.setattr(mlflow, "log_text", failing_log_text)

    _run_one_seed(session, make_config("bad_artifacts"))

    assert session.errors and "artifact location" in session.errors[0]
    assert mlflow.active_run() is None
    assert _run_named(settings, "bad_artifacts").info.status == "FINISHED"


def test_cleanup_after_a_crash_ends_nested_runs_innermost_first_and_marks_them_failed(
    settings: MlflowSettings, launch: Launch
) -> None:
    session = logged_training.MlflowSession(settings, launch)
    base = make_config("crashed")
    session.begin_group(base)
    config = dataclasses.replace(base, name="crashed_seed1", seed=1)
    session.begin_run(config, sweep.build_model(config), make_stages())

    session.cleanup(failed=True)

    assert mlflow.active_run() is None
    assert _run_named(settings, "crashed").info.status == "FAILED"
    assert _run_named(settings, "crashed_seed1").info.status == "FAILED"


def test_a_crashed_job_does_not_leak_an_active_run_into_the_next_job(
    settings: MlflowSettings, launch: Launch
) -> None:
    crashed = logged_training.MlflowSession(settings, launch)
    config = make_config("first_job")
    crashed.begin_run(config, sweep.build_model(config), make_stages())
    crashed.cleanup(failed=True)

    second = logged_training.MlflowSession(settings, launch)
    _run_one_seed(second, make_config("second_job"))

    second_run = _run_named(settings, "second_job")
    assert "mlflow.parentRunId" not in second_run.data.tags
    assert mlflow.active_run() is None


def test_teardown_errors_are_recorded_not_raised(
    settings: MlflowSettings, launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = logged_training.MlflowSession(settings, launch)
    config = make_config("teardown")
    session.begin_run(config, sweep.build_model(config), make_stages())

    def failing_end_run(*args, **kwargs):
        raise RuntimeError("cannot end")

    monkeypatch.setattr(mlflow, "end_run", failing_end_run)

    session.cleanup(failed=True)

    assert session.errors and "cannot end" in session.errors[0]


def test_log_final_skips_unset_metrics(settings: MlflowSettings, launch: Launch) -> None:
    session = logged_training.MlflowSession(settings, launch)
    config = make_config("record_type")
    session.begin_run(config, sweep.build_model(config), make_stages())

    session.log_final(TrainingRecord(accuracy_by_stage=[{0: 1.0}]), {"train_seconds": None})

    assert mlflow.active_run() is None
    assert session.errors == []
