"""Tests for experiment_jobs.py: running one config with all its seeds, in-process."""

import dataclasses
from pathlib import Path

import mlflow
import pytest
from conftest import run_git
from fakes import make_broken_config, make_cnn_config, make_config, tiny_stage_builder

import experiment_jobs
import run_store
from experiment_jobs import Job, build_seed_variant, run_job
from run_config import MlflowSettings
from run_store import Launch

EXPECTED_METRIC_KEYS = {"final_avg_accuracy", "final_backward_transfer", "train_seconds"}


def _job(launch: Launch, config=None, seeds=(0,), mlflow_settings=None, verbose=False) -> Job:
    return Job(
        config or make_config("job"),
        tuple(seeds),
        launch,
        mlflow=mlflow_settings,
        stage_builder=tiny_stage_builder,
        verbose=verbose,
    )


@pytest.fixture
def settings(tmp_path: Path) -> MlflowSettings:
    return MlflowSettings(
        tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        artifact_location=f"file:{tmp_path / 'artifacts'}",
    )


@pytest.fixture(autouse=True)
def _no_leftover_runs():
    yield
    leaked = mlflow.active_run()
    while mlflow.active_run() is not None:
        mlflow.end_run()
    assert leaked is None, "a test left an MLflow run active"


def test_seed_variant_names_keep_the_scheme_unchanged() -> None:
    mnist = make_config("cfg")
    fashion = dataclasses.replace(mnist, dataset="fashion_mnist")

    assert build_seed_variant(mnist, 0, 1).name == "cfg"
    assert build_seed_variant(fashion, 0, 1).name == "cfg_fashion_mnist"
    assert build_seed_variant(mnist, 17, 5).name == "cfg_seed17"
    assert build_seed_variant(fashion, 17, 5).name == "cfg_fashion_mnist_seed17"
    assert build_seed_variant(mnist, 17, 5).seed == 17
    assert mnist.name == "cfg"  # the base config is not mutated


def test_single_seed_job_writes_one_attributable_file(launch: Launch) -> None:
    config = make_config("job", initial_classes=1, classes_per_stage=1)

    outcome = run_job(_job(launch, config))

    assert outcome.error is None and outcome.mlflow_errors == ()
    [run] = run_store.list_runs(launch.directory)
    assert run.path == launch.directory / "job.h5"
    meta = run.meta
    assert meta["run_id"] == f"{launch.launch_id}/job"
    assert meta["config_name"] == "job"
    assert meta["config"]["seed"] == 0
    assert meta["config"]["initial_classes"] == 1
    assert meta["git"]["commit"] == launch.git.commit
    assert meta["git"]["dirty"] is False
    assert set(meta["metrics"]) == EXPECTED_METRIC_KEYS
    assert meta["metrics"]["train_seconds"] > 0
    assert run.array("accuracy").shape == (2, 10)
    assert run.array("train_loss").shape == (2,)
    assert "y_self_absmean/hidden_layers.0" in run.dataset_paths()


def test_cnn_job_has_no_y_self_datasets(launch: Launch) -> None:
    run_job(_job(launch, make_cnn_config("cnn_job")))

    [run] = run_store.list_runs(launch.directory)
    assert run.dataset_paths() == ["accuracy", "train_loss"]


def test_replicate_job_writes_one_file_per_seed_with_the_given_seeds(launch: Launch) -> None:
    outcome = run_job(_job(launch, seeds=(11, 22, 33)))

    assert outcome.error is None
    runs = run_store.list_runs(launch.directory)
    assert [run.meta["run_name"] for run in runs] == ["job_seed11", "job_seed22", "job_seed33"]
    assert [run.meta["config"]["seed"] for run in runs] == [11, 22, 33]
    assert {run.meta["config_name"] for run in runs} == {"job"}


def test_metrics_in_the_file_match_the_arrays(launch: Launch) -> None:
    run_job(_job(launch))

    [run] = run_store.list_runs(launch.directory)
    accuracy = run.array("accuracy").read()
    assert run.meta["metrics"]["final_avg_accuracy"] == pytest.approx(accuracy[-1, :2].mean())
    assert run.meta["metrics"]["final_backward_transfer"] == pytest.approx(
        accuracy[-1, 0] - accuracy[0, 0]
    )


def test_verbose_job_prints_summary_and_accuracy_matrix(
    launch: Launch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_job(_job(launch, verbose=True))

    out = capsys.readouterr().out
    assert "job: seeds=[0]" in out
    assert "job: avg_accuracy=" in out
    assert "stage |" in out


def test_quiet_job_prints_nothing(launch: Launch, capsys: pytest.CaptureFixture[str]) -> None:
    run_job(_job(launch, verbose=False))

    assert capsys.readouterr().out == ""


def test_existing_file_on_a_first_attempt_is_an_error_and_is_left_untouched(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_job(_job(launch))
    path = launch.directory / "job.h5"
    original = path.read_bytes()
    monkeypatch.setattr(
        experiment_jobs,
        "train_and_record",
        lambda *a, **k: pytest.fail("must not train when the target exists"),
    )

    outcome = run_job(_job(launch), skip_existing=False)

    assert outcome.error is not None and "never overwritten" in outcome.error
    assert path.read_bytes() == original


def test_retry_skips_finished_seeds_and_reruns_only_the_rest(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_train = experiment_jobs.train_and_record
    calls = []

    def crash_on_second_seed(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash mid-launch")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(experiment_jobs, "train_and_record", crash_on_second_seed)
    job = _job(launch, seeds=(5, 6, 7))

    first = run_job(job)
    first_file = launch.directory / "job_seed5.h5"
    first_bytes = first_file.read_bytes()
    second = run_job(job, skip_existing=True)

    assert first.error is not None and "crash mid-launch" in first.error
    assert second.error is None
    assert len(calls) == 4  # seed 5 (ok), 6 (crash), then retry: 6 and 7 only
    names = sorted(path.name for path in launch.directory.glob("*.h5"))
    assert names == ["job_seed5.h5", "job_seed6.h5", "job_seed7.h5"]
    assert first_file.read_bytes() == first_bytes
    assert not list(launch.directory.glob("*.tmp"))


def test_retried_replicate_group_still_aggregates_all_seeds_in_mlflow(
    launch: Launch, settings: MlflowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_train = experiment_jobs.train_and_record
    calls = []

    def crash_on_second_seed(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash mid-launch")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(experiment_jobs, "train_and_record", crash_on_second_seed)
    job = _job(launch, seeds=(5, 6), mlflow_settings=settings)

    run_job(job)
    outcome = run_job(job, skip_existing=True)

    assert outcome.error is None and outcome.mlflow_errors == ()
    parents = [
        run
        for run in mlflow.search_runs(experiment_names=["ai2-mnist"], output_format="list")
        if run.info.run_name == "job" and "mean_final_avg_accuracy" in run.data.metrics
    ]
    assert len(parents) == 1


def test_job_failure_is_returned_as_text_not_raised_and_leaves_no_file(launch: Launch) -> None:
    outcome = run_job(_job(launch, make_broken_config("broken")))

    assert outcome.error is not None and "AssertionError" in outcome.error
    assert list(launch.directory.iterdir()) == []


def test_mlflow_failure_after_training_leaves_the_h5_intact_and_is_reported(
    launch: Launch, settings: MlflowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_log_text(*args, **kwargs):
        raise OSError("artifact location not writable")

    monkeypatch.setattr(mlflow, "log_text", failing_log_text)

    outcome = run_job(_job(launch, mlflow_settings=settings))

    assert outcome.error is None
    assert len(outcome.mlflow_errors) == 1 and "not writable" in outcome.mlflow_errors[0]
    [run] = run_store.list_runs(launch.directory)
    assert run.array("accuracy").shape == (2, 10)
    assert mlflow.active_run() is None


def test_mlflow_failure_during_training_still_produces_the_h5(
    launch: Launch, settings: MlflowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_log_metric(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mlflow, "log_metric", failing_log_metric)

    outcome = run_job(_job(launch, mlflow_settings=settings))

    assert outcome.error is None
    assert "database is locked" in outcome.mlflow_errors[0]
    assert len(run_store.list_runs(launch.directory)) == 1


def test_source_edit_after_launch_makes_the_job_fail_before_training(
    launch: Launch, git_repo: Path
) -> None:
    (git_repo / "src" / "a.py").write_text("x = 99\n")

    outcome = run_job(_job(launch))

    assert outcome.error is not None and "DirtyTreeError" in outcome.error
    assert list(launch.directory.iterdir()) == []


def test_committing_only_the_lab_book_after_launch_does_not_fail_the_job(
    launch: Launch, git_repo: Path
) -> None:
    run_git(git_repo, "add", "docs/lab-book.md")
    run_git(git_repo, "commit", "-q", "-m", "log launch")

    assert run_job(_job(launch)).error is None


def test_seeds_are_fixed_by_the_job_not_redrawn_on_rerun(launch: Launch) -> None:
    job = _job(launch, seeds=(101, 202))

    run_job(job)
    outcome = run_job(job, skip_existing=True)

    assert outcome.error is None
    seeds = [run.meta["config"]["seed"] for run in run_store.list_runs(launch.directory)]
    assert seeds == [101, 202]


def test_interrupted_job_closes_its_mlflow_runs_as_failed_and_propagates(
    launch: Launch, settings: MlflowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment_jobs, "train_and_record", interrupted)

    with pytest.raises(KeyboardInterrupt):
        run_job(_job(launch, seeds=(1, 2), mlflow_settings=settings))

    runs = mlflow.search_runs(experiment_names=["ai2-mnist"], output_format="list")
    assert {run.info.status for run in runs} == {"FAILED"}
    assert len(runs) == 2  # the group run and the seed run
