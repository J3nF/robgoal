"""Tests for run_experiments.py: launch flow, planning, --workers, retry and reporting."""

import dataclasses
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pytest
from fakes import (
    DIE_FOR_INITIAL_CLASSES,
    FAIL_MARKER_ENV,
    MLFLOW_PROBE_ENV,
    SKIP_PID_ENV,
    die_for_marked_job_stage_builder,
    die_once_stage_builder,
    fail_once_stage_builder,
    make_broken_config,
    make_cnn_config,
    make_config,
    tiny_stage_builder,
)

import experiment_jobs
import run_experiments
import run_store
from experiment_jobs import Job, LaunchError
from run_config import MlflowSettings
from run_experiments import LaunchEnv, LaunchRequest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"


def _request(**overrides) -> LaunchRequest:
    return dataclasses.replace(
        LaunchRequest(
            goal="test goal", method="test method", code_changes="none", command="pytest"
        ),
        **overrides,
    )


@pytest.fixture
def env(git_repo: Path, tmp_path: Path) -> LaunchEnv:
    return LaunchEnv(
        results_root=tmp_path / "results",
        lab_book=git_repo / "docs" / "lab-book.md",
        repo_root=git_repo,
        mlflow_settings=MlflowSettings(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow-test.db'}",
            artifact_location=f"file:{tmp_path / 'artifacts'}",
        ),
        stage_builder=tiny_stage_builder,
    )


@pytest.fixture(autouse=True)
def _no_leftover_runs():
    yield
    leaked = mlflow.active_run()
    while mlflow.active_run() is not None:
        mlflow.end_run()
    assert leaked is None, "a test left an MLflow run active"


def _configs() -> list:
    return [make_config("fcnn_a"), make_cnn_config("cnn_b")]


def _h5_names(env: LaunchEnv) -> list[str]:
    return sorted(path.name for path in env.results_root.glob("*/*.h5"))


def _in_new_root(env: LaunchEnv, tmp_path: Path, name: str) -> LaunchEnv:
    return dataclasses.replace(env, results_root=tmp_path / name)


# --- planning ----------------------------------------------------------------------------


def test_draw_seeds_is_deterministic_at_one_and_random_above() -> None:
    assert run_experiments._draw_seeds(1) == [0]

    seeds_a = run_experiments._draw_seeds(5)
    seeds_b = run_experiments._draw_seeds(5)
    assert len(seeds_a) == len(set(seeds_a)) == 5
    assert seeds_a != seeds_b  # random draws essentially never collide


@pytest.mark.parametrize("num_seeds", [0, -1])
def test_draw_seeds_rejects_non_positive_num_seeds(num_seeds: int) -> None:
    with pytest.raises(ValueError, match="num_seeds"):
        run_experiments._draw_seeds(num_seeds)


def test_plan_sets_dataset_and_split_on_every_config_and_draws_seeds_once() -> None:
    request = _request(num_seeds=3, dataset="fashion_mnist", initial_classes=1)

    plan = run_experiments.plan_jobs(_configs(), request)

    assert [config.name for config, _ in plan] == ["fcnn_a", "cnn_b"]
    for config, seeds in plan:
        assert config.dataset == "fashion_mnist"
        assert (config.initial_classes, config.classes_per_stage) == (1, 1)
        assert len(set(seeds)) == 3


def test_plan_rejects_two_runs_with_the_same_name() -> None:
    with pytest.raises(ValueError, match="both be named 'fcnn_a'"):
        run_experiments.plan_jobs([make_config("fcnn_a"), make_config("fcnn_a")], _request())


def test_sweep_and_size_scan_entry_points_hand_over_their_config_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    monkeypatch.setattr(
        run_experiments,
        "run_launch",
        lambda configs, request, scan, env: captured.append((len(configs), request, scan)),
    )

    run_experiments.main(goal="g", method="m", code_changes="c", num_seeds=2, workers=3)
    run_experiments.run_size_scan(goal="g", method="m", code_changes="c", use_mlflow=False)

    (sweep_count, sweep_request, sweep_scan), (size_count, size_request, size_scan) = captured
    assert (sweep_count, sweep_scan) == (14, "sweep")
    assert (size_count, size_scan) == (56, "size")
    assert sweep_request.workers == 3 and sweep_request.num_seeds == 2
    assert "run_experiments.main(" in sweep_request.command
    assert "num_seeds=2" in sweep_request.command
    assert size_request.use_mlflow is False


# --- refusals happen before anything is created -------------------------------------------


def _assert_nothing_created(env: LaunchEnv, lab_book_before: str) -> None:
    assert not env.results_root.exists()
    assert env.lab_book.read_text() == lab_book_before


@pytest.mark.parametrize("field", ["goal", "method", "code_changes"])
def test_blank_lab_book_text_refuses_before_any_side_effect(env: LaunchEnv, field: str) -> None:
    before = env.lab_book.read_text()

    with pytest.raises(ValueError, match=field.replace("_", "-")):
        run_experiments.run_launch(_configs(), _request(**{field: " "}), "sweep", env)

    _assert_nothing_created(env, before)


@pytest.mark.parametrize("workers", [0, -2])
def test_workers_below_one_is_rejected(env: LaunchEnv, workers: int) -> None:
    before = env.lab_book.read_text()

    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_experiments.run_launch(_configs(), _request(workers=workers), "sweep", env)

    _assert_nothing_created(env, before)
    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_experiments.execute_jobs([], workers)


def test_dirty_tree_refuses_and_does_not_even_configure_mlflow(
    env: LaunchEnv, git_repo: Path
) -> None:
    (git_repo / "src" / "a.py").write_text("x = 2\n")
    before = env.lab_book.read_text()

    with pytest.raises(run_store.DirtyTreeError, match=r"src/a\.py"):
        run_experiments.run_launch(_configs(), _request(use_mlflow=True), "sweep", env)

    _assert_nothing_created(env, before)
    assert not Path(env.mlflow_settings.tracking_uri.removeprefix("sqlite:///")).exists()


def test_unbuildable_stages_refuse_before_the_launch_exists(env: LaunchEnv) -> None:
    def bad_split(dataset, initial_classes, classes_per_stage):
        raise ValueError("initial_classes must be in [1, 10]")

    before = env.lab_book.read_text()

    with pytest.raises(ValueError, match="initial_classes"):
        run_experiments.run_launch(
            _configs(), _request(), "sweep", dataclasses.replace(env, stage_builder=bad_split)
        )

    _assert_nothing_created(env, before)


# --- a launch end to end ----------------------------------------------------------------------


def test_sequential_launch_writes_files_and_a_lab_book_entry(
    env: LaunchEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    request = _request(initial_classes=1, command="uv run python src/run_experiments.py --x")

    run_experiments.run_launch(_configs(), request, "sweep", env)

    assert _h5_names(env) == ["cnn_b.h5", "fcnn_a.h5"]
    [launch_dir] = list(env.results_root.iterdir())
    entry = env.lab_book.read_text()
    assert f"## {launch_dir.name}" in entry
    assert "uv run python src/run_experiments.py --x" in entry
    assert "Planned runs:** 2" in entry
    assert "Workers:** 1" in entry
    assert "Initial classes / classes per stage:** 1 / 1" in entry
    assert "MLflow:** on" in entry
    out = capsys.readouterr().out
    assert "stage |" in out  # sequential launches still print the accuracy matrices


def test_launch_with_mlflow_tags_runs_with_launch_provenance(env: LaunchEnv) -> None:
    run_experiments.run_launch(_configs(), _request(), "sweep", env)

    mlflow.set_tracking_uri(env.mlflow_settings.tracking_uri)
    runs = mlflow.search_runs(experiment_names=["ai2-mnist"], output_format="list")
    [launch_dir] = list(env.results_root.iterdir())
    commit = run_store.get_git_info(env.repo_root).commit
    assert {run.info.run_name for run in runs} == {"fcnn_a", "cnn_b"}
    for run in runs:
        assert run.data.tags["launch_id"] == launch_dir.name
        assert run.data.tags["mlflow.source.git.commit"] == commit


def test_workers_two_produces_the_same_files_as_workers_one(
    env: LaunchEnv, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sequential = _in_new_root(env, tmp_path, "results_seq")
    parallel = _in_new_root(env, tmp_path, "results_par")

    run_experiments.run_launch(_configs(), _request(use_mlflow=False), "sweep", sequential)
    capsys.readouterr()
    run_experiments.run_launch(_configs(), _request(use_mlflow=False, workers=2), "sweep", parallel)
    out = capsys.readouterr().out

    assert _h5_names(sequential) == _h5_names(parallel) == ["cnn_b.h5", "fcnn_a.h5"]
    assert "[1/2]" in out and "[2/2]" in out
    assert "stage |" not in out  # parallel jobs do not print accuracy matrices
    for run in run_store.list_runs(parallel.results_root):
        assert set(run.meta["metrics"]) == {
            "final_avg_accuracy",
            "final_backward_transfer",
            "train_seconds",
        }


def test_workers_two_with_mlflow_on_a_fresh_database(env: LaunchEnv) -> None:
    request = _request(workers=2, num_seeds=2)

    run_experiments.run_launch(_configs(), request, "sweep", env)

    assert len(_h5_names(env)) == 4
    mlflow.set_tracking_uri(env.mlflow_settings.tracking_uri)
    runs = mlflow.search_runs(experiment_names=["ai2-mnist"], output_format="list")
    assert len(runs) == 2 + 4  # two parent runs plus four seed children


def test_replicate_seeds_are_drawn_once_and_reused_by_the_retry(
    env: LaunchEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_experiments, "_draw_seeds", lambda n: [101, 202])
    monkeypatch.setenv(FAIL_MARKER_ENV, str(tmp_path / "failed.marker"))
    monkeypatch.setenv(SKIP_PID_ENV, str(os.getpid()))
    failing = dataclasses.replace(env, stage_builder=fail_once_stage_builder)

    run_experiments.run_launch(
        [make_config("fcnn_a")],
        _request(num_seeds=2, use_mlflow=False, workers=2),
        "sweep",
        failing,
    )

    assert (tmp_path / "failed.marker").exists()
    assert _h5_names(env) == ["fcnn_a_seed101.h5", "fcnn_a_seed202.h5"]


def test_job_failing_once_is_retried_sequentially_too(
    env: LaunchEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FAIL_MARKER_ENV, str(tmp_path / "failed.marker"))
    monkeypatch.setenv(SKIP_PID_ENV, "-1")
    plan = run_experiments.plan_jobs([make_config("fcnn_a")], _request())
    launch = run_store.begin_launch(
        run_experiments._make_spec(_request(), "sweep", 1),
        results_root=env.results_root,
        lab_book=env.lab_book,
        repo_root=env.repo_root,
    )
    jobs = [
        Job(config, seeds, launch, stage_builder=fail_once_stage_builder, verbose=False)
        for config, seeds in plan
    ]

    [outcome] = run_experiments.execute_jobs(jobs, workers=1)

    assert outcome.error is None
    assert (tmp_path / "failed.marker").exists()
    assert _h5_names(env) == ["fcnn_a.h5"]


@pytest.mark.parametrize("workers", [1, 2])
def test_always_failing_job_is_reported_and_other_runs_are_kept(
    env: LaunchEnv, workers: int
) -> None:
    configs = [make_config("good_one"), make_broken_config("broken"), make_cnn_config("good_two")]

    with pytest.raises(LaunchError) as excinfo:
        run_experiments.run_launch(
            configs, _request(use_mlflow=False, workers=workers), "sweep", env
        )

    assert list(excinfo.value.failed) == ["broken"]
    assert "AssertionError" in excinfo.value.failed["broken"]
    assert "broken" in str(excinfo.value)
    assert _h5_names(env) == ["good_one.h5", "good_two.h5"]
    assert not list(env.results_root.glob("*/*.tmp"))


def test_worker_dying_breaks_the_pool_and_the_retry_round_recovers(
    env: LaunchEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FAIL_MARKER_ENV, str(tmp_path / "died.marker"))
    monkeypatch.setenv(SKIP_PID_ENV, str(os.getpid()))
    dying = dataclasses.replace(env, stage_builder=die_once_stage_builder)

    run_experiments.run_launch(_configs(), _request(use_mlflow=False, workers=2), "sweep", dying)

    assert (tmp_path / "died.marker").exists()
    assert _h5_names(env) == ["cnn_b.h5", "fcnn_a.h5"]


def test_mlflow_errors_fail_the_launch_but_keep_every_result(
    env: LaunchEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_log_text(*args, **kwargs):
        raise OSError("artifact location not writable")

    monkeypatch.setattr(mlflow, "log_text", failing_log_text)

    with pytest.raises(LaunchError) as excinfo:
        run_experiments.run_launch(_configs(), _request(), "sweep", env)

    assert excinfo.value.failed == {}
    assert set(excinfo.value.mlflow_errors) == {"fcnn_a", "cnn_b"}
    assert "all results were still written" in str(excinfo.value)
    assert _h5_names(env) == ["cnn_b.h5", "fcnn_a.h5"]


def test_second_launch_in_the_same_minute_on_the_same_commit_is_refused(
    env: LaunchEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 14, 12, 30, 5).astimezone()

    monkeypatch.setattr(run_store, "datetime", FrozenDatetime)
    run_experiments.run_launch(_configs(), _request(use_mlflow=False), "sweep", env)
    entry_after_first = env.lab_book.read_text()

    with pytest.raises(run_store.LaunchExistsError):
        run_experiments.run_launch(_configs(), _request(use_mlflow=False), "sweep", env)

    assert env.lab_book.read_text() == entry_after_first
    assert len(_h5_names(env)) == 2


def test_no_mlflow_launch_never_imports_mlflow_in_parent_or_workers(
    git_repo: Path, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.txt"
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import fakes, run_experiments\n"
        "repo, tmp = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "env = run_experiments.LaunchEnv(results_root=tmp / 'results',\n"
        "    lab_book=repo / 'docs' / 'lab-book.md', repo_root=repo,\n"
        "    stage_builder=fakes.mlflow_probe_stage_builder)\n"
        "request = run_experiments.LaunchRequest('g', 'm', 'c', workers=2, use_mlflow=False,\n"
        "    command='x')\n"
        "configs = [fakes.make_config('a'), fakes.make_config('b')]\n"
        "run_experiments.run_launch(configs, request, 'sweep', env)\n"
        "print('PARENT_HAS_MLFLOW', 'mlflow' in sys.modules)\n"
    )
    child_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(SRC_DIR), str(TESTS_DIR)]),
        MLFLOW_PROBE_ENV: str(probe),
    }

    result = subprocess.run(
        [sys.executable, "-c", code, str(git_repo), str(tmp_path)],
        capture_output=True,
        text=True,
        env=child_env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PARENT_HAS_MLFLOW False" in result.stdout
    probe_lines = probe.read_text().splitlines()
    assert len(probe_lines) >= 3  # the parent's fail-fast build plus one per job
    assert all(line.endswith("False") for line in probe_lines)


# --- command line -------------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["run_experiments.py", "--goal", "g", "--method", "m", "--code-changes", "c"]
    )

    args = run_experiments._parse_args()

    assert args.scan == "sweep"
    assert args.num_seeds == run_experiments.NUM_SEEDS
    assert args.dataset == "mnist"
    assert args.initial_classes == 5
    assert args.classes_per_stage == 1
    assert args.workers == 1
    assert args.no_mlflow is False


def test_parse_args_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiments.py",
            "--scan", "size",
            "--num-seeds", "5",
            "--dataset", "fashion_mnist",
            "--initial-classes", "1",
            "--classes-per-stage", "2",
            "--workers", "12",
            "--no-mlflow",
            "--goal", "g",
            "--method", "m",
            "--code-changes", "c",
        ],
    )  # fmt: skip

    args = run_experiments._parse_args()

    assert (args.scan, args.num_seeds, args.dataset) == ("size", 5, "fashion_mnist")
    assert (args.initial_classes, args.classes_per_stage) == (1, 2)
    assert args.workers == 12 and args.no_mlflow is True


@pytest.mark.parametrize("missing", ["--goal", "--method", "--code-changes"])
def test_parse_args_requires_the_lab_book_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    argv = ["run_experiments.py", "--goal", "g", "--method", "m", "--code-changes", "c"]
    index = argv.index(missing)
    monkeypatch.setattr("sys.argv", argv[:index] + argv[index + 2 :])

    with pytest.raises(SystemExit) as excinfo:
        run_experiments._parse_args()

    assert excinfo.value.code == 2
    assert missing in capsys.readouterr().err


def test_parse_args_rejects_workers_below_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["r.py", "--workers", "0", "--goal", "g", "--method", "m", "--code-changes", "c"],
    )

    with pytest.raises(SystemExit):
        run_experiments._parse_args()

    assert "must be >= 1" in capsys.readouterr().err


def test_cli_records_the_typed_command_and_maps_outcomes_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["src/run_experiments.py", "--no-mlflow", "--goal", "my goal"]
    argv += ["--method", "m", "--code-changes", "c"]
    monkeypatch.setattr("sys.argv", argv)
    seen = []
    outcomes = iter(
        [
            None,
            LaunchError({"x": "boom"}, {}),
            LaunchError({}, {"x": ["mlflow down"]}),
            ValueError("bad input"),
            run_store.DirtyTreeError("dirty"),
            run_store.LaunchExistsError("exists"),
        ]
    )

    def fake_run_launch(configs, request, scan, env):
        seen.append(request)
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(run_experiments, "run_launch", fake_run_launch)

    codes = [run_experiments._cli() for _ in range(6)]

    assert codes == [0, 1, 1, 2, 2, 2]
    assert seen[0].command == "python src/run_experiments.py --no-mlflow --goal 'my goal' " + (
        "--method m --code-changes c"
    )
    assert seen[0].use_mlflow is False
    err = capsys.readouterr().err
    assert "boom" in err and "mlflow down" in err and "error: bad input" in err


def test_retry_round_resumes_a_job_that_crashed_after_writing_some_seeds(
    env: LaunchEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_train = experiment_jobs.train_and_record
    calls = []

    def crash_on_second_seed_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash after first seed")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(experiment_jobs, "train_and_record", crash_on_second_seed_once)
    monkeypatch.setattr(run_experiments, "_draw_seeds", lambda n: [7, 8, 9])

    run_experiments.run_launch(
        [make_config("fcnn_a")], _request(num_seeds=3, use_mlflow=False), "sweep", env
    )

    assert _h5_names(env) == ["fcnn_a_seed7.h5", "fcnn_a_seed8.h5", "fcnn_a_seed9.h5"]
    assert len(calls) == 4  # seed 7, crashing seed 8, then seeds 8 and 9 on the retry


def test_size_scan_entry_point_honours_width_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = []
    monkeypatch.setattr(
        run_experiments,
        "run_launch",
        lambda configs, request, scan, env: counts.append(len(configs)),
    )

    run_experiments.run_size_scan(
        goal="g",
        method="m",
        code_changes="c",
        hidden_sizes_scan=[(4, 2), (2, 1)],
        channels_scan=[(2, 4), (1, 1)],
    )

    assert counts == [14]  # two steps x seven configs


def test_empty_config_list_is_refused_before_any_side_effect(env: LaunchEnv) -> None:
    before = env.lab_book.read_text()

    with pytest.raises(ValueError, match="Nothing to run"):
        run_experiments.run_launch([], _request(workers=2), "sweep", env)

    _assert_nothing_created(env, before)


def test_parallel_and_sequential_launches_give_the_same_numbers_and_record_thread_counts(
    env: LaunchEnv, tmp_path: Path
) -> None:
    sequential = _in_new_root(env, tmp_path, "numbers_seq")
    parallel = _in_new_root(env, tmp_path, "numbers_par")

    run_experiments.run_launch(_configs(), _request(use_mlflow=False), "sweep", sequential)
    run_experiments.run_launch(_configs(), _request(use_mlflow=False, workers=2), "sweep", parallel)

    seq_runs = {r.meta["run_name"]: r for r in run_store.list_runs(sequential.results_root)}
    par_runs = {r.meta["run_name"]: r for r in run_store.list_runs(parallel.results_root)}
    assert seq_runs.keys() == par_runs.keys()
    for name, seq in seq_runs.items():
        for dataset in ("accuracy", "train_loss"):
            np.testing.assert_allclose(
                par_runs[name].array(dataset).read(), seq.array(dataset).read(), atol=1e-5
            )
        assert par_runs[name].meta["torch_num_threads"] == 1
        assert seq.meta["torch_num_threads"] >= 1


def test_a_permanently_dying_job_does_not_take_the_healthy_jobs_down_with_it(
    env: LaunchEnv, launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SKIP_PID_ENV, str(os.getpid()))
    crasher = make_config("crasher", initial_classes=DIE_FOR_INITIAL_CLASSES)
    configs = [crasher] + [
        make_config(f"good{i}") for i in range(5)
    ]  # crasher first: queue is lost
    jobs = [
        Job(config, (0,), launch, stage_builder=die_for_marked_job_stage_builder, verbose=False)
        for config in configs
    ]

    outcomes = {o.name: o for o in run_experiments.execute_jobs(jobs, workers=2)}

    assert outcomes["crasher"].error is not None
    assert all(outcomes[f"good{i}"].error is None for i in range(5))
    assert sorted(p.name for p in launch.directory.glob("*.h5")) == [
        f"good{i}.h5" for i in range(5)
    ]
