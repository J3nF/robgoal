"""Tests for run_store.py: git provenance, launches, lab book, and the HDF5 run files."""

import dataclasses
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import run_git

import run_store
from run_config import RunConfig
from run_store import (
    DirtyTreeError,
    LaunchChangedError,
    LaunchExistsError,
    LaunchSpec,
    RunArray,
    RunExistsError,
    RunFileError,
)

NOW = datetime(2026, 10, 14, 12, 30, 5, tzinfo=timezone(timedelta(hours=2)))


BASE_SPEC = LaunchSpec(
    goal="test goal",
    method="test method",
    code_changes="none",
    command="uv run python src/run_experiments.py --scan size",
    scan="size",
    planned_runs=4,
    dataset="mnist",
    seeds_mode="single seed 0",
    initial_classes=1,
    classes_per_stage=1,
    workers=2,
    use_mlflow=False,
)


def _spec(**overrides) -> LaunchSpec:
    return dataclasses.replace(BASE_SPEC, **overrides)


def _begin(repo: Path, tmp_path: Path, spec: LaunchSpec | None = None, now=NOW):
    return run_store.begin_launch(
        spec or _spec(),
        results_root=tmp_path / "results",
        lab_book=repo / "docs" / "lab-book.md",
        repo_root=repo,
        now=now,
    )


def _config(name: str = "cfg") -> RunConfig:
    return RunConfig(
        name=name,
        sweep_axis="base",
        model_kind="fcnn",
        epochs_per_stage=1,
        ai2_variant="ai2_diff",
        activation_scheme="relu_sigmoid",
        T=2,
        hidden_sizes=(4, 2),
    )


def _write(launch: run_store.Launch, name: str = "cfg", arrays=None, metrics=None) -> Path:
    if arrays is None:
        arrays = {"accuracy": RunArray(np.arange(20.0).reshape(2, 10), ("epoch", "digit"))}
    return run_store.write_run(
        launch,
        _config(name),
        arrays,
        config_name=name,
        metrics=metrics
        or {"final_avg_accuracy": 0.5, "final_backward_transfer": -0.1, "train_seconds": 1.5},
        started_at=NOW.isoformat(),
        device="cpu",
    )


# --- git provenance ---------------------------------------------------------------------


def test_clean_tree_passes_and_reports_full_commit(git_repo: Path) -> None:
    run_store.require_clean_tree(git_repo)

    info = run_store.get_git_info(git_repo)

    assert info.commit == run_git(git_repo, "rev-parse", "HEAD")
    assert len(info.commit) == 40
    assert info.branch == "main"


def test_modified_tracked_file_is_dirty_and_listed(git_repo: Path) -> None:
    (git_repo / "src" / "a.py").write_text("x = 2\n")

    with pytest.raises(DirtyTreeError, match=r"src/a\.py"):
        run_store.require_clean_tree(git_repo)


def test_staged_change_is_dirty(git_repo: Path) -> None:
    (git_repo / "src" / "a.py").write_text("x = 2\n")
    run_git(git_repo, "add", "src/a.py")

    with pytest.raises(DirtyTreeError, match=r"src/a\.py"):
        run_store.require_clean_tree(git_repo)


def test_untracked_file_is_dirty_even_inside_new_directory(git_repo: Path) -> None:
    (git_repo / "newdir").mkdir()
    (git_repo / "newdir" / "b.py").write_text("y = 1\n")

    with pytest.raises(DirtyTreeError, match=r"newdir/b\.py"):
        run_store.require_clean_tree(git_repo)


def test_ignored_file_does_not_make_tree_dirty(git_repo: Path) -> None:
    (git_repo / "results").mkdir()
    (git_repo / "results" / "x.h5").write_text("data")

    run_store.require_clean_tree(git_repo)


def test_lab_book_only_change_counts_as_clean(git_repo: Path) -> None:
    lab_book = git_repo / "docs" / "lab-book.md"
    lab_book.write_text("# Lab book\n\n## entry\n")

    run_store.require_clean_tree(git_repo, lab_book)


def test_untracked_lab_book_counts_as_clean(git_repo: Path) -> None:
    lab_book = git_repo / "docs" / "new-lab-book.md"
    lab_book.write_text("# Lab book\n")

    run_store.require_clean_tree(git_repo, lab_book)


def test_lab_book_exemption_does_not_hide_other_changes(git_repo: Path) -> None:
    lab_book = git_repo / "docs" / "lab-book.md"
    lab_book.write_text("changed\n")
    (git_repo / "src" / "a.py").write_text("x = 2\n")

    with pytest.raises(DirtyTreeError) as excinfo:
        run_store.require_clean_tree(git_repo, lab_book)

    assert "src/a.py" in str(excinfo.value)
    assert "lab-book.md" not in str(excinfo.value)


def test_rename_reports_source_and_destination(git_repo: Path) -> None:
    run_git(git_repo, "mv", "src/a.py", "src/b.py")

    with pytest.raises(DirtyTreeError) as excinfo:
        run_store.require_clean_tree(git_repo)

    assert "src/a.py" in str(excinfo.value)
    assert "src/b.py" in str(excinfo.value)


def test_lab_book_outside_repo_needs_no_exemption(git_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "lab-book.md"

    run_store.require_clean_tree(git_repo, outside)


def test_detached_head_is_reported(git_repo: Path) -> None:
    run_git(git_repo, "checkout", "-q", "--detach")

    assert run_store.get_git_info(git_repo).branch == "detached"


def test_repo_without_commits_fails_with_actionable_message(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    run_git(repo, "init", "-q")

    with pytest.raises(RuntimeError, match="at least one commit"):
        run_store.get_git_info(repo)


# --- launches and the lab book ---------------------------------------------------------


def test_begin_launch_creates_directory_and_lab_book_entry(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)

    commit = run_git(git_repo, "rev-parse", "HEAD")
    assert launch.launch_id == f"2026-10-14-12-30_{commit[:7]}"
    assert launch.directory == tmp_path / "results" / launch.launch_id
    assert launch.directory.is_dir()
    entry = (git_repo / "docs" / "lab-book.md").read_text()
    for expected in (
        f"## {launch.launch_id}",
        commit,
        "main",
        "uv run python src/run_experiments.py --scan size",
        "test goal",
        "test method",
        "Initial classes / classes per stage:** 1 / 1",
        "Workers:** 2",
        "MLflow:** off",
        "2026-10-14T12:30:05+02:00",
    ):
        assert expected in entry


@pytest.mark.parametrize("field", ["goal", "method", "code_changes"])
@pytest.mark.parametrize("value", ["", "   \n"])
def test_blank_required_text_refuses_before_creating_anything(
    git_repo: Path, tmp_path: Path, field: str, value: str
) -> None:
    lab_book_before = (git_repo / "docs" / "lab-book.md").read_text()

    with pytest.raises(ValueError, match=field.replace("_", "-")):
        _begin(git_repo, tmp_path, _spec(**{field: value}))

    assert not (tmp_path / "results").exists()
    assert (git_repo / "docs" / "lab-book.md").read_text() == lab_book_before


def test_dirty_tree_refuses_lists_paths_and_creates_nothing(git_repo: Path, tmp_path: Path) -> None:
    (git_repo / "src" / "a.py").write_text("x = 2\n")
    lab_book_before = (git_repo / "docs" / "lab-book.md").read_text()

    with pytest.raises(DirtyTreeError, match=r"src/a\.py"):
        _begin(git_repo, tmp_path)

    assert not (tmp_path / "results").exists()
    assert (git_repo / "docs" / "lab-book.md").read_text() == lab_book_before


def test_existing_launch_directory_refuses_and_keeps_lab_book(
    git_repo: Path, tmp_path: Path
) -> None:
    first = _begin(git_repo, tmp_path)
    (first.directory / "keep.txt").write_text("precious")
    lab_book_after_first = (git_repo / "docs" / "lab-book.md").read_text()

    with pytest.raises(LaunchExistsError, match="already exists"):
        _begin(git_repo, tmp_path)

    assert (first.directory / "keep.txt").read_text() == "precious"
    assert (git_repo / "docs" / "lab-book.md").read_text() == lab_book_after_first


def test_second_launch_appends_and_leaves_first_entry_intact(
    git_repo: Path, tmp_path: Path
) -> None:
    first = _begin(git_repo, tmp_path)
    after_first = (git_repo / "docs" / "lab-book.md").read_text()

    second = _begin(git_repo, tmp_path, now=NOW + timedelta(minutes=1))

    text = (git_repo / "docs" / "lab-book.md").read_text()
    assert text.startswith(after_first)
    assert f"## {second.launch_id}" in text[len(after_first) :]
    assert first.launch_id != second.launch_id


def test_missing_lab_book_is_created_with_header(git_repo: Path, tmp_path: Path) -> None:
    lab_book = tmp_path / "fresh" / "lab-book.md"

    run_store.begin_launch(
        _spec(), results_root=tmp_path / "results", lab_book=lab_book, repo_root=git_repo, now=NOW
    )

    text = lab_book.read_text()
    assert text.startswith("# Lab book")
    assert text.count("## 2026-") == 1


def test_failed_lab_book_append_removes_the_launch_directory(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_append(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(run_store, "append_lab_book_entry", failing_append)

    with pytest.raises(OSError, match="disk full"):
        _begin(git_repo, tmp_path)

    assert list((tmp_path / "results").iterdir()) == []


def test_launch_is_picklable_for_worker_processes(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)

    assert pickle.loads(pickle.dumps(launch)) == launch


def test_verify_unchanged_passes_when_nothing_happened(git_repo: Path, tmp_path: Path) -> None:
    _begin(git_repo, tmp_path).verify_unchanged()


def test_verify_unchanged_allows_committing_only_the_lab_book(
    git_repo: Path, tmp_path: Path
) -> None:
    launch = _begin(git_repo, tmp_path)
    run_git(git_repo, "add", "docs/lab-book.md")
    run_git(git_repo, "commit", "-q", "-m", "log launch")

    launch.verify_unchanged()


def test_verify_unchanged_rejects_source_edit(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    (git_repo / "src" / "a.py").write_text("x = 99\n")

    with pytest.raises(DirtyTreeError, match=r"src/a\.py"):
        launch.verify_unchanged()


def test_verify_unchanged_rejects_source_commit_after_launch(
    git_repo: Path, tmp_path: Path
) -> None:
    launch = _begin(git_repo, tmp_path)
    (git_repo / "src" / "a.py").write_text("x = 99\n")
    run_git(git_repo, "commit", "-q", "-am", "change code")

    with pytest.raises(LaunchChangedError, match=r"src/a\.py"):
        launch.verify_unchanged()


def test_verify_unchanged_rejects_rename_hidden_behind_lab_book(
    git_repo: Path, tmp_path: Path
) -> None:
    launch = _begin(git_repo, tmp_path)
    run_git(git_repo, "mv", "-f", "src/a.py", "docs/lab-book.md")
    run_git(git_repo, "commit", "-q", "-m", "rename onto lab book")

    with pytest.raises(LaunchChangedError, match=r"src/a\.py"):
        launch.verify_unchanged()


# --- run files ---------------------------------------------------------------------------


def test_round_trip_preserves_meta_arrays_and_dims(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    accuracy = np.arange(20.0).reshape(2, 10)
    loss = np.array([0.9, 0.5, 0.3])
    per_neuron = np.arange(12.0).reshape(3, 4)
    path = _write(
        launch,
        arrays={
            "accuracy": RunArray(accuracy, ("epoch", "digit")),
            "train_loss": RunArray(loss, ("epoch",)),
            "y_self_per_neuron_batch/hidden_layers.0": RunArray(per_neuron, ("batch", "neuron")),
        },
    )

    run = run_store.open_run(path)

    assert run.run_id == f"{launch.launch_id}/cfg"
    assert run.meta["launch_id"] == launch.launch_id
    assert run.meta["config_name"] == "cfg"
    assert run.meta["config"]["hidden_sizes"] == [4, 2]
    assert run.meta["config"]["initial_classes"] == 5
    assert run.meta["git"] == {
        "commit": launch.git.commit,
        "branch": "main",
        "dirty": False,
    }
    assert run.meta["metrics"] == {
        "final_avg_accuracy": 0.5,
        "final_backward_transfer": -0.1,
        "train_seconds": 1.5,
    }
    assert run.meta["device"] == "cpu"
    for key in ("hostname", "python", "torch", "numpy", "h5py", "started_at"):
        assert run.meta[key]
    assert run.dataset_paths() == [
        "accuracy",
        "train_loss",
        "y_self_per_neuron_batch/hidden_layers.0",
    ]
    ref = run.array("accuracy")
    assert (ref.shape, ref.dims, ref.dtype) == ((2, 10), ("epoch", "digit"), np.float64)
    np.testing.assert_array_equal(ref.read(), accuracy)
    np.testing.assert_array_equal(run.array("train_loss").read(), loss)
    batch = run.array("y_self_per_neuron_batch/hidden_layers.0")
    assert batch.dims == ("batch", "neuron")
    np.testing.assert_array_equal(np.asarray(batch), per_neuron)


def test_slicing_an_array_ref_returns_the_requested_part(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    data = np.arange(60.0).reshape(6, 10)
    path = _write(launch, arrays={"a": RunArray(data, ("time", "neuron"))})

    ref = run_store.open_run(path).array("a")

    np.testing.assert_array_equal(ref[2:4, 3], data[2:4, 3])
    np.testing.assert_array_equal(ref[-1], data[-1])


def test_existing_target_is_never_overwritten(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    path = _write(launch)
    original = path.read_bytes()

    with pytest.raises(RunExistsError, match="never overwritten"):
        _write(launch, arrays={"other": RunArray(np.zeros(3), ("x",))})

    assert path.read_bytes() == original
    assert not path.with_name(path.name + ".tmp").exists()


def test_failure_while_writing_leaves_neither_tmp_nor_final(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)

    with pytest.raises(TypeError, match="not JSON serializable"):
        _write(launch, metrics={"final_avg_accuracy": object()})

    assert list(launch.directory.iterdir()) == []


def test_numpy_scalars_in_metrics_are_stored_as_plain_numbers(
    git_repo: Path, tmp_path: Path
) -> None:
    launch = _begin(git_repo, tmp_path)

    path = _write(launch, metrics={"final_avg_accuracy": np.float32(0.25), "n": np.int64(3)})

    assert run_store.open_run(path).meta["metrics"] == {"final_avg_accuracy": 0.25, "n": 3}


@pytest.mark.parametrize("data", [np.array(["a", "b"]), np.array([object()])])
def test_run_array_rejects_dtypes_h5py_cannot_store(data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Cannot store dtype"):
        RunArray(data, ("x",))


@pytest.mark.parametrize("name", ["", "a/b", ".hidden"])
def test_invalid_run_names_are_rejected_before_writing(
    git_repo: Path, tmp_path: Path, name: str
) -> None:
    launch = _begin(git_repo, tmp_path)

    with pytest.raises(ValueError, match="Invalid run name"):
        _write(launch, name=name)

    assert list(launch.directory.iterdir()) == []


def test_write_fsyncs_before_publishing(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _begin(git_repo, tmp_path)
    calls = []
    monkeypatch.setattr(run_store.os, "fsync", lambda fd: calls.append(fd))

    _write(launch)

    assert len(calls) == 1


def test_empty_and_zero_dimensional_arrays_round_trip(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    path = _write(
        launch,
        arrays={
            "empty": RunArray(np.zeros((0, 3)), ("epoch", "neuron")),
            "scalar": RunArray(np.array(2.5), ()),
        },
    )

    run = run_store.open_run(path)

    assert run.array("empty").shape == (0, 3)
    assert run.array("empty").dims == ("epoch", "neuron")
    assert run.array("scalar").dims == ()
    assert run.array("scalar").read() == 2.5


def test_dataset_that_is_also_a_group_prefix_is_rejected(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    clashing = {"a": RunArray(np.zeros(2), ("x",)), "a/b": RunArray(np.zeros(2), ("x",))}

    with pytest.raises(ValueError, match="group prefix"):
        _write(launch, arrays=clashing)

    assert list(launch.directory.iterdir()) == []


def test_failure_at_replace_leaves_no_tmp_and_no_final(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _begin(git_repo, tmp_path)

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(run_store.os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        _write(launch)

    assert list(launch.directory.iterdir()) == []


def test_stale_tmp_from_a_crashed_attempt_is_replaced(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    stale = launch.run_path("cfg").with_name("cfg.h5.tmp")
    stale.write_bytes(b"partial garbage")

    path = _write(launch)

    assert run_store.open_run(path).meta["run_name"] == "cfg"
    assert not stale.exists()


@pytest.mark.parametrize("name", ["", "/abs", "a//b", "meta_json", "trailing/"])
def test_invalid_dataset_paths_are_rejected(git_repo: Path, tmp_path: Path, name: str) -> None:
    launch = _begin(git_repo, tmp_path)

    with pytest.raises(ValueError, match="Invalid dataset path"):
        _write(launch, arrays={name: RunArray(np.zeros(2), ("x",))})

    assert list(launch.directory.iterdir()) == []


def test_run_array_rejects_dims_that_do_not_match_ndim() -> None:
    with pytest.raises(ValueError, match="do not match"):
        RunArray(np.zeros((2, 3)), ("only_one",))


def test_unknown_dataset_error_lists_available_ones(git_repo: Path, tmp_path: Path) -> None:
    launch = _begin(git_repo, tmp_path)
    run = run_store.open_run(_write(launch, arrays={"group/inner": RunArray(np.zeros(2), ("x",))}))

    with pytest.raises(RunFileError, match=r"group/inner"):
        run.array("nope")
    with pytest.raises(RunFileError, match="no dataset 'group'"):
        run.array("group")


def test_missing_meta_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "foreign.h5"
    with h5py.File(path, "x") as f:
        f.create_dataset("accuracy", data=np.zeros(3))

    with pytest.raises(RunFileError, match="meta_json"):
        run_store.open_run(path)


def test_corrupt_meta_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.h5"
    with h5py.File(path, "x") as f:
        f.create_dataset("meta_json", data="{not json", dtype=h5py.string_dtype())

    with pytest.raises(RunFileError, match="not valid JSON"):
        run_store.open_run(path)


def test_meta_without_required_keys_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "thin.h5"
    with h5py.File(path, "x") as f:
        f.create_dataset("meta_json", data='{"run_id": "x"}', dtype=h5py.string_dtype())

    with pytest.raises(RunFileError, match="required keys"):
        run_store.open_run(path)


def test_non_hdf5_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "truncated.h5"
    path.write_bytes(b"not hdf5 at all")

    with pytest.raises(RunFileError, match="cannot open"):
        run_store.open_run(path)


def test_list_runs_covers_launch_dirs_and_roots_and_skips_tmp(
    git_repo: Path, tmp_path: Path
) -> None:
    first = _begin(git_repo, tmp_path)
    _write(first, "b_run")
    _write(first, "a_run")
    first.run_path("c_run").with_name("c_run.h5.tmp").write_bytes(b"in progress")
    second = _begin(git_repo, tmp_path, now=NOW + timedelta(minutes=1))
    _write(second, "z_run")

    in_launch = run_store.list_runs(first.directory)
    in_root = run_store.list_runs(tmp_path / "results")

    assert [run.meta["run_name"] for run in in_launch] == ["a_run", "b_run"]
    assert [run.run_id for run in in_root] == [
        f"{first.launch_id}/a_run",
        f"{first.launch_id}/b_run",
        f"{second.launch_id}/z_run",
    ]


def test_list_runs_raises_on_a_corrupt_file_instead_of_skipping(
    git_repo: Path, tmp_path: Path
) -> None:
    launch = _begin(git_repo, tmp_path)
    _write(launch, "good")
    (launch.directory / "bad.h5").write_bytes(b"junk")

    with pytest.raises(RunFileError, match=r"bad\.h5"):
        run_store.list_runs(launch.directory)


def test_list_runs_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_store.list_runs(tmp_path / "nope")


def test_meta_json_is_not_exposed_as_an_array(git_repo: Path, tmp_path: Path) -> None:
    run = run_store.open_run(_write(_begin(git_repo, tmp_path)))

    with pytest.raises(RunFileError, match="no dataset 'meta_json'"):
        run.array("meta_json")


@pytest.mark.parametrize("payload", ["null", "[1, 2]", "42"])
def test_meta_json_that_is_not_an_object_is_reported(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "odd.h5"
    with h5py.File(path, "x") as f:
        f.create_dataset("meta_json", data=payload, dtype=h5py.string_dtype())

    with pytest.raises(RunFileError, match="JSON object"):
        run_store.open_run(path)


def test_meta_json_of_the_wrong_hdf5_type_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "odd.h5"
    with h5py.File(path, "x") as f:
        f.create_dataset("meta_json", data=np.zeros(3))
    group_path = tmp_path / "group.h5"
    with h5py.File(group_path, "x") as f:
        f.create_group("meta_json")

    with pytest.raises(RunFileError, match="meta_json"):
        run_store.open_run(path)
    with pytest.raises(RunFileError, match="meta_json"):
        run_store.open_run(group_path)


def test_array_ref_has_length_but_cannot_be_iterated_by_accident(
    git_repo: Path, tmp_path: Path
) -> None:
    ref = run_store.open_run(_write(_begin(git_repo, tmp_path))).array("accuracy")

    assert len(ref) == 2
    with pytest.raises(TypeError):
        iter(ref)


def test_branch_name_is_right_when_a_tag_has_the_same_name(git_repo: Path) -> None:
    run_git(git_repo, "tag", "main")

    assert run_store.get_git_info(git_repo).branch == "main"


def test_get_dirty_paths_is_public_and_excludes_the_lab_book(git_repo: Path) -> None:
    (git_repo / "src" / "a.py").write_text("x = 2\n")
    (git_repo / "docs" / "lab-book.md").write_text("changed\n")

    assert run_store.get_dirty_paths(git_repo) == ["docs/lab-book.md", "src/a.py"]
    assert run_store.get_dirty_paths(git_repo, git_repo / "docs" / "lab-book.md") == ["src/a.py"]


def test_dirty_listing_is_capped_and_quotes_odd_names(git_repo: Path) -> None:
    for index in range(60):
        (git_repo / f"new {index:02d}.txt").write_text("x")

    with pytest.raises(DirtyTreeError) as excinfo:
        run_store.require_clean_tree(git_repo)

    message = str(excinfo.value)
    assert "'new 00.txt'" in message
    assert "... and 10 more" in message


def test_paths_with_unicode_and_spaces_are_handled(git_repo: Path) -> None:
    (git_repo / "sp ace").mkdir()
    (git_repo / "sp ace" / "ünï.py").write_text("x")

    assert run_store.get_dirty_paths(git_repo) == ["sp ace/ünï.py"]


def test_repo_root_may_be_a_subdirectory_of_the_git_toplevel(git_repo: Path) -> None:
    lab_book = git_repo / "docs" / "lab-book.md"
    lab_book.write_text("changed\n")

    run_store.require_clean_tree(git_repo / "src", lab_book)


def test_default_launch_time_is_local_with_utc_offset(git_repo: Path, tmp_path: Path) -> None:
    before = datetime.now().astimezone()

    launch = run_store.begin_launch(
        _spec(),
        results_root=tmp_path / "results",
        lab_book=git_repo / "docs" / "lab-book.md",
        repo_root=git_repo,
    )

    entry = (git_repo / "docs" / "lab-book.md").read_text()
    offset = before.strftime("%z")
    assert f"{offset[:3]}:{offset[3:]}" in entry
    assert launch.launch_id.startswith(before.strftime("%Y-%m-"))


def test_command_with_backticks_cannot_break_out_of_its_fence(
    git_repo: Path, tmp_path: Path
) -> None:
    _begin(git_repo, tmp_path, _spec(command="echo ```danger``` `x`"))

    entry = (git_repo / "docs" / "lab-book.md").read_text()
    assert "````sh\necho ```danger``` `x`\n````" in entry


def test_appending_to_a_lab_book_without_trailing_newline_keeps_entries_separate(
    git_repo: Path, tmp_path: Path
) -> None:
    lab_book = git_repo / "docs" / "lab-book.md"
    lab_book.write_text("# Lab book\nno newline at end")

    _begin(git_repo, tmp_path)

    assert "no newline at end\n\n## " in lab_book.read_text()


def test_extra_meta_is_stored_and_cannot_replace_standard_keys(
    git_repo: Path, tmp_path: Path
) -> None:
    launch = _begin(git_repo, tmp_path)
    config = _config("extra")
    arrays = {"accuracy": RunArray(np.zeros((1, 10)), ("epoch", "digit"))}
    common = {
        "config_name": "extra",
        "metrics": {},
        "started_at": NOW.isoformat(),
        "device": "cpu",
    }

    path = run_store.write_run(
        launch, config, arrays, extra_meta={"torch_num_threads": 3}, **common
    )
    with pytest.raises(ValueError, match="standard meta keys"):
        run_store.build_meta(launch, config, extra_meta={"device": "gpu"}, **common)

    assert run_store.open_run(path).meta["torch_num_threads"] == 3


def test_unset_optional_fields_are_left_out_of_the_lab_book_entry(
    git_repo: Path, tmp_path: Path
) -> None:
    spec = _spec(initial_classes=None, classes_per_stage=None, workers=None, use_mlflow=None)

    _begin(git_repo, tmp_path, spec)

    entry = (git_repo / "docs" / "lab-book.md").read_text()
    assert "None" not in entry
    assert "Initial classes" not in entry and "Workers" not in entry and "MLflow" not in entry
