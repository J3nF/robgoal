"""Run provenance, launch bookkeeping and the HDF5 run store.

One launch = one directory `results/{launch_id}/` holding one `{run_name}.h5` per run, plus one
entry in the tracked lab book (`docs/lab-book.md`). Each `.h5` carries its whole config and
provenance in the scalar string dataset `/meta_json`, so attribution cannot be separated from
the data. Arrays are datasets with a `dims` attribute naming their axes; dataset paths are
free-form (`y_self_absmean/hidden_layers.0`), so new groups such as `/activations/{layer}` need
no format change.

Deliberately independent of torch and mlflow, so plotting code can import it cheaply.
"""

import dataclasses
import importlib.metadata
import json
import os
import platform
import re
import shlex
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from run_config import RunConfig

__all__ = [
    "ArrayRef",
    "DirtyTreeError",
    "GitInfo",
    "Launch",
    "LaunchChangedError",
    "LaunchExistsError",
    "LaunchSpec",
    "RunArray",
    "RunExistsError",
    "RunFile",
    "RunFileError",
    "append_lab_book_entry",
    "begin_launch",
    "build_meta",
    "get_dirty_paths",
    "get_git_info",
    "list_runs",
    "make_launch_id",
    "open_run",
    "require_clean_tree",
    "write_run",
    "write_run_file",
]

META_DATASET = "meta_json"
_STRING = h5py.string_dtype()
_NUMERIC_KINDS = "biufc"  # dtype kinds h5py stores natively: bool, int, uint, float, complex
_MAX_LISTED_PATHS = 50
REQUIRED_META_KEYS = ("run_id", "launch_id", "config", "git", "started_at", "metrics")
LAB_BOOK_HEADER = """# Lab book

Run log: one entry per launch of `src/run_experiments.py`, appended automatically by
`run_store.begin_launch` (append-only; never edit past entries). Each entry is the launch's
`launch_id`, its provenance, and the `--goal`, `--method` and `--code-changes` text given at
launch. The research reasoning lives in `old-lab-book.md`.
"""


class DirtyTreeError(RuntimeError):
    """The git working tree has changes that the recorded commit hash would not describe."""


class LaunchChangedError(RuntimeError):
    """The repository changed (edits or another commit) after the launch started."""


class LaunchExistsError(FileExistsError):
    """The launch directory already exists."""


class RunExistsError(FileExistsError):
    """The run file already exists; run files are never overwritten."""


class RunFileError(ValueError):
    """A run file is missing, corrupt, or lacks the requested dataset."""


@dataclass(frozen=True)
class GitInfo:
    """Commit and branch of a code state. `None` means unknown (legacy imports)."""

    commit: str | None
    branch: str | None


@dataclass(frozen=True)
class LaunchSpec:
    """What a launch records in the lab book; the text fields come from the user, verbatim.

    Attributes:
        goal, method, code_changes: Required, non-blank. `code_changes` is how big source
            changes get recorded; it is never auto-generated.
        command: The command line, or the equivalent call, as typed.
        scan: Scan type ("sweep" or "size").
        planned_runs: Number of `.h5` files the launch is meant to produce.
        dataset, seeds_mode: Free-form descriptions shown in the entry.
        initial_classes, classes_per_stage: The class split every run in the launch uses.
        workers: Parallel worker processes, `None` if not applicable.
        use_mlflow: Whether live MLflow tracking is on, `None` if not applicable.
        notes: Optional extra paragraph appended to the entry.
    """

    goal: str
    method: str
    code_changes: str
    command: str
    scan: str
    planned_runs: int
    dataset: str
    seeds_mode: str
    initial_classes: int | None = None
    classes_per_stage: int | None = None
    workers: int | None = None
    use_mlflow: bool | None = None
    notes: str = ""

    def validate(self) -> None:
        """Raises `ValueError` naming the flag if a required text field is missing or blank."""
        for flag, text in (
            ("--goal", self.goal),
            ("--method", self.method),
            ("--code-changes", self.code_changes),
        ):
            if not text or not text.strip():
                raise ValueError(
                    f"{flag} is required and must not be empty: it is copied into the lab book "
                    "entry, which is how a launch stays attributable."
                )


@dataclass(frozen=True)
class Launch:
    """One launch's identity and location; small and picklable so workers can receive it.

    Attributes:
        launch_id: `YYYY-MM-DD-HH-mm_{git7}` in local time.
        directory: `results/{launch_id}/`, holding the launch's `.h5` files.
        repo_root: Repository the commit hash refers to.
        lab_book: Lab-book path, exempt from the dirty-tree check (it is rewritten by every
            launch); `None` if there is none.
        git: Commit and branch at launch start.
    """

    launch_id: str
    directory: Path
    repo_root: Path
    lab_book: Path | None
    git: GitInfo

    def run_path(self, run_name: str) -> Path:
        """Path of `run_name`'s `.h5` file inside this launch."""
        return self.directory / f"{run_name}.h5"

    def run_id(self, run_name: str) -> str:
        """`{launch_id}/{run_name}`, the key that identifies a run across launches."""
        return f"{self.launch_id}/{run_name}"

    def verify_unchanged(self) -> None:
        """Raises if the code a worker is about to run differs from what the launch recorded.

        Long, parallel launches import code from the working tree minutes after the commit
        hash was read. Committing only the lab book (the normal workflow while a scan runs)
        is allowed; any other edit or HEAD movement is not.
        """
        require_clean_tree(self.repo_root, self.lab_book)
        head = _git(self.repo_root, "rev-parse", "HEAD").strip()
        if head == self.git.commit:
            return
        diff = _git(
            self.repo_root, "diff", "--no-renames", "--name-only", "-z", str(self.git.commit), head
        )
        allowed = _lab_book_relative(self.repo_root, self.lab_book)
        changed = sorted(path for path in diff.split("\0") if path and path != allowed)
        if changed:
            raise LaunchChangedError(
                f"HEAD moved from {self.git.commit} to {head} since launch {self.launch_id} "
                "started, changing files other than the lab book: " + ", ".join(changed)
            )


@dataclass(frozen=True)
class RunArray:
    """An array to store in a run file, with the names of its axes.

    Attributes:
        data: The values; stored as given (no compression, no chunking).
        dims: One axis name per dimension of `data`, e.g. `("stage", "digit")`.
    """

    data: np.ndarray
    dims: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.dims) != self.data.ndim:
            raise ValueError(f"dims {self.dims} do not match array of shape {self.data.shape}")
        if self.data.dtype.kind not in _NUMERIC_KINDS:
            raise ValueError(
                f"Cannot store dtype {self.data.dtype}: only bool/int/float/complex arrays are "
                "supported, and a write failing after training would lose the run."
            )


@dataclass(frozen=True)
class ArrayRef:
    """Lazy handle to one dataset of a run file; every read reopens the file.

    Reopening per read keeps no file handle alive, so plotting hundreds of runs cannot exhaust
    file descriptors, and slicing reads only the requested part from disk.
    """

    path: Path
    dataset: str
    shape: tuple[int, ...]
    dtype: np.dtype
    dims: tuple[str, ...]

    def __getitem__(self, key: Any) -> np.ndarray:
        with _open_h5(self.path) as f:
            return np.asarray(f[self.dataset][key])

    def read(self) -> np.ndarray:
        """The whole array."""
        return self[()]

    def __len__(self) -> int:
        return self.shape[0]

    __iter__ = None  # iterating would reopen the file per row; use `read()` or slices

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        data = self.read()
        return data if dtype is None else data.astype(dtype)


@dataclass(frozen=True)
class RunFile:
    """A run file's parsed `meta` plus lazy access to its arrays.

    Attributes:
        path: The `.h5` file.
        meta: Parsed `/meta_json`; see `build_meta` for the keys.
    """

    path: Path
    meta: dict[str, Any]

    @property
    def run_id(self) -> str:
        """`{launch_id}/{run_name}`."""
        return self.meta["run_id"]

    def array(self, dataset: str) -> ArrayRef:
        """Lazy handle to `dataset` (e.g. `"accuracy"` or `"y_self_absmean/hidden_layers.0"`).

        Raises:
            RunFileError: if `dataset` does not exist or is a group; lists what does exist.
        """
        with _open_h5(self.path) as f:
            node = f.get(dataset) if dataset != META_DATASET else None
            if not isinstance(node, h5py.Dataset):
                available = ", ".join(_dataset_paths(f)) or "(none)"
                raise RunFileError(
                    f"{self.path}: no dataset {dataset!r}. Available datasets: {available}"
                )
            return ArrayRef(
                self.path,
                dataset,
                tuple(node.shape),
                node.dtype,
                _read_dims(node),
            )

    def dataset_paths(self) -> list[str]:
        """All array dataset paths in the file, excluding `meta_json`."""
        with _open_h5(self.path) as f:
            return _dataset_paths(f)


# --- launch ---------------------------------------------------------------------------------


def begin_launch(
    spec: LaunchSpec,
    *,
    results_root: Path,
    lab_book: Path,
    repo_root: Path,
    now: datetime | None = None,
) -> Launch:
    """Starts a launch: checks inputs and tree, creates its directory, logs it in the lab book.

    Everything that can refuse (blank text, dirty tree, existing launch directory) happens
    before anything is created, so a refused launch leaves no trace. If the lab-book append
    fails, the just-created (still empty) directory is removed again.

    Raises:
        ValueError: a required text field is blank.
        DirtyTreeError: the tree has changes other than the lab book.
        LaunchExistsError: a launch with the same minute and commit already exists.
    """
    spec.validate()
    require_clean_tree(repo_root, lab_book)
    git = get_git_info(repo_root)
    started = now if now is not None else datetime.now().astimezone()
    launch_id = make_launch_id(git, started)
    directory = Path(results_root) / launch_id
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise LaunchExistsError(
            f"Launch directory {directory} already exists: an earlier launch ran at the same "
            "minute on the same commit. Wait for the next minute, or inspect/remove the "
            "directory; launches are never merged or suffixed."
        ) from exc
    try:
        append_lab_book_entry(lab_book, launch_id, started, git, spec)
    except BaseException:
        directory.rmdir()
        raise
    return Launch(launch_id, directory, Path(repo_root), Path(lab_book), git)


def make_launch_id(git: GitInfo, started: datetime) -> str:
    """`YYYY-MM-DD-HH-mm_{git7}` from the local start time and the 7-char commit hash."""
    if git.commit is None:
        raise ValueError("A launch needs a known commit hash.")
    return f"{started.strftime('%Y-%m-%d-%H-%M')}_{git.commit[:7]}"


# --- git provenance -----------------------------------------------------------------------


def get_git_info(repo_root: Path) -> GitInfo:
    """Full HEAD commit hash and branch name of `repo_root` ("detached" if HEAD is detached).

    Read by our own code, not by MLflow, so it is right whatever script launched the run.
    """
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    branch = _git(repo_root, "branch", "--show-current").strip()
    return GitInfo(commit=commit, branch=branch or "detached")


def require_clean_tree(repo_root: Path, lab_book: Path | None = None) -> None:
    """Raises `DirtyTreeError` listing every dirty path, except the lab book.

    Dirty = tracked modification (staged or not) or untracked-but-not-ignored file, from
    `git status --porcelain`. The lab book is exempt because every launch rewrites it.
    Gitignored paths (results, mlflow.db) never count.
    """
    dirty = get_dirty_paths(repo_root, lab_book)
    if dirty:
        listing = "\n".join(f"  - {shlex.quote(path)}" for path in dirty[:_MAX_LISTED_PATHS])
        if len(dirty) > _MAX_LISTED_PATHS:
            listing += f"\n  ... and {len(dirty) - _MAX_LISTED_PATHS} more"
        raise DirtyTreeError(
            f"Refusing to run: the git tree at {repo_root} is dirty, so the commit hash would "
            f"not describe the code that runs. Commit or stash these paths first:\n{listing}"
        )


def get_dirty_paths(repo_root: Path, lab_book: Path | None = None) -> list[str]:
    """Sorted paths with uncommitted changes, relative to the git top level, minus the lab book.

    Tracked modifications (staged or not), untracked-but-not-ignored files, and both ends of a
    rename count. Files marked `assume-unchanged`/`skip-worktree` are invisible to git status.
    """
    allowed = _lab_book_relative(repo_root, lab_book)
    return sorted(path for path in _get_status_paths(repo_root) if path != allowed)


def _get_status_paths(repo_root: Path) -> set[str]:
    """Every path named in `git status --porcelain=v1 -z`, rename sources included."""
    out = _git(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries = out.split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(entries) and entries[index]:
        status, path = entries[index][:2], entries[index][3:]
        paths.add(path)
        index += 1
        if "R" in status or "C" in status:
            paths.add(entries[index])
            index += 1
    return paths


def _lab_book_relative(repo_root: Path, lab_book: Path | None) -> str | None:
    """`lab_book` relative to the git top level, or `None` if absent or outside the repo."""
    if lab_book is None:
        return None
    top = Path(_git(repo_root, "rev-parse", "--show-toplevel").rstrip("\n")).resolve()
    try:
        return Path(lab_book).resolve().relative_to(top).as_posix()
    except ValueError:
        return None


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    """Runs git in `repo_root`; `--no-optional-locks` so a status never blocks a user's commit."""
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_INDEX_FILE")}
    env.pop("GIT_WORK_TREE", None)
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        errors="surrogateescape",
        env=env,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"`git {shlex.join(args)}` failed in {repo_root} (exit {result.returncode}): "
            f"{result.stderr.strip()}. Is this a git repository with at least one commit?"
        )
    return result.stdout


# --- lab book -------------------------------------------------------------------------------


def append_lab_book_entry(
    lab_book: Path, launch_id: str, started: datetime, git: GitInfo, spec: LaunchSpec
) -> None:
    """Appends one launch entry to `lab_book`, creating it with a header if missing.

    Append-only by construction: one `"a"`-mode write, existing text is never rewritten.
    """
    lab_book = Path(lab_book)
    lab_book.parent.mkdir(parents=True, exist_ok=True)
    entry = _format_lab_book_entry(launch_id, started, git, spec)
    existing = lab_book.read_text(encoding="utf-8") if lab_book.exists() else None
    prefix = LAB_BOOK_HEADER if existing is None else ("\n" if not existing.endswith("\n") else "")
    with lab_book.open("a", encoding="utf-8") as f:
        f.write(f"{prefix}\n{entry}")


def _format_lab_book_entry(
    launch_id: str, started: datetime, git: GitInfo, spec: LaunchSpec
) -> str:
    """Markdown for one entry: auto-filled fields, then the user's text verbatim."""
    split = f"{spec.initial_classes} / {spec.classes_per_stage}"
    fields = [
        ("Started", started.isoformat(timespec="seconds")),
        ("Branch", git.branch),
        ("Commit", git.commit),
        ("Scan", spec.scan),
        ("Planned runs", spec.planned_runs),
        ("Dataset", spec.dataset),
        ("Seeds", spec.seeds_mode),
        ("Initial classes / classes per stage", split),
        ("Workers", spec.workers),
        ("MLflow", None if spec.use_mlflow is None else ("on" if spec.use_mlflow else "off")),
        ("Results", f"`results/{launch_id}/`"),
    ]
    lines = [f"## {launch_id}", ""]
    lines += [f"- **{label}:** {value}" for label, value in fields if value is not None]
    fence = "`" * max(3, _longest_backtick_run(spec.command) + 1)
    lines += ["", "Command:", "", f"{fence}sh", spec.command, fence, ""]
    for heading, text in (
        ("Goal", spec.goal),
        ("Method", spec.method),
        ("Code changes", spec.code_changes),
        ("Notes", spec.notes),
    ):
        if text.strip():
            lines += [f"### {heading}", "", text.strip(), ""]
    return "\n".join(lines)


def _longest_backtick_run(text: str) -> int:
    return max((len(run) for run in re.findall(r"`+", text)), default=0)


# --- writing --------------------------------------------------------------------------------


def write_run(
    launch: Launch,
    config: RunConfig,
    arrays: Mapping[str, RunArray],
    *,
    config_name: str,
    metrics: Mapping[str, float | None],
    started_at: str,
    device: str,
    extra_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Writes `config`'s run file inside `launch`: meta from `config` and provenance, plus `arrays`.

    `config.name` is the run name (the file stem); `config_name` is the same config before the
    dataset/seed suffixes, so seed replicates can be grouped without parsing names.

    Raises:
        RunExistsError: the file already exists (never overwritten).
    """
    _check_run_name(config.name)
    meta = build_meta(
        launch,
        config,
        config_name=config_name,
        metrics=metrics,
        started_at=started_at,
        device=device,
        extra_meta=extra_meta,
    )
    path = launch.run_path(config.name)
    write_run_file(path, meta, arrays)
    return path


def build_meta(
    launch: Launch,
    config: RunConfig,
    *,
    config_name: str,
    metrics: Mapping[str, float | None],
    started_at: str,
    device: str,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The `meta_json` content for a run of `config` inside `launch`.

    `extra_meta` adds caller-specific top-level keys (e.g. `torch_num_threads`); it cannot
    replace the standard ones.

    `git.dirty` is always `false`: a launch refuses to start on a dirty tree and
    `Launch.verify_unchanged` re-checks it per job.
    """
    meta = {
        "run_id": launch.run_id(config.name),
        "run_name": config.name,
        "launch_id": launch.launch_id,
        "config_name": config_name,
        "config": dataclasses.asdict(config),
        "git": {"commit": launch.git.commit, "branch": launch.git.branch, "dirty": False},
        "started_at": started_at,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "device": device,
        "metrics": dict(metrics),
    }
    if extra_meta:
        clashes = sorted(set(extra_meta) & set(meta))
        if clashes:
            raise ValueError(f"extra_meta may not replace standard meta keys: {clashes}")
        meta.update(extra_meta)
    return meta


def _check_run_name(name: str) -> None:
    if not name or "/" in name or name.startswith("."):
        raise ValueError(
            f"Invalid run name {name!r}: it becomes a file name, so it must be non-empty, "
            "contain no '/' and not start with '.'."
        )


def write_run_file(path: Path, meta: Mapping[str, Any], arrays: Mapping[str, RunArray]) -> None:
    """Atomically creates the run file `path` from `meta` and `arrays`; never overwrites.

    Writes `{path}.tmp` (mode `"x"`, so never `"w"`), fsyncs, then `os.replace`s it into
    place: a file visible under its final name is always complete, whatever crashes or reads
    concurrently. A stale `.tmp` from a crashed earlier attempt is partial by construction and
    is removed first. `os.replace` overwrites by design, hence the explicit existence check;
    no race exists because each run name is written by exactly one job per launch.

    Raises:
        RunExistsError: `path` already exists.
        ValueError: `meta` lacks a required key or an array path is invalid.
    """
    path = Path(path)
    _check_meta(meta)
    _check_dataset_names(list(arrays))
    if path.exists():
        raise RunExistsError(f"{path} already exists; run files are never overwritten.")
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with h5py.File(tmp, "x") as f:
            f.create_dataset(
                META_DATASET, data=json.dumps(meta, indent=2, default=_json_default), dtype=_STRING
            )
            for name, array in arrays.items():
                dataset = f.create_dataset(name, data=array.data)
                if array.dims:
                    dataset.attrs.create("dims", list(array.dims), dtype=_STRING)
        _fsync(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _json_default(value: Any) -> Any:
    """Lets numpy scalars/arrays into `meta_json`, so a stray `np.float32` cannot lose a run."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _check_meta(meta: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"meta lacks required keys {missing}; got {sorted(meta)}")


def _check_dataset_names(names: list[str]) -> None:
    for name in names:
        if name == META_DATASET or name.startswith("/") or not all(name.split("/")):
            raise ValueError(
                f"Invalid dataset path {name!r}: use non-empty '/'-separated names, "
                f"and not the reserved {META_DATASET!r}."
            )
    for name in names:
        clashes = [other for other in names if other.startswith(name + "/")]
        if clashes:
            raise ValueError(f"Dataset path {name!r} is also a group prefix of {clashes}.")


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --- reading --------------------------------------------------------------------------------


def open_run(path: Path) -> RunFile:
    """Opens a run file: parses and validates `meta_json`, then closes the file again.

    Raises:
        RunFileError: the file is missing, not HDF5, or its `meta_json` is absent or corrupt.
    """
    path = Path(path)
    meta = _read_meta(path)
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise RunFileError(f"{path}: meta_json lacks required keys {missing}.")
    return RunFile(path, meta)


def _read_meta(path: Path) -> dict[str, Any]:
    """Parses `/meta_json` of `path`; any way of it being absent or corrupt is a `RunFileError`."""
    try:
        with _open_h5(path) as f:
            node = f.get(META_DATASET)
            if not isinstance(node, h5py.Dataset):
                raise RunFileError(
                    f"{path}: no {META_DATASET!r} dataset; this is not a run file written by "
                    "run_store (or it was written incompletely)."
                )
            meta = json.loads(node.asstr()[()])
    except (OSError, TypeError, AttributeError, ValueError, KeyError) as exc:
        if isinstance(exc, RunFileError):
            raise
        raise RunFileError(
            f"{path}: {META_DATASET!r} is unreadable or not valid JSON ({exc})."
        ) from exc
    if not isinstance(meta, dict):
        raise RunFileError(
            f"{path}: {META_DATASET!r} must hold a JSON object, got {type(meta).__name__}."
        )
    return meta


def list_runs(path: Path) -> list[RunFile]:
    """Opens every `.h5` in `path` and in its immediate subdirectories, sorted by `run_id`.

    Accepts a launch directory or a results root (or a mix). `.tmp` files, i.e. runs still
    being written, never match. A corrupt file raises rather than being skipped.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory.")
    files = sorted([*path.glob("*.h5"), *path.glob("*/*.h5")])
    return sorted((open_run(file) for file in files), key=lambda run: run.run_id)


def _open_h5(path: Path) -> h5py.File:
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise RunFileError(f"{path}: cannot open as an HDF5 file ({exc}).") from exc


def _dataset_paths(f: h5py.File) -> list[str]:
    paths: list[str] = []

    def collect(name: str, node: Any) -> None:
        if isinstance(node, h5py.Dataset) and name != META_DATASET:
            paths.append(name)

    f.visititems(collect)
    return sorted(paths)


def _read_dims(dataset: h5py.Dataset) -> tuple[str, ...]:
    if "dims" not in dataset.attrs:
        return ()
    raw = np.atleast_1d(dataset.attrs["dims"])
    return tuple(item.decode() if isinstance(item, bytes) else str(item) for item in raw)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
