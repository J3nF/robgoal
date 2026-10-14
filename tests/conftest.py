"""Shared fixtures: a throwaway git repo to run launch/provenance code against."""

import os
import subprocess
from pathlib import Path

import matplotlib
import pytest

import run_store

matplotlib.use("Agg")  # no test may open a window


def run_git(repo: Path, *args: str) -> str:
    """Runs git in `repo` with a fixed identity and without the caller's GIT_* variables."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A clean git repo with one commit: tracked `src/a.py` and `docs/lab-book.md`.

    `results/` is gitignored, like in the real repo.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "docs" / "lab-book.md").write_text("# Lab book\n")
    (repo / ".gitignore").write_text("results/\n")
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def launch(git_repo: Path, tmp_path: Path):
    """A started launch in `git_repo` (results under `tmp_path/results`, lab book in the repo)."""
    return run_store.begin_launch(
        run_store.LaunchSpec(
            goal="test goal",
            method="test method",
            code_changes="none",
            command="pytest",
            scan="sweep",
            planned_runs=1,
            dataset="mnist",
            seeds_mode="single seed 0 per config",
        ),
        results_root=tmp_path / "results",
        lab_book=git_repo / "docs" / "lab-book.md",
        repo_root=git_repo,
    )
