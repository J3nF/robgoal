"""Tests for seed splitting, coarse overrides and mixed-dataset planning in the launcher."""

import argparse
import dataclasses
from typing import Any

import pytest
from fakes import make_config, make_stages

import experiment_jobs
import run_experiments
from data import MixedDataset  # noqa: F401  (import check: the mixed dataset is public)
from experiment_jobs import get_seed_stages
from run_config import RunConfig
from run_experiments import LaunchRequest


def _request(**overrides: Any) -> LaunchRequest:
    request = LaunchRequest(goal="g", method="m", code_changes="c", num_seeds=3)
    return dataclasses.replace(request, **overrides)


def test_plan_applies_epochs_fraction_and_mixed_settings() -> None:
    plan = run_experiments.plan_jobs(
        [make_config("a")],
        _request(dataset="mixed", epochs_per_stage=2, train_fraction=0.25, initial_classes=1),
    )
    ((config, seeds),) = plan
    assert (config.epochs_per_stage, config.train_fraction) == (2, 0.25)
    assert (config.num_classes, config.shuffle_class_order) == (20, True)
    assert len(seeds) == 3


def test_plan_keeps_defaults_for_the_ten_class_datasets() -> None:
    ((config, _),) = run_experiments.plan_jobs([make_config("a", epochs_per_stage=4)], _request())
    assert (config.epochs_per_stage, config.train_fraction) == (4, 1.0)
    assert (config.num_classes, config.shuffle_class_order) == (10, False)


def test_split_seeds_gives_one_job_per_seed_with_unique_keys_and_suffixed_names(
    git_repo, launch
) -> None:
    config = make_config("a")
    plan: list[tuple[RunConfig, tuple[int, ...]]] = [(config, (11, 22, 33))]
    env = run_experiments.LaunchEnv(repo_root=git_repo)
    jobs = run_experiments._make_jobs(plan, launch, None, env, _request(split_seeds=True))
    assert [job.seeds for job in jobs] == [(11,), (22,), (33,)]
    assert len({job.key for job in jobs}) == 3
    names = [
        experiment_jobs.build_seed_variant(job.config, job.seeds[0], job.num_seeds or 1).name
        for job in jobs
    ]
    assert names == ["a_seed11", "a_seed22", "a_seed33"]


def test_unsplit_jobs_keep_one_job_per_config(git_repo, launch) -> None:
    env = run_experiments.LaunchEnv(repo_root=git_repo)
    (job,) = run_experiments._make_jobs([(make_config("a"), (1, 2))], launch, None, env, _request())
    assert job.seeds == (1, 2) and job.key == "a" and job.num_seeds is None


@pytest.mark.parametrize("text", ["0", "-0.1", "1.5", "abc"])
def test_fraction_argument_rejects_values_outside_zero_one(text: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        run_experiments._fraction(text)


def test_fraction_argument_accepts_one() -> None:
    assert run_experiments._fraction("1") == 1.0


def test_get_seed_stages_is_deterministic_per_seed_and_differs_between_seeds() -> None:
    stages = make_stages(2)
    config = make_config("a", shuffle_class_order=True, num_classes=10)
    one = get_seed_stages(stages, dataclasses.replace(config, seed=1))
    again = get_seed_stages(stages, dataclasses.replace(config, seed=1))
    assert [s.digits_seen for s in one] == [s.digits_seen for s in again]
    assert len(one) == len(stages)


def test_get_seed_stages_without_options_returns_the_stages() -> None:
    stages = make_stages(2)
    assert get_seed_stages(stages, make_config("a")) is stages
