"""Launches the autapsic-inhibition sweeps defined in sweep.py / size_scan.py.

One invocation is one launch: `run_store.begin_launch` refuses a dirty git tree, creates
`results/{launch_id}/` and appends the launch to the lab book; each config (with all its seeds)
then runs as one job -- in-process, or on `--workers N` spawned processes -- writing one `.h5`
per run. MLflow is an optional live viewer (`--no-mlflow` to drop it). See old-lab-book.md,
"Implemented tests", and scan-list.md.
"""

import argparse
import dataclasses
import multiprocessing
import random
import shlex
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import incremental_scan
import size_scan
import sweep
from data import NUM_CLASSES, DatasetName
from experiment_jobs import (
    Job,
    JobOutcome,
    LaunchError,
    StageBuilder,
    build_seed_variant,
    build_stages_for,
    init_worker,
    run_job,
)
from run_config import MlflowSettings, RunConfig
from run_store import (
    DirtyTreeError,
    Launch,
    LaunchExistsError,
    LaunchSpec,
    begin_launch,
    require_clean_tree,
)

__all__ = [
    "LaunchEnv",
    "LaunchRequest",
    "execute_jobs",
    "main",
    "plan_jobs",
    "run_launch",
    "run_size_scan",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
# Set number of seed replicates per config for variance estimates to default 1.
NUM_SEEDS = 1
RESULTS_DIR = REPO_ROOT / "results"
LAB_BOOK = REPO_ROOT / "docs" / "lab-book.md"
# Draws span [0, _SEED_UPPER_BOUND) -- generous enough that collisions between
# distinct draws are practically impossible without needing dedup logic.
_SEED_UPPER_BOUND = 2**31
_ATTEMPTS = 2  # the first run plus one retry


SCAN_BUILDERS: dict[str, Callable[[], list[RunConfig]]] = {
    "sweep": sweep.build_configs,
    "size": size_scan.build_size_scan_configs,
    "weights": sweep.build_weight_scan_configs,
}
# Scans that need extra command-line parameters, built by `_build_scan_configs`.
PARAMETRIC_SCANS = ("width", "incremental")


def _default_mlflow_settings() -> MlflowSettings:
    return MlflowSettings(
        tracking_uri=f"sqlite:///{REPO_ROOT / 'mlflow.db'}",
        artifact_location=f"file:{REPO_ROOT / 'mlartifacts'}",
    )


@dataclass(frozen=True)
class LaunchEnv:
    """Where a launch reads and writes; defaults are the repo's, tests inject temporary ones.

    Everything a worker needs is copied into its `Job`, because module-level state does not
    reach spawned processes.
    """

    results_root: Path = RESULTS_DIR
    lab_book: Path = LAB_BOOK
    repo_root: Path = REPO_ROOT
    mlflow_settings: MlflowSettings = field(default_factory=_default_mlflow_settings)
    stage_builder: StageBuilder = build_stages_for


@dataclass(frozen=True)
class LaunchRequest:
    """What the user asked for; the text fields end up verbatim in the lab book."""

    goal: str
    method: str
    code_changes: str
    num_seeds: int = NUM_SEEDS
    dataset: DatasetName = "mnist"
    initial_classes: int = 5
    classes_per_stage: int = 1
    workers: int = 1
    use_mlflow: bool = True
    command: str = ""
    epochs_per_stage: int | None = None
    train_fraction: float = 1.0
    split_seeds: bool = False


def main(
    *,
    goal: str,
    method: str,
    code_changes: str,
    num_seeds: int = NUM_SEEDS,
    dataset: DatasetName = "mnist",
    initial_classes: int = 5,
    classes_per_stage: int = 1,
    workers: int = 1,
    use_mlflow: bool = True,
    command: str | None = None,
    env: LaunchEnv | None = None,
) -> None:
    """Launches every config in sweep.build_configs().

    Args:
        goal, method, code_changes: Required free text, copied into the lab book entry.
        num_seeds: Seed replicates per config, see `_draw_seeds`.
        dataset: "mnist" (default) or "fashion_mnist".
        initial_classes: How many classes stage 0 introduces, see `data.build_stage_digits`.
        classes_per_stage: How many new classes each later stage introduces.
        workers: Parallel worker processes; 1 runs in-process.
        use_mlflow: Live-log to MLflow as well.
        command: The command line to record; defaults to a description of this call.
        env: Paths and injectables, see `LaunchEnv`.

    Raises:
        LaunchError: some job failed twice or MLflow logging failed (results are kept).
    """
    request = _make_request(locals(), command, "main")
    run_launch(sweep.build_configs(), request, "sweep", env or LaunchEnv())


def run_size_scan(
    *,
    goal: str,
    method: str,
    code_changes: str,
    num_seeds: int = NUM_SEEDS,
    dataset: DatasetName = "mnist",
    initial_classes: int = 5,
    classes_per_stage: int = 1,
    workers: int = 1,
    use_mlflow: bool = True,
    command: str | None = None,
    env: LaunchEnv | None = None,
    hidden_sizes_scan: list[tuple[int, ...]] | None = None,
    channels_scan: list[tuple[int, int]] | None = None,
) -> None:
    """Launches the shrinking-network scan from size_scan.py.

    Same args as `main`, plus `hidden_sizes_scan`/`channels_scan` (see
    `size_scan.build_size_scan_configs`) to override the scan's widths;
    `None` (default) uses `size_scan`'s own defaults for both.
    """
    request = _make_request(locals(), command, "run_size_scan")
    configs = size_scan.build_size_scan_configs(
        hidden_sizes_scan if hidden_sizes_scan is not None else size_scan.FCNN_HIDDEN_SIZES_SCAN,
        channels_scan if channels_scan is not None else size_scan.CNN_CHANNELS_SCAN,
    )
    run_launch(configs, request, "size", env or LaunchEnv())


def _make_request(arguments: dict, command: str | None, function: str) -> LaunchRequest:
    """The `LaunchRequest` for a `main`/`run_size_scan` call, given that function's `locals()`.

    Without an explicit `command`, the recorded command is a description of the call.
    """
    fields = {f.name for f in dataclasses.fields(LaunchRequest)} - {"command"}
    values = {key: arguments[key] for key in fields if key in arguments}
    if command is None:
        shown = dict(values)
        for key in ("hidden_sizes_scan", "channels_scan"):
            if arguments.get(key) is not None:
                shown[key] = arguments[key]
        call = ", ".join(f"{key}={value!r}" for key, value in sorted(shown.items()))
        command = f"run_experiments.{function}({call})"
    return LaunchRequest(**values, command=command)


def run_launch(configs: list[RunConfig], request: LaunchRequest, scan: str, env: LaunchEnv) -> None:
    """Runs one launch: validate, plan, refuse early, begin the launch, execute, report.

    Everything that can refuse (bad arguments, blank text, dirty tree, unbuildable stages,
    unusable MLflow backend) happens before the launch directory or the lab-book entry exist.
    This is also the entry point for injected config lists (tests, ad-hoc scans).

    Raises:
        ValueError: bad arguments or blank text.
        DirtyTreeError: the git tree is dirty.
        LaunchError: some job failed twice or MLflow logging failed.
    """
    if request.workers < 1:
        raise ValueError(f"workers must be >= 1, got {request.workers}")
    if not configs:
        raise ValueError("Nothing to run: the config list is empty.")
    plan = plan_jobs(configs, request)
    spec = _make_spec(request, scan, planned_runs=sum(len(seeds) for _, seeds in plan))
    spec.validate()
    require_clean_tree(env.repo_root, env.lab_book)
    env.stage_builder(request.dataset, request.initial_classes, request.classes_per_stage)
    settings = _prepare_mlflow(request, env)
    launch = begin_launch(
        spec, results_root=env.results_root, lab_book=env.lab_book, repo_root=env.repo_root
    )
    jobs = _make_jobs(plan, launch, settings, env, request)
    print(f"Launch {launch.launch_id}: {spec.planned_runs} runs -> {launch.directory}")
    _raise_if_problems(execute_jobs(jobs, request.workers))
    print(f"Launch {launch.launch_id}: done.")


def _make_jobs(
    plan: list[tuple[RunConfig, tuple[int, ...]]],
    launch: Launch,
    settings: MlflowSettings | None,
    env: LaunchEnv,
    request: LaunchRequest,
) -> list[Job]:
    """One job per config, or one per config and seed with `request.split_seeds`.

    Splitting gives a launch with few configs enough jobs to keep every worker busy; split jobs
    carry the total seed count so run names keep their seed suffix, and they are not grouped
    into an MLflow parent run.
    """
    jobs = []
    for config, seeds in plan:
        groups = [(seed,) for seed in seeds] if request.split_seeds else [seeds]
        for group in groups:
            jobs.append(
                Job(
                    config,
                    group,
                    launch,
                    mlflow=settings,
                    stage_builder=env.stage_builder,
                    verbose=request.workers == 1,
                    num_seeds=len(seeds) if request.split_seeds else None,
                )
            )
    return jobs


def plan_jobs(
    configs: list[RunConfig], request: LaunchRequest
) -> list[tuple[RunConfig, tuple[int, ...]]]:
    """Pairs each config (with the launch's dataset and class split set) with its drawn seeds.

    Seeds are drawn here, once, so a retried job reruns the same seeds.

    Raises:
        ValueError: if two planned runs would share a file name.
    """
    plan = []
    names: set[str] = set()
    for config in configs:
        full = dataclasses.replace(
            config,
            dataset=request.dataset,
            initial_classes=request.initial_classes,
            classes_per_stage=request.classes_per_stage,
            num_classes=NUM_CLASSES[request.dataset],
            shuffle_class_order=request.dataset == "mixed",
            train_fraction=request.train_fraction,
        )
        if request.epochs_per_stage is not None:
            full = dataclasses.replace(full, epochs_per_stage=request.epochs_per_stage)
        seeds = tuple(_draw_seeds(request.num_seeds))
        for seed in seeds:
            name = build_seed_variant(full, seed, len(seeds)).name
            if name in names:
                raise ValueError(f"Two planned runs would both be named {name!r}.")
            names.add(name)
        plan.append((full, seeds))
    return plan


def _make_spec(request: LaunchRequest, scan: str, planned_runs: int) -> LaunchSpec:
    seeds_mode = (
        f"{request.num_seeds} random seeds per config"
        if request.num_seeds > 1
        else "single seed 0 per config"
    )
    return LaunchSpec(
        goal=request.goal,
        method=request.method,
        code_changes=request.code_changes,
        command=request.command,
        scan=scan,
        planned_runs=planned_runs,
        dataset=request.dataset,
        seeds_mode=seeds_mode,
        initial_classes=request.initial_classes,
        classes_per_stage=request.classes_per_stage,
        workers=request.workers,
        use_mlflow=request.use_mlflow,
    )


def _prepare_mlflow(request: LaunchRequest, env: LaunchEnv) -> MlflowSettings | None:
    """MLflow settings for the jobs, configured once here before any worker exists.

    First use of a fresh MLflow database runs schema migrations that fail when several worker
    processes race to do it, and a bad artifact location should fail the launch, not every job.
    """
    if not request.use_mlflow:
        return None
    import logged_training  # deferred: --no-mlflow must not import mlflow

    logged_training.configure(env.mlflow_settings)
    if request.workers > 1:
        return logged_training.with_busy_timeout(env.mlflow_settings)
    return env.mlflow_settings


def execute_jobs(jobs: list[Job], workers: int) -> list[JobOutcome]:
    """Runs `jobs`, retrying each failed job once; returns the final outcome per job.

    The retry round skips seeds whose `.h5` exists. Swallowed MLflow errors are collected
    across both rounds and not retried (a retry cannot repair them).
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    final: dict[str, JobOutcome] = {}
    mlflow_errors: dict[str, list[str]] = {}
    pending = list(jobs)
    for attempt in range(_ATTEMPTS):
        outcomes = _run_round(pending, workers, skip_existing=attempt > 0)
        for name, outcome in outcomes.items():
            mlflow_errors.setdefault(name, []).extend(outcome.mlflow_errors)
            final[name] = outcome
        pending = [job for job in pending if outcomes[job.key].error is not None]
        if not pending:
            break
    return [
        JobOutcome(name, outcome.error, tuple(mlflow_errors[name]))
        for name, outcome in final.items()
    ]


def _run_round(jobs: list[Job], workers: int, skip_existing: bool) -> dict[str, JobOutcome]:
    """One pass over `jobs`: in-process if `workers == 1`, else on spawned worker processes.

    The first pass shares one pool. The retry pass (`skip_existing`) gives each job its own
    single-worker pool, `workers` at a time: a worker that dies (OOM kill, segfault) makes the
    executor fail every unfinished job, so a shared retry pool would let one crasher take all
    the others down a second time.
    """
    if workers == 1:
        return {job.key: run_job(job, skip_existing) for job in jobs}
    max_workers = min(workers, len(jobs))
    if skip_existing:
        with ThreadPoolExecutor(max_workers=max_workers) as threads:
            return _collect({threads.submit(_run_in_own_process, job): job for job in jobs})
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
    ) as pool:
        return _collect({pool.submit(run_job, job, False): job for job in jobs})


def _collect(futures: dict[Future[JobOutcome], Job]) -> dict[str, JobOutcome]:
    """Waits for the futures, printing one progress line per finished job."""
    outcomes: dict[str, JobOutcome] = {}
    for done, future in enumerate(as_completed(futures), start=1):
        job = futures[future]
        outcome = _outcome_of(future, job)
        outcomes[job.key] = outcome
        status = "FAILED" if outcome.error else "ok"
        print(f"[{done}/{len(futures)}] {job.key}: {status}", flush=True)
    return outcomes


def _run_in_own_process(job: Job) -> JobOutcome:
    """Retries `job` in a fresh single-worker spawned pool, isolated from other jobs."""
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker
    ) as pool:
        return _outcome_of(pool.submit(run_job, job, True), job)


def _outcome_of(future: Future[JobOutcome], job: Job) -> JobOutcome:
    """The job's outcome, or a failure outcome if its worker died (BrokenProcessPool)."""
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 -- any worker failure becomes a reported outcome
        return JobOutcome(job.key, f"worker failed: {type(exc).__name__}: {exc}")


def _raise_if_problems(outcomes: list[JobOutcome]) -> None:
    failed = {o.name: o.error for o in outcomes if o.error is not None}
    mlflow_errors = {o.name: list(o.mlflow_errors) for o in outcomes if o.mlflow_errors}
    if failed or mlflow_errors:
        raise LaunchError(failed, mlflow_errors)


def _draw_seeds(num_seeds: int) -> list[int]:
    """Draws `num_seeds` distinct random seeds for reproducible multi-seed runs.

    `num_seeds == 1` stays deterministic (`[0]`, matching this repo's
    single-seed behavior before random seeds existed) -- only opting into
    multiple replicates opts into randomized seeds. The drawn seeds are recorded in every
    run file's `meta["config"]["seed"]` and its name.

    Raises:
        ValueError: if `num_seeds < 1` -- fails fast rather than silently
            running zero seeds (an empty seed list would otherwise make a job
            do nothing: no training, no results, no error, just a misleadingly
            "successful" no-op run).
    """
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    if num_seeds == 1:
        return [0]
    rng = random.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < num_seeds:
        seeds.add(rng.randrange(_SEED_UPPER_BOUND))
    return sorted(seeds)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return value


def _build_scan_configs(args: argparse.Namespace) -> list[RunConfig]:
    """The configs of the scan named in `args`.

    Raises:
        ValueError: if the 'incremental' scan is chosen without `--width`.
    """
    if args.scan == "width":
        return incremental_scan.build_width_pilot_configs(tuple(args.widths))
    if args.scan == "incremental":
        if args.width is None:
            raise ValueError("--scan incremental needs --width")
        return incremental_scan.build_incremental_configs(args.width, tuple(args.leaky_decays))
    return SCAN_BUILDERS[args.scan]()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        choices=[*SCAN_BUILDERS, *PARAMETRIC_SCANS],
        default="sweep",
        help="'sweep' (default): sweep.py's 14 configs. 'size': size_scan.py's "
        "shrinking-network scan. 'weights': sweep.py's ai2_weight scan (28 configs). "
        "'width': FCNN and CNN controls over --widths (size pilot). 'incremental': controls, "
        "unrolled ai2_gate and leaky layers at --width.",
    )
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--dataset", choices=list(NUM_CLASSES), default="mnist")
    parser.add_argument("--initial-classes", type=int, default=5)
    parser.add_argument("--classes-per-stage", type=int, default=1)
    parser.add_argument(
        "--epochs-per-stage",
        type=_positive_int,
        default=None,
        help="Override every config's epochs per stage (coarse runs use 1).",
    )
    parser.add_argument(
        "--train-fraction",
        type=_fraction,
        default=1.0,
        help="Keep this fraction of every stage's training examples (coarse runs only).",
    )
    parser.add_argument(
        "--split-seeds",
        action="store_true",
        help="Run every seed as its own job, so launches with few configs use all workers.",
    )
    parser.add_argument(
        "--widths",
        type=_positive_int,
        nargs="+",
        default=list(incremental_scan.PILOT_WIDTHS),
        help="Hidden widths of the 'width' scan.",
    )
    parser.add_argument(
        "--width", type=_positive_int, default=None, help="Hidden width of the 'incremental' scan."
    )
    parser.add_argument(
        "--leaky-decays",
        type=float,
        nargs="+",
        default=list(incremental_scan.DEFAULT_LEAKY_DECAYS),
        help="Decays of the leaky layers in the 'incremental' scan.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Parallel worker processes (default 1: sequential, in-process).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Do not import or configure MLflow at all."
    )
    for flag, what in (
        ("--goal", "why this launch is run"),
        ("--method", "how it is run"),
        ("--code-changes", "what source changes it includes"),
    ):
        parser.add_argument(
            flag, required=True, help=f"Required lab-book text: {what}. Copied verbatim."
        )
    return parser.parse_args()


def _cli() -> int:
    """Parses the command line, launches, and returns the process exit code.

    Exit 0: all runs written and logged. 1: a job failed twice, or MLflow logging failed (the
    `.h5` results of everything that finished are kept). 2: refused before starting.
    """
    args = _parse_args()
    request = LaunchRequest(
        goal=args.goal,
        method=args.method,
        code_changes=args.code_changes,
        num_seeds=args.num_seeds,
        dataset=args.dataset,
        initial_classes=args.initial_classes,
        classes_per_stage=args.classes_per_stage,
        workers=args.workers,
        use_mlflow=not args.no_mlflow,
        command="python " + shlex.join(sys.argv),
        epochs_per_stage=args.epochs_per_stage,
        train_fraction=args.train_fraction,
        split_seeds=args.split_seeds,
    )
    try:
        configs = _build_scan_configs(args)
        run_launch(configs, request, args.scan, LaunchEnv())
    except (ValueError, DirtyTreeError, LaunchExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except LaunchError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
