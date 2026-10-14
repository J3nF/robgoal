"""Experiment-run configuration. See old-lab-book.md, "Logging: MLflow".

Split out from `run_experiments.py` so `logged_training.py` can depend on
`RunConfig` without either of those two modules importing each other.
"""

from dataclasses import dataclass
from typing import Any, Literal

from autapse import Activation_Scheme, AI2_Variant
from data import DatasetName

ModelKind = Literal["cnn", "fcnn"]


@dataclass
class RunConfig:
    """One sweep entry: a model config plus everything worth logging about it.

    Attributes:
        name: Run name, also the stem of the run's `.h5` file in its launch directory.
        sweep_axis: Which dimension this config varies relative to the anchor
            config ("base", "T", "epochs_per_stage", "hidden_sizes"); logged
            as an MLflow tag, not a param.
        model_kind: Discriminates which model `build_model` constructs.
            Explicit rather than inferred from `ai2_variant is None`, since
            the latter would silently accept a half-specified FCNN config.
        dataset: Which dataset this run trains/evaluates on ("mnist" or
            "fashion_mnist"). Logged like any other param so switching
            datasets never silently collides with a same-named run on the
            other dataset -- see `run_experiments._build_seed_variant`.
        initial_classes: How many classes stage 0 introduces, see
            `data.build_stage_digits`. Recorded so a 1-class-start run is never
            confused with a 5-class-start run of the same config.
        classes_per_stage: How many new classes each later stage introduces.
        ai2_variant: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        activation_scheme: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        T: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        hidden_sizes: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        channels: `None` for `model_kind="fcnn"`; optional for `"cnn"` (falls
            back to `models.CNN`'s own default `(16, 32)` when unset).
        capture_y_self_per_epoch: Log a per-neuron y_self history artifact,
            one snapshot per epoch. See `dynamics_insights.get_y_self_per_neuron`.
        capture_y_self_per_batch: Log a per-neuron y_self history artifact,
            one snapshot per training batch. Opt-in: on a full sweep this can
            accumulate a large number of records.
    """

    name: str
    sweep_axis: str
    model_kind: ModelKind
    epochs_per_stage: int
    dataset: DatasetName = "mnist"
    initial_classes: int = 5
    classes_per_stage: int = 1
    seed: int = 0
    lr: float = 0.1
    batch_size: int = 128
    ai2_variant: AI2_Variant | None = None
    activation_scheme: Activation_Scheme | None = None
    T: int | None = None
    hidden_sizes: tuple[int, ...] | None = None
    channels: tuple[int, int] | None = None
    capture_y_self_per_epoch: bool = False
    capture_y_self_per_batch: bool = False


def config_from_dict(fields: dict[str, Any]) -> RunConfig:
    """Rebuilds a `RunConfig` from its JSON form (e.g. a run file's `meta["config"]`).

    JSON turns the tuple fields (`hidden_sizes`, `channels`) into lists; this restores them,
    since `sweep.build_model` and equality checks expect tuples.
    """
    restored: dict[str, Any] = {
        key: tuple(value) if isinstance(value, list) else value for key, value in fields.items()
    }
    return RunConfig(**restored)


@dataclass(frozen=True)
class MlflowSettings:
    """Where the optional MLflow viewer logs to; carried inside jobs sent to worker processes.

    Attributes:
        tracking_uri: MLflow tracking backend, e.g. `sqlite:///{repo}/mlflow.db`.
        artifact_location: Artifact root given to the experiment the one time it is created.
        experiment_name: MLflow experiment all runs go into.
    """

    tracking_uri: str
    artifact_location: str
    experiment_name: str = "ai2-mnist"
