"""Experiment-run configuration. See lab-book.md, "Logging: MLflow".

Split out from `run_experiments.py` so `logged_training.py` can depend on
`RunConfig` without either of those two modules importing each other.
"""

from dataclasses import dataclass
from typing import Literal

from autapse import Activation_Scheme, AI2_Variant

ModelKind = Literal["cnn", "fcnn"]


@dataclass
class RunConfig:
    """One sweep entry: a model config plus everything worth logging about it.

    Attributes:
        name: Run name, also used for the results JSON filename.
        sweep_axis: Which dimension this config varies relative to the anchor
            config ("base", "T", "epochs_per_stage", "hidden_sizes"); logged
            as an MLflow tag, not a param.
        model_kind: Discriminates which model `build_model` constructs.
            Explicit rather than inferred from `ai2_variant is None`, since
            the latter would silently accept a half-specified FCNN config.
        ai2_variant: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        activation_scheme: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        T: `None` for `model_kind="cnn"`; required for `"fcnn"`.
        hidden_sizes: `None` for `model_kind="cnn"`; required for `"fcnn"`.
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
    seed: int = 0
    lr: float = 1e-3
    batch_size: int = 128
    ai2_variant: AI2_Variant | None = None
    activation_scheme: Activation_Scheme | None = None
    T: int | None = None
    hidden_sizes: tuple[int, ...] | None = None
    capture_y_self_per_epoch: bool = False
    capture_y_self_per_batch: bool = False
