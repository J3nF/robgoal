"""Defines the autapsic-inhibition experiment sweep: what configs exist, and
how to build the model each one describes. See lab-book.md, "Implemented
tests"."""

from torch import nn

from autapse import Activation_Scheme, AI2_Variant
from models import CNN, FCNN_AI2
from run_config import RunConfig

VARIANTS: list[AI2_Variant] = ["ai2_diff", "ai2_gate"]
SCHEMES: list[Activation_Scheme] = ["sigmoid", "relu_sigmoid"]
BASE_T = 3
BASE_EPOCHS_PER_STAGE = 5
BASE_HIDDEN_SIZES = (256, 128)
ANCHOR_VARIANT: AI2_Variant = "ai2_diff"
ANCHOR_SCHEME: Activation_Scheme = "relu_sigmoid"
T_SWEEP = [2, 5, 8]
EPOCHS_SWEEP = [2, 10]
HIDDEN_SIZES_SWEEP = [(64, 32), (512, 256)]


def build_configs() -> list[RunConfig]:
    """The 14 sweep configs: 7 base tests plus a one-factor-at-a-time sweep.

    The 7 base tests: 3 controls (no self-inhibition) + 2 variants x 2 schemes.
    The remaining 7 vary T, epochs_per_stage, and hidden_sizes one at a time
    off the ai2_diff/relu_sigmoid anchor config.
    """
    return [
        *_build_base_configs(),
        *_build_variant_scheme_grid(),
        *_build_T_sweep(),
        *_build_epochs_sweep(),
        *_build_hidden_sizes_sweep(),
    ]


def _build_base_configs() -> list[RunConfig]:
    """The 3 controls: cnn_control, fcnn_control_relu, fcnn_control_sigmoid (all T=1)."""
    return [
        RunConfig(
            name="cnn_control",
            sweep_axis="base",
            model_kind="cnn",
            epochs_per_stage=BASE_EPOCHS_PER_STAGE,
        ),
        RunConfig(
            name="fcnn_control_relu",
            sweep_axis="base",
            model_kind="fcnn",
            epochs_per_stage=BASE_EPOCHS_PER_STAGE,
            ai2_variant="ai2_diff",
            activation_scheme="relu_sigmoid",
            T=1,
            hidden_sizes=BASE_HIDDEN_SIZES,
        ),
        RunConfig(
            name="fcnn_control_sigmoid",
            sweep_axis="base",
            model_kind="fcnn",
            epochs_per_stage=BASE_EPOCHS_PER_STAGE,
            ai2_variant="ai2_diff",
            activation_scheme="sigmoid",
            T=1,
            hidden_sizes=BASE_HIDDEN_SIZES,
        ),
    ]


def _build_variant_scheme_grid() -> list[RunConfig]:
    """The 2x2 ai2_variant x activation_scheme grid at BASE_T/BASE_EPOCHS_PER_STAGE/BASE_HIDDEN_SIZES."""
    configs = []
    for activation_scheme in SCHEMES:
        for ai2_variant in VARIANTS:
            configs.append(
                RunConfig(
                    name=f"fcnn_{ai2_variant}_{activation_scheme}",
                    sweep_axis="base",
                    model_kind="fcnn",
                    epochs_per_stage=BASE_EPOCHS_PER_STAGE,
                    ai2_variant=ai2_variant,
                    activation_scheme=activation_scheme,
                    T=BASE_T,
                    hidden_sizes=BASE_HIDDEN_SIZES,
                )
            )
    return configs


def _build_T_sweep() -> list[RunConfig]:
    """Sweeps T over T_SWEEP off the anchor config.

    T=1 is skipped: it degenerates to the fcnn_control_relu/fcnn_control_sigmoid
    controls regardless of ai2_variant/activation_scheme, and
    relu_sigmoid+ai2_diff+T=1 is already covered by fcnn_control_relu.
    """
    configs = []
    for T in T_SWEEP:
        configs.append(
            RunConfig(
                name=f"fcnn_{ANCHOR_VARIANT}_{ANCHOR_SCHEME}_T{T}",
                sweep_axis="T",
                model_kind="fcnn",
                epochs_per_stage=BASE_EPOCHS_PER_STAGE,
                ai2_variant=ANCHOR_VARIANT,
                activation_scheme=ANCHOR_SCHEME,
                T=T,
                hidden_sizes=BASE_HIDDEN_SIZES,
            )
        )
    return configs


def _build_epochs_sweep() -> list[RunConfig]:
    """Sweeps epochs_per_stage over EPOCHS_SWEEP off the anchor config."""
    configs = []
    for epochs_per_stage in EPOCHS_SWEEP:
        configs.append(
            RunConfig(
                name=f"fcnn_{ANCHOR_VARIANT}_{ANCHOR_SCHEME}_epochs{epochs_per_stage}",
                sweep_axis="epochs_per_stage",
                model_kind="fcnn",
                epochs_per_stage=epochs_per_stage,
                ai2_variant=ANCHOR_VARIANT,
                activation_scheme=ANCHOR_SCHEME,
                T=BASE_T,
                hidden_sizes=BASE_HIDDEN_SIZES,
            )
        )
    return configs


def _build_hidden_sizes_sweep() -> list[RunConfig]:
    """Sweeps hidden_sizes over HIDDEN_SIZES_SWEEP off the anchor config."""
    configs = []
    for hidden_sizes in HIDDEN_SIZES_SWEEP:
        name_suffix = "x".join(str(size) for size in hidden_sizes)
        configs.append(
            RunConfig(
                name=f"fcnn_{ANCHOR_VARIANT}_{ANCHOR_SCHEME}_hidden{name_suffix}",
                sweep_axis="hidden_sizes",
                model_kind="fcnn",
                epochs_per_stage=BASE_EPOCHS_PER_STAGE,
                ai2_variant=ANCHOR_VARIANT,
                activation_scheme=ANCHOR_SCHEME,
                T=BASE_T,
                hidden_sizes=hidden_sizes,
            )
        )
    return configs


def build_model(config: RunConfig) -> nn.Module:
    """Constructs the model `config` describes.

    Asserts (doesn't gracefully handle) `config.T`/`ai2_variant`/
    `activation_scheme`/`hidden_sizes` being set for `model_kind="fcnn"` --
    `build_configs` always sets all four together, so this is a fail-fast
    guard against a malformed `RunConfig`, not an expected input.
    """
    if config.model_kind == "cnn":
        return CNN(channels=config.channels) if config.channels is not None else CNN()
    assert config.T is not None
    assert config.ai2_variant is not None
    assert config.activation_scheme is not None
    assert config.hidden_sizes is not None
    return FCNN_AI2(
        T=config.T,
        ai2_variant=config.ai2_variant,
        activation_scheme=config.activation_scheme,
        hidden_sizes=config.hidden_sizes,
    )
