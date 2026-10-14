"""Shrinking-network scan: same model families as `sweep.py`, at progressively
smaller widths. See lab-book.md, "ToDos 2026-08-26 or so" -- ai2 is hypothesized
to handle catastrophic interference better than controls once capacity is
scarce, so this stresses both ai2 and control architectures with less capacity
to see how that gap changes.

Pure config/model-factory definitions, mirroring sweep.py -- no I/O, no
MLflow, no torch training.
"""

from run_config import RunConfig
from sweep import ANCHOR_VARIANT, BASE_EPOCHS_PER_STAGE, BASE_T, SCHEMES, VARIANTS

# Each step shrinks both families by roughly half, down to widths that should
# genuinely overwhelm a 10-class classifier's capacity (FCNN (2, 1), CNN
# (1, 1)) rather than just "smaller". channels_scan is CNN's equivalent of
# hidden_sizes_scan -- both lists must be the same length (validated in
# build_size_scan_configs), but their absolute numbers aren't meant to match:
# CNN's default (16, 32) is already narrow compared to FCNN's default
# (256, 128).
FCNN_HIDDEN_SIZES_SCAN: list[tuple[int, ...]] = [
    (256, 128),
    (128, 64),
    (64, 32),
    (32, 16),
    (16, 8),
    (8, 4),
    (4, 2),
    (2, 1),
]
CNN_CHANNELS_SCAN: list[tuple[int, int]] = [
    (16, 32),
    (12, 24),
    (8, 16),
    (6, 12),
    (4, 8),
    (2, 4),
    (1, 2),
    (1, 1),
]


def build_size_scan_configs(
    hidden_sizes_scan: list[tuple[int, ...]] | None = None,
    channels_scan: list[tuple[int, int]] | None = None,
) -> list[RunConfig]:
    """Builds one shrinking-network scan step per `(hidden_sizes, channels)` pair.

    Each step emits: the CNN control at `channels`, the 2 FCNN controls
    (`T=1`) at `hidden_sizes`, and the `ai2_variant` x `activation_scheme`
    grid (`T=sweep.BASE_T`) at `hidden_sizes` -- i.e. every family in
    `sweep.build_configs()`'s base tests, just at a shrinking width.

    Names are prefixed `sizescan_step{i}_` (`i` = position in the scan) so
    they can never collide with `sweep.build_configs()`'s own names, which
    also use raw hidden-size numbers in their own names (e.g. both scans can
    include a `(64, 32)` step).

    Args:
        hidden_sizes_scan: FCNN widths per step, `None` (default) uses
            `FCNN_HIDDEN_SIZES_SCAN`.
        channels_scan: CNN channels per step, `None` (default) uses
            `CNN_CHANNELS_SCAN`.
    """
    if hidden_sizes_scan is None:
        hidden_sizes_scan = FCNN_HIDDEN_SIZES_SCAN
    if channels_scan is None:
        channels_scan = CNN_CHANNELS_SCAN
    if len(hidden_sizes_scan) != len(channels_scan):
        raise ValueError(
            f"hidden_sizes_scan ({len(hidden_sizes_scan)} steps) and channels_scan "
            f"({len(channels_scan)} steps) must have the same number of steps"
        )
    configs = []
    for step, (hidden_sizes, channels) in enumerate(zip(hidden_sizes_scan, channels_scan)):
        prefix = f"sizescan_step{step}"
        configs.append(_build_cnn_step(prefix, channels))
        configs.extend(_build_fcnn_controls_step(prefix, hidden_sizes))
        configs.extend(_build_fcnn_ai2_grid_step(prefix, hidden_sizes))
    return configs


def _build_cnn_step(prefix: str, channels: tuple[int, int]) -> RunConfig:
    """The CNN control at `channels`."""
    return RunConfig(
        name=f"{prefix}_cnn_control_c{channels[0]}x{channels[1]}",
        sweep_axis="size_scan",
        model_kind="cnn",
        epochs_per_stage=BASE_EPOCHS_PER_STAGE,
        channels=channels,
    )


def _build_fcnn_controls_step(prefix: str, hidden_sizes: tuple[int, ...]) -> list[RunConfig]:
    """The 2 FCNN controls (relu, sigmoid; both T=1) at `hidden_sizes`."""
    name_suffix = "x".join(str(size) for size in hidden_sizes)
    configs = []
    for activation_scheme in SCHEMES:
        configs.append(
            RunConfig(
                name=f"{prefix}_fcnn_control_{activation_scheme}_hidden{name_suffix}",
                sweep_axis="size_scan",
                model_kind="fcnn",
                epochs_per_stage=BASE_EPOCHS_PER_STAGE,
                ai2_variant=ANCHOR_VARIANT,
                activation_scheme=activation_scheme,
                T=1,
                hidden_sizes=hidden_sizes,
            )
        )
    return configs


def _build_fcnn_ai2_grid_step(prefix: str, hidden_sizes: tuple[int, ...]) -> list[RunConfig]:
    """The 2x2 ai2_variant x activation_scheme grid (T=BASE_T) at `hidden_sizes`."""
    name_suffix = "x".join(str(size) for size in hidden_sizes)
    configs = []
    for activation_scheme in SCHEMES:
        for ai2_variant in VARIANTS:
            configs.append(
                RunConfig(
                    name=f"{prefix}_fcnn_{ai2_variant}_{activation_scheme}_hidden{name_suffix}",
                    sweep_axis="size_scan",
                    model_kind="fcnn",
                    epochs_per_stage=BASE_EPOCHS_PER_STAGE,
                    ai2_variant=ai2_variant,
                    activation_scheme=activation_scheme,
                    T=BASE_T,
                    hidden_sizes=hidden_sizes,
                )
            )
    return configs
