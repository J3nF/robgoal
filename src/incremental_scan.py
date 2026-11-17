"""Configs for the incremental-inhibition experiments: the width pilot and the main scan.

Pure config definitions, like `sweep.py` and `size_scan.py`: no I/O, no torch training.
All models have four equal-width hidden layers of width `W` (`WEIGHT_SCAN_HIDDEN_SIZES`'s shape),
so per-layer plots have several layers.

CNN matching convention (reused from `size_scan.py`): that scan pairs each FCNN width step with
CNN channels `(c, 2c)` by step index, which is not a parameter-count match.
Keyed by the last FCNN width, the pairs give `c2` = 32, 24, 16, 12, 8, 4, 2, 1 for widths
128, 64, 32, 16, 8, 4, 2, 1.
`matched_channels(W)` interpolates `c2` log-log between those table points (and extends the
last segment beyond them), rounds it, and sets `c1 = max(1, round(c2 / 2))`.
"""

import math

from run_config import RunConfig
from size_scan import CNN_CHANNELS_SCAN, FCNN_HIDDEN_SIZES_SCAN
from sweep import BASE_EPOCHS_PER_STAGE, SCHEMES, WEIGHT_SCAN_T, weight_label

__all__ = [
    "DEFAULT_LEAKY_DECAYS",
    "PILOT_WIDTHS",
    "build_incremental_configs",
    "build_width_pilot_configs",
    "matched_channels",
]

PILOT_WIDTHS = (2, 4, 8, 16, 32, 64)
DEFAULT_LEAKY_DECAYS = (0.9,)


def matched_channels(width: int) -> tuple[int, int]:
    """CNN channels `(c1, c2)` matched to FCNN hidden width `width`, see the module docstring.

    Raises:
        ValueError: if `width < 1`.
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    points = sorted(
        (hidden[-1], channels[1])
        for hidden, channels in zip(FCNN_HIDDEN_SIZES_SCAN, CNN_CHANNELS_SCAN, strict=True)
    )
    if width <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    else:
        index = next((i for i, (x, _) in enumerate(points) if x >= width), len(points) - 1)
        (x0, y0), (x1, y1) = points[max(index - 1, 0)], points[max(index, 1)]
    slope = math.log(y1 / y0) / math.log(x1 / x0)
    c2 = max(1, round(y0 * (width / x0) ** slope))
    return max(1, round(c2 / 2)), c2


def build_width_pilot_configs(
    widths: tuple[int, ...] = PILOT_WIDTHS, epochs_per_stage: int = 1
) -> list[RunConfig]:
    """The size pilot: both FCNN controls (`T=1`) and the matched CNN control at each width."""
    configs = []
    for width in widths:
        configs.extend(_build_controls(f"width{width}", width, epochs_per_stage, "width"))
    return configs


def build_incremental_configs(
    width: int,
    decays: tuple[float, ...] = DEFAULT_LEAKY_DECAYS,
    epochs_per_stage: int = BASE_EPOCHS_PER_STAGE,
) -> list[RunConfig]:
    """The main scan at one width: controls, unrolled `ai2_gate`, and leaky layers.

    Per activation scheme: the FCNN control, the unrolled `ai2_gate` (`T=5`, weight 1) and one
    leaky layer per decay; plus the CNN control. So every run has its controls in its launch.
    """
    hidden_sizes = (width,) * 4
    configs = _build_controls("inc", width, epochs_per_stage, "incremental")
    for scheme in SCHEMES:
        configs.append(
            RunConfig(
                name=f"inc_ai2_gate_{scheme}_T{WEIGHT_SCAN_T}",
                sweep_axis="incremental",
                model_kind="fcnn",
                epochs_per_stage=epochs_per_stage,
                ai2_variant="ai2_gate",
                activation_scheme=scheme,
                T=WEIGHT_SCAN_T,
                hidden_sizes=hidden_sizes,
            )
        )
        for decay in decays:
            configs.append(
                RunConfig(
                    name=f"inc_leaky_{scheme}_lam{weight_label(decay)}",
                    sweep_axis="incremental",
                    model_kind="fcnn",
                    epochs_per_stage=epochs_per_stage,
                    ai2_variant="ai2_gate",
                    activation_scheme=scheme,
                    T=1,
                    hidden_sizes=hidden_sizes,
                    layer_kind="leaky",
                    decay=decay,
                )
            )
    return configs


def _build_controls(
    prefix: str, width: int, epochs_per_stage: int, sweep_axis: str
) -> list[RunConfig]:
    """The CNN control with matched channels and the two FCNN controls (`T=1`) at `width`."""
    channels = matched_channels(width)
    configs = [
        RunConfig(
            name=f"{prefix}_cnn_control_c{channels[0]}x{channels[1]}",
            sweep_axis=sweep_axis,
            model_kind="cnn",
            epochs_per_stage=epochs_per_stage,
            channels=channels,
        )
    ]
    for scheme in SCHEMES:
        configs.append(
            RunConfig(
                name=f"{prefix}_fcnn_control_{scheme}_hidden{width}",
                sweep_axis=sweep_axis,
                model_kind="fcnn",
                epochs_per_stage=epochs_per_stage,
                ai2_variant="ai2_gate",
                activation_scheme=scheme,
                T=1,
                hidden_sizes=(width,) * 4,
            )
        )
    return configs
