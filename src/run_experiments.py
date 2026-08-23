"""Runs the 7 autapsic-inhibition tests. See lab-book.md, "Implemented tests"."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from autapse import Activation_Scheme, AI2_Variant
from data import STAGE_DIGITS, build_stages, get_mnist_data
from metrics import average_accuracy, backward_transfer, format_accuracy_matrix
from models import CNN, FCNN_AI2
from train import run_incremental_training

VARIANTS: list[AI2_Variant] = ["ai2_diff", "ai2_gate"]
SCHEMES: list[Activation_Scheme] = ["sigmoid", "relu_sigmoid"]
T = 3
EPOCHS_PER_STAGE = 5
RESULTS_DIR = Path("../results")


def build_configs() -> list[tuple[str, Callable[[], nn.Module]]]:
    """The 7 tests: 3 controls (no self-inhibition) + 2 variants x 2 schemes."""
    configs: list[tuple[str, Callable[[], nn.Module]]] = [
        ("cnn_control", CNN),
        # Skip ai2 inhibition by setting T=1
        (
            "fcnn_control_relu",
            lambda: FCNN_AI2(T=1, ai2_variant="ai2_diff", activation_scheme="relu_sigmoid"),
        ),
        (
            "fcnn_control_sigmoid",
            lambda: FCNN_AI2(T=1, ai2_variant="ai2_diff", activation_scheme="sigmoid"),
        ),
    ]
    for activation_scheme in SCHEMES:
        for ai2_variant in VARIANTS:
            name = f"fcnn_{ai2_variant}_{activation_scheme}"
            configs.append(
                (
                    name,
                    lambda ai2_variant=ai2_variant, activation_scheme=activation_scheme: FCNN_AI2(
                        T=T, ai2_variant=ai2_variant, activation_scheme=activation_scheme
                    ),
                )
            )
    return configs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = get_mnist_data()
    stages = build_stages(train_data, test_data)
    first_stage_digits = frozenset(STAGE_DIGITS[0])
    RESULTS_DIR.mkdir(exist_ok=True)
    configs = build_configs()

    for name, model_fn in configs:
        model = model_fn()
        accuracy_by_stage = run_incremental_training(model, stages, device, EPOCHS_PER_STAGE)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(accuracy_by_stage, indent=2))
        avg_acc = average_accuracy(accuracy_by_stage[-1], stages[-1].digits_seen)
        bwt = backward_transfer(accuracy_by_stage, first_stage_digits)
        print(f"{name}: avg_accuracy={avg_acc:.4f} backward_transfer={bwt:+.4f}")
        print(format_accuracy_matrix(accuracy_by_stage))


if __name__ == "__main__":
    main()
