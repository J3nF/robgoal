"""Per-layer inhibition statistics: small scalar series recorded during training and testing.

The inhibition signal is the one the weight multiplies: `last_y_self` of an unrolled
`AutapticLayer` with `T > 1`, and `last_inhibition` of a `LeakyLayer`. Controls (`T=1`, CNN)
have no inhibition and record nothing. Only means and variances over a batch (all samples and
neurons) are kept, never per-neuron histories, so run files stay small.
"""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autapse import AutapticLayer, LeakyLayer, reset_inhibition_state
from data import LabeledDataset
from run_store import RunArray

__all__ = [
    "InhibitionRecord",
    "build_inhibition_arrays",
    "collect_test_inhibition",
    "get_inhibition_layers",
    "get_inhibition_signal",
    "record_test_stage",
    "record_train_batch",
]

TEST_BATCH_SIZE = 256


@dataclass
class InhibitionRecord:
    """Per-layer inhibition scalars of one run, keyed by layer name.

    Attributes:
        train_mean: Mean of the signal over each training batch, in batch order of the whole run.
        train_var: Variance (ddof=0) of the signal over each training batch.
        train_stage: Stage index of each training batch, shared by all layers.
        test_mean: Mean of the signal over the test set, one value per stage end.
        test_var: Variance (ddof=0) of the signal over the test set, one value per stage end.
        test_class_mean: Mean of the signal over the test samples of each class, one row per
            stage end; NaN for classes without test samples.
    """

    train_mean: dict[str, list[float]] = field(default_factory=dict)
    train_var: dict[str, list[float]] = field(default_factory=dict)
    train_stage: list[int] = field(default_factory=list)
    test_mean: dict[str, list[float]] = field(default_factory=dict)
    test_var: dict[str, list[float]] = field(default_factory=dict)
    test_class_mean: dict[str, list[list[float]]] = field(default_factory=dict)


def get_inhibition_layers(model: nn.Module) -> dict[str, nn.Module]:
    """The layers of `model` that inhibit, by submodule name; empty for controls."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LeakyLayer) or (isinstance(module, AutapticLayer) and module.T > 1)
    }


def get_inhibition_signal(layer: nn.Module) -> torch.Tensor | None:
    """The layer's last inhibition signal (batch, neurons) before the weight factor.

    `None` while the layer has not run a forward pass yet.
    """
    if isinstance(layer, LeakyLayer):
        return layer.last_inhibition
    return cast(AutapticLayer, layer).last_y_self


def record_train_batch(model: nn.Module, record: InhibitionRecord, stage_index: int) -> None:
    """Appends each inhibiting layer's batch mean and variance (mutates `record`).

    Call right after a training batch; reads the signal of that batch's forward pass.
    """
    layers = get_inhibition_layers(model)
    if not layers:
        return
    record.train_stage.append(stage_index)
    for name, layer in layers.items():
        signal = get_inhibition_signal(layer)
        assert signal is not None, "record_train_batch called before any forward pass"
        record.train_mean.setdefault(name, []).append(signal.mean().item())
        record.train_var.setdefault(name, []).append(signal.var(unbiased=False).item())


@torch.no_grad()
def collect_test_inhibition(
    model: nn.Module, test_data: LabeledDataset, device: torch.device, num_classes: int
) -> dict[str, tuple[float, float, np.ndarray]]:
    """Mean, variance and per-class mean of each inhibiting layer's signal over the test set.

    Runs in test order with batch size `TEST_BATCH_SIZE` after clearing leaky state, so it sees
    the same stream as `metrics.evaluate_per_digit_accuracy`. The CPU RNG is forked because the
    `DataLoader` draws from the global RNG, which would change later training shuffles.
    Classes without test samples get NaN.
    """
    layers = get_inhibition_layers(model)
    if not layers:
        return {}
    model.eval()
    reset_inhibition_state(model, training=False)
    totals = {name: torch.zeros(3, dtype=torch.float64) for name in layers}  # sum, sum sq, count
    class_sums = {name: torch.zeros(num_classes, dtype=torch.float64) for name in layers}
    class_counts = {name: torch.zeros(num_classes, dtype=torch.float64) for name in layers}
    loader = DataLoader(
        cast(Dataset[tuple[torch.Tensor, int]], test_data), batch_size=TEST_BATCH_SIZE
    )
    with torch.random.fork_rng(devices=[]):
        for images, labels in loader:
            model(images.to(device))
            for name, layer in layers.items():
                signal = get_inhibition_signal(layer)
                assert signal is not None
                signal = signal.double().cpu()
                totals[name] += torch.tensor([signal.sum(), (signal**2).sum(), signal.numel()])
                class_sums[name].index_add_(0, labels, signal.sum(dim=1))
                class_counts[name].index_add_(
                    0,
                    labels,
                    torch.full((len(labels),), float(signal.shape[1]), dtype=torch.float64),
                )
    stats = {}
    for name in layers:
        total, total_sq, count = totals[name].tolist()
        mean = total / count
        with np.errstate(invalid="ignore", divide="ignore"):
            class_mean = (class_sums[name] / class_counts[name]).numpy()
        stats[name] = (mean, total_sq / count - mean**2, class_mean)
    return stats


def record_test_stage(
    model: nn.Module,
    record: InhibitionRecord,
    test_data: LabeledDataset,
    device: torch.device,
    num_classes: int,
) -> None:
    """Appends one stage-end test pass of every inhibiting layer to `record` (mutates it)."""
    for name, (mean, var, class_mean) in collect_test_inhibition(
        model, test_data, device, num_classes
    ).items():
        record.test_mean.setdefault(name, []).append(mean)
        record.test_var.setdefault(name, []).append(var)
        record.test_class_mean.setdefault(name, []).append(class_mean.tolist())


def build_inhibition_arrays(record: InhibitionRecord) -> dict[str, RunArray]:
    """The `inhibition/*` datasets of a run file; empty when the model has no inhibition."""
    arrays: dict[str, RunArray] = {}
    if record.train_stage:
        arrays["inhibition/train_batch_stage"] = RunArray(
            np.asarray(record.train_stage, dtype=np.int64), ("batch",)
        )
    for prefix, series, dims in (
        ("train_mean", record.train_mean, ("batch",)),
        ("train_var", record.train_var, ("batch",)),
        ("test_mean", record.test_mean, ("stage",)),
        ("test_var", record.test_var, ("stage",)),
        ("test_class_mean", record.test_class_mean, ("stage", "class")),
    ):
        for layer, values in series.items():
            arrays[f"inhibition/{prefix}/{layer}"] = RunArray(
                np.asarray(values, dtype=np.float64), dims
            )
    return arrays
