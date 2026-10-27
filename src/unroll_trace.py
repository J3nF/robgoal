"""Replays the unroll of `AutapticLayer`s step by step, to see how inhibition and output evolve.

`AutapticLayer.forward` only returns the final step. The trace recomputes every step with the
layer's own helper methods (no duplicated formulas) and checks that it ends where `forward` does.
`build_trace_arrays` condenses the trace of a trained model into the datasets a run file stores.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autapse import AutapticLayer
from data import LabeledDataset
from run_store import RunArray

__all__ = ["LayerTrace", "build_trace_arrays", "get_layer_trace", "get_model_trace"]

NUM_CLASSES = 10


@dataclass(frozen=True)
class LayerTrace:
    """What one `AutapticLayer` computed at every unroll step for one batch.

    Attributes:
        y: The uninhibited activation, shape (batch, neurons); constant over the steps.
        y_self_used: The inhibition signal after the `ai2_weight` factor (the value the layer
            combined with `y`), shape (steps, batch, neurons); all zeros at step 0.
        y_tilde: The inhibited output, shape (steps, batch, neurons).
    """

    y: torch.Tensor
    y_self_used: torch.Tensor
    y_tilde: torch.Tensor


@torch.no_grad()
def get_layer_trace(layer: AutapticLayer, x: torch.Tensor) -> LayerTrace:
    """Replays `layer`'s unroll on `x` and returns `y`, `y_self_used` and `y_tilde` per step.

    Calls `layer(x)` once, so `layer.last_y_self` afterwards belongs to `x`.

    Raises:
        AssertionError: if the replay's last step differs from `layer(x)`, or its last
            `y_self_used` from `ai2_weight * layer.last_y_self`. NaNs compare equal, so a
            diverged network is still traced.
    """
    y = layer._get_y_activation()(layer.linear(x))
    y_self = x.new_zeros(y.shape)
    y_selfs_used, y_tildes = [], []
    for step in range(layer.T):
        if step > 0:
            y_self = layer._get_y_self_activation(y_tildes[-1])
        y_selfs_used.append(layer.ai2_weight * y_self)
        y_tildes.append(layer._get_inhibited_output(y, y_self))
    trace = LayerTrace(y, torch.stack(y_selfs_used), torch.stack(y_tildes))
    forward_output = layer(x)
    assert layer.last_y_self is not None
    torch.testing.assert_close(
        trace.y_tilde[-1], forward_output, rtol=1e-5, atol=1e-6, equal_nan=True
    )
    torch.testing.assert_close(
        trace.y_self_used[-1],
        layer.ai2_weight * layer.last_y_self,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    )
    return trace


@torch.no_grad()
def get_model_trace(model: nn.Module, images: torch.Tensor) -> list[LayerTrace]:
    """Traces every `AutapticLayer` of `model` in order; each layer reads the previous output.

    Returns an empty list for models without `AutapticLayer`s (e.g. `CNN`).
    """
    layers = [module for module in model.modules() if isinstance(module, AutapticLayer)]
    traces = []
    x = images.view(images.shape[0], -1)
    for layer in layers:
        trace = get_layer_trace(layer, x)
        traces.append(trace)
        x = trace.y_tilde[-1]
    return traces


@torch.no_grad()
def build_trace_arrays(
    model: nn.Module, test_data: LabeledDataset, device: torch.device
) -> dict[str, RunArray]:
    """The `unroll/*` datasets of a run file, from the whole test set through the final model.

    Means are signed means over samples (of the class, where the dims say `class`) and neurons.
    `neuron_std_y_tilde` is the mean over neurons of the per-neuron standard deviation over
    all test samples (ddof=0), i.e. the input-dependent part of `y_tilde`.
    Empty for models without `AutapticLayer`s.
    """
    model.eval()
    images, labels = next(
        iter(
            DataLoader(
                cast(Dataset[tuple[torch.Tensor, int]], test_data), batch_size=len(test_data)
            )
        )
    )
    traces = get_model_trace(model, images.to(device))
    if not traces:
        return {}
    labels = labels.to(device)
    class_masks = [labels == digit for digit in range(NUM_CLASSES)]
    per_layer: dict[str, list[torch.Tensor]] = {name: [] for name in _TRACE_DIMS}
    for trace in traces:
        sample_y_tilde = trace.y_tilde.mean(dim=-1)  # (steps, batch)
        sample_abs_y_tilde = trace.y_tilde.abs().mean(dim=-1)
        sample_y_self = trace.y_self_used.mean(dim=-1)
        per_layer["mean_y"].append(_class_means(trace.y.mean(dim=-1), class_masks))
        per_layer["mean_y_self_used"].append(_class_means(sample_y_self, class_masks))
        per_layer["mean_y_tilde"].append(_class_means(sample_y_tilde, class_masks))
        per_layer["mean_abs_y_tilde_class"].append(_class_means(sample_abs_y_tilde, class_masks))
        per_layer["mean_abs_y_tilde"].append(sample_abs_y_tilde.mean(dim=-1))
        per_layer["neuron_std_y_tilde"].append(
            trace.y_tilde.std(dim=1, unbiased=False).mean(dim=-1)
        )
    return {
        f"unroll/{name}": RunArray(torch.stack(values).cpu().double().numpy(), _TRACE_DIMS[name])
        for name, values in per_layer.items()
    }


_TRACE_DIMS: dict[str, tuple[str, ...]] = {
    "mean_y": ("layer", "class"),
    "mean_y_self_used": ("layer", "class", "step"),
    "mean_y_tilde": ("layer", "class", "step"),
    "mean_abs_y_tilde_class": ("layer", "class", "step"),
    "mean_abs_y_tilde": ("layer", "step"),
    "neuron_std_y_tilde": ("layer", "step"),
}


def _class_means(per_sample: torch.Tensor, class_masks: list[torch.Tensor]) -> torch.Tensor:
    """Mean of `per_sample` (last dim = batch) over each class's samples: (class, ...)."""
    return torch.stack([per_sample[..., mask].mean(dim=-1) for mask in class_masks])
