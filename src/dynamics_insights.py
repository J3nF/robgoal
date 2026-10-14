"""Records autapsic inhibition (y_self) dynamics during training. See old-lab-book.md."""

from torch import nn

from autapse import AutapticLayer


def get_y_self_absmeans(model: nn.Module) -> dict[str, float]:
    """Mean |y_self| of each AutapticLayer's last forward call, by submodule name.

    Skips layers that haven't run a forward pass yet (`last_y_self` still
    `None`) and models with no `AutapticLayer`s (e.g. `CNN`).
    Both return an empty entry for that layer/model rather than raising, since
    both are legitimate call states, not errors.
    """
    absmeans = {
        name: module.last_y_self.abs().mean().item()
        for name, module in model.named_modules()
        if isinstance(module, AutapticLayer) and module.last_y_self is not None
    }
    return absmeans


def get_y_self_per_neuron(model: nn.Module) -> dict[str, list[float]]:
    """Per-neuron mean y_self of each AutapticLayer's last forward call, by submodule name.

    Unlike `get_y_self_absmeans`, keeps the sign and the neuron dimension --
    only the batch dimension is averaged out. Same skip-empty behavior for
    layers with no forward pass yet and models with no `AutapticLayer`s.
    """
    y_self_per_neuron = {
        name: module.last_y_self.mean(dim=0).tolist()
        for name, module in model.named_modules()
        if isinstance(module, AutapticLayer) and module.last_y_self is not None
    }
    return y_self_per_neuron
