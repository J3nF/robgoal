"""Records autapsic inhibition (y_self) dynamics during training. See lab-book.md."""

from torch import nn

from autapse import AutapticLayer


def get_y_self_magnitudes(model: nn.Module) -> dict[str, float]:
    """Mean |y_self| of each AutapticLayer's last forward call, by submodule name.

    Skips layers that haven't run a forward pass yet (`last_y_self` still
    `None`) and models with no `AutapticLayer`s (e.g. `CNN`) -- both return an
    empty entry for that layer/model rather than raising, since both are
    legitimate call states, not errors.
    """
    return {
        name: module.last_y_self.abs().mean().item()
        for name, module in model.named_modules()
        if isinstance(module, AutapticLayer) and module.last_y_self is not None
    }
