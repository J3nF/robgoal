"""Autapsic self-inhibition layer.

Implementing two activation schemes and two inhibition modes.
See `lab-book.md` for more details.
"""

from typing import Literal

import torch
from torch import nn

Mode = Literal["ai2_diff", "ai2_gate"]
Scheme = Literal["sigmoid", "relu_sigmoid"]


class AutapticLayer(nn.Module):
    """A linear layer with self-recurrent (autapsic) inhibition.

    Unrolls the same input through `T` internal pseudo-timesteps. At each step,
    a neuron's own output from the previous step ("y_self") feeds back in
    to inhibit it.
    For `T=1`, it degenerates exactly to a plain `Linear` + activation, since
    the inhibition signal is initialized to zero.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        T: Number of internal unroll steps.
        mode: How the inhibition signal is combined with the raw output:
            - "ai2_diff": subtract inhibition from the activated output.
            - "ai2_gate": multiplicatively gate the activated output by
              `(1 - inhibition)`.
        scheme: Which activation drives the main path:
            - "sigmoid": bounded output, inhibition signal is the raw output.
            - "relu_sigmoid": unbounded ReLU output; inhibition signal is
              `tanh(y / 2)` instead of `sigmoid(y)`, see lab-book.md,
              "Activation functions".
    """

    def __init__(
        self, in_features: int, out_features: int, T: int, mode: Mode, scheme: Scheme
    ) -> None:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.T = T
        self.mode = mode
        self.scheme = scheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_activation = self.get_f_activation()
        y_self = x.new_zeros(x.shape[0], self.linear.out_features)
        y = self._step(x, y_self, f_activation)  # Get uninhibited activation
        for _ in range(self.T - 1):
            y_self = self.get_y_self(y)
            y = self._step(x, y_self, f_activation)
        return y

    def get_f_activation(self):
        if self.scheme == "sigmoid":
            f_activation = torch.sigmoid
        else:
            f_activation = torch.relu
        return f_activation

    def get_y_self(self, y):
        if self.scheme == "sigmoid":
            y_self = y
        else:
            y_self = torch.tanh(y/2)
        return y_self

    #TODO
    def _step(self, x: torch.Tensor, y_self: torch.Tensor, f_activation) -> torch.Tensor:
        y_raw = f_activation(self.linear(x))
        if self.mode == "ai2_diff":
            y = y_raw - y_self
            return torch.relu(y) if self.scheme == "relu_sigmoid" else y
        return y_raw * (1 - y_self)  # ai2_gate
