"""Autapsic self-inhibition layer.

Implementing two activation schemes and two inhibition modes.
See `lab-book.md` for more details.
"""

from typing import Literal

import torch
from torch import nn

AI2_Variant= Literal["ai2_diff", "ai2_gate"]
Activation_Scheme = Literal["sigmoid", "relu_sigmoid"]


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
        self, in_features: int, out_features: int, T: int, mode: AI2_Variant, scheme: Activation_Scheme
    ) -> None:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.T = T
        self.mode = mode
        self.scheme = scheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_activation = self.get_y_activation()
        y_self = x.new_zeros(x.shape[0], self.linear.out_features)
        y = self._step(x, y_self, y_activation)  # Get uninhibited activation
        for _ in range(self.T - 1):
            y_self = self.get_y_self_activation(y)
            y = self._step(x, y_self, y_activation)
        return y

    def get_y_activation(self):
        if self.scheme == "sigmoid":
            y_activation = torch.sigmoid
        elif self.scheme == "relu_sigmoid":
            y_activation = torch.relu
        else:
            raise ValueError(f"Unknown scheme {self.scheme}")
        return y_activation

    def get_y_self_activation(self, y):
        if self.scheme == "sigmoid":
            y_self = y
        elif self.scheme == "relu_sigmoid":
            y_self = torch.tanh(y / 2)
        else:
            raise ValueError(f"Unknown scheme {self.scheme}")
        return y_self

    def _step(self, x: torch.Tensor, y_self: torch.Tensor, y_activation) -> torch.Tensor:
        y = y_activation(self.linear(x))
        if self.mode == "ai2_diff":
            y_tilde = y - y_self
        elif self.mode == "ai2_gate":
            y_tilde = y * (1 - y_self)
        else:
            raise ValueError(f"Unknown ai2 mode {self.mode}")
        return y_tilde