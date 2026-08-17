"""Autapsic self-inhibition layer.

See lab-book.md for the derivation of the two activation schemes and two
inhibition modes implemented here.
"""

from typing import Literal

import torch
from torch import nn

Mode = Literal["output_diff", "mult_gate"]
Scheme = Literal["sigmoid", "relu_sigmoid"]


class AutapticLayer(nn.Module):
    """A linear layer with self-recurrent (autapsic) inhibition.

    Unrolls the same input through `T` internal pseudo-timesteps. At each step,
    a neuron's own output from the previous step (y_tilde_prev, i.e. \\tilde
    y_i(t-1) in lab-book.md notation) feeds back in to inhibit it, per `mode`.
    `T=1` degenerates exactly to a plain `Linear` + activation, since the
    inhibition signal is initialized to zero.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        T: Number of internal unroll steps.
        mode: How the inhibition signal is combined with the raw output:
            - "output_diff": subtract inhibition from the activated output.
            - "mult_gate": multiplicatively gate the activated output by
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
        if self.scheme == "sigmoid":
            activation = torch.sigmoid
        else:
            activation = torch.relu
        y_tilde_prev = x.new_zeros(x.shape[0], self.linear.out_features)
        y = self._step(x, y_tilde_prev, activation)  # T is >= 1, so this always runs
        for _ in range(self.T - 1):
            y_tilde_prev = y if self.scheme == "sigmoid" else torch.tanh(y / 2)
            y = self._step(x, y_tilde_prev, activation)
        return y

    def _step(self, x: torch.Tensor, y_tilde_prev: torch.Tensor, activation) -> torch.Tensor:
        y_raw = activation(self.linear(x))
        if self.mode == "output_diff":
            y = y_raw - y_tilde_prev
            return torch.relu(y) if self.scheme == "relu_sigmoid" else y
        return y_raw * (1 - y_tilde_prev)  # mult_gate
