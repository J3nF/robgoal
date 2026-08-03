"""Autapsic self-inhibition layer.

See lab-book.md for the derivation of the two activation schemes and three
inhibition modes implemented here.
"""

from typing import Literal

import torch
from torch import nn

Mode = Literal["output_diff", "bias_diff", "mult_gate"]
Scheme = Literal["sigmoid", "relu_sigmoid_gate"]


class AutapticLayer(nn.Module):
    """A linear layer with self-recurrent (autapsic) inhibition.

    Unrolls the same input through `T` internal pseudo-timesteps. At each step,
    a neuron's own activity from the previous step feeds back in to inhibit it,
    per `mode`. `T=1` degenerates exactly to a plain `Linear` + activation,
    since the inhibition signal is initialized to zero.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        T: Number of internal unroll steps.
        mode: How the inhibition signal is combined with the raw output:
            - "output_diff": subtract inhibition from the activated output.
            - "bias_diff": subtract inhibition from the pre-activation bias.
            - "mult_gate": multiplicatively gate the activated output by
              `(1 - inhibition)`.
        scheme: Which activation drives the main path:
            - "sigmoid": bounded output, inhibition signal is the raw output.
            - "relu_sigmoid_gate": unbounded ReLU output; the inhibition signal
              fed to the next step is `tanh(output / 2)`, a bounded proxy that
              maps 0 activity to 0 inhibition (unlike `sigmoid(output)`, which
              floors at 0.5 for any nonnegative output and would inhibit even
              a unit that fired zero last step).
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
        activation = torch.sigmoid if self.scheme == "sigmoid" else torch.relu
        inhibition_prev = x.new_zeros(x.shape[0], self.linear.out_features)
        y = self._step(x, inhibition_prev, activation)  # T is >= 1, so this always runs
        for _ in range(self.T - 1):
            inhibition_prev = y if self.scheme == "sigmoid" else torch.tanh(y / 2)
            y = self._step(x, inhibition_prev, activation)
        return y

    def _step(self, x: torch.Tensor, inhibition_prev: torch.Tensor, activation) -> torch.Tensor:
        if self.mode == "bias_diff":
            raw = x @ self.linear.weight.T + (self.linear.bias - inhibition_prev)
            return activation(raw)

        y_raw = activation(self.linear(x))
        if self.mode == "output_diff":
            y = y_raw - inhibition_prev
            return torch.relu(y) if self.scheme == "relu_sigmoid_gate" else y
        return y_raw * (1 - inhibition_prev)  # mult_gate
