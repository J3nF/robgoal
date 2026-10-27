"""Autaptic self-inhibition layer.

Implementing two activation schemes and two inhibition variants.
See `old-lab-book.md` for more details.
"""

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

AI2_Variant = Literal["ai2_diff", "ai2_gate"]
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
        ai2_variant: How the inhibition signal is combined with the raw output:
            - "ai2_diff": subtract inhibition from the activated output.
            - "ai2_gate": multiplicatively gate the activated output by
              `(1 - inhibition)`.
        ai2_weight: Factor applied to the inhibition signal before it is combined with
            the raw output; `1.0` is the unweighted mechanism, `0.0` removes the inhibition
            (equivalent to `T=1`), negative values turn inhibition into excitation.
        activation_scheme: Which activation drives the main path:
            - "sigmoid": bounded output, inhibition signal is the raw output.
            - "relu_sigmoid": unbounded ReLU output; inhibition signal is
              `tanh(y / 2)` instead of `sigmoid(y)`, see old-lab-book.md,
              "Activation functions".

    Attributes:
        last_y_self: The inhibition signal that fed the *last* unroll step of
            the most recent `forward` call (i.e. `\\tilde y_i(T-1)` in
            old-lab-book.md notation) -- the "settled" value the `T`-step unroll
            exists to produce. `None` until `forward` has run at least once.
            For `T=1`, stays the all-zeros initial value forever (no
            inhibition took place, matching this class's `T=1` degeneracy).
            A plain tensor attribute, not a parameter or buffer -- doesn't
            affect `state_dict()`, `.parameters()`, or gradients; see
            `dynamics_insights.get_y_self_absmeans`.
            Stored *before* the `ai2_weight` factor, so it is the same for every weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        T: int,
        ai2_variant: AI2_Variant,
        activation_scheme: Activation_Scheme,
        ai2_weight: float = 1.0,
    ) -> None:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.T = T
        self.ai2_variant = ai2_variant
        self.activation_scheme = activation_scheme
        self.ai2_weight = ai2_weight
        self.last_y_self: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_activation = self._get_y_activation()
        y_self = x.new_zeros(x.shape[0], self.linear.out_features)
        y = self._step(x, y_self, y_activation)  # i.e., get uninhibited activation
        for _ in range(self.T - 1):
            y_self = self._get_y_self_activation(y)
            y = self._step(x, y_self, y_activation)
        self.last_y_self = y_self.detach()
        return y

    def _get_y_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.activation_scheme == "sigmoid":
            y_activation = torch.sigmoid
        elif self.activation_scheme == "relu_sigmoid":
            y_activation = torch.relu
        else:
            raise ValueError(f"Unknown activation_scheme {self.activation_scheme}")
        return y_activation

    def _get_y_self_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Derives next step's inhibition signal from this step's output `y`.

        "sigmoid": the raw output. "relu_sigmoid": `tanh(y / 2)` instead, so
        the signal stays in `[0, 1[` even though `y` is unbounded ReLU output
        -- see old-lab-book.md, "Activation functions".
        """
        if self.activation_scheme == "sigmoid":
            y_self = y
        elif self.activation_scheme == "relu_sigmoid":
            y_self = torch.tanh(y / 2)
        else:
            raise ValueError(f"Unknown activation_scheme {self.activation_scheme}")
        return y_self

    def _step(
        self,
        x: torch.Tensor,
        y_self: torch.Tensor,
        y_activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        y = y_activation(self.linear(x))
        y_tilde = self._get_inhibited_output(y, y_self)
        return y_tilde

    def _get_inhibited_output(self, y: torch.Tensor, y_self: torch.Tensor) -> torch.Tensor:
        """Combines `y` with the inhibition signal `y_self` per `self.ai2_variant`.

        `y_self` is scaled by `self.ai2_weight` first, so the weight enters only here.
        """
        y_self = self.ai2_weight * y_self
        if self.ai2_variant == "ai2_diff":
            y_tilde = y - y_self
        elif self.ai2_variant == "ai2_gate":
            y_tilde = y * (1 - y_self)
        else:
            raise ValueError(f"Unknown ai2 variant {self.ai2_variant}")
        return y_tilde
