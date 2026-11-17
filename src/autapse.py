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
LayerKind = Literal["unrolled", "leaky"]


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


def reset_inhibition_state(model: nn.Module, training: bool | None = None) -> None:
    """Clears the carried state of every `LeakyLayer` in `model` (no-op for other modules).

    Called with `training=None` (both states) at each stage start and with `training=False`
    (eval state only) at the start of each evaluation pass, so a stream never leaks inhibition
    across a stage boundary, and an evaluation neither inherits nor erases the training stream.
    """
    for module in model.modules():
        if isinstance(module, LeakyLayer):
            module.reset_state(training)


class LeakyLayer(nn.Module):
    """A linear layer whose inhibition is a leaky integral of the preceding samples' signals.

    One forward pass per input (no unroll).
    The inhibition of sample `i` of the stream is
    `I_i = (1 - decay) * sum_{k>=1} decay^(k-1) * a_{i-k}`, where `a_j` is the activation signal
    of the uninhibited output of sample `j` (same scheme handling as `AutapticLayer`), per neuron.
    The output is gated like `ai2_gate`: `y * (1 - ai2_weight * I)`.

    Within a batch `I` is one matmul with a strictly lower-triangular decay matrix, so sample `i`
    sees samples before it only, never itself or later ones.
    The state `I_next` after the last sample is carried to the next batch, detached, so gradients
    stop at the batch boundary but flow between samples inside a batch.
    The `(1 - decay)` factor keeps `I` in `[0, 1)` for signals in `[0, 1)`, like `y_self`.
    Train and eval modes keep separate states, so per-epoch evaluation does not disturb the
    training stream; see `reset_inhibition_state` for when states are cleared.

    Gotcha: with shuffled training batches neighbouring samples are random classes, so `I` is a
    running population-level activity estimate; evaluation results depend on test order and
    batch size.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        activation_scheme: As in `AutapticLayer`; `"relu_sigmoid"` uses `tanh(y / 2)` as signal.
        decay: Per-sample decay `lambda` in `[0, 1)`; the memory is about `1 / (1 - decay)` samples.
        ai2_weight: Factor applied to `I` before gating.

    Attributes:
        last_inhibition: `I` (before the weight factor) of the most recent `forward` call, shape
            (batch, neurons), detached. `None` until `forward` has run.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_scheme: Activation_Scheme,
        decay: float = 0.9,
        ai2_weight: float = 1.0,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if activation_scheme not in ("sigmoid", "relu_sigmoid"):
            raise ValueError(f"Unknown activation_scheme {activation_scheme}")
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation_scheme = activation_scheme
        self.decay = decay
        self.ai2_weight = ai2_weight
        self.last_inhibition: torch.Tensor | None = None
        self._states: dict[bool, torch.Tensor | None] = {True: None, False: None}

    def reset_state(self, training: bool | None = None) -> None:
        """Forgets the carried inhibition of train mode, eval mode, or (`None`) both."""
        for mode in (True, False) if training is None else (training,):
            self._states[mode] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (
            torch.sigmoid(self.linear(x))
            if self.activation_scheme == "sigmoid"
            else torch.relu(self.linear(x))
        )
        signal = y if self.activation_scheme == "sigmoid" else torch.tanh(y / 2)
        inhibition = self._get_inhibition(signal)
        self.last_inhibition = inhibition.detach()
        return y * (1 - self.ai2_weight * inhibition)

    def _get_inhibition(self, signal: torch.Tensor) -> torch.Tensor:
        """`I` for every sample of the batch, and updates the carried state for the next batch."""
        batch = signal.shape[0]
        positions = torch.arange(batch, device=signal.device)
        lag = positions[:, None] - positions[None, :] - 1  # i - j - 1; >= 0 only for j < i
        decay = torch.tensor(self.decay, dtype=signal.dtype, device=signal.device)
        weights = torch.where(lag >= 0, decay ** lag.clamp(min=0).to(signal.dtype), 0.0)
        carried = self._states[self.training]
        if carried is None:
            carried = signal.new_zeros(signal.shape[1])
        inhibition = (1 - self.decay) * (weights @ signal) + (decay ** positions.to(signal.dtype))[
            :, None
        ] * carried[None, :]
        next_state = self.decay * inhibition[-1] + (1 - self.decay) * signal[-1]
        self._states[self.training] = next_state.detach()
        return inhibition
