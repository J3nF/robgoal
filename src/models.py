"""Model architectures for the autapsic-inhibition MNIST tests. See old-lab-book.md."""

import torch
from torch import nn

from autapse import Activation_Scheme, AI2_Variant, AutapticLayer


class FCNN_AI2(nn.Module):
    """784-...-10 MLP whose hidden layers are `AutapticLayer`s.

    `T=1` gives the FCNN_AI2 controls (tests 2-3); see `autapse.AutapticLayer`
    for why ai2_variant is irrelevant there.

    Args:
        T: Number of internal unroll steps.
        ai2_variant: Inhibition variant, see `autapse.AutapticLayer`.
        activation_scheme: Activation scheme, see `autapse.AutapticLayer`.
        hidden_sizes: Output width of each hidden `AutapticLayer`, in order.
        ai2_weight: Inhibition weight of every hidden layer, see `autapse.AutapticLayer`.
    """

    def __init__(
        self,
        T: int,
        ai2_variant: AI2_Variant,
        activation_scheme: Activation_Scheme,
        hidden_sizes: tuple[int, ...] = (256, 128),
        ai2_weight: float = 1.0,
    ) -> None:
        super().__init__()
        layer_sizes = (28 * 28, *hidden_sizes)
        self.hidden_layers = nn.ModuleList(
            AutapticLayer(in_features, out_features, T, ai2_variant, activation_scheme, ai2_weight)
            for in_features, out_features in zip(layer_sizes, hidden_sizes)
        )
        self.output = nn.Linear(hidden_sizes[-1], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output(x)


class CNN(nn.Module):
    """Small conv net control: 2x(conv+relu+pool) + FC, no self-inhibition.

    Args:
        channels: Output channel count of (conv1, conv2), in order. Shrinking
            this is the CNN control's equivalent of FCNN_AI2's `hidden_sizes`
            for the network-size scan (see `size_scan.py`).
    """

    def __init__(self, channels: tuple[int, int] = (16, 32)) -> None:
        super().__init__()
        conv1_channels, conv2_channels = channels
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.output = nn.Linear(conv2_channels * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        return self.output(x)
