"""Model architectures for the autapsic-inhibition MNIST tests. See lab-book.md."""

import torch
from torch import nn

from autapse import AutapticLayer, AI2_Variant, Activation_Scheme


class FCNN_AI2(nn.Module):
    """784-256-128-10 MLP whose hidden layers are `AutapticLayer`s.

    `T=1` gives the FCNN_AI2 controls (tests 2-3); see `autapse.AutapticLayer`
    for why mode is irrelevant there.

    Args:
        T: Number of internal unroll steps.
        mode: Inhibition mode, see `autapse.AutapticLayer`.
        scheme: Activation scheme, see `autapse.AutapticLayer`.
    """

    def __init__(self, T: int, mode: AI2_Variant, scheme: Activation_Scheme) -> None:
        super().__init__()
        self.hidden1 = AutapticLayer(28 * 28, 256, T, mode, scheme)
        self.hidden2 = AutapticLayer(256, 128, T, mode, scheme)
        self.output = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.output(x)


class CNN(nn.Module):
    """Small conv net control: 2x(conv+relu+pool) + FC, no self-inhibition."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        return self.output(x)
