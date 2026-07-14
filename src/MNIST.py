# %%
import ssl

import certifi
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Some systems' OpenSSL trust store is missing the CA that signs the MNIST
# mirror's cert, causing CERTIFICATE_VERIFY_FAILED on download. Use certifi's
# bundle instead, which is kept up to date independently of the OS.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# %%
def get_tt_data():
    # Define how to treat loaded data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Copied MNIST mean and std values
        ]
    )

    train_data = torchvision.datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )

    test_data = torchvision.datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform,
    )

    return train_data, test_data


# %%
train_data, test_data = get_tt_data()

# %%
