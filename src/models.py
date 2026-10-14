"""CNN model architectures for the IAOR final project."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Required baseline CNN for 28 x 28 grayscale images."""

    def __init__(self, number_of_classes=26):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, number_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ComplexCNN(nn.Module):
    """Deeper CNN used for experiment E1."""

    def __init__(self, number_of_classes=26):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ConfigurableCNN(nn.Module):
    """
    Reusable two-block CNN.

    Parameters allow experiments with:
    - number of feature maps
    - convolution kernel size
    - hidden-layer size
    - number of output classes
    """

    def __init__(
        self,
        number_of_classes=26,
        first_channels=16,
        second_channels=32,
        kernel_size=3,
        hidden_features=128,
    ):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        padding = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=first_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=first_channels,
                out_channels=second_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                second_channels * 7 * 7,
                hidden_features,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_features,
                number_of_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_trainable_parameters(model):
    """Return the number of parameters currently marked as trainable."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_all_parameters(model):
    """Return the total number of model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


if __name__ == "__main__":
    baseline = BaselineCNN()
    complex_model = ComplexCNN()
    configurable = ConfigurableCNN(
        first_channels=16,
        second_channels=32,
        kernel_size=5,
    )

    print("BaselineCNN parameters    :", count_all_parameters(baseline))
    print("ComplexCNN parameters     :", count_all_parameters(complex_model))
    print("ConfigurableCNN parameters:", count_all_parameters(configurable))

    sample = torch.randn(1, 1, 28, 28)

    print("Baseline output shape     :", baseline(sample).shape)
    print("Complex output shape      :", complex_model(sample).shape)
    print("Configurable output shape :", configurable(sample).shape)