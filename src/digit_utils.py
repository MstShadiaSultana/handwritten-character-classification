"""Shared utilities for MNIST digit-classification experiments."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_VALIDATION_FRACTION = 0.10


def set_seed(seed=DEFAULT_SEED):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_mnist_data_loaders(
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
    data_dir=None,
    validation_fraction=DEFAULT_VALIDATION_FRACTION,
):
    """Create stratified MNIST train, validation, and test loaders."""

    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data"
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    indices = np.arange(len(full_train))
    targets = full_train.targets.numpy()

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=targets,
    )

    train_dataset = Subset(full_train, train_indices)
    validation_dataset = Subset(full_train, validation_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    metadata = {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
    }

    return train_loader, validation_loader, test_loader, metadata


def count_parameters(model):
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable