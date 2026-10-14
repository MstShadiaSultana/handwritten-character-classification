"""Create reproducible EMNIST Letters DataLoaders."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_NUM_WORKERS = 0


def correct_emnist_orientation(image):
    """Rotate and horizontally flip raw EMNIST images to upright form."""

    image = image.rotate(-90)
    return ImageOps.mirror(image)


def create_transform():
    """Create the preprocessing pipeline used for EMNIST Letters."""

    return transforms.Compose(
        [
            transforms.Lambda(correct_emnist_orientation),
            transforms.ToTensor(),
        ]
    )


def create_data_loaders(
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
    data_dir=None,
    validation_fraction=DEFAULT_VALIDATION_FRACTION,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """
    Create stratified EMNIST Letters train, validation, and test loaders.

    Default split:
        training   : 112,320 images
        validation : 12,480 images
        test       : 20,800 images
    """

    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data"
    transform = create_transform()

    full_train = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=True,
        download=True,
        transform=transform,
        target_transform=lambda label: label - 1,
    )

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=False,
        download=True,
        transform=transform,
        target_transform=lambda label: label - 1,
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
        num_workers=num_workers,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metadata = {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "number_of_classes": 26,
        "image_shape": (1, 28, 28),
        "batch_size": batch_size,
        "seed": seed,
    }

    return train_loader, validation_loader, test_loader, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect the EMNIST Letters data-loader configuration."
    )

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_loader, validation_loader, test_loader, metadata = (
        create_data_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=args.data_dir,
        )
    )

    print("=" * 60)
    print("EMNIST LETTERS DATA LOADERS")
    print("=" * 60)

    print(f"Training images    : {metadata['train_size']:,}")
    print(f"Validation images  : {metadata['validation_size']:,}")
    print(f"Test images        : {metadata['test_size']:,}")
    print(f"Number of classes  : {metadata['number_of_classes']}")
    print(f"Image shape        : {metadata['image_shape']}")
    print(f"Batch size         : {metadata['batch_size']}")
    print()

    print(f"Training batches   : {len(train_loader):,}")
    print(f"Validation batches : {len(validation_loader):,}")
    print(f"Test batches       : {len(test_loader):,}")


if __name__ == "__main__":
    main()