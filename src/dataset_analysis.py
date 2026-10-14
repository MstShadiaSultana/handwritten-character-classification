"""Download, inspect, and visualize the EMNIST Letters dataset."""

import argparse
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from data_loader import PROJECT_ROOT, create_transform


CLASS_NAMES = list(string.ascii_uppercase)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the EMNIST Letters dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_datasets(data_dir):
    """Load corrected EMNIST Letters training and test datasets."""

    transform = create_transform()

    train_dataset = datasets.EMNIST(
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

    return train_dataset, test_dataset


def print_dataset_information(train_dataset, test_dataset):
    """Print basic dataset properties."""

    sample_image, sample_label = train_dataset[0]

    print("=" * 65)
    print("EMNIST LETTERS DATASET ANALYSIS")
    print("=" * 65)
    print(f"Training images : {len(train_dataset):,}")
    print(f"Test images     : {len(test_dataset):,}")
    print(f"Total images    : {len(train_dataset) + len(test_dataset):,}")
    print()
    print(f"Image shape     : {tuple(sample_image.shape)}")
    print(f"Pixel minimum   : {sample_image.min().item():.1f}")
    print(f"Pixel maximum   : {sample_image.max().item():.1f}")
    print(f"First label     : {sample_label}")
    print(f"Classes         : {len(CLASS_NAMES)}")
    print(f"Label range     : 0-{len(CLASS_NAMES) - 1}")
    print()


def save_random_samples(dataset, output_path, number_of_samples=20, seed=42):
    """Save random corrected EMNIST samples."""

    rng = np.random.default_rng(seed)
    sample_count = min(number_of_samples, len(dataset))
    indices = rng.choice(len(dataset), size=sample_count, replace=False)

    columns = 5
    rows = int(np.ceil(sample_count / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 2 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, index in zip(axes.flat, indices):
        image, label = dataset[int(index)]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(f"{CLASS_NAMES[label]} ({label})")
        axis.axis("off")

    figure.suptitle("Random Corrected Samples from EMNIST Letters", fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def get_class_counts(dataset):
    """Return image count for each class."""

    labels = dataset.targets.numpy() - 1
    return np.bincount(labels, minlength=len(CLASS_NAMES))


def save_class_distribution(train_dataset, test_dataset, output_path):
    """Save training and test class distributions."""

    train_counts = get_class_counts(train_dataset)
    test_counts = get_class_counts(test_dataset)

    positions = np.arange(len(CLASS_NAMES))
    width = 0.4

    figure, axis = plt.subplots(figsize=(13, 6))

    axis.bar(positions - width / 2, train_counts, width=width, label="Training")
    axis.bar(positions + width / 2, test_counts, width=width, label="Test")

    axis.set_title("EMNIST Letters Class Distribution")
    axis.set_xlabel("Letter")
    axis.set_ylabel("Number of images")
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASS_NAMES)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return train_counts, test_counts


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = load_datasets(args.data_dir)
    print_dataset_information(train_dataset, test_dataset)

    random_samples_path = args.output_dir / "emnist_letters_random_samples.png"
    distribution_path = args.output_dir / "emnist_letters_class_distribution.png"

    save_random_samples(
        train_dataset,
        random_samples_path,
        number_of_samples=args.samples,
        seed=args.seed,
    )

    train_counts, test_counts = save_class_distribution(
        train_dataset,
        test_dataset,
        distribution_path,
    )

    print("Class distribution:")
    print(f"Training per class : min={train_counts.min():,}, max={train_counts.max():,}")
    print(f"Test per class     : min={test_counts.min():,}, max={test_counts.max():,}")
    print()
    print(f"Random samples figure : {random_samples_path}")
    print(f"Distribution figure   : {distribution_path}")
    print("=" * 65)
    print("DATASET ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()