"""Finalize and evaluate the best EMNIST letter classifier."""

import argparse
import json
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from data_loader import create_data_loaders
from models import ConfigurableCNN, count_trainable_parameters
from path_utils import result_path
from training_utils import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42

DEFAULT_FIRST_CHANNELS = 16
DEFAULT_SECOND_CHANNELS = 32
DEFAULT_KERNEL_SIZE = 5
DEFAULT_HIDDEN_FEATURES = 128

NUMBER_OF_CLASSES = 26
CLASS_NAMES = list(string.ascii_uppercase)


def parse_args():
    """Read optional command-line settings."""

    parser = argparse.ArgumentParser(
        description="Finalize and evaluate the selected EMNIST letter model."
    )

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--source-model",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "letters_e8_epochs10_kernel5_channels16-32_lr0p001_bs64.pth",
    )

    parser.add_argument(
        "--experiment-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "e8_epochs10_training_summary.json",
    )

    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_letter_classifier.pth",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )

    parser.add_argument(
        "--first-channels",
        type=int,
        default=DEFAULT_FIRST_CHANNELS,
    )
    parser.add_argument(
        "--second-channels",
        type=int,
        default=DEFAULT_SECOND_CHANNELS,
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
    )
    parser.add_argument(
        "--hidden-features",
        type=int,
        default=DEFAULT_HIDDEN_FEATURES,
    )

    return parser.parse_args()


def collect_predictions(model, data_loader, device):
    """Collect true and predicted labels from a data loader."""

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    return np.asarray(true_labels), np.asarray(predicted_labels)


def calculate_per_class_accuracy(true_labels, predicted_labels):
    """Calculate test accuracy separately for each letter A-Z."""

    rows = []

    for class_index, letter in enumerate(CLASS_NAMES):
        mask = true_labels == class_index

        total = int(mask.sum())
        correct = int(
            (predicted_labels[mask] == true_labels[mask]).sum()
        )

        accuracy = 100 * correct / total if total else 0.0

        rows.append(
            {
                "class_index": class_index,
                "letter": letter,
                "correct_predictions": correct,
                "total_images": total,
                "accuracy_percent": accuracy,
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix(true_labels, predicted_labels, output_path):
    """Save a normalized A-Z confusion matrix."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(NUMBER_OF_CLASSES)),
        normalize="true",
    )

    figure, axis = plt.subplots(figsize=(13, 11))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Proportion of true class",
    )

    axis.set_title(
        "Normalized Confusion Matrix — Final Letter Classifier"
    )
    axis.set_xlabel("Predicted letter")
    axis.set_ylabel("True letter")

    positions = np.arange(NUMBER_OF_CLASSES)

    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(CLASS_NAMES, fontsize=8)
    axis.set_yticklabels(CLASS_NAMES, fontsize=8)
    axis.tick_params(axis="x", rotation=90)

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_model(args, device):
    """Create the architecture used by the selected E8 experiment."""

    return ConfigurableCNN(
        number_of_classes=NUMBER_OF_CLASSES,
        first_channels=args.first_channels,
        second_channels=args.second_channels,
        kernel_size=args.kernel_size,
        hidden_features=args.hidden_features,
    ).to(device)


def save_final_checkpoint(
    model,
    args,
    experiment_summary,
    test_result,
    parameter_count,
):
    """Save the structured final letter-classifier checkpoint."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "ConfigurableCNN",
        "model_config": {
            "number_of_classes": NUMBER_OF_CLASSES,
            "first_channels": args.first_channels,
            "second_channels": args.second_channels,
            "kernel_size": args.kernel_size,
            "hidden_features": args.hidden_features,
        },
        "class_names": CLASS_NAMES,
        "source_experiment": "E8",
        "selected_epoch": experiment_summary["best_epoch"],
        "validation_accuracy": experiment_summary[
            "best_validation_accuracy"
        ],
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "trainable_parameters": parameter_count,
        "preprocessing": {
            "image_size": [28, 28],
            "channels": 1,
            "pixel_range": [0.0, 1.0],
            "emnist_orientation_corrected": True,
            "label_mapping": "0=A, 1=B, ..., 25=Z",
        },
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_model)


def main():
    args = parse_args()

    results_dir = args.results_dir.resolve()
    figure_dir = results_dir / "figures"

    final_summary_path = results_dir / "final_letter_evaluation.json"
    per_class_path = results_dir / "final_letter_per_class_accuracy.csv"
    confusion_matrix_path = (
        figure_dir / "final_letter_confusion_matrix.png"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not args.source_model.exists():
        raise FileNotFoundError(
            f"Selected model was not found: {args.source_model}"
        )

    if not args.experiment_summary.exists():
        raise FileNotFoundError(
            f"Experiment summary was not found: "
            f"{args.experiment_summary}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 75)
    print("FINAL LETTER CLASSIFIER EVALUATION")
    print("=" * 75)

    print(f"Device             : {device}")
    print("Source experiment  : E8")
    print(f"Source model       : {args.source_model}")
    print()

    with open(
        args.experiment_summary,
        "r",
        encoding="utf-8",
    ) as file:
        experiment_summary = json.load(file)

    _, _, test_loader, metadata = create_data_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = create_model(args, device)

    state_dict = torch.load(
        args.source_model,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    parameter_count = count_trainable_parameters(model)

    print(model)
    print()

    print(f"Trainable parameters     : {parameter_count:,}")
    print(
        f"Best validation epoch    : "
        f"{experiment_summary['best_epoch']}"
    )
    print(
        f"Best validation accuracy : "
        f"{experiment_summary['best_validation_accuracy']:.2f}%"
    )
    print()

    loss_function = nn.CrossEntropyLoss()

    test_result = evaluate_model(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        dataset_name="final letter test set",
    )

    true_labels, predicted_labels = collect_predictions(
        model,
        test_loader,
        device,
    )

    per_class_results = calculate_per_class_accuracy(
        true_labels,
        predicted_labels,
    )

    per_class_results.to_csv(
        per_class_path,
        index=False,
    )

    save_confusion_matrix(
        true_labels,
        predicted_labels,
        confusion_matrix_path,
    )

    save_final_checkpoint(
        model=model,
        args=args,
        experiment_summary=experiment_summary,
        test_result=test_result,
        parameter_count=parameter_count,
    )

    final_summary = {
        "selected_experiment": "E8",
        "selected_epoch": experiment_summary["best_epoch"],
        "best_validation_accuracy": experiment_summary[
            "best_validation_accuracy"
        ],
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_samples": metadata["test_size"],
        "trainable_parameters": parameter_count,
        "source_model": result_path(args.source_model),
        "final_model": result_path(args.output_model),
        "confusion_matrix": result_path(confusion_matrix_path),
        "per_class_results": result_path(per_class_path),
    }

    with open(
        final_summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_summary, file, indent=4)

    best_class = per_class_results.loc[
        per_class_results["accuracy_percent"].idxmax()
    ]

    worst_class = per_class_results.loc[
        per_class_results["accuracy_percent"].idxmin()
    ]

    print()
    print("=" * 75)
    print("FINAL LETTER TEST RESULTS")
    print("=" * 75)

    print(f"Test samples              : {metadata['test_size']:,}")
    print(f"Test loss                 : {test_result['loss']:.4f}")
    print(f"Test accuracy             : {test_result['accuracy']:.2f}%")
    print(
        f"Highest class accuracy    : "
        f"{best_class['letter']} "
        f"({best_class['accuracy_percent']:.2f}%)"
    )
    print(
        f"Lowest class accuracy     : "
        f"{worst_class['letter']} "
        f"({worst_class['accuracy_percent']:.2f}%)"
    )

    print()
    print(f"Final letter model        : {args.output_model}")
    print(f"Final evaluation summary  : {final_summary_path}")
    print(f"Per-class results         : {per_class_path}")
    print(f"Confusion matrix          : {confusion_matrix_path}")

    print("=" * 75)
    print("FINAL LETTER MODEL SAVED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
