"""Task B Experiment E4: test a lower learning rate."""

import argparse
from pathlib import Path

from letter_experiment_utils import (
    PROJECT_ROOT,
    compare_experiments,
    run_letter_experiment,
)
from models import ConfigurableCNN


DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run E4: test learning rate 0.0001 on EMNIST Letters."
    )

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    figure_dir = args.results_dir / "figures"

    model = ConfigurableCNN(
        number_of_classes=26,
        first_channels=16,
        second_channels=32,
        kernel_size=5,
        hidden_features=128,
    )

    result = run_letter_experiment(
        experiment_id="E4",
        title="EXPERIMENT E4 — LOWER LEARNING RATE",
        model=model,
        model_name="ConfigurableCNN",
        model_path=(
            args.models_dir
            / "letters_e4_lr0p0001_kernel5_channels16-32_bs64_ep5.pth"
        ),
        summary_path=(
            args.results_dir
            / "e4_lr0p0001_training_summary.json"
        ),
        batch_figure=(
            figure_dir
            / "e4_lr0p0001_training_loss_per_batch.png"
        ),
        epoch_figure=(
            figure_dir
            / "e4_lr0p0001_training_loss_per_epoch.png"
        ),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        model_title="E4 — Learning Rate 0.0001",
        architecture={
            "number_of_classes": 26,
            "first_channels": 16,
            "second_channels": 32,
            "kernel_size": 5,
            "hidden_features": 128,
        },
        extra_summary={
            "main_modification": "Reduced learning rate from 0.001 to 0.0001",
            "reference_experiment": "E3",
        },
    )

    compare_experiments(
        current_name="E4",
        current_accuracy=result["best_validation_accuracy"],
        current_time=result["training_time"],
        reference_summary_path=(
            args.results_dir
            / "e3_kernel5_training_summary.json"
        ),
        reference_name="E3",
    )


if __name__ == "__main__":
    main()