"""Train and evaluate the required baseline EMNIST CNN."""

import argparse
from pathlib import Path

from letter_experiment_utils import (
    PROJECT_ROOT,
    run_letter_experiment,
)
from models import BaselineCNN


DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN on EMNIST Letters."
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

    model = BaselineCNN(number_of_classes=26)

    run_letter_experiment(
        experiment_id="Baseline",
        title="BASELINE CNN — EMNIST LETTER CLASSIFICATION",
        model=model,
        model_name="BaselineCNN",
        model_path=(
            args.models_dir
            / "letters_baseline_conv8_k3_lr0p001_bs64_ep5.pth"
        ),
        summary_path=(
            args.results_dir
            / "baseline_training_summary.json"
        ),
        batch_figure=(
            figure_dir
            / "baseline_training_loss_per_batch.png"
        ),
        epoch_figure=(
            figure_dir
            / "baseline_training_loss_per_epoch.png"
        ),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        model_title="Baseline CNN",
        architecture={
            "number_of_classes": 26,
            "convolution_layers": 1,
            "output_channels": 8,
            "kernel_size": 3,
            "padding": 1,
            "pool_size": 2,
        },
        extra_summary={
            "main_modification": "Required baseline architecture",
        },
        evaluate_test=True,
    )


if __name__ == "__main__":
    main()