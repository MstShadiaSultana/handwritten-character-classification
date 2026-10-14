"""Train Task B Experiment E1: a more complex CNN."""

import argparse
from pathlib import Path

from letter_experiment_utils import (
    PROJECT_ROOT,
    compare_experiments,
    run_letter_experiment,
)
from models import ComplexCNN


DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run E1: deeper CNN experiment on EMNIST Letters."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args()


def main():
    args = parse_args()
    figure_dir = args.results_dir / "figures"

    model = ComplexCNN(number_of_classes=26)

    result = run_letter_experiment(
        experiment_id="E1",
        title="EXPERIMENT E1 — DEEPER CNN",
        model=model,
        model_name="ComplexCNN",
        model_path=(
            args.models_dir
            / "letters_complex_e1_conv16-32_fc128_k3_lr0p001_bs64_ep5.pth"
        ),
        summary_path=args.results_dir / "e1_complex_training_summary.json",
        batch_figure=figure_dir / "e1_complex_training_loss_per_batch.png",
        epoch_figure=figure_dir / "e1_complex_training_loss_per_epoch.png",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        model_title="E1 — Deeper CNN",
        architecture={
            "number_of_classes": 26,
            "first_channels": 16,
            "second_channels": 32,
            "kernel_size": 3,
            "hidden_features": 128,
        },
        extra_summary={
            "architecture": (
                "Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2,2) -> "
                "Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d(2,2) -> "
                "Flatten -> Linear(1568,128) -> ReLU -> Linear(128,26)"
            ),
            "main_modification": (
                "Added a second convolution-pooling block "
                "and a 128-unit hidden layer"
            ),
            "reference_experiment": "Baseline",
        },
    )

    compare_experiments(
        current_name="E1",
        current_accuracy=result["best_validation_accuracy"],
        current_time=result["training_time"],
        reference_summary_path=args.results_dir / "baseline_training_summary.json",
        reference_name="Baseline",
    )


if __name__ == "__main__":
    main()