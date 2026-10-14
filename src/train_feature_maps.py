"""Task B Experiment E2: change convolution feature maps."""

import argparse
from pathlib import Path

from letter_experiment_utils import (
    PROJECT_ROOT,
    compare_experiments,
    run_letter_experiment,
)
from models import ConfigurableCNN


DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run E2: test reduced feature-map counts on EMNIST Letters."
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

    model = ConfigurableCNN(
        number_of_classes=26,
        first_channels=8,
        second_channels=16,
        kernel_size=3,
        hidden_features=128,
    )

    result = run_letter_experiment(
        experiment_id="E2",
        title="EXPERIMENT E2 — REDUCED FEATURE MAPS",
        model=model,
        model_name="ConfigurableCNN",
        model_path=(
            args.models_dir
            / "letters_e2_channels8-16_k3_fc128_lr0p001_bs64_ep5.pth"
        ),
        summary_path=args.results_dir / "e2_feature_maps_training_summary.json",
        batch_figure=figure_dir / "e2_feature_maps_training_loss_per_batch.png",
        epoch_figure=figure_dir / "e2_feature_maps_training_loss_per_epoch.png",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        model_title="E2 — Feature Maps 8 → 16",
        architecture={
            "number_of_classes": 26,
            "first_channels": 8,
            "second_channels": 16,
            "kernel_size": 3,
            "hidden_features": 128,
        },
        extra_summary={
            "main_modification": "Reduced feature maps from 16-32 to 8-16",
            "reference_experiment": "E1",
        },
    )

    compare_experiments(
        current_name="E2",
        current_accuracy=result["best_validation_accuracy"],
        current_time=result["training_time"],
        reference_summary_path=args.results_dir / "e1_complex_training_summary.json",
        reference_name="E1",
    )


if __name__ == "__main__":
    main()