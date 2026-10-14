"""Control experiment: train the MNIST digit CNN from scratch."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from digit_utils import (
    count_parameters,
    create_mnist_data_loaders,
    set_seed,
)
from models import ConfigurableCNN
from path_utils import result_path
from training_utils import (
    evaluate_model,
    plot_batch_losses,
    plot_epoch_history,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42

FIRST_CHANNELS = 16
SECOND_CHANNELS = 32
KERNEL_SIZE = 5
HIDDEN_FEATURES = 128


def parse_args():
    """Read optional command-line settings."""

    parser = argparse.ArgumentParser(
        description="Train the MNIST comparison model from scratch."
    )

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "digit_classifier_from_scratch.pth",
    )

    return parser.parse_args()


def compare_with_transfer_learning(
    results_dir,
    scratch_training_time,
    scratch_validation_accuracy,
    scratch_test_accuracy,
    scratch_trainable_parameters,
):
    """Compare the scratch model with the saved transfer-learning result."""

    transfer_summary_path = results_dir / "digit_transfer_learning_summary.json"

    if not transfer_summary_path.exists():
        print("\nTransfer-learning summary not found; comparison skipped.")
        return

    with open(transfer_summary_path, "r", encoding="utf-8") as file:
        transfer = json.load(file)

    transfer_time = float(transfer["training_time_seconds"])
    transfer_validation_accuracy = float(transfer["best_validation_accuracy"])
    transfer_test_accuracy = float(transfer["test_accuracy"])
    transfer_trainable_parameters = int(transfer["trainable_parameters"])

    time_difference = scratch_training_time - transfer_time
    time_reduction = (
        100 * time_difference / scratch_training_time
        if scratch_training_time > 0
        else 0.0
    )

    print()
    print("=" * 78)
    print("FROM-SCRATCH TRAINING VS TRANSFER LEARNING")
    print("=" * 78)

    print(f"Transfer trainable parameters : {transfer_trainable_parameters:,}")
    print(f"Scratch trainable parameters  : {scratch_trainable_parameters:,}")
    print()

    print(f"Transfer validation accuracy  : {transfer_validation_accuracy:.2f}%")
    print(f"Scratch validation accuracy   : {scratch_validation_accuracy:.2f}%")
    print(
        f"Validation difference         : "
        f"{scratch_validation_accuracy - transfer_validation_accuracy:+.2f} "
        f"percentage points"
    )
    print()

    print(f"Transfer test accuracy        : {transfer_test_accuracy:.2f}%")
    print(f"Scratch test accuracy         : {scratch_test_accuracy:.2f}%")
    print(
        f"Test difference               : "
        f"{scratch_test_accuracy - transfer_test_accuracy:+.2f} "
        f"percentage points"
    )
    print()

    print(f"Transfer training time        : {transfer_time:.2f} seconds")
    print(f"Scratch training time         : {scratch_training_time:.2f} seconds")
    print(f"Time difference               : {time_difference:+.2f} seconds")
    print(f"Transfer time reduction       : {time_reduction:.2f}%")
    print("=" * 78)


def main():
    args = parse_args()
    set_seed(args.seed)

    results_dir = args.results_dir.resolve()
    figure_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    temporary_model = (
        args.output_model.parent
        / "temporary_digit_from_scratch_state_dict.pth"
    )

    batch_figure = figure_dir / "digit_from_scratch_loss_per_batch.png"
    epoch_figure = figure_dir / "digit_from_scratch_loss_per_epoch.png"
    summary_path = results_dir / "digit_from_scratch_summary.json"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("CONTROL EXPERIMENT — DIGIT MODEL TRAINED FROM SCRATCH")
    print("=" * 78)

    print(f"Device             : {device}")
    print("Dataset            : MNIST")
    print(f"Batch size         : {args.batch_size}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Number of epochs   : {args.epochs}")
    print()

    train_loader, validation_loader, test_loader, metadata = (
        create_mnist_data_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=args.data_dir,
        )
    )

    print(f"Training images    : {metadata['train_size']:,}")
    print(f"Validation images  : {metadata['validation_size']:,}")
    print(f"Test images        : {metadata['test_size']:,}")
    print()

    model = ConfigurableCNN(
        number_of_classes=10,
        first_channels=FIRST_CHANNELS,
        second_channels=SECOND_CHANNELS,
        kernel_size=KERNEL_SIZE,
        hidden_features=HIDDEN_FEATURES,
    ).to(device)

    total_parameters, trainable_parameters = count_parameters(model)

    print(model)
    print()
    print(f"Total parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")

    if total_parameters != trainable_parameters:
        raise RuntimeError("Some parameters are unexpectedly frozen.")

    print("\nAll layers will be trained from random initialization.\n")

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history, training_time = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        number_of_epochs=args.epochs,
        best_model_path=temporary_model,
    )

    best_state = torch.load(
        temporary_model,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)

    test_result = evaluate_model(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        dataset_name="MNIST scratch-model test set",
    )

    best_validation_accuracy = max(history["validation_accuracy"])
    best_epoch = (
        history["validation_accuracy"].index(best_validation_accuracy) + 1
    )

    plot_batch_losses(
        batch_losses=history["batch_losses"],
        output_path=batch_figure,
        model_title="MNIST Digit Model Trained from Scratch",
    )

    plot_epoch_history(
        history=history,
        output_path=epoch_figure,
        model_title="MNIST Digit Model Trained from Scratch",
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "ConfigurableCNN",
        "model_config": {
            "number_of_classes": 10,
            "first_channels": FIRST_CHANNELS,
            "second_channels": SECOND_CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "hidden_features": HIDDEN_FEATURES,
        },
        "class_names": [str(digit) for digit in range(10)],
        "training_method": "from_scratch",
        "best_epoch": best_epoch,
        "validation_accuracy": best_validation_accuracy,
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "trainable_parameters": trainable_parameters,
    }

    torch.save(checkpoint, args.output_model)

    summary = {
        "experiment": "MNIST full training from scratch",
        "training_method": "from_scratch",
        "dataset": "MNIST",
        "training_samples": metadata["train_size"],
        "validation_samples": metadata["validation_size"],
        "test_samples": metadata["test_size"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "training_time_seconds": training_time,
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "train_loss_history": history["train_loss"],
        "validation_loss_history": history["validation_loss"],
        "train_accuracy_history": history["train_accuracy"],
        "validation_accuracy_history": history["validation_accuracy"],
        "saved_model": result_path(args.output_model),
        "batch_loss_figure": result_path(batch_figure),
        "epoch_loss_figure": result_path(epoch_figure),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print()
    print("=" * 78)
    print("FROM-SCRATCH DIGIT RESULTS")
    print("=" * 78)

    print(f"Best epoch               : {best_epoch}")
    print(f"Best validation accuracy : {best_validation_accuracy:.2f}%")
    print(f"Test loss                : {test_result['loss']:.4f}")
    print(f"Test accuracy            : {test_result['accuracy']:.2f}%")
    print(f"Training time            : {training_time:.2f} seconds")
    print(f"Saved comparison model   : {args.output_model}")
    print(f"Summary                  : {summary_path}")

    compare_with_transfer_learning(
        results_dir=results_dir,
        scratch_training_time=training_time,
        scratch_validation_accuracy=best_validation_accuracy,
        scratch_test_accuracy=test_result["accuracy"],
        scratch_trainable_parameters=trainable_parameters,
    )

    if temporary_model.exists():
        temporary_model.unlink()

    print()
    print("=" * 78)
    print("CONTROL EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
