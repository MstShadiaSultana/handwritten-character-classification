"""Task C: Transfer learning from letters to MNIST digits."""

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


def parse_args():
    """Read optional command-line settings."""

    parser = argparse.ArgumentParser(
        description="Transfer the best EMNIST letter CNN to MNIST digits."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--letter-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_letter_classifier.pth",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_digit_classifier.pth",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )

    return parser.parse_args()


def prepare_transfer_model(device, letter_model_path):
    """
    Load the complete letter model, replace its final layer with 10 outputs,
    and freeze all parameters except the new output layer.
    """

    letter_model_path = Path(letter_model_path)

    if not letter_model_path.exists():
        raise FileNotFoundError(
            f"Letter-model checkpoint not found: {letter_model_path}"
        )

    checkpoint = torch.load(
        letter_model_path,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint["model_config"]

    # Recreate the original 26-class letter model.
    model = ConfigurableCNN(
        number_of_classes=config["number_of_classes"],
        first_channels=config["first_channels"],
        second_channels=config["second_channels"],
        kernel_size=config["kernel_size"],
        hidden_features=config["hidden_features"],
    ).to(device)

    # Load all learned letter-model parameters.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Replace only the final classification layer.
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 10).to(device)

    # Freeze every parameter first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train only the new 10-class output layer.
    for parameter in model.classifier[3].parameters():
        parameter.requires_grad = True

    return model, checkpoint


def save_digit_checkpoint(
    model,
    output_path,
    best_epoch,
    validation_accuracy,
    test_result,
):
    """Save the final structured digit-classifier checkpoint."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "ConfigurableCNN",
        "model_config": {
            "number_of_classes": 10,
            "first_channels": 16,
            "second_channels": 32,
            "kernel_size": 5,
            "hidden_features": 128,
        },
        "class_names": [str(digit) for digit in range(10)],
        "training_method": "transfer_learning",
        "best_epoch": best_epoch,
        "validation_accuracy": validation_accuracy,
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
    }

    torch.save(checkpoint, output_path)


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
        / "temporary_digit_transfer_state_dict.pth"
    )

    batch_figure = (
        figure_dir
        / "digit_transfer_training_loss_per_batch.png"
    )

    epoch_figure = (
        figure_dir
        / "digit_transfer_training_loss_per_epoch.png"
    )

    summary_path = (
        results_dir
        / "digit_transfer_learning_summary.json"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("TASK C — TRANSFER LEARNING: EMNIST LETTERS TO MNIST DIGITS")
    print("=" * 78)

    print(f"Device            : {device}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"Epochs            : {args.epochs}")
    print(f"Letter checkpoint : {args.letter_model}")
    print()

    train_loader, validation_loader, test_loader, metadata = (
        create_mnist_data_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=args.data_dir,
        )
    )

    print(f"Training images   : {metadata['train_size']:,}")
    print(f"Validation images : {metadata['validation_size']:,}")
    print(f"Test images       : {metadata['test_size']:,}")
    print()

    model, letter_checkpoint = prepare_transfer_model(
        device=device,
        letter_model_path=args.letter_model,
    )

    total_parameters, trainable_parameters = count_parameters(model)
    frozen_parameters = total_parameters - trainable_parameters

    print(model)
    print()

    print(f"Total parameters     : {total_parameters:,}")
    print(f"Frozen parameters    : {frozen_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")
    print(
        f"Trainable percentage : "
        f"{100 * trainable_parameters / total_parameters:.2f}%"
    )

    print()
    print("Parameter training status:")

    for name, parameter in model.named_parameters():
        status = "TRAINABLE" if parameter.requires_grad else "frozen"
        print(f"  {name:<30} {status}")

    print()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
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
        dataset_name="MNIST transfer-learning test set",
    )

    best_validation_accuracy = max(
        history["validation_accuracy"]
    )

    best_epoch = (
        history["validation_accuracy"].index(
            best_validation_accuracy
        )
        + 1
    )

    plot_batch_losses(
        batch_losses=history["batch_losses"],
        output_path=batch_figure,
        model_title="MNIST Transfer Learning",
    )

    plot_epoch_history(
        history=history,
        output_path=epoch_figure,
        model_title="MNIST Transfer Learning",
    )

    save_digit_checkpoint(
        model=model,
        output_path=args.output_model,
        best_epoch=best_epoch,
        validation_accuracy=best_validation_accuracy,
        test_result=test_result,
    )

    summary = {
        "experiment": "MNIST transfer learning from EMNIST letter model",
        "training_method": "transfer_learning",
        "source_model": result_path(args.letter_model),
        "source_letter_validation_accuracy": letter_checkpoint.get(
            "validation_accuracy"
        ),
        "training_samples": metadata["train_size"],
        "validation_samples": metadata["validation_size"],
        "test_samples": metadata["test_size"],
        "total_parameters": total_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": (
            100 * trainable_parameters / total_parameters
        ),
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

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print()
    print("=" * 78)
    print("TRANSFER LEARNING RESULTS")
    print("=" * 78)

    print(f"Best epoch               : {best_epoch}")
    print(
        f"Best validation accuracy : "
        f"{best_validation_accuracy:.2f}%"
    )
    print(f"Test loss                : {test_result['loss']:.4f}")
    print(f"Test accuracy            : {test_result['accuracy']:.2f}%")
    print(f"Training time            : {training_time:.2f} seconds")
    print(f"Saved digit model        : {args.output_model}")
    print(f"Summary                  : {summary_path}")

    if temporary_model.exists():
        temporary_model.unlink()

    print()
    print("=" * 78)
    print("TRANSFER LEARNING COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
