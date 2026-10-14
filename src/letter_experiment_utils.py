"""Shared utilities for EMNIST letter-training experiments."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_loader import create_data_loaders
from models import count_trainable_parameters
from path_utils import result_path
from training_utils import (
    evaluate_model,
    plot_batch_losses,
    plot_epoch_history,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reset_model_parameters(model):
    """Reinitialize a fresh model after the random seed has been set."""

    def reset(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    model.apply(reset)


def save_json(data, output_path):
    """Save a dictionary as formatted JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def run_letter_experiment(
    experiment_id,
    title,
    model,
    model_name,
    model_path,
    summary_path,
    batch_figure,
    epoch_figure,
    batch_size=64,
    learning_rate=0.001,
    epochs=5,
    seed=DEFAULT_SEED,
    model_title="CNN",
    architecture=None,
    extra_summary=None,
    evaluate_test=False,
):
    """Run one reproducible EMNIST Letters training experiment."""

    set_seed(seed)
    reset_model_parameters(model)

    model_path = Path(model_path)
    summary_path = Path(summary_path)
    batch_figure = Path(batch_figure)
    epoch_figure = Path(epoch_figure)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    batch_figure.parent.mkdir(parents=True, exist_ok=True)
    epoch_figure.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"Device            : {device}")
    print(f"Batch size        : {batch_size}")
    print(f"Learning rate     : {learning_rate}")
    print(f"Epochs            : {epochs}")
    print(f"Random seed       : {seed}")
    print()

    train_loader, validation_loader, test_loader, metadata = create_data_loaders(
        batch_size=batch_size,
        seed=seed,
    )

    parameter_count = count_trainable_parameters(model)

    print(model)
    print()
    print(f"Trainable parameters: {parameter_count:,}")
    print()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history, total_training_time = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        number_of_epochs=epochs,
        best_model_path=model_path,
    )

    best_validation_accuracy = max(history["validation_accuracy"])
    best_epoch = history["validation_accuracy"].index(best_validation_accuracy) + 1

    plot_batch_losses(
        history["batch_losses"],
        batch_figure,
        model_title=model_title,
    )

    plot_epoch_history(
        history,
        epoch_figure,
        model_title=model_title,
    )

    summary = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "device": str(device),
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "random_seed": seed,
        "trainable_parameters": parameter_count,
        "training_samples": len(metadata["train_dataset"]),
        "validation_samples": len(metadata["validation_dataset"]),
        "test_samples": len(metadata["test_dataset"]),
        "training_batches": len(train_loader),
        "validation_batches": len(validation_loader),
        "test_batches": len(test_loader),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "total_training_time_seconds": total_training_time,
        "train_loss_history": history["train_loss"],
        "validation_loss_history": history["validation_loss"],
        "train_accuracy_history": history["train_accuracy"],
        "validation_accuracy_history": history["validation_accuracy"],
        "saved_model": result_path(model_path),
        "batch_loss_figure": result_path(batch_figure),
        "epoch_loss_figure": result_path(epoch_figure),
    }

    if architecture:
        summary.update(architecture)

    if extra_summary:
        summary.update(extra_summary)

    if evaluate_test:
        best_state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)

        test_result = evaluate_model(
            model,
            test_loader,
            loss_function,
            device,
            dataset_name="test set",
        )

        summary["test_accuracy"] = test_result["accuracy"]
        summary["test_loss"] = test_result["loss"]

    save_json(summary, summary_path)

    print()
    print("=" * 72)
    print(f"{experiment_id} RESULTS")
    print("=" * 72)
    print(f"Best epoch               : {best_epoch}")
    print(f"Best validation accuracy : {best_validation_accuracy:.2f}%")
    print(f"Final training accuracy  : {history['train_accuracy'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['validation_accuracy'][-1]:.2f}%")
    print(f"Final training loss      : {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss    : {history['validation_loss'][-1]:.4f}")
    print(f"Training time            : {total_training_time:.2f} seconds")

    if evaluate_test:
        print(f"Test accuracy             : {summary['test_accuracy']:.2f}%")
        print(f"Test loss                 : {summary['test_loss']:.4f}")

    print(f"Saved model               : {model_path}")
    print(f"Training summary          : {summary_path}")
    print("=" * 72)

    return {
        "best_validation_accuracy": best_validation_accuracy,
        "best_epoch": best_epoch,
        "training_time": total_training_time,
        "parameters": parameter_count,
        "history": history,
        "summary": summary,
    }


def compare_experiments(
    current_name,
    current_accuracy,
    current_time,
    reference_summary_path,
    reference_name,
):
    """Print a simple comparison with an earlier experiment."""

    reference_summary_path = Path(reference_summary_path)

    if not reference_summary_path.exists():
        print(f"{reference_name} summary not found. Comparison skipped.")
        return

    with open(reference_summary_path, "r", encoding="utf-8") as file:
        reference = json.load(file)

    reference_accuracy = reference["best_validation_accuracy"]
    reference_time = reference["total_training_time_seconds"]

    print()
    print("=" * 72)
    print(f"{current_name} VS {reference_name}")
    print("=" * 72)
    print(f"{reference_name} validation accuracy : {reference_accuracy:.2f}%")
    print(f"{current_name} validation accuracy : {current_accuracy:.2f}%")
    print(f"Accuracy difference       : {current_accuracy - reference_accuracy:+.2f} pp")
    print()
    print(f"{reference_name} training time       : {reference_time:.2f} seconds")
    print(f"{current_name} training time       : {current_time:.2f} seconds")
    print(f"Time difference           : {current_time - reference_time:+.2f} seconds")
    print("=" * 72)
