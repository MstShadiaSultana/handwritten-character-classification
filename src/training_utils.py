"""Reusable training, evaluation, saving, and plotting functions."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_accuracy(outputs, labels):
    """Return the number of correct and total predictions."""

    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return correct, total


def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
    epoch_number,
):
    """Train a model for one epoch."""

    model.train()

    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_losses = []

    progress_bar = tqdm(
        data_loader,
        desc=f"Training epoch {epoch_number}",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        batch_losses.append(loss.item())

        correct, total = calculate_accuracy(outputs, labels)

        total_correct += correct
        total_samples += total

        current_accuracy = 100 * total_correct / total_samples

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{current_accuracy:.2f}%",
        )

    return {
        "loss": running_loss / total_samples,
        "accuracy": 100 * total_correct / total_samples,
        "batch_losses": batch_losses,
    }


def evaluate_model(
    model,
    data_loader,
    loss_function,
    device,
    dataset_name="dataset",
):
    """Evaluate a model without updating its parameters."""

    model.eval()

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Evaluating {dataset_name}",
        leave=False,
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            correct, total = calculate_accuracy(outputs, labels)

            total_correct += correct
            total_samples += total

    return {
        "loss": running_loss / total_samples,
        "accuracy": 100 * total_correct / total_samples,
    }


def save_model(model, output_path):
    """Save a PyTorch state dictionary."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path)


def train_model(
    model,
    train_loader,
    validation_loader,
    loss_function,
    optimizer,
    device,
    number_of_epochs,
    best_model_path,
):
    """
    Train and validate a model.

    The checkpoint with the highest validation accuracy is saved.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "batch_losses": [],
    }

    best_validation_accuracy = -1.0
    training_start = perf_counter()

    for epoch in range(1, number_of_epochs + 1):
        epoch_start = perf_counter()

        train_result = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch_number=epoch,
        )

        validation_result = evaluate_model(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            dataset_name="validation set",
        )

        history["train_loss"].append(train_result["loss"])
        history["train_accuracy"].append(train_result["accuracy"])
        history["validation_loss"].append(validation_result["loss"])
        history["validation_accuracy"].append(validation_result["accuracy"])
        history["batch_losses"].append(train_result["batch_losses"])

        epoch_time = perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d}/{number_of_epochs:02d} | "
            f"Train loss: {train_result['loss']:.4f} | "
            f"Train accuracy: {train_result['accuracy']:.2f}% | "
            f"Validation loss: {validation_result['loss']:.4f} | "
            f"Validation accuracy: {validation_result['accuracy']:.2f}% | "
            f"Time: {epoch_time:.2f} seconds"
        )

        if validation_result["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_result["accuracy"]

            save_model(
                model=model,
                output_path=best_model_path,
            )

            print(
                "Best model saved with validation accuracy: "
                f"{best_validation_accuracy:.2f}%"
            )

    total_training_time = perf_counter() - training_start

    return history, total_training_time


def plot_batch_losses(
    batch_losses,
    output_path,
    model_title="CNN",
):
    """Plot training loss for each batch."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6))

    for epoch, losses in enumerate(batch_losses, start=1):
        axis.plot(
            range(1, len(losses) + 1),
            losses,
            label=f"Epoch {epoch}",
            linewidth=1,
        )

    axis.set_title(f"{model_title} Training Loss per Batch")
    axis.set_xlabel("Batch")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_epoch_history(
    history,
    output_path,
    model_title="CNN",
):
    """Plot average training and validation loss per epoch."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training loss",
    )

    axis.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    axis.set_title(f"{model_title} Loss per Epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.set_xticks(list(epochs))
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)