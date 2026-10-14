"""Task D: classify self-generated handwritten letters and digits."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from models import ConfigurableCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify custom handwritten letters and digits."
    )

    parser.add_argument(
        "--letter-dir",
        type=Path,
        default=PROJECT_ROOT / "test_images" / "letters",
    )
    parser.add_argument(
        "--digit-dir",
        type=Path,
        default=PROJECT_ROOT / "test_images" / "digits",
    )
    parser.add_argument(
        "--letter-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_letter_classifier.pth",
    )
    parser.add_argument(
        "--digit-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_digit_classifier.pth",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "custom_image_predictions.csv",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "figures"
        / "custom_image_predictions.png",
    )

    return parser.parse_args()


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint["model_config"]

    model = ConfigurableCNN(
        number_of_classes=config["number_of_classes"],
        first_channels=config["first_channels"],
        second_channels=config["second_channels"],
        kernel_size=config["kernel_size"],
        hidden_features=config["hidden_features"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["class_names"]


def correct_polarity(image):
    """Convert to light handwriting on a dark background."""

    array = np.asarray(image, dtype=np.float32)

    border = np.concatenate(
        [
            array[0, :],
            array[-1, :],
            array[:, 0],
            array[:, -1],
        ]
    )

    if border.mean() > 127:
        image = ImageOps.invert(image)

    return image


def crop_character(image):
    """Crop empty background around the handwritten character."""

    array = np.asarray(image, dtype=np.uint8)
    foreground = array > 20

    if not foreground.any():
        raise ValueError("No visible character detected.")

    rows, cols = np.where(foreground)

    left = int(cols.min())
    right = int(cols.max()) + 1
    top = int(rows.min())
    bottom = int(rows.max()) + 1

    return image.crop((left, top, right, bottom))


def resize_and_center(image, character_size=20, canvas_size=28):
    """Resize while preserving aspect ratio and center on a square canvas."""

    width, height = image.size
    scale = min(character_size / width, character_size / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (canvas_size, canvas_size), 0)

    x = (canvas_size - new_width) // 2
    y = (canvas_size - new_height) // 2

    canvas.paste(resized, (x, y))

    return canvas


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = correct_polarity(image)
    image = crop_character(image)
    image = resize_and_center(image)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

    return tensor, image


def expected_label(image_path, task):
    """Extract expected class from names like letter_A.png or digit_7.jpg."""

    filename = image_path.stem

    pattern = r"letter_([A-Za-z])" if task == "letter" else r"digit_([0-9])"
    match = re.match(pattern, filename, flags=re.IGNORECASE)

    if not match:
        return "Unknown"

    label = match.group(1)

    return label.upper() if task == "letter" else label


def predict(image_path, model, class_names, device):
    tensor, processed_image = preprocess_image(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    label = class_names[prediction.item()]
    confidence = confidence.item() * 100

    return label, confidence, processed_image


def find_images(directory):
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def evaluate_images(image_paths, task, model, class_names, device):
    results = []
    processed_images = []

    for image_path in image_paths:
        expected = expected_label(image_path, task)

        try:
            predicted, confidence, processed = predict(
                image_path,
                model,
                class_names,
                device,
            )

            if expected == "Unknown":
                status = "Expected label unknown"
            elif expected == predicted:
                status = "Correct"
            else:
                status = "Incorrect"

            result = {
                "filename": image_path.name,
                "task": task,
                "expected_label": expected,
                "predicted_label": predicted,
                "confidence_percent": confidence,
                "result": status,
            }

            processed_images.append(processed)

            print(
                f"{image_path.name:<24} | "
                f"Expected: {expected:<3} | "
                f"Predicted: {predicted:<3} | "
                f"Confidence: {confidence:6.2f}% | "
                f"{status}"
            )

        except Exception as error:
            result = {
                "filename": image_path.name,
                "task": task,
                "expected_label": expected,
                "predicted_label": "Error",
                "confidence_percent": 0.0,
                "result": str(error),
            }

            processed_images.append(None)
            print(f"{image_path.name}: ERROR — {error}")

        results.append(result)

    return results, processed_images


def save_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "filename",
        "task",
        "expected_label",
        "predicted_label",
        "confidence_percent",
        "result",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def save_prediction_figure(results, images, output_path):
    total = len(results)
    columns = min(6, total)
    rows = int(np.ceil(total / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, image, result in zip(axes.flat, images, results):
        if image is not None:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)

        axis.set_title(
            f"{result['filename']}\n"
            f"Expected: {result['expected_label']} | "
            f"Predicted: {result['predicted_label']}\n"
            f"{result['confidence_percent']:.1f}% | "
            f"{result['result']}",
            fontsize=8,
        )

    figure.suptitle(
        "Predictions for Self-Generated Handwritten Images",
        fontsize=15,
    )

    figure.tight_layout(rect=[0, 0, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def accuracy_summary(results, task):
    task_results = [
        row
        for row in results
        if row["task"] == task
        and row["expected_label"] != "Unknown"
        and row["predicted_label"] != "Error"
    ]

    correct = sum(row["result"] == "Correct" for row in task_results)
    total = len(task_results)
    accuracy = 100 * correct / total if total else 0.0

    return correct, total, accuracy


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("TASK D — CUSTOM HANDWRITTEN IMAGE CLASSIFICATION")
    print("=" * 78)
    print(f"Device       : {device}")
    print(f"Letter folder: {args.letter_dir}")
    print(f"Digit folder : {args.digit_dir}")
    print()

    letter_model, letter_classes = load_model(args.letter_model, device)
    digit_model, digit_classes = load_model(args.digit_model, device)

    letter_images = find_images(args.letter_dir)
    digit_images = find_images(args.digit_dir)

    print(f"Letter images found: {len(letter_images)}")
    print(f"Digit images found : {len(digit_images)}")
    print(f"Total images       : {len(letter_images) + len(digit_images)}")
    print()

    letter_results, letter_processed = evaluate_images(
        letter_images,
        "letter",
        letter_model,
        letter_classes,
        device,
    )

    digit_results, digit_processed = evaluate_images(
        digit_images,
        "digit",
        digit_model,
        digit_classes,
        device,
    )

    results = letter_results + digit_results
    processed_images = letter_processed + digit_processed

    save_csv(results, args.output_csv)
    save_prediction_figure(
        results,
        processed_images,
        args.output_figure,
    )

    letter_correct, letter_total, letter_accuracy = accuracy_summary(
        results,
        "letter",
    )

    digit_correct, digit_total, digit_accuracy = accuracy_summary(
        results,
        "digit",
    )

    total_correct = letter_correct + digit_correct
    total_images = letter_total + digit_total
    overall_accuracy = (
        100 * total_correct / total_images
        if total_images
        else 0.0
    )

    print()
    print("=" * 78)
    print("CUSTOM IMAGE RESULTS")
    print("=" * 78)

    print(
        f"Letters : {letter_correct}/{letter_total} correct "
        f"({letter_accuracy:.2f}%)"
    )
    print(
        f"Digits  : {digit_correct}/{digit_total} correct "
        f"({digit_accuracy:.2f}%)"
    )
    print(
        f"Overall : {total_correct}/{total_images} correct "
        f"({overall_accuracy:.2f}%)"
    )

    print()
    print(f"Prediction table  : {args.output_csv}")
    print(f"Prediction figure : {args.output_figure}")

    print("=" * 78)
    print("CUSTOM IMAGE INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()