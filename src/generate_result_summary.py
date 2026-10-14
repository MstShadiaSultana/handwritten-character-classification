"""Generate final experiment tables, comparison figures, and project summary."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("Baseline", "Required simple CNN", "8", 3, "baseline_training_summary.json"),
    ("E1", "Added second convolution and hidden layer", "16 → 32", 3, "e1_complex_training_summary.json"),
    ("E2", "Reduced feature maps", "8 → 16", 3, "e2_feature_maps_training_summary.json"),
    ("E3", "Increased kernel size", "16 → 32", 5, "e3_kernel5_training_summary.json"),
    ("E4", "Lower learning rate", "16 → 32", 5, "e4_lr0p0001_training_summary.json"),
    ("E5", "Higher learning rate", "16 → 32", 5, "e5_lr0p01_training_summary.json"),
    ("E6", "Smaller batch size", "16 → 32", 5, "e6_bs32_training_summary.json"),
    ("E7", "Larger batch size", "16 → 32", 5, "e7_bs128_training_summary.json"),
    ("E8", "Increased epochs", "16 → 32", 5, "e8_epochs10_training_summary.json"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate final IAOR experiment summaries and figures."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing JSON/CSV result files.",
    )
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required result file was not found:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def value(data, key, history_key=None, default=None):
    """Get a direct value or, optionally, the final value of a history list."""

    if key in data and data[key] is not None:
        return data[key]

    if history_key:
        history = data.get(history_key, [])
        if history:
            return history[-1]

    return default


def training_time(summary):
    """Support the timing key names used by the project scripts."""

    return float(
        summary.get(
            "total_training_time_seconds",
            summary.get("training_time_seconds", 0.0),
        )
    )


def best_epoch(summary):
    if summary.get("best_epoch") is not None:
        return int(summary["best_epoch"])

    history = summary.get("validation_accuracy_history", [])
    if history:
        return history.index(max(history)) + 1

    return int(summary.get("epochs", 0))


def create_task_b_table(results_dir):
    rows = []

    for name, change, channels, kernel, filename in EXPERIMENTS:
        summary = load_json(results_dir / filename)

        rows.append(
            {
                "Experiment": name,
                "Main modification": change,
                "Feature maps": channels,
                "Kernel size": kernel,
                "Learning rate": float(summary["learning_rate"]),
                "Batch size": int(summary["batch_size"]),
                "Epochs": int(summary["epochs"]),
                "Best epoch": best_epoch(summary),
                "Trainable parameters": int(summary["trainable_parameters"]),
                "Final training accuracy (%)": value(
                    summary,
                    "final_train_accuracy",
                    "train_accuracy_history",
                ),
                "Best validation accuracy (%)": float(
                    summary["best_validation_accuracy"]
                ),
                "Final validation loss": value(
                    summary,
                    "final_validation_loss",
                    "validation_loss_history",
                ),
                "Training time (seconds)": training_time(summary),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(
        results_dir / "task_b_experiment_comparison.csv",
        index=False,
        float_format="%.4f",
    )

    return dataframe


def create_digit_comparison(results_dir):
    transfer = load_json(results_dir / "digit_transfer_learning_summary.json")
    scratch = load_json(results_dir / "digit_from_scratch_summary.json")

    rows = [
        {
            "Training method": "Transfer learning",
            "Trainable parameters": int(transfer["trainable_parameters"]),
            "Best validation accuracy (%)": float(
                transfer["best_validation_accuracy"]
            ),
            "Test accuracy (%)": float(transfer["test_accuracy"]),
            "Test loss": float(transfer["test_loss"]),
            "Training time (seconds)": training_time(transfer),
        },
        {
            "Training method": "From scratch",
            "Trainable parameters": int(scratch["trainable_parameters"]),
            "Best validation accuracy (%)": float(
                scratch["best_validation_accuracy"]
            ),
            "Test accuracy (%)": float(scratch["test_accuracy"]),
            "Test loss": float(scratch["test_loss"]),
            "Training time (seconds)": training_time(scratch),
        },
    ]

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(
        results_dir / "digit_training_comparison.csv",
        index=False,
        float_format="%.4f",
    )

    return dataframe


def add_bar_labels(axis, decimals=2):
    for bar in axis.patches:
        height = bar.get_height()
        axis.annotate(
            f"{height:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_bar_chart(
    labels,
    values,
    title,
    ylabel,
    output_path,
    ylim=None,
    decimals=2,
):
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(labels, values)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.3)

    if ylim:
        axis.set_ylim(*ylim)

    add_bar_labels(axis, decimals)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_task_b_figures(dataframe, figure_dir):
    save_bar_chart(
        dataframe["Experiment"],
        dataframe["Best validation accuracy (%)"],
        "Task B Experiment Validation Accuracy",
        "Best validation accuracy (%)",
        figure_dir / "task_b_validation_accuracy_comparison.png",
        ylim=(0, 100),
    )

    save_bar_chart(
        dataframe["Experiment"],
        dataframe["Training time (seconds)"],
        "Task B Experiment Training Time",
        "Recorded training time (seconds)",
        figure_dir / "task_b_training_time_comparison.png",
        decimals=1,
    )

    figure, axis = plt.subplots(figsize=(10, 7))

    x = dataframe["Training time (seconds)"]
    y = dataframe["Best validation accuracy (%)"]

    axis.scatter(x, y, s=80)

    for _, row in dataframe.iterrows():
        axis.annotate(
            row["Experiment"],
            (
                row["Training time (seconds)"],
                row["Best validation accuracy (%)"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set_title("Validation Accuracy versus Training Time")
    axis.set_xlabel("Recorded training time (seconds)")
    axis.set_ylabel("Best validation accuracy (%)")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        figure_dir / "task_b_accuracy_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_digit_figures(dataframe, figure_dir):
    save_bar_chart(
        dataframe["Training method"],
        dataframe["Test accuracy (%)"],
        "MNIST Test Accuracy: Transfer Learning vs Full Training",
        "Test accuracy (%)",
        figure_dir / "digit_test_accuracy_comparison.png",
        ylim=(95, 100),
    )

    save_bar_chart(
        dataframe["Training method"],
        dataframe["Training time (seconds)"],
        "MNIST Training Time: Transfer Learning vs Full Training",
        "Training time (seconds)",
        figure_dir / "digit_training_time_comparison.png",
    )


def custom_test_summary(results_dir):
    csv_path = results_dir / "custom_image_predictions.csv"

    if not csv_path.exists():
        return None

    dataframe = pd.read_csv(csv_path)
    valid = dataframe[
        dataframe["result"].isin(["Correct", "Incorrect"])
    ].copy()

    def stats(task=None):
        subset = valid if task is None else valid[valid["task"] == task]
        total = len(subset)
        correct = int((subset["result"] == "Correct").sum())
        accuracy = 100 * correct / total if total else 0.0
        return correct, total, accuracy

    return {
        "overall": stats(),
        "letter": stats("letter"),
        "digit": stats("digit"),
    }


def create_text_summary(results_dir, task_b, digits):
    letter = load_json(results_dir / "final_letter_evaluation.json")

    best = task_b.loc[
        task_b["Best validation accuracy (%)"].idxmax()
    ]

    transfer = digits[
        digits["Training method"] == "Transfer learning"
    ].iloc[0]

    scratch = digits[
        digits["Training method"] == "From scratch"
    ].iloc[0]

    lines = [
        "IAOR FINAL PROJECT — RESULTS SUMMARY",
        "=" * 60,
        "",
        "BEST LETTER CLASSIFICATION EXPERIMENT",
        f"Experiment: {best['Experiment']}",
        f"Main modification: {best['Main modification']}",
        f"Best validation accuracy: {best['Best validation accuracy (%)']:.2f}%",
        f"Selected epoch: {int(best['Best epoch'])}",
        f"Recorded training time: {best['Training time (seconds)']:.2f} seconds",
        "",
        "FINAL LETTER TEST EVALUATION",
        f"Test accuracy: {letter['test_accuracy']:.2f}%",
        f"Test loss: {letter['test_loss']:.4f}",
        f"Test samples: {letter['test_samples']:,}",
        "",
        "DIGIT TRANSFER LEARNING",
        f"Trainable parameters: {int(transfer['Trainable parameters']):,}",
        f"Validation accuracy: {transfer['Best validation accuracy (%)']:.2f}%",
        f"Test accuracy: {transfer['Test accuracy (%)']:.2f}%",
        f"Training time: {transfer['Training time (seconds)']:.2f} seconds",
        "",
        "DIGIT FROM-SCRATCH CONTROL EXPERIMENT",
        f"Trainable parameters: {int(scratch['Trainable parameters']):,}",
        f"Test accuracy: {scratch['Test accuracy (%)']:.2f}%",
        f"Training time: {scratch['Training time (seconds)']:.2f} seconds",
    ]

    custom = custom_test_summary(results_dir)

    if custom:
        overall_correct, overall_total, overall_acc = custom["overall"]
        letter_correct, letter_total, letter_acc = custom["letter"]
        digit_correct, digit_total, digit_acc = custom["digit"]

        lines += [
            "",
            "SELF-GENERATED IMAGE EVALUATION",
            f"Images evaluated: {overall_total}",
            f"Correct predictions: {overall_correct}",
            f"Overall custom-image accuracy: {overall_acc:.2f}%",
            f"Letter accuracy: {letter_acc:.2f}% ({letter_correct}/{letter_total})",
            f"Digit accuracy: {digit_acc:.2f}% ({digit_correct}/{digit_total})",
        ]

    lines += ["", "=" * 60]

    output_path = results_dir / "final_project_results_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    figure_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GENERATING FINAL PROJECT RESULT SUMMARIES")
    print("=" * 75)

    task_b = create_task_b_table(results_dir)
    digits = create_digit_comparison(results_dir)

    create_task_b_figures(task_b, figure_dir)
    create_digit_figures(digits, figure_dir)
    create_text_summary(results_dir, task_b, digits)

    print()
    print("TASK B EXPERIMENT TABLE")
    print("-" * 75)

    columns = [
        "Experiment",
        "Learning rate",
        "Batch size",
        "Epochs",
        "Best validation accuracy (%)",
        "Training time (seconds)",
    ]
    print(task_b[columns].to_string(index=False))

    print()
    print("DIGIT TRAINING COMPARISON")
    print("-" * 75)
    print(digits.to_string(index=False))

    best = task_b.loc[
        task_b["Best validation accuracy (%)"].idxmax()
    ]

    print()
    print("=" * 75)
    print("KEY RESULT")
    print("=" * 75)
    print(f"Best letter experiment: {best['Experiment']}")
    print(
        f"Best letter validation accuracy: "
        f"{best['Best validation accuracy (%)']:.2f}%"
    )
    print(f"Selected epoch: {int(best['Best epoch'])}")

    print()
    print("Generated files:")
    generated = [
        results_dir / "task_b_experiment_comparison.csv",
        results_dir / "digit_training_comparison.csv",
        results_dir / "final_project_results_summary.txt",
        figure_dir / "task_b_validation_accuracy_comparison.png",
        figure_dir / "task_b_training_time_comparison.png",
        figure_dir / "task_b_accuracy_vs_time.png",
        figure_dir / "digit_test_accuracy_comparison.png",
        figure_dir / "digit_training_time_comparison.png",
    ]

    for path in generated:
        print(f"- {path}")

    print()
    print("=" * 75)
    print("FINAL RESULT SUMMARIES GENERATED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()