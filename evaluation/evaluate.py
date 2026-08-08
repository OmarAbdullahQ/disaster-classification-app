"""Evaluate saved models without retraining them."""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data.pipeline import build_datasets
from modeling.config import BATCH_SIZE, MODEL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
HISTORY_DIR = ARTIFACTS / "histories"
RESULT_DIR = ARTIFACTS / "results"
FIGURE_DIR = ARTIFACTS / "figures"


def plot_history(model_name: str, history: pd.DataFrame) -> None:
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["accuracy"], label="Training")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set(title=f"{model_name}: accuracy", xlabel="Epoch", ylabel="Accuracy")

    axes[1].plot(epochs, history["loss"], label="Training")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set(title=f"{model_name}: loss", xlabel="Epoch", ylabel="Loss")

    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{model_name}_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(model_name: str, matrix: np.ndarray, classes: list[str]) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axis,
    )
    axis.set(
        title=f"{model_name}: confusion matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_error_examples(
    model_name: str,
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    maximum: int = 12,
) -> None:
    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return

    confidence = probabilities[np.arange(len(y_pred)), y_pred]
    selected = wrong[np.argsort(confidence[wrong])[::-1][:maximum]]
    rows = int(np.ceil(len(selected) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for axis, index in zip(axes, selected):
        image = tf.keras.utils.load_img(paths[index])
        axis.imshow(image)
        axis.set_title(
            f"True: {classes[y_true[index]]}\n"
            f"Pred: {classes[y_pred[index]]} ({confidence[index]:.2f})"
        )
        axis.axis("off")
    for axis in axes[len(selected):]:
        axis.axis("off")

    fig.suptitle(f"{model_name}: highest-confidence errors")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{model_name}_error_examples.png", dpi=180)
    plt.close(fig)


def evaluate_all() -> pd.DataFrame:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets(batch_size=BATCH_SIZE)
    y_true = datasets.test_labels
    records = []
    weak_records = []
    confusion_records = []

    for model_name in MODEL_CONFIGS:
        model_path = MODEL_DIR / f"{model_name}.keras"
        history_path = HISTORY_DIR / f"{model_name}.csv"
        if not model_path.exists() or not history_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts for {model_name}. Run python -m modeling.train first."
            )

        print("Evaluating", model_name)
        model = tf.keras.models.load_model(model_path)
        history = pd.read_csv(history_path)
        plot_history(model_name, history)

        started = time.perf_counter()
        probabilities = model.predict(datasets.test, verbose=1)
        inference_seconds = time.perf_counter() - started
        y_pred = probabilities.argmax(axis=1)

        test_loss, test_accuracy = model.evaluate(datasets.test, verbose=0)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        report = classification_report(
            y_true,
            y_pred,
            target_names=datasets.class_names,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(RESULT_DIR / f"{model_name}_classification_report.csv")

        matrix = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(model_name, matrix, datasets.class_names)
        save_error_examples(
            model_name,
            datasets.test_paths,
            y_true,
            y_pred,
            probabilities,
            datasets.class_names,
        )

        class_f1 = {name: report[name]["f1-score"] for name in datasets.class_names}
        weakest_class = min(class_f1, key=class_f1.get)
        weak_records.append(
            {
                "model": model_name,
                "weakest_class": weakest_class,
                "weakest_class_f1": class_f1[weakest_class],
            }
        )

        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        true_index, predicted_index = np.unravel_index(
            off_diagonal.argmax(), off_diagonal.shape
        )
        confusion_records.append(
            {
                "model": model_name,
                "actual_class": datasets.class_names[true_index],
                "predicted_class": datasets.class_names[predicted_index],
                "error_count": int(off_diagonal[true_index, predicted_index]),
            }
        )

        records.append(
            {
                "model": model_name,
                "parameters": int(model.count_params()),
                "test_loss": float(test_loss),
                "test_accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "inference_time_seconds": float(inference_seconds),
            }
        )

    testing = pd.DataFrame(records)
    training = pd.read_csv(RESULT_DIR / "training_comparison.csv")
    comparison = training.merge(testing, on=["model", "parameters"])
    comparison.to_csv(RESULT_DIR / "final_model_comparison.csv", index=False)
    pd.DataFrame(weak_records).to_csv(RESULT_DIR / "weak_class_analysis.csv", index=False)
    pd.DataFrame(confusion_records).to_csv(
        RESULT_DIR / "most_confused_pairs.csv", index=False
    )

    summary = {
        "important_note": (
            "The deployment winner was selected from validation data before this test evaluation."
        ),
        "highest_test_macro_f1": str(
            comparison.sort_values("macro_f1", ascending=False).iloc[0]["model"]
        ),
    }
    (RESULT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return comparison


if __name__ == "__main__":
    print(evaluate_all().round(4).to_string(index=False))
