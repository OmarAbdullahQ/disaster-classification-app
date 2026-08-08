"""Train each required CNN once and persist every reusable artifact."""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data.pipeline import load_split
from modeling.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_CONFIGS,
    SEED,
)
from modeling.models import build_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
HISTORY_DIR = ARTIFACTS / "histories"
RESULT_DIR = ARTIFACTS / "results"


def set_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def gpu_summary() -> list[str]:
    return [device.name for device in tf.config.list_physical_devices("GPU")]


def train_all(force: bool = False) -> pd.DataFrame:
    set_reproducibility()
    for directory in (MODEL_DIR, HISTORY_DIR, RESULT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    gpus = gpu_summary()
    print("GPUs detected:", gpus or "None - training will use CPU")
    print("Loading dataset...")

    X_train, y_train = load_split("Train")
    X_val, y_val = load_split("Val")

    print("Training shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )

    train_ds = (
        train_ds
        .shuffle(
            buffer_size=len(y_train),
            seed=SEED,
            reshuffle_each_iteration=True
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )

    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    records = []
    for model_name, config in MODEL_CONFIGS.items():
        model_path = MODEL_DIR / f"{model_name}.keras"
        history_path = HISTORY_DIR / f"{model_name}.csv"
        metadata_path = RESULT_DIR / f"{model_name}_training.json"

        if not force and all(path.exists() for path in (model_path, history_path, metadata_path)):
            print(f"Skipping completed experiment: {model_name}")
            records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue

        print(f"\nTraining {model_name}")
        print("Hypothesis:", config["hypothesis"])
        keras.backend.clear_session()
        set_reproducibility()

        model = build_model(model_name, config)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
        training_seconds = time.perf_counter() - started

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_path, index=False)
        best_index = int(history_df["val_loss"].idxmin())

        record = {
            "model": model_name,
            "hypothesis": config["hypothesis"],
            "parameters": int(model.count_params()),
            "epochs_trained": int(len(history_df)),
            "best_epoch": best_index + 1,
            "training_time_seconds": float(training_seconds),
            "train_accuracy_at_best_epoch": float(history_df.loc[best_index, "accuracy"]),
            "best_validation_accuracy": float(history_df.loc[best_index, "val_accuracy"]),
            "best_validation_loss": float(history_df.loc[best_index, "val_loss"]),
            "generalization_gap": float(
                history_df.loc[best_index, "accuracy"]
                - history_df.loc[best_index, "val_accuracy"]
            ),
            "gpu_devices": gpus,
        }
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)

    comparison = pd.DataFrame(records).sort_values(
        "best_validation_accuracy", ascending=False
    )
    comparison.to_csv(RESULT_DIR / "training_comparison.csv", index=False)

    # Choose for deployment using validation data only; test data remains untouched.
    winner_name = str(comparison.iloc[0]["model"])
    shutil.copy2(MODEL_DIR / f"{winner_name}.keras", ARTIFACTS / "best_model.keras")
    selection = {
        "selected_model": winner_name,
        "selection_rule": "highest validation accuracy at the lowest-validation-loss epoch",
    }
    (RESULT_DIR / "model_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("\nSelected model for deployment:", winner_name)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when a completed model and history already exist.",
    )
    arguments = parser.parse_args()
    print(train_all(force=arguments.force).to_string(index=False))
