"""Deep learning model training for IoT IDS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.sparse import issparse
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def _sample_rows(x_data, y_data, max_samples: int | None):
    """Sample aligned feature/label rows when dataset is very large."""
    if max_samples is None or x_data.shape[0] <= max_samples:
        return x_data, y_data

    indices = np.random.default_rng(42).choice(x_data.shape[0], size=max_samples, replace=False)
    indices = np.sort(indices)
    return x_data[indices], y_data[indices]


def _to_dense_float32(x_data) -> np.ndarray:
    """Convert sparse/dense feature matrix into dense float32 array."""
    if issparse(x_data):
        x_data = x_data.toarray()
    return np.asarray(x_data, dtype=np.float32)


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Find threshold maximizing F1 on validation set."""
    candidate_thresholds = np.linspace(0.2, 0.8, 31)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidate_thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def build_dl_model(input_dim: int) -> tf.keras.Model:
    """Create a compact feed-forward neural network for binary classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dl_model(
    x_train,
    y_train,
    models_dir: Path,
    epochs: int = 30,
    batch_size: int = 256,
    max_samples: int | None = 60000,
):
    """Train and save deep neural network model with callbacks."""
    y_train_np = y_train.to_numpy(dtype=np.float32)
    x_train_sampled, y_train_sampled = _sample_rows(x_train, y_train_np, max_samples=max_samples)
    x_train_dense = _to_dense_float32(x_train_sampled)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_dense,
        y_train_sampled,
        test_size=0.2,
        random_state=42,
        stratify=y_train_sampled,
    )

    model = build_dl_model(input_dim=x_fit.shape[1])
    checkpoint_path = models_dir / "dl_best_model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    val_probs = model.predict(x_val, verbose=0).flatten()
    best_threshold = tune_threshold(y_val.astype(int), val_probs)

    final_model_path = models_dir / "dl_model.keras"
    model.save(final_model_path)

    return model, best_threshold


def predict_dl(model: tf.keras.Model, x_test, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Generate probabilities and binary predictions for deep learning model."""
    x_test_dense = _to_dense_float32(x_test)
    probabilities = model.predict(x_test_dense, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions
