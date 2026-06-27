"""
LSTM model for next-day attendance forecasting from 30-day sliding windows.
Also generates per-student attendance features for ensemble feature augmentation.
"""

import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path


def build_model(window: int = 30) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    X_seq_train: np.ndarray,
    y_seq_train: np.ndarray,
    X_seq_test: np.ndarray,
    y_seq_test: np.ndarray,
    window: int = 30,
    epochs: int = 8,
    batch_size: int = 4096,
    max_samples: int = 20_000,
    save_path: str = "models/lstm_attendance.keras",
) -> tuple[tf.keras.Model, dict]:
    print("  Building LSTM model...")
    model = build_model(window)

    if len(X_seq_train) > max_samples:
        idx = np.random.choice(len(X_seq_train), max_samples, replace=False)
        X_seq_train = X_seq_train[idx]
        y_seq_train = y_seq_train[idx]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]

    print(f"  Training on {len(X_seq_train):,} sequences...")
    model.fit(
        X_seq_train, y_seq_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=1,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"  LSTM saved to {save_path}")

    y_pred_prob = model.predict(X_seq_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = y_seq_test.astype(int)

    metrics = {
        "lstm_accuracy": float(np.mean(y_pred == y_true)),
        "lstm_mae":      float(np.mean(np.abs(y_pred_prob - y_seq_test))),
        "lstm_rmse":     float(np.sqrt(np.mean((y_pred_prob - y_seq_test) ** 2))),
    }
    print(f"  LSTM — accuracy: {metrics['lstm_accuracy']:.4f} | MAE: {metrics['lstm_mae']:.4f}")
    return model, metrics


def generate_features(
    lstm_model: tf.keras.Model,
    attendance_matrix: np.ndarray,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two per-student features for ensemble augmentation:
      attendance_trend_score : mean of last `window` days  (fast, no LSTM)
      lstm_next_day_prob     : LSTM predicted prob on the last window
    Both arrays are shape (n_students,).
    """
    n = len(attendance_matrix)
    trend_scores = attendance_matrix[:, -window:].mean(axis=1).astype(np.float32)

    last_windows = attendance_matrix[:, -window:].reshape(n, window, 1).astype(np.float32)
    lstm_probs = lstm_model.predict(last_windows, batch_size=512, verbose=0).ravel()

    return trend_scores, lstm_probs.astype(np.float32)


def compute_attendance_trend_scores(
    attendance_matrix: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Rolling mean of the last `window` days — used when LSTM model unavailable."""
    return attendance_matrix[:, -window:].mean(axis=1).astype(np.float32)
