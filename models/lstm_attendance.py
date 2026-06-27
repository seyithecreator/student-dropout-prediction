"""
LSTM model for next-day attendance forecasting from 30-day sliding windows.

Training (train.py only) — uses TensorFlow.
Inference (app.py)       — uses NumPy only; loads weights from lstm_weights.npz.
"""

import numpy as np
from pathlib import Path


# ── NumPy forward pass ────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_forward(X_batch: np.ndarray, weights: dict) -> np.ndarray:
    """
    Pure-NumPy LSTM forward pass matching Keras default (tanh cell, sigmoid gates).
    X_batch : (n, timesteps, 1)  float32
    Returns : (n,)  float32  probabilities
    """
    W  = weights["kernel"]           # (1, 128)
    U  = weights["recurrent_kernel"] # (32, 128)
    b  = weights["bias"]             # (128,)
    Wd = weights["dense_kernel"]     # (32, 1)
    bd = weights["dense_bias"]       # (1,)

    units = 32
    n = X_batch.shape[0]
    h = np.zeros((n, units), dtype=np.float32)
    c = np.zeros((n, units), dtype=np.float32)

    for t in range(X_batch.shape[1]):
        x      = X_batch[:, t, :]                  # (n, 1)
        gates  = x @ W + h @ U + b                # (n, 128)
        i_gate = _sigmoid(gates[:, :units])
        f_gate = _sigmoid(gates[:, units:2*units])
        g_gate = np.tanh(gates[:, 2*units:3*units])
        o_gate = _sigmoid(gates[:, 3*units:])
        c      = f_gate * c + i_gate * g_gate
        h      = o_gate * np.tanh(c)

    logit = h @ Wd + bd               # (n, 1)
    return _sigmoid(logit).ravel().astype(np.float32)


def load_weights(path: str = "models/lstm_weights.npz") -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ── Training (TensorFlow, only called from train.py) ─────────────────────────

def _build_keras_model(window: int = 30):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

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


def _extract_and_save_weights(keras_model, npz_path: str = "models/lstm_weights.npz"):
    lstm_layer  = keras_model.get_layer("lstm")
    dense_layer = keras_model.get_layer("dense")
    kernel, r_kernel, bias = lstm_layer.get_weights()
    W_dense, b_dense       = dense_layer.get_weights()
    np.savez(npz_path,
             kernel=kernel, recurrent_kernel=r_kernel, bias=bias,
             dense_kernel=W_dense, dense_bias=b_dense)
    print(f"  LSTM weights saved to {npz_path}")


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
    weights_path: str = "models/lstm_weights.npz",
) -> tuple[dict, dict]:
    """
    Trains the Keras LSTM, saves weights as .npz for NumPy inference,
    and returns (weights_dict, metrics).  The .keras file is also saved
    but is only needed for future re-training, not inference.
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("  Building LSTM model...")
    model = _build_keras_model(window)

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

    _extract_and_save_weights(model, weights_path)

    weights = load_weights(weights_path)
    y_pred_prob = _lstm_forward(
        X_seq_test[:, -window:].reshape(-1, window, 1).astype(np.float32), weights
    )
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = y_seq_test.astype(int)

    metrics = {
        "lstm_accuracy": float(np.mean(y_pred == y_true)),
        "lstm_mae":      float(np.mean(np.abs(y_pred_prob - y_seq_test))),
        "lstm_rmse":     float(np.sqrt(np.mean((y_pred_prob - y_seq_test) ** 2))),
    }
    print(f"  LSTM — accuracy: {metrics['lstm_accuracy']:.4f} | MAE: {metrics['lstm_mae']:.4f}")
    return weights, metrics


# ── Feature generation (used by both train.py and app.py) ─────────────────────

def generate_features(
    lstm_weights: dict,
    attendance_matrix: np.ndarray,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (attendance_trend_score, lstm_next_day_prob) — both shape (n_students,).
    lstm_weights : dict loaded from lstm_weights.npz
    """
    n = len(attendance_matrix)
    trend_scores = attendance_matrix[:, -window:].mean(axis=1).astype(np.float32)
    last_windows = attendance_matrix[:, -window:].reshape(n, window, 1).astype(np.float32)
    lstm_probs   = _lstm_forward(last_windows, lstm_weights)
    return trend_scores, lstm_probs


def compute_attendance_trend_scores(
    attendance_matrix: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Rolling mean of the last `window` days — used when LSTM weights unavailable."""
    return attendance_matrix[:, -window:].mean(axis=1).astype(np.float32)
