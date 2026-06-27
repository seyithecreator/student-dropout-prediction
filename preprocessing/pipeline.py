"""
Data preprocessing pipeline: encoding, scaling, SMOTE, and LSTM sequence preparation.
Supports both training mode (fit + transform) and inference mode (transform only).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

CATEGORICAL_COLS = [
    "application_mode", "course_id", "daytime_evening_attendance",
    "previous_qualification", "parental_education", "parental_occupation", "gender",
]
BINARY_COLS = [
    "scholarship_holder", "debtor", "tuition_fees_up_to_date", "displaced", "international",
]
DROP_COLS = ["student_id", "target", "dropout_probability"]
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio and flag features that carry stronger signal than raw counts."""
    df = df.copy()

    def _col(name, default):
        return df[name].astype(float) if name in df.columns else pd.Series(float(default), index=df.index)

    e1 = _col("curricular_units_1st_sem_enrolled", 1).clip(lower=1)
    e2 = _col("curricular_units_2nd_sem_enrolled", 1).clip(lower=1)

    df["approval_rate_sem1"] = (_col("curricular_units_1st_sem_approved", 0) / e1).clip(0, 1)
    df["approval_rate_sem2"] = (_col("curricular_units_2nd_sem_approved", 0) / e2).clip(0, 1)
    df["grade_delta"]        = _col("curricular_units_2nd_sem_grade", 0) - _col("curricular_units_1st_sem_grade", 0)
    df["financial_stress"]   = (
        (_col("debtor", 0).astype(int) == 1) & (_col("tuition_fees_up_to_date", 1).astype(int) == 0)
    ).astype(float)
    return df


def load_and_split(df, test_size=0.20, random_state=42):
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in BINARY_COLS:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return pd.get_dummies(
        X, columns=[c for c in CATEGORICAL_COLS if c in X.columns], drop_first=True
    ).astype(float)


def prepare_lstm_sequences(attendance_matrix: np.ndarray, window: int = 30):
    """Vectorised sliding window using stride tricks — much faster than nested loops."""
    from numpy.lib.stride_tricks import sliding_window_view
    all_X, all_y = [], []
    for seq in attendance_matrix:
        windows = sliding_window_view(seq, window)   # shape (T-window, window)
        all_X.append(windows[:-1])                   # inputs
        all_y.append(seq[window:])                   # targets (next-day)
    X_seq = np.concatenate(all_X, axis=0).astype(np.float32).reshape(-1, window, 1)
    y_seq = np.concatenate(all_y, axis=0).astype(np.float32)
    return X_seq, y_seq


# ── Training mode ─────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, attendance_matrix: np.ndarray) -> dict:
    """Full preprocessing for training: fit imputer + scaler, apply SMOTE."""
    print("Preprocessing tabular features...")
    df = engineer_features(df)
    X_train_raw, X_test_raw, y_train, y_test = load_and_split(df)

    X_train_enc = encode_features(X_train_raw)
    X_test_enc  = encode_features(X_test_raw)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    train_columns = list(X_train_enc.columns)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_enc)
    X_test_imp  = imputer.transform(X_test_enc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)
    feature_names  = train_columns

    # Save preprocessor for inference
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "train_columns": train_columns},
        PREPROCESSOR_PATH,
    )
    print(f"  Preprocessor saved to {PREPROCESSOR_PATH}")

    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    print("Preparing LSTM sequences...")
    X_seq, y_seq = prepare_lstm_sequences(attendance_matrix)
    split = int(len(X_seq) * 0.80)

    # Keep train indices for attendance feature alignment
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42,
        stratify=df["target"].values,
    )

    return {
        "X_train":         X_train_smote,
        "y_train":         y_train_smote,
        "X_train_unsmote": X_train_scaled,   # pre-SMOTE, same rows as attendance
        "X_test":          X_test_scaled,
        "y_test":          y_test.values,
        "feature_names":   feature_names,
        "scaler":          scaler,
        "student_ids_test": df.iloc[test_idx]["student_id"].values,
        "test_indices":    test_idx,
        "X_seq_train":     X_seq[:split],
        "y_seq_train":     y_seq[:split],
        "X_seq_test":      X_seq[split:],
        "y_seq_test":      y_seq[split:],
        "attendance_matrix": attendance_matrix,
    }


def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_r, y_r = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE: {len(y_train)} → {len(y_r)} samples (dropout: {y_r.mean():.1%})")
    return X_r, y_r


# ── Inference mode ─────────────────────────────────────────────────────────────

def load_preprocessor() -> dict:
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. Run train.py first."
        )
    return joblib.load(PREPROCESSOR_PATH)


def transform_for_inference(df: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    """Transform a new CSV using the saved imputer + scaler (no refitting)."""
    df = engineer_features(df)
    X = df.drop(columns=DROP_COLS, errors="ignore")
    X_enc = encode_features(X)

    # Align to training columns — add missing dummies as 0, drop extras
    train_cols = preprocessor["train_columns"]
    X_enc = X_enc.reindex(columns=train_cols, fill_value=0)

    X_imp    = preprocessor["imputer"].transform(X_enc)
    X_scaled = preprocessor["scaler"].transform(X_imp)
    return X_scaled.astype(np.float32)
