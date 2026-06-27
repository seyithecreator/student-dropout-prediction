"""
Offline training script — run once to train all models and save them to disk.

Usage:
    python train.py [--csv data/students.csv] [--attendance data/attendance_sequences.npy]

The app (app.py) loads the saved models for fast inference — no retraining needed.
"""

import os, sys, time, json, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(csv_path: str, attendance_path: str):
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")
    df = pd.read_csv(csv_path)
    if "student_id" not in df.columns:
        df.insert(0, "student_id", [f"STU{i:05d}" for i in range(len(df))])
    print(f"  {len(df)} students loaded from {csv_path}")

    if Path(attendance_path).exists():
        attendance_matrix = np.load(attendance_path)
        print(f"  Attendance matrix: {attendance_matrix.shape}")
    else:
        print("  Attendance file not found — generating synthetic sequences...")
        from data.synthetic_generator import generate_attendance_sequences
        attendance_matrix = generate_attendance_sequences(df)
        np.save(attendance_path, attendance_matrix)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/8] Preprocessing...")
    from preprocessing.pipeline import run as preprocess
    data = preprocess(df, attendance_matrix)

    # ── 3. Train LSTM ─────────────────────────────────────────────────────────
    print("\n[3/8] Training LSTM attendance model...")
    from models.lstm_attendance import train as train_lstm, generate_features
    lstm_model, lstm_metrics = train_lstm(
        data["X_seq_train"], data["y_seq_train"],
        data["X_seq_test"],  data["y_seq_test"],
    )

    # ── 4. Generate LSTM attendance features → augment tabular data ───────────
    print("\n[4/8] Generating attendance features + carving calibration split...")
    n = len(df)
    test_idx = data["test_indices"]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    att_train = attendance_matrix[train_mask]
    att_test  = attendance_matrix[test_idx]

    trend_train, lstm_p_train = generate_features(lstm_model, att_train)
    trend_test,  lstm_p_test  = generate_features(lstm_model, att_test)

    X_train_aug = np.column_stack([data["X_train_unsmote"], trend_train, lstm_p_train])
    X_test_aug  = np.column_stack([data["X_test"],          trend_test,  lstm_p_test])
    y_train_orig = pd.Series(df["target"].values[train_mask], name="target")

    # Carve out a calibration set (12.5% of total ≈ 10% of data) before SMOTE
    from sklearn.model_selection import train_test_split as _tts
    X_tr_aug, X_cal, y_tr_aug, y_cal = _tts(
        X_train_aug, y_train_orig,
        test_size=0.125, random_state=42, stratify=y_train_orig,
    )

    from preprocessing.pipeline import apply_smote
    X_train_final, y_train_final = apply_smote(X_tr_aug, y_tr_aug)

    feature_names = data["feature_names"] + ["attendance_trend_score", "lstm_next_day_prob"]
    print(f"  Feature matrix: {X_train_final.shape[1]} features "
          f"(tabular + engineered + 2 attendance features)")

    # ── 5. Train stacking ensemble ────────────────────────────────────────────
    print("\n[5/8] Training stacking ensemble...")
    from models.ensemble_model import train as train_ensemble
    ensemble = train_ensemble(X_train_final, y_train_final, feature_names)

    # ── 6. SHAP explainability ────────────────────────────────────────────────
    print("\n[6/8] SHAP explainability...")
    from explainability.shap_explainer import run as run_shap
    raw_probs_test = ensemble.predict_proba(X_test_aug)[:, 1]
    shap_explainer = run_shap(
        ensemble.get_xgb_estimator(),
        X_train_final, X_test_aug,
        feature_names, list(data["student_ids_test"]),
        raw_probs_test, n_local=2,
    )
    top_features = shap_explainer.get_top_feature_names(feature_names, n=10)

    # ── 7. Calibrate probabilities + compute optimal threshold ────────────────
    print("\n[7/8] Calibrating probabilities...")
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import precision_recall_curve
    calibrated = CalibratedClassifierCV(FrozenEstimator(ensemble.model), method="isotonic")
    calibrated.fit(X_cal, y_cal.values)
    joblib.dump(calibrated, "models/stacking_model.joblib")
    print("  Calibrated model saved.")

    dropout_probs = calibrated.predict_proba(X_test_aug)[:, 1]
    precisions, recalls, pr_thresholds = precision_recall_curve(data["y_test"], dropout_probs)
    f1_arr = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    optimal_threshold = float(round(pr_thresholds[np.argmax(f1_arr)], 4))
    tier_thresholds = {
        "High":   optimal_threshold,
        "Medium": round(optimal_threshold * 0.55, 4),
    }
    print(f"  Optimal threshold: {optimal_threshold:.4f}  "
          f"(High ≥ {tier_thresholds['High']}, Medium ≥ {tier_thresholds['Medium']})")

    # ── 8. Evaluate + risk stratification ─────────────────────────────────────
    print("\n[8/8] Evaluating and stratifying risk...")
    from evaluation.metrics import evaluate_ensemble
    ensemble_metrics = evaluate_ensemble(calibrated, X_test_aug, data["y_test"])

    from models.risk_stratification import stratify_students
    risk_df = stratify_students(
        student_ids=data["student_ids_test"],
        dropout_probs=dropout_probs,
        output_dir="reports",
        source_df=df.iloc[test_idx].reset_index(drop=True),
        attendance_trend_scores=trend_test,
        thresholds=tier_thresholds,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    results_payload = {
        "ensemble_metrics": ensemble_metrics,
        "lstm_metrics":     lstm_metrics,
        "top_features":     top_features,
        "feature_names":    feature_names,
        "tier_thresholds":  tier_thresholds,
        "run_timestamp":    datetime.now().strftime("%d %b %Y · %H:%M"),
        "training_time_s":  elapsed,
        "n_students":       len(df),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/pipeline_results.json").write_text(json.dumps(results_payload, indent=2))

    # Save feature names for inference
    joblib.dump(feature_names, "models/feature_names.joblib")

    print(f"\n✓ Training complete in {elapsed}s")
    print(f"  Accuracy : {ensemble_metrics['accuracy']}")
    print(f"  ROC-AUC  : {ensemble_metrics['roc_auc']}")
    print(f"  Models saved to models/")
    print(f"  Results  saved to reports/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dropout prediction models")
    parser.add_argument("--csv",        default="data/students.csv")
    parser.add_argument("--attendance", default="data/attendance_sequences.npy")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"Error: {args.csv} not found.")
        print("Run the app first and click 'Generate Dataset', then re-run train.py")
        sys.exit(1)

    main(args.csv, args.attendance)
