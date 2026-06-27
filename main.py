"""
End-to-end pipeline: data generation → preprocessing → ensemble training →
LSTM training → SHAP explainability → risk stratification → evaluation → HTML report.
"""

import os
import sys
import numpy as np

# Suppress TF/SHAP noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

# Ensure project root is on path when run from subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_generator import run as generate_data
from preprocessing.pipeline import run as preprocess
from models.ensemble_model import train as train_ensemble
from models.lstm_attendance import train as train_lstm, compute_attendance_trend_scores
from explainability.shap_explainer import run as run_shap
from evaluation.metrics import evaluate_ensemble
from models.risk_stratification import stratify_students
from reports.report_generator import generate as generate_report


def main():
    print("\n" + "=" * 60)
    print("  Student Dropout Prediction System")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n[1/8] Generating synthetic data...")
    df, attendance_matrix = generate_data(output_dir="data")

    # 2. Preprocess
    print("\n[2/8] Preprocessing features...")
    data = preprocess(df, attendance_matrix)

    # 3. Train stacking ensemble
    print("\n[3/8] Training stacking ensemble...")
    model = train_ensemble(data["X_train"], data["y_train"], data["feature_names"])

    # 4. Train LSTM attendance model
    print("\n[4/8] Training LSTM attendance model...")
    lstm_model, lstm_metrics = train_lstm(
        data["X_seq_train"], data["y_seq_train"],
        data["X_seq_test"], data["y_seq_test"],
    )

    # Compute attendance trend scores for test students
    attendance_trend_scores = compute_attendance_trend_scores(data["attendance_matrix"])
    # Map to test set indices (attendance_matrix rows match df order; test indices from stratified split)
    # We approximate by using full-dataset trend scores aligned to student order
    n_test = len(data["X_test"])
    trend_scores_test = attendance_trend_scores[-n_test:]

    # 5. SHAP explainability
    print("\n[5/8] Running SHAP explainability...")
    xgb_model = model.get_xgb_estimator()
    dropout_probs_test = model.predict_proba(data["X_test"])[:, 1]
    shap_explainer = run_shap(
        xgb_model,
        data["X_train"],
        data["X_test"],
        data["feature_names"],
        list(data["student_ids_test"]),
        dropout_probs_test,
        n_local=5,
    )
    top_features = shap_explainer.get_top_feature_names(data["feature_names"], n=10)
    print(f"  Top features: {top_features[:5]}")

    # 6. Evaluate ensemble
    print("\n[6/8] Evaluating ensemble classifier...")
    ensemble_metrics = evaluate_ensemble(
        model, data["X_test"], data["y_test"],
        data["X_train"], data["y_train"],
    )

    # 7. Risk stratification
    print("\n[7/8] Stratifying students by risk...")
    risk_df = stratify_students(
        student_ids=data["student_ids_test"],
        dropout_probs=dropout_probs_test,
        attendance_trend_scores=trend_scores_test,
        output_dir="reports",
    )

    # 8. Generate HTML report
    print("\n[8/8] Generating HTML report...")
    generate_report(
        ensemble_metrics=ensemble_metrics,
        lstm_metrics=lstm_metrics,
        risk_df=risk_df,
        top_features=top_features,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Reports: reports/final_report.html")
    print("           reports/risk_report.csv")
    print("           reports/risk_report.json")
    print("  Figures: reports/figures/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
