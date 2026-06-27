"""
Model evaluation: classification metrics and diagnostic plots (no CV — done at training time).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

FIGURES_DIR = Path("reports/figures")


def evaluate_ensemble(model, X_test: np.ndarray, y_test: np.ndarray,
                      X_train: np.ndarray = None, y_train: np.ndarray = None) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy      = accuracy_score(y_test, y_pred)
    roc_auc       = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    report        = classification_report(y_test, y_pred,
                                          target_names=["Non-Dropout", "Dropout"],
                                          output_dict=True)

    print(f"\n  Ensemble Test Results")
    print(f"  Accuracy : {accuracy:.4f}  |  ROC-AUC: {roc_auc:.4f}  |  Avg Prec: {avg_precision:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Non-Dropout", "Dropout"]))

    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_prob, roc_auc)
    _plot_pr_curve(y_test, y_prob, avg_precision)
    _plot_calibration_curve(y_test, y_prob)

    return {
        "accuracy":             round(accuracy, 4),
        "roc_auc":              round(roc_auc, 4),
        "avg_precision":        round(avg_precision, 4),
        "precision_dropout":    round(report["Dropout"]["precision"], 4),
        "recall_dropout":       round(report["Dropout"]["recall"], 4),
        "f1_dropout":           round(report["Dropout"]["f1-score"], 4),
        "precision_nondropout": round(report["Non-Dropout"]["precision"], 4),
        "recall_nondropout":    round(report["Non-Dropout"]["recall"], 4),
        "f1_nondropout":        round(report["Non-Dropout"]["f1-score"], 4),
    }


def _plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Dropout", "Dropout"],
                yticklabels=["Non-Dropout", "Dropout"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(y_test, y_prob, auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curve(y_test, y_prob, avg_prec):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR (AP = {avg_prec:.3f})")
    ax.axhline(y=y_test.mean(), color="grey", linestyle="--", label=f"Baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_curve(y_test, y_prob):
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="steelblue", label="Ensemble")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
