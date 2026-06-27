"""
SHAP-based explainability using TreeExplainer on the XGBoost base learner.
Generates global (summary, beeswarm) and local (waterfall) plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

FIGURES_DIR = Path("reports/figures")


class SHAPExplainer:
    def __init__(self):
        self.explainer    = None
        self.shap_values  = None
        self.X_background = None

    def fit(self, xgb_model, X_train: np.ndarray):
        print("  Computing SHAP values...")
        background = X_train[np.random.choice(len(X_train), min(150, len(X_train)), replace=False)]
        self.explainer    = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
        self.X_background = background
        self.shap_values  = self.explainer.shap_values(background)
        print("  SHAP values computed.")
        return self

    def explain_global(self, feature_names: list[str]):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(self.shap_values, self.X_background,
                          feature_names=feature_names, plot_type="bar",
                          max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary_bar.png", dpi=100, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 7))
        shap.summary_plot(self.shap_values, self.X_background,
                          feature_names=feature_names, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=100, bbox_inches="tight")
        plt.close()

        print(f"  Global SHAP plots saved to {FIGURES_DIR}/")

    def explain_local(self, student_id: str, X_row: np.ndarray, feature_names: list[str]):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        sv = self.explainer.shap_values(X_row.reshape(1, -1))
        explanation = shap.Explanation(
            values=sv[0],
            base_values=self.explainer.expected_value,
            data=X_row,
            feature_names=feature_names,
        )
        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{student_id}.png", dpi=100, bbox_inches="tight")
        plt.close()

    def get_top_feature_names(self, feature_names: list[str], n: int = 10) -> list[str]:
        if self.shap_values is None:
            return []
        mean_abs    = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:n]
        return [feature_names[i] for i in top_indices if i < len(feature_names)]

    def get_top_features(self, n: int = 10) -> list[str]:
        if self.shap_values is None:
            return []
        mean_abs    = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:n]
        return [str(i) for i in top_indices]


def run(xgb_model, X_train, X_test, feature_names, student_ids,
        dropout_probs, n_local=2) -> "SHAPExplainer":
    explainer = SHAPExplainer()
    explainer.fit(xgb_model, X_train)
    explainer.explain_global(feature_names)

    top_indices = np.argsort(dropout_probs)[::-1][:n_local]
    for idx in top_indices:
        sid = student_ids[idx] if idx < len(student_ids) else f"student_{idx}"
        explainer.explain_local(sid, X_test[idx], feature_names)
    print(f"  Local SHAP waterfall plots saved for top {n_local} students.")
    return explainer
