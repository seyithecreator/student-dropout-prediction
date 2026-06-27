"""
Stacking ensemble: Random Forest + XGBoost + LightGBM → Logistic Regression meta-learner.
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BASE_ESTIMATORS = [
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    (
        "xgboost",
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=2,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    ),
    (
        "lightgbm",
        LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
    ),
]

META_LEARNER = LogisticRegression(C=1.0, max_iter=1000, random_state=42)


class StackingDropoutModel:
    def __init__(self):
        self.model = StackingClassifier(
            estimators=BASE_ESTIMATORS,
            final_estimator=META_LEARNER,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []
        self._xgb_estimator = None  # cached reference for SHAP

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str] | None = None):
        if feature_names:
            self.feature_names = feature_names
        print("  Training stacking ensemble (RF + XGBoost + LightGBM → LR)...")
        self.model.fit(X_train, y_train)
        # Cache the fitted XGBoost base learner for SHAP
        self._xgb_estimator = self.model.named_estimators_["xgboost"]
        print("  Training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_xgb_estimator(self):
        return self._xgb_estimator

    def save(self, path: str = "models/stacking_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  Model saved to {path}")

    @staticmethod
    def load(path: str = "models/stacking_model.joblib") -> "StackingDropoutModel":
        return joblib.load(path)


def train(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> StackingDropoutModel:
    model = StackingDropoutModel()
    model.fit(X_train, y_train, feature_names)
    model.save()
    return model
