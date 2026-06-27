"""
Risk stratification: maps dropout probabilities to High/Medium/Low tiers
and assigns personalised interventions based on each student's actual risk drivers.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_THRESHOLDS = {"High": 0.70, "Medium": 0.40}


# Each rule: (label, condition_fn, intervention_text)
# condition_fn receives a row (pd.Series) and returns True if the factor is present
_INTERVENTION_RULES = [
    (
        "financial_fees",
        lambda r: r.get("tuition_fees_up_to_date", 1) == 0,
        "Emergency tuition fee review & payment plan setup",
    ),
    (
        "financial_debt",
        lambda r: r.get("debtor", 0) == 1,
        "Financial counselling referral — outstanding debt identified",
    ),
    (
        "no_scholarship",
        lambda r: r.get("scholarship_holder", 1) == 0 and r.get("debtor", 0) == 1,
        "Scholarship & bursary eligibility assessment",
    ),
    (
        "sem1_academic",
        lambda r: (r.get("curricular_units_1st_sem_enrolled", 1) > 0 and
                   r.get("curricular_units_1st_sem_approved", 1) /
                   max(r.get("curricular_units_1st_sem_enrolled", 1), 1) < 0.5),
        "Semester 1 academic recovery plan & tutoring referral",
    ),
    (
        "sem2_academic",
        lambda r: (r.get("curricular_units_2nd_sem_enrolled", 1) > 0 and
                   r.get("curricular_units_2nd_sem_approved", 1) /
                   max(r.get("curricular_units_2nd_sem_enrolled", 1), 1) < 0.5),
        "Semester 2 academic recovery plan & tutoring referral",
    ),
    (
        "low_grades",
        lambda r: ((r.get("curricular_units_1st_sem_grade", 10) +
                    r.get("curricular_units_2nd_sem_grade", 10)) / 2) < 8,
        "Academic skills workshop — low grade average detected",
    ),
    (
        "missed_evaluations",
        lambda r: r.get("curricular_units_1st_sem_without_evaluations", 0) > 2,
        "Engagement alert — missed assessments in semester 1",
    ),
    (
        "low_attendance",
        lambda r: r.get("attendance_trend_score", 1.0) < 0.65,
        "Attendance support plan — declining attendance pattern",
    ),
    (
        "very_low_attendance",
        lambda r: r.get("attendance_trend_score", 1.0) < 0.50,
        "Urgent attendance intervention — below 50% attendance",
    ),
    (
        "displaced",
        lambda r: r.get("displaced", 0) == 1,
        "Relocation & housing support services referral",
    ),
    (
        "mature_student",
        lambda r: r.get("age_at_enrollment", 20) >= 30,
        "Mature student support group & flexible study options",
    ),
    (
        "international",
        lambda r: r.get("international", 0) == 1,
        "International student welfare & visa/academic support",
    ),
    (
        "low_parental_education",
        lambda r: r.get("parental_education", 3) <= 1,
        "First-generation student mentoring programme",
    ),
    (
        "low_admission",
        lambda r: r.get("admission_grade", 150) < 110,
        "Foundation skills assessment — low entry qualification",
    ),
]

# Fallback interventions when no specific rules fire
_FALLBACK = {
    "High":   "Immediate academic counselling referral & weekly advisor check-in",
    "Medium": "Bi-weekly progress monitoring with academic advisor",
    "Low":    "Standard monthly check-in & resource awareness",
}


def _personalise(row: pd.Series, tier: str, max_interventions: int = 3) -> str:
    """Return a personalised intervention string for one student."""
    fired = []
    for key, condition, text in _INTERVENTION_RULES:
        try:
            if condition(row):
                fired.append(text)
        except Exception:
            continue
        if len(fired) >= max_interventions:
            break

    if not fired:
        return _FALLBACK[tier]
    return "; ".join(fired)


def _assign_tier(prob: float, thresholds: dict) -> str:
    if prob >= thresholds["High"]:
        return "High"
    if prob >= thresholds["Medium"]:
        return "Medium"
    return "Low"


def stratify_students(
    student_ids: np.ndarray,
    dropout_probs: np.ndarray,
    output_dir: str = "reports",
    source_df: pd.DataFrame | None = None,
    attendance_trend_scores: np.ndarray | None = None,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    tiers = [_assign_tier(p, thresholds) for p in dropout_probs]

    # Build a lookup df for personalisation
    if source_df is not None:
        lookup = source_df.copy().reset_index(drop=True)
        if "student_id" in lookup.columns:
            lookup = lookup.set_index("student_id")
    else:
        lookup = None

    interventions = []
    for sid, tier in zip(student_ids, tiers):
        if lookup is not None and sid in lookup.index:
            row = lookup.loc[sid]
            # Attach attendance trend score if available
            if attendance_trend_scores is not None:
                idx = list(student_ids).index(sid)
                row = row.copy()
                row["attendance_trend_score"] = attendance_trend_scores[idx]
            interventions.append(_personalise(row, tier))
        else:
            interventions.append(_FALLBACK[tier])

    df = pd.DataFrame({
        "student_id":          student_ids,
        "dropout_probability": np.round(dropout_probs, 4),
        "risk_tier":           tiers,
        "interventions":       interventions,
    })
    df = df.sort_values("dropout_probability", ascending=False).reset_index(drop=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "risk_report.csv", index=False)
    df.to_json(out / "risk_report.json", orient="records", indent=2)

    counts = df["risk_tier"].value_counts()
    print(f"  Risk distribution — High: {counts.get('High',0)}, "
          f"Medium: {counts.get('Medium',0)}, Low: {counts.get('Low',0)}")
    return df
