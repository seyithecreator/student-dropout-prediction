"""
Generates synthetic student data for dropout prediction.

Produces two outputs:
  - data/students.csv         : tabular student records
  - data/attendance_sequences.npy : (n_students, 180) binary daily attendance
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_STUDENTS = 5000
N_DAYS = 180
DROPOUT_RATE = 0.30


def _dropout_probability(row: dict) -> float:
    """Compute a latent dropout score (0-1) from a student feature dict."""
    score = 0.0

    # Socio-economic risk factors
    if not row["scholarship_holder"]:
        score += 0.10
    if row["debtor"]:
        score += 0.12
    if not row["tuition_fees_up_to_date"]:
        score += 0.15
    if row["parental_education"] <= 1:
        score += 0.05
    if row["displaced"]:
        score += 0.04
    if row["unemployment_rate"] > 12:
        score += 0.05

    # Academic risk factors
    score += max(0, (120 - row["admission_grade"]) / 120) * 0.15
    approval_rate_1 = (
        row["curricular_units_1st_sem_approved"]
        / max(row["curricular_units_1st_sem_enrolled"], 1)
    )
    approval_rate_2 = (
        row["curricular_units_2nd_sem_approved"]
        / max(row["curricular_units_2nd_sem_enrolled"], 1)
    )
    score += (1 - approval_rate_1) * 0.18
    score += (1 - approval_rate_2) * 0.18

    grade_1 = row["curricular_units_1st_sem_grade"]
    grade_2 = row["curricular_units_2nd_sem_grade"]
    avg_grade = (grade_1 + grade_2) / 2
    score += max(0, (10 - avg_grade) / 10) * 0.10

    if row["curricular_units_1st_sem_without_evaluations"] > 2:
        score += 0.08

    return float(np.clip(score + RNG.normal(0, 0.05), 0.0, 1.0))


def generate_students(n: int = N_STUDENTS) -> pd.DataFrame:
    records = []
    for i in range(n):
        enrolled_1 = int(RNG.integers(3, 9))
        approved_1 = int(RNG.integers(0, enrolled_1 + 1))
        enrolled_2 = int(RNG.integers(3, 9))
        approved_2 = int(RNG.integers(0, enrolled_2 + 1))

        row = {
            "student_id": f"STU{i:05d}",
            # Socio-economic
            "age_at_enrollment": int(RNG.integers(17, 55)),
            "gender": int(RNG.integers(0, 2)),
            "scholarship_holder": bool(RNG.random() < 0.30),
            "debtor": bool(RNG.random() < 0.20),
            "tuition_fees_up_to_date": bool(RNG.random() < 0.75),
            "parental_education": int(RNG.integers(0, 5)),
            "parental_occupation": int(RNG.integers(0, 10)),
            "displaced": bool(RNG.random() < 0.12),
            "international": bool(RNG.random() < 0.05),
            "gdp": float(RNG.uniform(-4.0, 4.0)),
            "inflation_rate": float(RNG.uniform(0.5, 5.0)),
            "unemployment_rate": float(RNG.uniform(6.0, 18.0)),
            # Enrollment-time academic
            "application_mode": int(RNG.integers(1, 18)),
            "course_id": int(RNG.integers(1, 20)),
            "daytime_evening_attendance": int(RNG.integers(0, 2)),
            "previous_qualification": int(RNG.integers(1, 12)),
            "admission_grade": float(RNG.uniform(95.0, 190.0)),
            # 1st semester
            "curricular_units_1st_sem_credited": int(RNG.integers(0, 3)),
            "curricular_units_1st_sem_enrolled": enrolled_1,
            "curricular_units_1st_sem_evaluations": int(RNG.integers(0, enrolled_1 * 3 + 1)),
            "curricular_units_1st_sem_approved": approved_1,
            "curricular_units_1st_sem_grade": float(RNG.uniform(0, 18)),
            "curricular_units_1st_sem_without_evaluations": int(RNG.integers(0, 4)),
            # 2nd semester
            "curricular_units_2nd_sem_credited": int(RNG.integers(0, 3)),
            "curricular_units_2nd_sem_enrolled": enrolled_2,
            "curricular_units_2nd_sem_evaluations": int(RNG.integers(0, enrolled_2 * 3 + 1)),
            "curricular_units_2nd_sem_approved": approved_2,
            "curricular_units_2nd_sem_grade": float(RNG.uniform(0, 18)),
            "curricular_units_2nd_sem_without_evaluations": int(RNG.integers(0, 4)),
        }
        records.append(row)

    df = pd.DataFrame(records)

    # Compute dropout probabilities and assign labels
    probs = np.array([_dropout_probability(r) for r in records])
    threshold = np.quantile(probs, 1 - DROPOUT_RATE)
    df["dropout_probability"] = probs
    df["target"] = (probs >= threshold).astype(int)  # 1 = Dropout

    return df


def generate_attendance_sequences(df: pd.DataFrame) -> np.ndarray:
    """
    Returns (n_students, N_DAYS) binary attendance matrix.
    Dropout students have progressively lower attendance in the final 60 days.
    """
    n = len(df)
    sequences = np.ones((n, N_DAYS), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        base_prob = 0.90 - row["dropout_probability"] * 0.25  # 0.65–0.90

        for day in range(N_DAYS):
            p = base_prob

            # Weekly pattern: lower on day 4 (Friday, 0-indexed Mon–Fri cycle)
            if day % 5 == 4:
                p -= 0.08

            # Seasonal dips: weeks 6–8 (days 25–39) and weeks 14–16 (days 65–79)
            if 25 <= day <= 39 or 65 <= day <= 79:
                p -= 0.06

            # Dropout students trail off in the final 60 days
            if row["target"] == 1 and day >= (N_DAYS - 60):
                decay = (day - (N_DAYS - 60)) / 60
                p -= decay * 0.35

            sequences[i, day] = float(RNG.random() < np.clip(p, 0.05, 1.0))

    return sequences


def run(output_dir: str = "data", n_students: int = N_STUDENTS) -> tuple[pd.DataFrame, np.ndarray]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic student records...")
    df = generate_students(n_students)
    df.to_csv(out / "students.csv", index=False)
    print(f"  Saved {len(df)} records to {out / 'students.csv'}")
    print(f"  Dropout rate: {df['target'].mean():.1%}")

    print("Generating attendance sequences...")
    sequences = generate_attendance_sequences(df)
    np.save(out / "attendance_sequences.npy", sequences)
    print(f"  Saved shape {sequences.shape} to {out / 'attendance_sequences.npy'}")

    return df, sequences


if __name__ == "__main__":
    run()
