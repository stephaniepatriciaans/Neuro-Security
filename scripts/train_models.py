from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_subjects(value: str) -> list[int]:
    value = value.strip()
    if "-" in value:
        start_str, end_str = value.split("-", maxsplit=1)
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_feature_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return pd.read_csv(path)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("f")]
    if not cols:
        raise ValueError("No feature columns found (expected f000..f319)")
    return df[cols].to_numpy(dtype=np.float64)


def require_non_empty_features(features: np.ndarray, subject_id: int, role: str) -> np.ndarray:
    if len(features) == 0:
        raise ValueError(f"Subject {subject_id:03d} has no {role} feature windows. Re-run extraction with artifact rejection disabled or a looser threshold.")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-user verification models.")
    parser.add_argument("--subjects", type=str, default="1-40", help="Subject IDs, e.g. '1-40' or '1,2,3'.")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Directory of extracted feature CSVs.")
    parser.add_argument("--models-dir", type=str, default="models", help="Output directory for user models.")
    parser.add_argument(
        "--train-run",
        type=int,
        default=1,
        help="Run ID used for enrollment training (default 1: eyes-open baseline).",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Feature split used for training positives/negatives.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, int]] = []

    for s in subjects:
        pos_path = features_dir / f"S{s:03d}_R{args.train_run:02d}_{args.train_split}.csv"
        pos_df = load_feature_frame(pos_path)
        pos_x = require_non_empty_features(feature_matrix(pos_df), s, "positive")

        neg_arrays: list[np.ndarray] = []
        for j in subjects:
            if j == s:
                continue
            neg_path = features_dir / f"S{j:03d}_R{args.train_run:02d}_{args.train_split}.csv"
            neg_df = load_feature_frame(neg_path)
            neg_x_j = feature_matrix(neg_df)
            if len(neg_x_j) == 0:
                print(f"Skipping subject {j:03d} as a negative source because it has no training windows.")
                continue
            neg_arrays.append(neg_x_j)

        if not neg_arrays:
            raise ValueError(f"No negative enrollment windows available to train user {s:03d}.")

        neg_x = np.vstack(neg_arrays)

        x_train = np.vstack([pos_x, neg_x])
        y_train = np.hstack([np.ones(len(pos_x), dtype=int), np.zeros(len(neg_x), dtype=int)])

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)),
            ]
        )
        pipeline.fit(x_train, y_train)

        model_path = models_dir / f"user_{s:03d}.pkl"
        joblib.dump(pipeline, model_path)

        print(
            f"Trained user_{s:03d}: positives={len(pos_x)}, negatives={len(neg_x)}, "
            f"saved={model_path}"
        )
        summary_rows.append(
            {
                "subject_id": s,
                "n_positive": int(len(pos_x)),
                "n_negative": int(len(neg_x)),
                "train_run": int(args.train_run),
                "train_split": args.train_split,
            }
        )

    pd.DataFrame(summary_rows).to_csv(models_dir / "training_summary.csv", index=False)
    print("Training complete.")


if __name__ == "__main__":
    main()
