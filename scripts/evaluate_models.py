from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def parse_subjects(value: str) -> list[int]:
    value = value.strip()
    if "-" in value:
        start_str, end_str = value.split("-", maxsplit=1)
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_feature_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("f")]
    if not cols:
        raise ValueError(f"No feature columns in {path}")
    return df[cols].to_numpy(dtype=np.float64)


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx]), float(fpr[idx]), float(fnr[idx])


def strict_threshold_far(scores_neg: np.ndarray, target_far: float = 0.01) -> float:
    # Using only impostor scores, pick the score quantile so accept-rate among impostors is target_far.
    # Accept if score >= threshold, so threshold is high quantile of impostor scores.
    q = 1.0 - target_far
    return float(np.quantile(scores_neg, q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-user authentication models and compute thresholds.")
    parser.add_argument("--subjects", type=str, default="1-40", help="Subject IDs, e.g. '1-40' or '1,2,3'.")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Directory of feature CSVs.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory of trained user models.")
    parser.add_argument("--thresholds-dir", type=str, default="thresholds", help="Output directory for thresholds JSONs.")
    parser.add_argument("--protocol1-run", type=int, default=1, help="Run ID for within-condition test.")
    parser.add_argument("--protocol2-run", type=int, default=2, help="Run ID for cross-condition test.")
    parser.add_argument("--target-far", type=float, default=0.01, help="Strict threshold target FAR.")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    thresholds_dir = Path(args.thresholds_dir)
    thresholds_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float]] = []

    pooled_scores_p1: list[float] = []
    pooled_labels_p1: list[int] = []
    pooled_scores_p2: list[float] = []
    pooled_labels_p2: list[int] = []

    for s in subjects:
        model_path = models_dir / f"user_{s:03d}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = joblib.load(model_path)

        def score_subject_test(run_id: int, subject_id: int) -> np.ndarray:
            x = load_feature_matrix(features_dir / f"S{subject_id:03d}_R{run_id:02d}_test.csv")
            return model.decision_function(x)

        p1_genuine = score_subject_test(args.protocol1_run, s)
        p1_impostor = np.concatenate(
            [score_subject_test(args.protocol1_run, j) for j in subjects if j != s], axis=0
        )

        p2_genuine = score_subject_test(args.protocol2_run, s)
        p2_impostor = np.concatenate(
            [score_subject_test(args.protocol2_run, j) for j in subjects if j != s], axis=0
        )

        y1 = np.hstack([np.ones(len(p1_genuine)), np.zeros(len(p1_impostor))])
        sc1 = np.hstack([p1_genuine, p1_impostor])
        y2 = np.hstack([np.ones(len(p2_genuine)), np.zeros(len(p2_impostor))])
        sc2 = np.hstack([p2_genuine, p2_impostor])

        p1_eer, p1_thr, p1_far_at_eer, p1_frr_at_eer = eer_from_scores(y1, sc1)
        p2_eer, p2_thr, p2_far_at_eer, p2_frr_at_eer = eer_from_scores(y2, sc2)

        p1_strict = strict_threshold_far(p1_impostor, target_far=args.target_far)

        payload = {
            "subject_id": s,
            "protocol1": {
                "run": int(args.protocol1_run),
                "eer": p1_eer,
                "eer_threshold": p1_thr,
                "far_at_eer_threshold": p1_far_at_eer,
                "frr_at_eer_threshold": p1_frr_at_eer,
                "strict_threshold": p1_strict,
                "strict_target_far": float(args.target_far),
                "auc": float(roc_auc_score(y1, sc1)),
            },
            "protocol2": {
                "run": int(args.protocol2_run),
                "eer": p2_eer,
                "eer_threshold": p2_thr,
                "far_at_eer_threshold": p2_far_at_eer,
                "frr_at_eer_threshold": p2_frr_at_eer,
                "auc": float(roc_auc_score(y2, sc2)),
            },
            "demo_threshold": p1_strict,
        }

        out_path = thresholds_dir / f"user_{s:03d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary_rows.append(
            {
                "subject_id": s,
                "p1_eer": p1_eer,
                "p1_auc": float(roc_auc_score(y1, sc1)),
                "p2_eer": p2_eer,
                "p2_auc": float(roc_auc_score(y2, sc2)),
                "p1_demo_threshold": p1_strict,
            }
        )

        pooled_scores_p1.extend(sc1.tolist())
        pooled_labels_p1.extend(y1.astype(int).tolist())
        pooled_scores_p2.extend(sc2.tolist())
        pooled_labels_p2.extend(y2.astype(int).tolist())

        print(
            f"user_{s:03d}: P1 EER={p1_eer:.4f}, AUC={roc_auc_score(y1, sc1):.4f} | "
            f"P2 EER={p2_eer:.4f}, AUC={roc_auc_score(y2, sc2):.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(thresholds_dir / "summary.csv", index=False)

    pooled_y1 = np.asarray(pooled_labels_p1, dtype=int)
    pooled_s1 = np.asarray(pooled_scores_p1, dtype=np.float64)
    pooled_y2 = np.asarray(pooled_labels_p2, dtype=int)
    pooled_s2 = np.asarray(pooled_scores_p2, dtype=np.float64)

    p1_eer, p1_thr, _, _ = eer_from_scores(pooled_y1, pooled_s1)
    p2_eer, p2_thr, _, _ = eer_from_scores(pooled_y2, pooled_s2)

    report = {
        "n_subjects": len(subjects),
        "protocol1": {
            "pooled_eer": p1_eer,
            "pooled_eer_threshold": p1_thr,
            "pooled_auc": float(roc_auc_score(pooled_y1, pooled_s1)),
            "mean_subject_eer": float(summary_df["p1_eer"].mean()),
            "std_subject_eer": float(summary_df["p1_eer"].std(ddof=0)),
        },
        "protocol2": {
            "pooled_eer": p2_eer,
            "pooled_eer_threshold": p2_thr,
            "pooled_auc": float(roc_auc_score(pooled_y2, pooled_s2)),
            "mean_subject_eer": float(summary_df["p2_eer"].mean()),
            "std_subject_eer": float(summary_df["p2_eer"].std(ddof=0)),
        },
    }

    with (thresholds_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
