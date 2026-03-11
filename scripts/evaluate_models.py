from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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


def feature_path(features_dir: Path, subject_id: int, session_id: int, split: str) -> Path:
    return features_dir / f"S{subject_id:03d}_S{session_id}_{split}.csv"


def load_feature_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("f")]
    if not cols:
        raise ValueError(f"No feature columns in {path}")
    return df[cols].to_numpy(dtype=np.float64)


def score_non_empty(model, features: np.ndarray, subject_id: int, session_id: int, split: str) -> np.ndarray:
    if len(features) == 0:
        raise ValueError(
            f"Subject {subject_id:03d} session {session_id} split '{split}' has no windows after preprocessing. "
            "Re-run extraction with artifact rejection disabled or a looser threshold."
        )
    return model.decision_function(features)


def far_frr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    decisions = (scores >= threshold).astype(int)
    positive_mask = y_true == 1
    negative_mask = y_true == 0

    if np.sum(negative_mask) == 0 or np.sum(positive_mask) == 0:
        raise ValueError("Both positive and negative samples are required to compute FAR/FRR")

    far = float(np.mean(decisions[negative_mask] == 1))
    frr = float(np.mean(decisions[positive_mask] == 0))
    return far, frr


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx]), float(fpr[idx]), float(fnr[idx])


def save_score_histogram(genuine: np.ndarray, impostor: np.ndarray, threshold: float, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(genuine, bins=30, alpha=0.7, density=True, label="Genuine")
    plt.hist(impostor, bins=30, alpha=0.7, density=True, label="Impostor")
    plt.axvline(threshold, linestyle="--", label=f"Mean threshold = {threshold:.3f}")
    plt.xlabel("Decision score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_plot(y_true: np.ndarray, scores: np.ndarray, out_path: Path, title: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return auc


def save_far_frr_plot(y_true: np.ndarray, scores: np.ndarray, out_path: Path, title: str) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, fpr, label="FAR")
    plt.plot(thresholds, fnr, label="FRR")
    plt.axvline(thresholds[idx], linestyle="--", label=f"Approx EER thr = {thresholds[idx]:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {"eer": float((fpr[idx] + fnr[idx]) / 2.0), "threshold": float(thresholds[idx])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-user authentication models on Lee2019 resting-state sessions.")
    parser.add_argument("--subjects", type=str, default="1-40", help="Subject IDs, e.g. '1-40' or '1,2,3'.")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Directory of feature CSVs.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory of trained user models.")
    parser.add_argument("--thresholds-dir", type=str, default="thresholds", help="Output directory for thresholds JSONs.")
    parser.add_argument("--validation-session", type=int, default=1, help="Session ID used for threshold selection.")
    parser.add_argument("--protocol1-session", type=int, default=1, help="Session ID for within-session testing.")
    parser.add_argument("--protocol2-session", type=int, default=2, help="Session ID for cross-session testing.")
    parser.add_argument("--validation-split", type=str, default="val", help="Feature split used for threshold selection.")
    parser.add_argument("--test-split", type=str, default="test", help="Feature split used for final evaluation.")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    thresholds_dir = Path(args.thresholds_dir)
    plots_dir = thresholds_dir / "plots"
    thresholds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float]] = []
    cohort_p1_genuine: list[np.ndarray] = []
    cohort_p1_impostor: list[np.ndarray] = []
    cohort_p2_genuine: list[np.ndarray] = []
    cohort_p2_impostor: list[np.ndarray] = []
    thresholds: list[float] = []

    for s in subjects:
        model_path = models_dir / f"user_{s:03d}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = joblib.load(model_path)

        def score_subject_split(session_id: int, subject_id: int, split: str) -> np.ndarray:
            x = load_feature_matrix(feature_path(features_dir, subject_id, session_id, split))
            return score_non_empty(model, x, subject_id, session_id, split)

        val_genuine = score_subject_split(args.validation_session, s, args.validation_split)
        val_impostor = np.concatenate(
            [score_subject_split(args.validation_session, j, args.validation_split) for j in subjects if j != s],
            axis=0,
        )
        y_val = np.hstack([np.ones(len(val_genuine)), np.zeros(len(val_impostor))])
        sc_val = np.hstack([val_genuine, val_impostor])
        val_eer, val_thr, _, _ = eer_from_scores(y_val, sc_val)
        val_auc = float(roc_auc_score(y_val, sc_val))

        p1_genuine = score_subject_split(args.protocol1_session, s, args.test_split)
        p1_impostor = np.concatenate(
            [score_subject_split(args.protocol1_session, j, args.test_split) for j in subjects if j != s],
            axis=0,
        )
        p2_genuine = score_subject_split(args.protocol2_session, s, args.test_split)
        p2_impostor = np.concatenate(
            [score_subject_split(args.protocol2_session, j, args.test_split) for j in subjects if j != s],
            axis=0,
        )

        y1 = np.hstack([np.ones(len(p1_genuine)), np.zeros(len(p1_impostor))])
        sc1 = np.hstack([p1_genuine, p1_impostor])
        y2 = np.hstack([np.ones(len(p2_genuine)), np.zeros(len(p2_impostor))])
        sc2 = np.hstack([p2_genuine, p2_impostor])

        p1_far_at_val_thr, p1_frr_at_val_thr = far_frr_at_threshold(y1, sc1, val_thr)
        p2_far_at_val_thr, p2_frr_at_val_thr = far_frr_at_threshold(y2, sc2, val_thr)
        p1_auc = float(roc_auc_score(y1, sc1))
        p2_auc = float(roc_auc_score(y2, sc2))

        payload = {
            "subject_id": s,
            "decision": {
                "session": int(args.validation_session),
                "split": args.validation_split,
                "threshold": val_thr,
                "validation_eer": val_eer,
                "validation_auc": val_auc,
            },
            "protocol1": {
                "name": "within_session",
                "session": int(args.protocol1_session),
                "split": args.test_split,
                "auc": p1_auc,
                "far": p1_far_at_val_thr,
                "frr": p1_frr_at_val_thr,
            },
            "protocol2": {
                "name": "cross_session",
                "session": int(args.protocol2_session),
                "split": args.test_split,
                "auc": p2_auc,
                "far": p2_far_at_val_thr,
                "frr": p2_frr_at_val_thr,
            },
        }

        with (thresholds_dir / f"user_{s:03d}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary_rows.append(
            {
                "subject_id": s,
                "decision_threshold": val_thr,
                "val_eer": val_eer,
                "val_auc": val_auc,
                "p1_auc": p1_auc,
                "p1_far": p1_far_at_val_thr,
                "p1_frr": p1_frr_at_val_thr,
                "p2_auc": p2_auc,
                "p2_far": p2_far_at_val_thr,
                "p2_frr": p2_frr_at_val_thr,
            }
        )

        thresholds.append(val_thr)
        cohort_p1_genuine.append(p1_genuine)
        cohort_p1_impostor.append(p1_impostor)
        cohort_p2_genuine.append(p2_genuine)
        cohort_p2_impostor.append(p2_impostor)

        print(
            f"user_{s:03d}: threshold={val_thr:.4f}, val_auc={val_auc:.4f} | "
            f"P1 FAR={p1_far_at_val_thr:.4f}, FRR={p1_frr_at_val_thr:.4f}, AUC={p1_auc:.4f} | "
            f"P2 FAR={p2_far_at_val_thr:.4f}, FRR={p2_frr_at_val_thr:.4f}, AUC={p2_auc:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(thresholds_dir / "summary.csv", index=False)

    mean_threshold = float(np.mean(thresholds))

    p1_gen = np.concatenate(cohort_p1_genuine)
    p1_imp = np.concatenate(cohort_p1_impostor)
    p2_gen = np.concatenate(cohort_p2_genuine)
    p2_imp = np.concatenate(cohort_p2_impostor)

    p1_y = np.hstack([np.ones(len(p1_gen)), np.zeros(len(p1_imp))])
    p1_scores = np.hstack([p1_gen, p1_imp])
    p2_y = np.hstack([np.ones(len(p2_gen)), np.zeros(len(p2_imp))])
    p2_scores = np.hstack([p2_gen, p2_imp])

    save_score_histogram(
        p1_gen,
        p1_imp,
        mean_threshold,
        plots_dir / "protocol1_score_hist.png",
        "Protocol 1 score distributions (within session)",
    )
    save_score_histogram(
        p2_gen,
        p2_imp,
        mean_threshold,
        plots_dir / "protocol2_score_hist.png",
        "Protocol 2 score distributions (cross session)",
    )
    p1_auc_cohort = save_roc_plot(p1_y, p1_scores, plots_dir / "protocol1_roc.png", "Protocol 1 ROC")
    p2_auc_cohort = save_roc_plot(p2_y, p2_scores, plots_dir / "protocol2_roc.png", "Protocol 2 ROC")
    p1_det = save_far_frr_plot(p1_y, p1_scores, plots_dir / "protocol1_far_frr.png", "Protocol 1 FAR / FRR vs threshold")
    p2_det = save_far_frr_plot(p2_y, p2_scores, plots_dir / "protocol2_far_frr.png", "Protocol 2 FAR / FRR vs threshold")

    report = {
        "n_subjects": len(subjects),
        "threshold_selection": {
            "session": int(args.validation_session),
            "split": args.validation_split,
        },
        "validation": {
            "mean_eer": float(summary_df["val_eer"].mean()),
            "mean_auc": float(summary_df["val_auc"].mean()),
        },
        "protocol1": {
            "name": "within_session",
            "mean_auc": float(summary_df["p1_auc"].mean()),
            "mean_far": float(summary_df["p1_far"].mean()),
            "mean_frr": float(summary_df["p1_frr"].mean()),
            "cohort_auc": p1_auc_cohort,
            "cohort_eer": p1_det["eer"],
        },
        "protocol2": {
            "name": "cross_session",
            "mean_auc": float(summary_df["p2_auc"].mean()),
            "mean_far": float(summary_df["p2_far"].mean()),
            "mean_frr": float(summary_df["p2_frr"].mean()),
            "cohort_auc": p2_auc_cohort,
            "cohort_eer": p2_det["eer"],
        },
        "plots": {
            "protocol1_score_hist": str(plots_dir / "protocol1_score_hist.png"),
            "protocol2_score_hist": str(plots_dir / "protocol2_score_hist.png"),
            "protocol1_roc": str(plots_dir / "protocol1_roc.png"),
            "protocol2_roc": str(plots_dir / "protocol2_roc.png"),
            "protocol1_far_frr": str(plots_dir / "protocol1_far_frr.png"),
            "protocol2_far_frr": str(plots_dir / "protocol2_far_frr.png"),
        },
    }

    with (thresholds_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()