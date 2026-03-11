from __future__ import annotations

import argparse
import subprocess
import sys


def run_step(args: list[str]) -> None:
    print(f"Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Lee2019 resting-state authentication pipeline.")
    parser.add_argument("--subjects", type=str, default="1-40", help="Subject IDs, e.g. '1-40' or '1,2,3'.")
    parser.add_argument("--data-root", type=str, default="data/raw_lee2019_mi", help="MOABB download/cache root.")
    parser.add_argument("--features-dir", type=str, default="data/features_lee2019_mi", help="Output directory for feature CSVs.")
    parser.add_argument("--models-dir", type=str, default="models_lee2019_mi", help="Output directory for trained models.")
    parser.add_argument("--thresholds-dir", type=str, default="thresholds_lee2019_mi", help="Output directory for evaluation results.")
    args = parser.parse_args()

    python_exe = sys.executable

    run_step(
        [
            python_exe,
            "scripts/download_data.py",
            "--subjects",
            args.subjects,
            "--data-root",
            args.data_root,
        ]
    )
    run_step(
        [
            python_exe,
            "scripts/extract_features.py",
            "--subjects",
            args.subjects,
            "--data-root",
            args.data_root,
            "--out-dir",
            args.features_dir,
        ]
    )
    run_step(
        [
            python_exe,
            "scripts/train_models.py",
            "--subjects",
            args.subjects,
            "--features-dir",
            args.features_dir,
            "--models-dir",
            args.models_dir,
        ]
    )
    run_step(
        [
            python_exe,
            "scripts/evaluate_models.py",
            "--subjects",
            args.subjects,
            "--features-dir",
            args.features_dir,
            "--models-dir",
            args.models_dir,
            "--thresholds-dir",
            args.thresholds_dir,
        ]
    )


if __name__ == "__main__":
    main()