from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from mne.datasets import eegbci
from mne.io import read_raw_edf
from scipy.signal import welch


BANDS: list[tuple[str, float, float]] = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 40.0),
]


def parse_subjects(value: str) -> list[int]:
    value = value.strip()
    if "-" in value:
        start_str, end_str = value.split("-", maxsplit=1)
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def to_windows(data: np.ndarray, win_len: int, step: int) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    start = 0
    n_samples = data.shape[1]
    while start + win_len <= n_samples:
        windows.append(data[:, start : start + win_len])
        start += step
    return windows


def bandpower_features(window: np.ndarray, sfreq: float) -> np.ndarray:
    # window shape: (n_channels, n_samples)
    features: list[float] = []
    for ch in range(window.shape[0]):
        freqs, psd = welch(window[ch], fs=sfreq, nperseg=int(sfreq), noverlap=0)
        for _, fmin, fmax in BANDS:
            mask = (freqs >= fmin) & (freqs < fmax)
            if not np.any(mask):
                features.append(0.0)
            else:
                # NumPy 2.x removed trapz; use trapezoid and fallback for older versions.
                if hasattr(np, "trapezoid"):
                    area = np.trapezoid(psd[mask], freqs[mask])
                else:
                    area = np.trapz(psd[mask], freqs[mask])
                features.append(float(area))
    return np.asarray(features, dtype=np.float64)


def extract_block_features(
    block: np.ndarray,
    sfreq: float,
    win_sec: float,
    step_sec: float,
    artifact_uv: float,
) -> np.ndarray:
    win_len = int(round(win_sec * sfreq))
    step = int(round(step_sec * sfreq))
    artifact_v = artifact_uv * 1e-6 if artifact_uv > 0 else None

    windows = to_windows(block, win_len=win_len, step=step)
    feats: list[np.ndarray] = []
    for window in windows:
        if artifact_v is not None and np.max(np.abs(window)) > artifact_v:
            continue
        feats.append(bandpower_features(window, sfreq))

    if not feats:
        return np.empty((0, block.shape[0] * len(BANDS)), dtype=np.float64)
    return np.vstack(feats)


def make_dataframe(features: np.ndarray, subject_id: int, run_id: int, split: str) -> pd.DataFrame:
    n_features = features.shape[1] if features.ndim == 2 else 0
    columns = [f"f{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "window_idx", np.arange(len(df), dtype=int))
    df.insert(0, "split", split)
    df.insert(0, "run", run_id)
    df.insert(0, "subject_id", subject_id)
    return df


def process_subject_run(
    subject_id: int,
    run_id: int,
    data_root: Path,
    out_dir: Path,
    block_sec: float,
    win_sec: float,
    step_sec: float,
    artifact_uv: float,
) -> None:
    edf_paths = eegbci.load_data(subject_id, [run_id], path=str(data_root), update_path=False)
    if not edf_paths:
        raise RuntimeError(f"No EDF found for S{subject_id:03d} R{run_id:02d}")

    raw = read_raw_edf(edf_paths[0], preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.filter(1.0, 40.0, method="iir", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    data = raw.get_data()  # shape: (n_channels, n_samples), in volts
    sfreq = float(raw.info["sfreq"])

    block_len = int(round(block_sec * sfreq))
    if data.shape[1] < 2 * block_len:
        raise RuntimeError(
            f"S{subject_id:03d} R{run_id:02d} too short: {data.shape[1]} samples at {sfreq} Hz"
        )

    enroll_block = data[:, :block_len]
    test_block = data[:, block_len : 2 * block_len]

    enroll_feats = extract_block_features(enroll_block, sfreq, win_sec, step_sec, artifact_uv)
    test_feats = extract_block_features(test_block, sfreq, win_sec, step_sec, artifact_uv)

    enroll_df = make_dataframe(enroll_feats, subject_id, run_id, "enroll")
    test_df = make_dataframe(test_feats, subject_id, run_id, "test")

    out_dir.mkdir(parents=True, exist_ok=True)
    enroll_path = out_dir / f"S{subject_id:03d}_R{run_id:02d}_enroll.csv"
    test_path = out_dir / f"S{subject_id:03d}_R{run_id:02d}_test.csv"

    enroll_df.to_csv(enroll_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"S{subject_id:03d} R{run_id:02d} -> "
        f"enroll_windows={len(enroll_df)}, test_windows={len(test_df)}"
    )


def iter_runs(runs_arg: str) -> Iterable[int]:
    for item in runs_arg.split(","):
        item = item.strip()
        if item:
            yield int(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 320D PSD-bandpower features from EEGMMIDB baseline runs.")
    parser.add_argument("--subjects", type=str, default="1-40", help="Subject IDs, e.g. '1-40' or '1,2,3'.")
    parser.add_argument("--runs", type=str, default="1,2", help="Runs to process, default '1,2'.")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Root folder containing EDF files.")
    parser.add_argument("--out-dir", type=str, default="data/features", help="Where feature CSVs are saved.")
    parser.add_argument("--block-sec", type=float, default=30.0, help="Seconds for enrollment and test blocks.")
    parser.add_argument("--win-sec", type=float, default=2.0, help="Window length in seconds.")
    parser.add_argument("--step-sec", type=float, default=2.0, help="Window step in seconds.")
    parser.add_argument(
        "--artifact-uv",
        type=float,
        default=300.0,
        help="Drop windows whose absolute amplitude exceeds this threshold in microvolts. Set <=0 to disable.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = list(iter_runs(args.runs))
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    print(
        "Starting feature extraction with "
        f"subjects={subjects[0]}..{subjects[-1]} (n={len(subjects)}), runs={runs}, out={out_dir}"
    )

    for subject_id in subjects:
        for run_id in runs:
            process_subject_run(
                subject_id=subject_id,
                run_id=run_id,
                data_root=data_root,
                out_dir=out_dir,
                block_sec=args.block_sec,
                win_sec=args.win_sec,
                step_sec=args.step_sec,
                artifact_uv=args.artifact_uv,
            )

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
