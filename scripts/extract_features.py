from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from moabb import set_download_dir
from moabb.datasets import Lee2019_MI
from scipy.signal import welch


BANDS: list[tuple[str, float, float]] = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 40.0),
]

SESSION1_TRAIN_RUNS = ("0preTrainRest", "2postTrainRest")
SESSION1_VAL_RUNS = ("3preTestRest",)
SESSION1_TEST_RUNS = ("5postTestRest",)
SESSION2_TEST_RUNS = ("0preTrainRest", "2postTrainRest", "3preTestRest", "5postTestRest")


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
    features: list[float] = []
    nperseg = min(window.shape[1], int(round(sfreq)))
    for ch in range(window.shape[0]):
        freqs, psd = welch(window[ch], fs=sfreq, nperseg=nperseg, noverlap=0)
        for _, fmin, fmax in BANDS:
            mask = (freqs >= fmin) & (freqs < fmax)
            if not np.any(mask):
                features.append(0.0)
            else:
                if hasattr(np, "trapezoid"):
                    area = np.trapezoid(psd[mask], freqs[mask])
                else:
                    area = np.trapz(psd[mask], freqs[mask])
                features.append(float(area))
    return np.asarray(features, dtype=np.float64)


def preprocess_raw(raw, l_freq: float, h_freq: float, resample_hz: float):
    raw = raw.copy().pick("eeg")
    raw.filter(l_freq, h_freq, method="iir", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if resample_hz > 0:
        raw.resample(resample_hz, verbose=False)
    return raw


def concatenate_runs(
    session_runs: dict,
    run_names: Iterable[str],
    *,
    l_freq: float,
    h_freq: float,
    resample_hz: float,
) -> tuple[np.ndarray, float]:
    arrays: list[np.ndarray] = []
    sfreq: float | None = None
    for run_name in run_names:
        if run_name not in session_runs:
            raise KeyError(f"Missing resting-state run '{run_name}' in session data")
        raw = preprocess_raw(session_runs[run_name], l_freq=l_freq, h_freq=h_freq, resample_hz=resample_hz)
        data = raw.get_data()
        current_sfreq = float(raw.info["sfreq"])
        if sfreq is None:
            sfreq = current_sfreq
        elif abs(sfreq - current_sfreq) > 1e-9:
            raise ValueError("Sampling frequency changed across runs after preprocessing")
        arrays.append(data)
    if not arrays or sfreq is None:
        raise ValueError("No resting-state arrays were collected for concatenation")
    return np.concatenate(arrays, axis=1), sfreq


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


def make_dataframe(features: np.ndarray, subject_id: int, session_id: int, split: str) -> pd.DataFrame:
    n_features = features.shape[1] if features.ndim == 2 else 0
    columns = [f"f{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "window_idx", np.arange(len(df), dtype=int))
    df.insert(0, "split", split)
    df.insert(0, "session_id", session_id)
    df.insert(0, "subject_id", subject_id)
    return df


def save_features(out_dir: Path, subject_id: int, session_id: int, split: str, features: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_dataframe(features, subject_id, session_id, split)
    out_path = out_dir / f"S{subject_id:03d}_S{session_id}_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"S{subject_id:03d} S{session_id} {split}: windows={len(df)} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract resting-state PSD bandpower features from MOABB Lee2019_MI."
    )
    parser.add_argument("--subjects", type=str, default="1-40", help="Subject IDs, e.g. '1-40' or '1,2,3'.")
    parser.add_argument("--data-root", type=str, default="data/raw", help="MOABB download/cache root.")
    parser.add_argument("--out-dir", type=str, default="data/features", help="Where feature CSVs are saved.")
    parser.add_argument("--win-sec", type=float, default=2.0, help="Window length in seconds.")
    parser.add_argument("--step-sec", type=float, default=2.0, help="Window step in seconds.")
    parser.add_argument(
        "--artifact-uv",
        type=float,
        default=0.0,
        help="Drop windows whose absolute amplitude exceeds this threshold in microvolts. Set <=0 to disable.",
    )
    parser.add_argument("--l-freq", type=float, default=1.0, help="High-pass cutoff in Hz.")
    parser.add_argument("--h-freq", type=float, default=40.0, help="Low-pass cutoff in Hz.")
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=0.0,
        help="Optional resampling rate after filtering. Set <=0 to keep the native 1000 Hz.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    set_download_dir(str(data_root))
    dataset = Lee2019_MI(
        train_run=True,
        test_run=False,
        resting_state=True,
        sessions=[1, 2],
    )

    print(
        "Starting Lee2019_MI resting-state feature extraction with "
        f"subjects={subjects[0]}..{subjects[-1]} (n={len(subjects)}), out={out_dir}"
    )

    for subject_id in subjects:
        subject_data = dataset.get_data(subjects=[subject_id])[subject_id]
        session1_runs = subject_data["0"]
        session2_runs = subject_data["1"]
        
        s1_train_block, sfreq = concatenate_runs(
            session1_runs,
            SESSION1_TRAIN_RUNS,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            resample_hz=args.resample_hz,
        )
        s1_val_block, sfreq_val = concatenate_runs(
            session1_runs,
            SESSION1_VAL_RUNS,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            resample_hz=args.resample_hz,
        )
        s1_test_block, sfreq_test = concatenate_runs(
            session1_runs,
            SESSION1_TEST_RUNS,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            resample_hz=args.resample_hz,
        )
        s2_test_block, sfreq_s2 = concatenate_runs(
            session2_runs,
            SESSION2_TEST_RUNS,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            resample_hz=args.resample_hz,
        )

        if len({round(sfreq, 6), round(sfreq_val, 6), round(sfreq_test, 6), round(sfreq_s2, 6)}) != 1:
            raise ValueError(f"Inconsistent sampling rates for subject {subject_id:03d}")

        s1_train_feats = extract_block_features(s1_train_block, sfreq, args.win_sec, args.step_sec, args.artifact_uv)
        s1_val_feats = extract_block_features(s1_val_block, sfreq, args.win_sec, args.step_sec, args.artifact_uv)
        s1_test_feats = extract_block_features(s1_test_block, sfreq, args.win_sec, args.step_sec, args.artifact_uv)
        s2_test_feats = extract_block_features(s2_test_block, sfreq, args.win_sec, args.step_sec, args.artifact_uv)

        save_features(out_dir, subject_id, 1, "train", s1_train_feats)
        save_features(out_dir, subject_id, 1, "val", s1_val_feats)
        save_features(out_dir, subject_id, 1, "test", s1_test_feats)
        save_features(out_dir, subject_id, 2, "test", s2_test_feats)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()