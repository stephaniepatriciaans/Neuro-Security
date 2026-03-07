from __future__ import annotations

import argparse
from pathlib import Path

from mne.datasets import eegbci


def parse_subjects(value: str) -> list[int]:
    """Parse subjects from formats like '1-40' or '1,2,3'."""
    value = value.strip()
    if "-" in value:
        start_str, end_str = value.split("-", maxsplit=1)
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EEGMMIDB runs for selected subjects.")
    parser.add_argument(
        "--subjects",
        type=str,
        default="1-40",
        help="Subject IDs, e.g. '1-40' or '1,2,3'.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="1,2",
        help="Run IDs, e.g. '1,2'. Default uses baseline eyes-open and eyes-closed.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root folder where EEGMMIDB files are downloaded.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = [int(x.strip()) for x in args.runs.split(",") if x.strip()]

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading runs {runs} for {len(subjects)} subjects into: {data_root}")

    total_files = 0
    for subject in subjects:
        paths = eegbci.load_data(subject, runs, path=str(data_root), update_path=False)
        total_files += len(paths)
        print(f"S{subject:03d}: downloaded/found {len(paths)} files")

    print(f"Done. Total files processed: {total_files}")


if __name__ == "__main__":
    main()
