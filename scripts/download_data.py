from __future__ import annotations

import argparse
from pathlib import Path

from moabb import set_download_dir
from moabb.datasets import Lee2019_MI


def parse_subjects(value: str) -> list[int]:
    value = value.strip()
    if "-" in value:
        start_str, end_str = value.split("-", maxsplit=1)
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/cache MOABB Lee2019_MI resting-state data for selected subjects."
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="1-40",
        help="Subject IDs, e.g. '1-40' or '1,2,3'.",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="1,2",
        help="Sessions to cache, default '1,2'.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root folder where MOABB will cache downloaded files.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    sessions = [int(x.strip()) for x in args.sessions.split(",") if x.strip()]
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    set_download_dir(str(data_root))
    dataset = Lee2019_MI(
        train_run=True,
        test_run=False,
        resting_state=True,
        sessions=sessions,
    )

    print(
        f"Caching Lee2019_MI resting-state data for {len(subjects)} subject(s) "
        f"across sessions {sessions} into {data_root}"
    )

    for subject in subjects:
        subject_data = dataset.get_data(subjects=[subject])
        session_names = sorted(subject_data[subject].keys())
        total_runs = sum(len(subject_data[subject][sess]) for sess in session_names)
        print(f"S{subject:03d}: cached {len(session_names)} session(s), {total_runs} run object(s)")

    print("Download/cache complete.")


if __name__ == "__main__":
    main()