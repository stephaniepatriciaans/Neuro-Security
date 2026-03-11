from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from mne import set_config
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


def parse_sessions(value: str) -> list[int]:
    sessions = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not sessions:
        raise ValueError("at least one session must be provided")
    invalid = [session for session in sessions if session not in (1, 2)]
    if invalid:
        raise ValueError(f"invalid Lee2019 session(s): {invalid}; valid sessions are 1 and 2")
    return sessions


def configure_download_dir(data_root: Path) -> None:
    resolved = str(data_root.resolve())
    set_download_dir(resolved)
    set_config("MNE_DATA", resolved)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message='Setting non-standard config type: "MNE_DATASETS_LEE2019-MI_PATH"',
            category=RuntimeWarning,
        )
        set_config("MNE_DATASETS_LEE2019-MI_PATH", resolved)


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
        default="data/raw_lee2019_mi",
        help="Root folder where MOABB will cache downloaded files.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    sessions = parse_sessions(args.sessions)
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    configure_download_dir(data_root)
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

    total_files = 0
    for subject in subjects:
        paths = dataset.data_path(subject, path=str(data_root), force_update=False, update_path=False)
        total_files += len(paths)
        print(f"S{subject:03d}: cached {len(paths)} session file(s)")

    print(f"Download/cache complete. Total files processed: {total_files}")


if __name__ == "__main__":
    main()