#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/ADL_videos"


def download(participant: int, output: Path) -> Path:
    destination = output / f"P_{participant:02d}.MP4"
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "10",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            f"{BASE_URL}/{destination.name}",
        ],
        check=True,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the 20 official ADL egocentric videos")
    parser.add_argument("--output", type=Path, default=Path("data/raw/adl/videos"))
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(download, index, args.output): index for index in range(1, 21)}
        for future in as_completed(futures):
            print(f"completed participant {futures[future]:02d}: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
