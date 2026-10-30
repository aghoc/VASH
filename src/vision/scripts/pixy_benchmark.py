#!/usr/bin/env python3
"""Measure Pixy2 raw-frame throughput and verify frames actually advance.

charmedlabs does not document whether a single stop() lets getRawFrame() observe
new sensor frames, so both grab modes are measured here and judged on two axes:

  fps     - achieved grab rate
  novelty - mean absolute difference between consecutive frames. A mode that
            returns a stale buffer scores ~0 on a moving scene and is unusable
            regardless of how fast it is.

Wave a hand in front of the lens while this runs, otherwise a static scene will
report low novelty for both modes and prove nothing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.vision.frame_source import PixyFrameSource


def measure(cycle: bool, frames: int, settle_us: int) -> dict[str, float | int | bool]:
    source = PixyFrameSource(cycle=cycle, settle_us=settle_us)
    previous: np.ndarray | None = None
    novelties: list[float] = []
    failures = 0
    started = time.perf_counter()
    try:
        for _ in range(frames):
            ok, frame = source.read()
            if not ok or frame is None:
                failures += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
            if previous is not None:
                novelties.append(float(np.mean(np.abs(gray - previous))))
            previous = gray
        elapsed = time.perf_counter() - started
    finally:
        source.release()

    captured = frames - failures
    return {
        "mode": "cycled" if cycle else "persistent",
        "settle_us": settle_us,
        "frames_requested": frames,
        "frames_captured": captured,
        "failures": failures,
        "elapsed_s": round(elapsed, 4),
        "fps": round(captured / elapsed, 3) if elapsed > 0 else 0.0,
        "mean_novelty": round(float(np.mean(novelties)), 4) if novelties else 0.0,
        "max_novelty": round(float(np.max(novelties)), 4) if novelties else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--settle-us", type=int, default=20_000)
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports/pixy_capture.json"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "artifacts/deployment/pixy/bayer_reference.png",
        help="saved frame for confirming the demosaic pattern against a known target",
    )
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("need at least 2 frames to measure novelty")

    results = [
        measure(cycle=False, frames=args.frames, settle_us=args.settle_us),
        measure(cycle=True, frames=args.frames, settle_us=args.settle_us),
    ]

    usable = [item for item in results if float(item["mean_novelty"]) > 0.5]
    recommended = max(usable, key=lambda item: float(item["fps"]))["mode"] if usable else None
    recommended_cycle = recommended == "cycled"
    source = PixyFrameSource(cycle=recommended_cycle, settle_us=args.settle_us)
    try:
        ok, frame = source.read()
    finally:
        source.release()
    if not ok or frame is None:
        raise RuntimeError("Could not capture a reference frame")
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.reference), frame):
        raise RuntimeError(f"Failed to write {args.reference}")

    payload = {
        "frame_size": list(source.frame_size),
        "measurements": results,
        "reference_frame": str(args.reference),
        "recommended_mode": recommended,
        "note": (
            "mean_novelty near zero means the mode returns a stale buffer. "
            "Re-run while moving an object in front of the lens before trusting it."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
