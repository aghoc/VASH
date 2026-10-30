#!/usr/bin/env python3
"""Check a UVC camera (e.g. ELP IMX291) on this machine: negotiate mode, measure FPS, save a snapshot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.vision.frame_source import UVCFrameSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", help="stable /dev/v4l/by-id path; overrides --index")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--fourcc", default="MJPG", help="MJPG or YUYV")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs/camera_check")
    parser.add_argument("--view", action="store_true", help="show a live window (q to quit)")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("frames must be at least 2")

    capture = UVCFrameSource(
        args.device or args.index, args.width, args.height, args.fps, args.fourcc
    )
    got_fourcc = capture.mode["fourcc"]
    got_w, got_h = capture.mode["width"], capture.mode["height"]
    print(f"requested {args.width}x{args.height}@{args.fps} {args.fourcc}")
    print(f"negotiated {got_w}x{got_h}@{capture.fps:.1f} {got_fourcc}")

    for _ in range(10):  # warm-up: auto exposure settles
        capture.read()

    failures = 0
    consecutive = 0
    stamps: list[float] = []
    last = None
    novelty: list[float] = []
    while len(stamps) < args.frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            failures += 1
            consecutive += 1
            if consecutive > 30:
                break
            continue
        consecutive = 0
        stamps.append(time.perf_counter())
        if last is not None:
            novelty.append(float(np.mean(cv2.absdiff(frame, last))))
        last = frame
        if args.view:
            cv2.imshow("uvc_camera_check", cv2.resize(frame, None, fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    capture.release()
    if args.view:
        cv2.destroyAllWindows()

    if len(stamps) < 2 or last is None:
        raise SystemExit(f"프레임 수신 실패 (failures={failures})")

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / f"snapshot_{got_w}x{got_h}.jpg"
    cv2.imwrite(str(snapshot), last)
    elapsed = stamps[-1] - stamps[0]
    report = {
        "requested": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "fourcc": args.fourcc,
        },
        "negotiated": capture.mode,
        "frames": len(stamps),
        "failures": failures,
        "measured_fps": (len(stamps) - 1) / elapsed,
        "mean_novelty": float(np.mean(novelty)),
        "mean_brightness": float(last.mean()),
        "snapshot": str(snapshot),
    }
    (args.out / f"capture_{got_w}x{got_h}_{got_fourcc}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"frames={len(stamps)} failures={failures} "
        f"fps={(len(stamps) - 1) / elapsed:.2f} mean_novelty={np.mean(novelty):.3f} "
        f"mean_brightness={float(last.mean()):.1f}"
    )
    print(f"snapshot: {snapshot}")


if __name__ == "__main__":
    main()
