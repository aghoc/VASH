#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def fit_frame(image, width: int, height: int):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-sequence egocentric demo video")
    parser.add_argument(
        "--images", type=Path, default=Path("data/processed/egoobjects_15/images/val")
    )
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/deployment/yolo26s_demo/input_sequences.mp4")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase6_demo_source.json")
    )
    args = parser.parse_args()

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in args.images.iterdir():
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        source = image.resolve()
        groups[source.stem.rsplit("_", 1)[0]].append(source)
    selected = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[: args.sequences]
    frames = [
        frame
        for _, sequence_frames in selected
        for frame in sorted(sequence_frames, key=frame_number)
    ]
    if not frames:
        raise RuntimeError("No egocentric sequence frames found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video {args.output}")
    try:
        for source in frames:
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"Unable to decode {source}")
            writer.write(fit_frame(image, args.width, args.height))
    finally:
        writer.release()
    report = {
        "input_video": str(args.output.resolve()),
        "unique_source_frames": len(frames),
        "fps": args.fps,
        "duration_seconds": len(frames) / args.fps,
        "resolution": [args.width, args.height],
        "sequences": [
            {
                "id": sequence_id,
                "frames": len(sequence_frames),
                "first": str(min(sequence_frames, key=frame_number)),
                "last": str(max(sequence_frames, key=frame_number)),
            }
            for sequence_id, sequence_frames in selected
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
