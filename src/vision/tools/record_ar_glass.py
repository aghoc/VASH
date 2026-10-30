#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.vision.frame_source import open_frame_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled AR-glass domain frames")
    parser.add_argument(
        "--source",
        required=True,
        help="camera index, stream URL, video, 'elp' (IMX291), or 'pixy'",
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--lighting", choices=("bright", "dark"), required=True)
    parser.add_argument("--distance", choices=("near", "far"), required=True)
    parser.add_argument("--background", choices=("plain", "cluttered"), required=True)
    parser.add_argument("--rotation", choices=("front", "rotated"), required=True)
    parser.add_argument("--occlusion", choices=("clear", "partial"), required=True)
    parser.add_argument("--multi-object", choices=("single", "multiple"), required=True)
    parser.add_argument("--motion", choices=("static", "walking_blur"), required=True)
    parser.add_argument("--every", type=int, default=5, help="save every Nth frame")
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/domain/ar_glass"
    )
    args = parser.parse_args()
    config = load_yaml(PROJECT_ROOT / "configs/domain_capture.yaml")
    if args.class_name not in config["required_classes"]:
        raise ValueError(f"Unknown target class: {args.class_name}")
    if args.every <= 0 or args.max_images <= 0:
        raise ValueError("every and max-images must be positive")

    session_dir = args.output / "images" / args.session
    if session_dir.exists():
        raise FileExistsError(f"Session already exists: {session_dir}")
    session_dir.mkdir(parents=True)
    manifest_path = args.output / "manifest.jsonl"
    capture = open_frame_source(args.source)
    metadata = {
        "class_name": args.class_name,
        "session": args.session,
        "lighting": args.lighting,
        "distance": args.distance,
        "background": args.background,
        "rotation": args.rotation,
        "occlusion": args.occlusion,
        "multi_object": args.multi_object,
        "motion": args.motion,
    }
    source_frame = 0
    saved = 0
    records = []
    try:
        while saved < args.max_images:
            ok, frame = capture.read()
            if not ok:
                break
            if source_frame % args.every == 0:
                relative = Path("images") / args.session / f"frame_{source_frame:06d}.jpg"
                if not cv2.imwrite(str(args.output / relative), frame):
                    raise RuntimeError(f"Failed to write {relative}")
                records.append(
                    {
                        **metadata,
                        "image": relative.as_posix(),
                        "source_frame": source_frame,
                        "captured_unix_ns": time.time_ns(),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
                saved += 1
            source_frame += 1
    finally:
        capture.release()
    if not records:
        raise RuntimeError("Source produced no frames")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"session": args.session, "saved": saved, "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
