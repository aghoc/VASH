#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from vagt_orca.deployment import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a rendered inference video and JSONL")
    parser.add_argument(
        "--source-report", type=Path, default=Path("reports/yolo26s_phase6_demo_source.json")
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("artifacts/deployment/yolo26s_demo/result_fp16_desktop.mp4"),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("artifacts/deployment/yolo26s_demo/result_fp16_desktop.jsonl"),
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/yolo26s_best.pt"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase6_demo_result.json")
    )
    args = parser.parse_args()

    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines()]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {args.video}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    duration = frames / fps if fps else 0.0
    expected_frames = int(source["unique_source_frames"])
    candidate_counts = Counter(
        record["candidate"] for record in records if record.get("candidate") is not None
    )
    target_counts = Counter(
        record["target"] for record in records if record.get("target") is not None
    )
    plan_counts = Counter(
        record["plan"]["grip"] for record in records if record.get("plan") is not None
    )
    checks = {
        "video_frame_count_matches_source": frames == expected_frames,
        "jsonl_record_count_matches_source": len(records) == expected_frames,
        "frame_indices_are_contiguous": [row.get("frame") for row in records]
        == list(range(expected_frames)),
        "duration_is_at_least_10_seconds": duration >= 10.0,
        "resolution_matches_source": [width, height] == source["resolution"],
    }
    report = {
        "accepted": all(checks.values()),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "checks": checks,
        "source_sequences": len(source["sequences"]),
        "unique_source_frames": expected_frames,
        "video": {
            "path": str(args.video.resolve()),
            "sha256": sha256(args.video),
            "bytes": args.video.stat().st_size,
            "frames": frames,
            "fps": fps,
            "duration_seconds": duration,
            "resolution": [width, height],
        },
        "jsonl": {
            "path": str(args.jsonl.resolve()),
            "sha256": sha256(args.jsonl),
            "records": len(records),
        },
        "inference_summary": {
            "detections": sum(int(row["detections"]) for row in records),
            "frames_with_candidate": sum(row.get("candidate") is not None for row in records),
            "frames_with_target_lock": sum(row.get("target") is not None for row in records),
            "frames_with_plan": sum(row.get("plan") is not None for row in records),
            "new_plan_updates": sum(bool(row.get("plan_updated")) for row in records),
            "candidate_classes": dict(sorted(candidate_counts.items())),
            "target_classes": dict(sorted(target_counts.items())),
            "grip_frames": dict(sorted(plan_counts.items())),
        },
        "scope": "RTX FP16 development demonstration; not Jetson acceptance evidence",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
