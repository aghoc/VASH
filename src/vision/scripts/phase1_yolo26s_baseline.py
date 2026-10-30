#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

from vagt_orca.deployment import sha256
from vagt_orca.gpu import gpu_summary, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pretrained YOLO26s CUDA baseline")
    parser.add_argument("--model", type=Path, default=Path("models/pretrained/yolo26s.pt"))
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/phase1_baseline_input.mp4")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/yolo26s_phase1_baseline_visualized.mp4")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase1_baseline.json")
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    require_cuda(args.device)
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    model = YOLO(str(args.model))
    end2end = bool(getattr(model.model, "end2end", False))
    if not end2end:
        raise RuntimeError("YOLO26 default end-to-end/NMS-free head is not active")
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 10.0
    writer = None
    latencies: list[float] = []
    detections = 0
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            torch.cuda.synchronize(args.device)
            started = time.perf_counter_ns()
            result = model.predict(
                frame,
                imgsz=args.image_size,
                device=args.device,
                quantize=32,
                nms=False,
                verbose=False,
            )[0]
            torch.cuda.synchronize(args.device)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            detections += 0 if result.boxes is None else len(result.boxes)
            rendered = result.plot()
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                height, width = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to create {args.output}")
            writer.write(rendered)
            frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not frames:
        raise RuntimeError("Baseline source contained no frames")
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "YOLO26s",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "training_precision_target": "fp32",
        "deployment_precision_target": "fp16",
        "default_head": "end-to-end / NMS-free",
        "end2end_nms_free_verified": end2end,
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "gpu": gpu_summary(args.device),
        "frames": frames,
        "detections": detections,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "input": str(args.source.resolve()),
        "visualization": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
