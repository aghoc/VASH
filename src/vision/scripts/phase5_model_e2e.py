#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import yaml

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.deployment import sha256
from vagt_orca.evaluation import select_class_balanced_images
from vagt_orca.gpu import gpu_summary, require_cuda
from vagt_orca.grip.planner import GripPlanner
from vagt_orca.types import Detection
from vagt_orca.vision.detector import YoloDetector
from vagt_orca.vision.pipeline import VisionGripPipeline
from vagt_orca.vision.target_selector import CenterBiasSelector
from vagt_orca.vision.target_stabilizer import TargetStabilizer
from vagt_orca.visualization import render


def detections_from_result(result: Any) -> list[Detection]:
    boxes = result.boxes
    if boxes is None:
        return []
    return [
        Detection(
            class_id=int(class_id),
            class_name=str(result.names[int(class_id)]),
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox),
        )
        for class_id, confidence, bbox in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained-model Phase 5 video acceptance on CUDA"
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/yolo26s_best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/egoobjects_15/data.yaml"))
    parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    parser.add_argument("--candidate-images", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/yolo26s_phase5_e2e"))
    parser.add_argument("--report", type=Path, default=Path("reports/yolo26s_phase5_e2e.json"))
    args = parser.parse_args()

    runtime = load_yaml(args.runtime)
    device = int(str(runtime["device"]).split(":", 1)[1])
    require_cuda(device)
    dataset: dict[str, Any] = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(dataset["path"])
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    image_dir = root / dataset["val"]
    label_dir = root / "labels" / "val"
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    images = select_class_balanced_images(
        images, label_dir, min(args.candidate_images, len(images)), len(dataset["names"])
    )

    from ultralytics import YOLO

    selector = CenterBiasSelector()
    search_model = YOLO(str(args.model))
    best: tuple[float, Path, Detection] | None = None
    for start in range(0, len(images), args.batch):
        batch = images[start : start + args.batch]
        results = search_model.predict(
            source=[str(path) for path in batch],
            device=device,
            conf=float(runtime["confidence_threshold"]),
            iou=float(runtime["iou_threshold"]),
            imgsz=int(runtime["image_size"]),
            quantize=32,
            verbose=False,
        )
        for image, result in zip(batch, results):
            height, width = result.orig_shape
            thresholds = runtime.get("class_confidence_thresholds", {})
            accepted_detections = [
                item
                for item in detections_from_result(result)
                if item.confidence
                >= thresholds.get(item.class_name, float(runtime["confidence_threshold"]))
            ]
            candidate = selector.select(accepted_detections, (width, height))
            if candidate is not None and (best is None or candidate.confidence > best[0]):
                best = (candidate.confidence, image, candidate)
    if best is None:
        raise RuntimeError("No detection above the runtime confidence threshold in candidates")

    _, source_image, searched_candidate = best
    frame = cv2.imread(str(source_image))
    if frame is None:
        raise RuntimeError(f"Unable to decode selected image {source_image}")
    target_config = runtime["target"]
    pipeline = VisionGripPipeline(
        YoloDetector(
            args.model,
            confidence=float(runtime["confidence_threshold"]),
            iou=float(runtime["iou_threshold"]),
            image_size=int(runtime["image_size"]),
            device=device,
            fp16=False,
            class_confidence_thresholds=runtime.get("class_confidence_thresholds"),
        ),
        TargetStabilizer(
            int(target_config["debounce_window"]),
            int(target_config["required_votes"]),
            float(target_config["switch_margin"]),
            float(target_config["identity_iou_threshold"]),
            int(target_config.get("missing_grace_frames", 8)),
        ),
        GripPlanner(
            float(runtime["confidence_threshold"]),
            class_confidence_thresholds=runtime.get("class_confidence_thresholds"),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_video = args.output_dir / "input.mp4"
    output_video = args.output_dir / "visualized.mp4"
    height, width = frame.shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    input_writer = cv2.VideoWriter(str(input_video), codec, 10.0, (width, height))
    output_writer = cv2.VideoWriter(str(output_video), codec, 10.0, (width, height))
    if not input_writer.isOpened() or not output_writer.isOpened():
        raise RuntimeError("Unable to create Phase 5 E2E videos")
    payloads = []
    try:
        for frame_number in range(args.frames):
            result = pipeline.process(frame)
            payload = {
                "frame": frame_number,
                "detections": len(result.detections),
                "candidate": result.candidate.class_name if result.candidate else None,
                "target": result.target.class_name if result.target else None,
                "plan": result.plan.proposal.as_dict() if result.plan.proposal else None,
                "plan_updated": result.plan.updated,
                "reason": result.plan.reason,
            }
            payloads.append(payload)
            input_writer.write(frame)
            output_writer.write(render(frame, result))
    finally:
        input_writer.release()
        output_writer.release()

    locked = [payload for payload in payloads if payload["target"] is not None]
    if not locked or locked[-1]["plan"] is None or not locked[-1]["plan_updated"]:
        raise RuntimeError("Trained-model E2E did not produce a stable target and grip proposal")
    report = {
        "gpu": gpu_summary(device),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "source_image": str(source_image.resolve()),
        "candidates_searched": len(images),
        "search_candidate": {
            "class": searched_candidate.class_name,
            "confidence": searched_candidate.confidence,
            "bbox": searched_candidate.bbox,
        },
        "input_video": str(input_video.resolve()),
        "visualized_video": str(output_video.resolve()),
        "first_locked_frame": locked[0]["frame"],
        "frames": payloads,
        "accepted": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
