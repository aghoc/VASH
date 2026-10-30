#!/usr/bin/env python3
"""Measure rare-class detection at the actual runtime confidence, with size bins."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from vagt_orca.deployment import sha256
from vagt_orca.gpu import gpu_summary, require_cuda


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0)
    intersection = np.maximum(
        0.0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])
    ).prod(axis=1)
    area1 = np.maximum(0.0, box[2:] - box[:2]).prod()
    area2 = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area1 + area2 - intersection, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    thresholds = (
        yaml.safe_load(args.runtime.read_text()).get("class_confidence_thresholds", {})
        if args.runtime
        else {}
    )
    require_cuda(0)
    from ultralytics import YOLO

    config = yaml.safe_load(args.data.read_text())
    root = Path(config.get("path", args.data.resolve().parent))
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    images = sorted(
        p
        for p in (root / config[args.split]).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    counts = {1: Counter(), 13: Counter()}
    bins = {1: defaultdict(Counter), 13: defaultdict(Counter)}
    model = YOLO(str(args.model))
    for start in range(0, len(images), args.batch):
        batch = images[start : start + args.batch]
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=640,
            conf=min([args.confidence, *thresholds.values()]),
            iou=0.45,
            device=0,
            quantize=32,
            rect=False,
            verbose=False,
        )
        for path, result in zip(batch, results):
            height, width = result.orig_shape
            gt = defaultdict(list)
            label = root / "labels" / args.split / f"{path.stem}.txt"
            for line in label.read_text().splitlines():
                fields = line.split()
                if not fields or int(fields[0]) not in counts:
                    continue
                cx, cy, w, h = map(float, fields[1:])
                gt[int(fields[0])].append(
                    [
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        (cx + w / 2) * width,
                        (cy + h / 2) * height,
                    ]
                )
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for class_id, count in counts.items():
                targets = np.array(gt[class_id], dtype=float).reshape(-1, 4)
                threshold = thresholds.get(config["names"][class_id], args.confidence)
                predicted = np.flatnonzero((classes == class_id) & (scores >= threshold))
                predicted = predicted[np.argsort(-scores[predicted])]
                matched = set()
                false_positives = 0
                for index in predicted:
                    overlaps = iou_one(boxes[index], targets)
                    if matched:
                        overlaps[list(matched)] = -1
                    best = int(overlaps.argmax()) if len(overlaps) else -1
                    if best >= 0 and overlaps[best] >= 0.5:
                        matched.add(best)
                    else:
                        false_positives += 1
                count.update(
                    tp=len(matched),
                    fp=false_positives,
                    fn=len(targets) - len(matched),
                    images=1,
                    negative_images=int(not len(targets)),
                    false_positive_negative_images=int(not len(targets) and false_positives > 0),
                )
                for index, target in enumerate(targets):
                    side = (
                        np.sqrt((target[2] - target[0]) * (target[3] - target[1]))
                        * 640
                        / max(width, height)
                    )
                    size = (
                        "small_lt32px"
                        if side < 32
                        else "medium_32to96px"
                        if side < 96
                        else "large_ge96px"
                    )
                    bins[class_id][size].update(instances=1, tp=int(index in matched))
        if start % (args.batch * 20) == 0:
            print(
                f"Operating point: {min(start + args.batch, len(images))}/{len(images)}", flush=True
            )
    summary = {
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "data": str(args.data.resolve()),
        "split": args.split,
        "confidence": args.confidence,
        "class_confidence_thresholds": thresholds,
        "match_iou": 0.5,
        "gpu": gpu_summary(0),
        "classes": {},
    }
    for class_id, count in counts.items():
        summary["classes"][config["names"][class_id]] = {
            **dict(count),
            "precision": count["tp"] / max(1, count["tp"] + count["fp"]),
            "recall": count["tp"] / max(1, count["tp"] + count["fn"]),
            "size_bins": {
                name: {**dict(value), "recall": value["tp"] / value["instances"]}
                for name, value in bins[class_id].items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
