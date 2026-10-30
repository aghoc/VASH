#!/usr/bin/env python3
"""Choose rare-class thresholds on validation positives and frozen negative frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from vagt_orca.gpu import require_cuda


def load_predictions(model, data: Path, negative_only: bool) -> list[dict]:
    config = yaml.safe_load(data.read_text())
    root = Path(config["path"])
    images = sorted(
        p for p in (root / config["val"]).iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    )
    records = []
    batch_size = (
        1 if isinstance(model.model, str | Path) and Path(model.model).suffix == ".engine" else 16
    )
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=640,
            conf=0.001,
            device=0,
            quantize=32,
            rect=False,
            batch=batch_size,
            verbose=False,
        )
        for image, result in zip(batch, results):
            height, width = result.orig_shape
            targets = {1: [], 13: []}
            for line in (root / "labels" / "val" / f"{image.stem}.txt").read_text().splitlines():
                fields = line.split()
                if not fields or int(fields[0]) not in targets:
                    continue
                x, y, w, h = map(float, fields[1:])
                targets[int(fields[0])].append(
                    [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        (x + w / 2) * width,
                        (y + h / 2) * height,
                    ]
                )
            pred = result.boxes.data.cpu().numpy()
            records.append(
                {"targets": targets, "predictions": pred, "negative_only": negative_only}
            )
        if start % 1000 < batch_size:
            print(
                f"Calibration {data}: {min(start + batch_size, len(images))}/{len(images)}",
                flush=True,
            )
    return records


def measure(records: list[dict], class_id: int, threshold: float) -> dict:
    tp = fp = fn = negatives = fp_negatives = 0
    for record in records:
        gt = np.array(record["targets"][class_id]).reshape(-1, 4)
        if record["negative_only"] and len(gt):
            continue
        pred = record["predictions"]
        pred = pred[(pred[:, 5] == class_id) & (pred[:, 4] >= threshold)]
        pred = pred[np.argsort(-pred[:, 4])]
        negatives += int(not len(gt))
        fp_negatives += int(not len(gt) and len(pred) > 0)
        if record["negative_only"]:
            continue
        matched = set()
        for box in pred:
            if not len(gt):
                fp += 1
                continue
            inter = np.maximum(
                0, np.minimum(box[2:4], gt[:, 2:]) - np.maximum(box[:2], gt[:, :2])
            ).prod(axis=1)
            union = (
                (box[2] - box[0]) * (box[3] - box[1])
                + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
                - inter
            )
            overlaps = inter / np.maximum(union, 1e-9)
            if matched:
                overlaps[list(matched)] = -1
            index = int(overlaps.argmax())
            if overlaps[index] >= 0.5:
                matched.add(index)
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(matched)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "negative_images": negatives,
        "false_positive_negative_images": fp_negatives,
        "negative_image_fp_rate": fp_negatives / max(1, negatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_cuda(0)
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    public = load_predictions(model, args.public, False)
    frozen = load_predictions(model, args.frozen, True)
    thresholds = [
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.35,
        0.5,
        0.65,
        0.75,
        0.9,
    ]
    report = {"model": str(args.model.resolve()), "split": "validation only", "classes": {}}
    for class_id, name, min_precision in [(1, "can", 0.7), (13, "coin", 0.9)]:
        curve = []
        for threshold in thresholds:
            positive = measure(public, class_id, threshold)
            negative = measure(frozen, class_id, threshold)
            curve.append(
                {
                    **positive,
                    "frozen_negative_images": negative["negative_images"],
                    "frozen_negative_fp_images": negative["false_positive_negative_images"],
                    "frozen_negative_fp_rate": negative["negative_image_fp_rate"],
                }
            )
        feasible = [
            row
            for row in curve
            if row["precision"] >= min_precision and row["frozen_negative_fp_rate"] <= 0.01
        ]
        selected = max(feasible, key=lambda row: row["f1"]) if feasible else None
        report["classes"][name] = {
            "minimum_public_precision": min_precision,
            "maximum_frozen_negative_fp_rate": 0.01,
            "selected": selected,
            "curve": curve,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
