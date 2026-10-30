#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from vagt_orca.evaluation import select_class_balanced_images
from vagt_orca.gpu import gpu_summary, require_cuda


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def load_ground_truth(
    image: Path, label_dir: Path, width: int, height: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    label = label_dir / f"{image.stem}.txt"
    output = []
    for line in label.read_text(encoding="utf-8").splitlines():
        class_id_text, cx_text, cy_text, width_text, height_text = line.split()
        cx, cy = float(cx_text) * width, float(cy_text) * height
        box_width, box_height = float(width_text) * width, float(height_text) * height
        output.append(
            (
                int(class_id_text),
                (
                    cx - box_width / 2,
                    cy - box_height / 2,
                    cx + box_width / 2,
                    cy + box_height / 2,
                ),
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save representative YOLO false positive/negative cases"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-save", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("artifacts/failures"))
    args = parser.parse_args()
    require_cuda(args.device)

    config: dict[str, Any] = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    image_dir = root / config[args.split]
    label_dir = root / "labels" / args.split
    all_images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    names = {int(index): name for index, name in config["names"].items()}
    images = select_class_balanced_images(
        all_images, label_dir, min(args.max_images, len(all_images)), len(names)
    )

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    misses: Counter[int] = Counter()
    false_positives: Counter[int] = Counter()
    ground_truth_counts: Counter[int] = Counter()
    saved: list[dict[str, object]] = []

    indexed_results = (
        (image, result)
        for start in range(0, len(images), args.batch)
        for image, result in zip(
            images[start : start + args.batch],
            model.predict(
                source=[str(path) for path in images[start : start + args.batch]],
                device=args.device,
                conf=args.confidence,
                quantize=32,
                verbose=False,
            ),
        )
    )
    for image, result in indexed_results:
        height, width = result.orig_shape
        ground_truth = load_ground_truth(image, label_dir, width, height)
        ground_truth_counts.update(class_id for class_id, _ in ground_truth)
        predictions = [
            (int(class_id), tuple(float(value) for value in bbox))
            for class_id, bbox in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())
        ]
        matched_predictions: set[int] = set()
        missed_boxes = []
        for class_id, gt_box in ground_truth:
            candidates = [
                (index, iou(gt_box, pred_box))
                for index, (pred_class, pred_box) in enumerate(predictions)
                if pred_class == class_id and index not in matched_predictions
            ]
            best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
            if best_iou >= args.iou_match:
                matched_predictions.add(best_index)
            else:
                misses[class_id] += 1
                missed_boxes.append((class_id, gt_box))
        unmatched = [
            item for index, item in enumerate(predictions) if index not in matched_predictions
        ]
        for class_id, _ in unmatched:
            false_positives[class_id] += 1

        if (missed_boxes or unmatched) and len(saved) < args.max_save:
            rendered = result.plot()
            for class_id, box in missed_boxes:
                x1, y1, x2, y2 = (int(value) for value in box)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    rendered,
                    f"MISSED {result.names[class_id]}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            output_image = args.output / f"{len(saved):04d}_{image.name}"
            cv2.imwrite(str(output_image), rendered)
            saved.append(
                {
                    "source": str(image.resolve()),
                    "rendered": str(output_image.resolve()),
                    "misses": len(missed_boxes),
                    "false_positives": len(unmatched),
                }
            )

    report = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "gpu": gpu_summary(args.device),
        "split": args.split,
        "images_analyzed": len(images),
        "gpu_batch": args.batch,
        "sampling": "deterministic_class_balanced_round_robin",
        "iou_match": args.iou_match,
        "ground_truth_by_class": {
            names[index]: ground_truth_counts[index] for index in sorted(names)
        },
        "misses_by_class": {names[index]: count for index, count in sorted(misses.items())},
        "false_positives_by_class": {
            names[index]: count for index, count in sorted(false_positives.items())
        },
        "saved_cases": saved,
    }
    (args.output / "failure_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
