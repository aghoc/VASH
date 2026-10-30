#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml

from vagt_orca.evaluation import select_class_balanced_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portable Phase 6 benchmark images")
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/egoobjects_15/data.yaml")
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmark/yolo26s_phase6/images")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase6_benchmark_dataset.json")
    )
    args = parser.parse_args()
    if args.count < 15:
        raise ValueError("Use at least one representative image per target class")
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    image_dir = root / config[args.split]
    label_dir = root / "labels" / args.split
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    selected = select_class_balanced_images(images, label_dir, args.count, len(config["names"]))
    if args.output.exists():
        existing = [path for path in args.output.iterdir() if path.is_file()]
        if existing:
            raise FileExistsError(f"Benchmark output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in selected:
        destination = args.output / source.name
        shutil.copy2(source, destination)
        if cv2.imread(str(destination)) is None:
            raise RuntimeError(f"Benchmark image failed to decode: {destination}")
    report = {
        "dataset": str(args.output.resolve()),
        "source_data": str(args.data.resolve()),
        "split": args.split,
        "selection": "deterministic_class_balanced_round_robin",
        "images": len(selected),
        "all_images_decoded": True,
        "model_family": "YOLO26s",
        "deployment_precision": "fp16",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
