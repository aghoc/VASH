#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from vagt_orca.config import load_yaml
from vagt_orca.deployment import sha256
from vagt_orca.evaluation import select_class_balanced_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable class-balanced INT8 bundle")
    parser.add_argument("--config", type=Path, default=Path("configs/deployment.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--count", type=int)
    parser.add_argument("--ar-glass-dir", type=Path, default=Path("data/raw/ar_glass"))
    parser.add_argument("--domain-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/calibration/jetson_int8"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase6_calibration.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_path = args.data or Path(config["data"])
    count = args.count or int(config["calibration"]["images"])
    if count < 500:
        raise ValueError("Use at least 500 representative calibration images")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Calibration output already exists: {args.output}")
        shutil.rmtree(args.output)

    dataset: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(dataset["path"])
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    image_dir = root / dataset[args.split]
    label_dir = root / "labels" / args.split
    class_names = {int(index): name for index, name in dataset["names"].items()}
    public_images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    ar_images = (
        sorted(
            path
            for path in args.ar_glass_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if args.ar_glass_dir.is_dir()
        else []
    )
    supplemental_images = (
        sorted(
            path
            for path in args.domain_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if args.domain_dir and args.domain_dir.is_dir()
        else []
    )
    domain_images = ar_images or supplemental_images
    domain_kind = "ar_glass" if ar_images else "public_egocentric_supplement"
    domain_limit = min(len(domain_images), count // 4)
    public_count = count - domain_limit
    selected = select_class_balanced_images(
        public_images, label_dir, public_count, len(class_names)
    )
    if len(selected) != public_count:
        raise RuntimeError(f"Requested {public_count} public images, selected {len(selected)}")

    images_out = args.output / "images"
    labels_out = args.output / "labels"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)
    class_counts: Counter[int] = Counter()
    entries = []
    for index, source in enumerate(selected):
        destination = images_out / f"public_{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        if cv2.imread(str(destination)) is None:
            raise RuntimeError(f"Calibration image failed to decode: {destination}")
        source_label = label_dir / f"{source.stem}.txt"
        destination_label = labels_out / f"{destination.stem}.txt"
        shutil.copy2(source_label, destination_label)
        classes = sorted(
            {int(line.split(maxsplit=1)[0]) for line in source_label.read_text().splitlines()}
        )
        class_counts.update(classes)
        entries.append(
            {
                "file": destination.name,
                "source": str(source.resolve()),
                "source_kind": "public_egocentric_or_lvis",
                "classes": classes,
                "sha256": sha256(destination),
            }
        )
    for offset, source in enumerate(domain_images[:domain_limit], start=len(selected)):
        destination = images_out / f"domain_{offset:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        if cv2.imread(str(destination)) is None:
            raise RuntimeError(f"Calibration image failed to decode: {destination}")
        (labels_out / f"{destination.stem}.txt").write_text("", encoding="utf-8")
        entries.append(
            {
                "file": destination.name,
                "source": str(source.resolve()),
                "source_kind": domain_kind,
                "classes": [],
                "sha256": sha256(destination),
            }
        )

    calibration_yaml = args.output / "data.yaml"
    calibration_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "images",
                "val": "images",
                "names": class_names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        "dataset": str(args.output.resolve()),
        "data_yaml": str(calibration_yaml.resolve()),
        "images": len(entries),
        "minimum_recommended_images_met": len(entries) >= 500,
        "selection": "class_balanced_public_plus_up_to_25_percent_domain_data",
        "source_split": args.split,
        "public_images": len(selected),
        "domain_images": domain_limit,
        "domain_kind": domain_kind if domain_limit else None,
        "ar_glass_images": domain_limit if domain_kind == "ar_glass" else 0,
        "ar_glass_available": len(ar_images),
        "supplemental_domain_available": len(supplemental_images),
        "class_image_counts": {class_names[index]: class_counts[index] for index in class_names},
        "all_classes_present": all(class_counts[index] > 0 for index in class_names),
        "all_images_decoded": True,
        "portable_yaml": "path" not in yaml.safe_load(calibration_yaml.read_text(encoding="utf-8")),
        "entries": entries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
