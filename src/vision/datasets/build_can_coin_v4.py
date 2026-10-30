#!/usr/bin/env python3
"""Mix rare-class public data with retention examples and frozen evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml

from vagt_orca.datasets.lvis import LVIS_TO_PROJECT, _download, _filename, _load_lvis
from vagt_orca.datasets.validate import validate_yolo_dataset


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise ValueError(f"Conflicting destination {destination}")
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def link_pair(image: Path, label: Path, output: Path, split: str, prefix: str = "") -> None:
    link(image, output / "images" / split / f"{prefix}{image.name}")
    link(label, output / "labels" / split / f"{prefix}{label.name}")


def lvis_lines(image: dict, annotations: list[dict], ids: dict[str, int]) -> list[str]:
    width, height = image["width"], image["height"]
    lines = []
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 > x1 and y2 > y1:
            lines.append(
                f"{ids[LVIS_TO_PROJECT[annotation['category_name']]]} "
                f"{(x1 + x2) / 2 / width:.8f} {(y1 + y2) / 2 / height:.8f} "
                f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
            )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--supplement", type=Path, required=True)
    p.add_argument("--lvis-train", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=750)
    p.add_argument("--coin-repeat", type=int, default=4)
    p.add_argument("--can-repeat", type=int, default=2)
    args = p.parse_args()
    base, supplement, output = args.base.resolve(), args.supplement.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    ids = {name: index for index, name in names.items()}
    by_class = defaultdict(list)
    images = {i.stem: i for i in (base / "images" / "train").iterdir()}
    for label in sorted((base / "labels" / "train").glob("*.txt")):
        for class_id in {
            int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()
        }:
            by_class[class_id].append(label)
    selected = set()
    for class_id, labels in by_class.items():
        random.Random(42 + class_id).shuffle(labels)
        selected.update(labels if class_id in (1, 13) else labels[: args.per_class])
    for label in selected:
        link_pair(images[label.stem], label, output, "train", "base_")
    for split in ("val", "test"):
        for image in (base / "images" / split).iterdir():
            link_pair(
                image, base / "labels" / split / f"{image.stem}.txt", output, split, "frozen_"
            )
    # Heldout Open Images must also be absent from the old model's training images.
    old_train_pixels = set()
    for image in (base / "images" / "train").iterdir():
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Invalid retention image {image}")
        old_train_pixels.add(hashlib.sha256(frame.tobytes()).hexdigest())
    excluded = []
    source_manifest = [
        json.loads(line) for line in (supplement / "manifest.jsonl").read_text().splitlines()
    ]
    for record in source_manifest:
        image = Path(record["image"])
        split = record["split"]
        if record["pixel_sha256"] in old_train_pixels:
            excluded.append(record["image"])
            continue
        label = supplement / "labels" / split / f"{image.stem}.txt"
        repeat = 1
        if split == "train":
            class_ids = {int(line.split()[0]) for line in record["labels"]}
            repeat = (
                args.coin_repeat if 13 in class_ids else args.can_repeat if 1 in class_ids else 1
            )
        for occurrence in range(repeat):
            link_pair(image, label, output, split, f"public_r{occurrence}_")
    # Recover all mapped labels for training images affected by the two formerly missing LVIS classes.
    lvis_images, annotations = _load_lvis(args.lvis_train)
    rare_ids = {
        image_id
        for image_id, records in annotations.items()
        if any(r["category_name"] in {"beer_can", "penny_(coin)"} for r in records)
    }
    holdout_ids = set()
    base_lvis = {}
    for split in ("train", "val", "test"):
        for image in (base / "images" / split).iterdir():
            match = re.search(r"lvis_(\d+)_", image.name)
            if match:
                if split == "train":
                    base_lvis[int(match[1])] = image
                else:
                    holdout_ids.add(int(match[1]))
    jobs = []
    fixed = []
    for image_id in sorted(rare_ids - holdout_ids):
        image = lvis_images[image_id]
        dest = output / "lvis_recovered" / "images" / _filename(image)
        if image_id in base_lvis:
            link(base_lvis[image_id], dest)
        else:
            jobs.append((str(image["coco_url"]), dest))
        label = output / "lvis_recovered" / "labels" / (dest.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = lvis_lines(image, annotations[image_id], ids)
        label.write_text("\n".join(lines) + "\n")
        fixed.append({"image_id": image_id, "boxes": len(lines), "existing": image_id in base_lvis})
        # Replace any selected old label for this image, keeping one instance rather than contradictory labels.
        for old_label in (output / "labels" / "train").glob(f"*lvis_{image_id}_*.txt"):
            (output / "images" / "train" / (old_label.stem + ".jpg")).unlink()
            old_label.unlink()
        link_pair(dest, label, output, "train", "recovered_")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda job: _download(*job), jobs))
    config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    integrity = validate_yolo_dataset(output / "data.yaml")
    if not integrity["valid"] or any(integrity["missing_classes_by_split"].values()):
        raise ValueError(f"Invalid mixed dataset: {integrity}")
    report = {
        "base_retention_images": len(selected),
        "base_per_class": args.per_class,
        "repeats": {
            "coin_positive": args.coin_repeat,
            "can_positive": args.can_repeat,
            "negative": 1,
        },
        "old_train_duplicates_excluded": excluded,
        "lvis_recovery": fixed,
        "integrity": integrity,
        "unique_train_sources": len({i.resolve() for i in (output / "images" / "train").iterdir()}),
        "source_counts": dict(Counter(r["source"] for r in source_manifest)),
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"lvis_recovery", "old_train_duplicates_excluded"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
