#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml

from vagt_orca.datasets.validate import validate_yolo_dataset


def link_split(source: Path, destination: Path, prefix: str = "") -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        name = f"{prefix}{item.name}"
        target = destination / name
        relative = os.path.relpath(item.resolve(), target.parent.resolve())
        if target.is_symlink():
            if target.resolve() != item.resolve():
                raise RuntimeError(f"Conflicting link: {target}")
        elif target.exists():
            raise RuntimeError(f"Refusing to overwrite: {target}")
        else:
            target.symlink_to(relative)
        count += 1
    return count


def link_base_focus(base: Path, output: Path, per_class: int, seed: int) -> dict[str, int]:
    image_by_stem = {
        item.stem: item for item in (base / "images" / "train").iterdir() if item.is_file()
    }
    labels_by_class: dict[int, list[Path]] = defaultdict(list)
    for label in sorted((base / "labels" / "train").iterdir()):
        if not label.is_file():
            continue
        class_ids = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        for class_id in class_ids:
            labels_by_class[class_id].append(label)

    selected: set[Path] = set()
    selected_by_class: dict[int, int] = {}
    for class_id, labels in sorted(labels_by_class.items()):
        random.Random(seed + class_id).shuffle(labels)
        chosen = labels[:per_class]
        selected.update(chosen)
        selected_by_class[class_id] = len(chosen)

    image_output = output / "images" / "train"
    label_output = output / "labels" / "train"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    for label in sorted(selected):
        image = image_by_stem.get(label.stem)
        if image is None:
            raise FileNotFoundError(f"Image missing for selected label: {label}")
        for source, destination in ((image, image_output / image.name), (label, label_output / label.name)):
            relative = os.path.relpath(source.resolve(), destination.parent.resolve())
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(f"Refusing to overwrite: {destination}")
            destination.symlink_to(relative)
    return {
        "base_images": len(selected),
        "base_labels": len(selected),
        **{f"selected_class_{key}": value for key, value in selected_by_class.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the frozen EgoObjects/LVIS dataset with a train-only public supplement"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_yaml = yaml.safe_load((args.base / "data.yaml").read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    if args.base_per_class is None:
        counts["train"] = {
            "base_images": link_split(
                args.base / "images" / "train", args.output / "images" / "train"
            ),
            "base_labels": link_split(
                args.base / "labels" / "train", args.output / "labels" / "train"
            ),
        }
    else:
        counts["train"] = link_base_focus(
            args.base, args.output, args.base_per_class, args.seed
        )
    for split in ("val", "test"):
        counts[split] = {
            "base_images": link_split(
                args.base / "images" / split, args.output / "images" / split
            ),
            "base_labels": link_split(
                args.base / "labels" / split, args.output / "labels" / split
            ),
        }

    counts["train"]["supplement_images"] = link_split(
        args.supplement / "images" / "train",
        args.output / "images" / "train",
        prefix="public_v2_",
    )
    counts["train"]["supplement_labels"] = link_split(
        args.supplement / "labels" / "train",
        args.output / "labels" / "train",
        prefix="public_v2_",
    )

    dataset = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": base_yaml["names"],
    }
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )
    integrity = validate_yolo_dataset(args.output / "data.yaml")
    report = {"counts": counts, "integrity": integrity}
    (args.output / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not integrity["valid"] or integrity["missing_classes"]:
        raise RuntimeError(f"Combined dataset integrity failed: {integrity}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
