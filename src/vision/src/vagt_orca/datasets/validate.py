from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_yolo_dataset(data_yaml_path: Path) -> dict[str, Any]:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    names = config["names"]
    class_count = len(names)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}
    class_counts: Counter[int] = Counter()
    split_class_counts: dict[str, Counter[int]] = {}
    seen_resolved_images: dict[Path, str] = {}

    for split in ("train", "val", "test"):
        current_class_counts: Counter[int] = Counter()
        split_class_counts[split] = current_class_counts
        image_dir = root / str(config[split])
        label_dir = root / "labels" / split
        images = sorted(
            path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        stats[split] = {"images": len(images), "labels": len(labels)}
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for stem in sorted(image_stems - label_stems):
            errors.append(f"{split}: missing label for image {stem}")
        for stem in sorted(label_stems - image_stems):
            errors.append(f"{split}: missing image for label {stem}")
        for image in images:
            if image.is_symlink() and not image.exists():
                errors.append(f"{split}: broken image symlink {image}")
                continue
            resolved = image.resolve()
            previous_split = seen_resolved_images.get(resolved)
            if previous_split and previous_split != split:
                errors.append(f"split leakage: {resolved} occurs in {previous_split} and {split}")
            seen_resolved_images[resolved] = split
        for label in labels:
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_number}: non-numeric value")
                    continue
                if not 0 <= class_id < class_count:
                    errors.append(f"{label}:{line_number}: invalid class {class_id}")
                if any(not 0.0 <= value <= 1.0 for value in coords):
                    errors.append(f"{label}:{line_number}: bbox outside normalized range")
                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"{label}:{line_number}: non-positive bbox size")
                cx, cy, width, height = coords
                if (
                    cx - width / 2 < -1e-6
                    or cy - height / 2 < -1e-6
                    or cx + width / 2 > 1.0 + 1e-6
                    or cy + height / 2 > 1.0 + 1e-6
                ):
                    errors.append(f"{label}:{line_number}: bbox edges outside image")
                class_counts[class_id] += 1
                current_class_counts[class_id] += 1

    missing_classes = [index for index in range(class_count) if class_counts[index] == 0]
    if missing_classes:
        warnings.append(f"classes with zero annotations: {missing_classes}")
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "missing_classes": missing_classes,
        "splits": stats,
        "class_counts": dict(sorted(class_counts.items())),
        "class_counts_by_split": {
            split: {index: counts[index] for index in range(class_count)}
            for split, counts in split_class_counts.items()
        },
        "missing_classes_by_split": {
            split: [index for index in range(class_count) if counts[index] == 0]
            for split, counts in split_class_counts.items()
        },
    }
    (root / "integrity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
