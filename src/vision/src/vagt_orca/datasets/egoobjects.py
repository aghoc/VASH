from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from vagt_orca.config import PROJECT_ROOT, load_yaml


@dataclass(slots=True)
class ConversionStats:
    images: int = 0
    annotations: int = 0
    skipped_categories: int = 0
    invalid_boxes: int = 0
    missing_images: int = 0
    missing_category_ids: int = 0


def normalize_category(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def load_class_mapping(path: str | Path) -> tuple[list[str], dict[str, int]]:
    config = load_yaml(path)
    names_map = {int(key): str(value) for key, value in config["names"].items()}
    names = [names_map[index] for index in range(len(names_map))]
    aliases: dict[str, int] = {}
    for class_id, class_name in enumerate(names):
        for alias in config["aliases"].get(class_name, [class_name]):
            normalized = normalize_category(str(alias))
            if normalized in aliases and aliases[normalized] != class_id:
                raise ValueError(f"Alias {alias!r} maps to multiple classes")
            aliases[normalized] = class_id
    return names, aliases


def _group_key(image: dict[str, Any]) -> str:
    # EgoObjects `video_id` is only unique inside `group_id` (often 01..10).
    # Prefer the globally unique capture/object group to prevent near-frame leakage.
    for key in ("group_id", "video_uid", "sequence_id", "clip_id", "video_id"):
        if image.get(key) is not None:
            return f"{key}:{image[key]}"
    # EgoObjects frames normally preserve their video directory in file_name.
    image_name = image.get("file_name", image.get("url", image["id"]))
    parent = Path(str(image_name)).parent.as_posix()
    if parent not in ("", "."):
        return f"parent:{parent}"
    # Conservative fallback: one image per group, explicit in the report.
    return f"image:{image['id']}"


def _assign_splits(
    group_keys: Iterable[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("split ratios must be three non-negative values")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_cut = ratios[0] / total
    val_cut = (ratios[0] + ratios[1]) / total
    groups = sorted(set(group_keys))
    random.Random(seed).shuffle(groups)
    assignments: dict[str, str] = {}
    for index, group in enumerate(groups):
        fraction = (index + 0.5) / max(1, len(groups))
        assignments[group] = "train" if fraction < train_cut else "val" if fraction < val_cut else "test"
    return assignments


def _ensure_class_coverage(
    assignments: dict[str, str], group_classes: dict[str, set[int]]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Move the fewest complete groups needed to cover each class in every split."""
    splits = ("train", "val", "test")
    class_groups: dict[int, set[str]] = defaultdict(set)
    for group, class_ids in group_classes.items():
        for class_id in class_ids:
            class_groups[class_id].add(group)
    impossible = sorted(class_id for class_id, groups in class_groups.items() if len(groups) < 3)
    changes: list[dict[str, Any]] = []

    # Rarest classes go first so common classes cannot consume their only safe donor groups.
    for class_id, groups in sorted(class_groups.items(), key=lambda item: (len(item[1]), item[0])):
        if len(groups) < 3:
            continue
        for target_split in splits:
            if any(assignments[group] == target_split for group in groups):
                continue
            coverage = {
                split: Counter(
                    class_value
                    for group, class_ids in group_classes.items()
                    if assignments[group] == split
                    for class_value in class_ids
                )
                for split in splits
            }
            candidates = [
                group
                for group in groups
                if assignments[group] != target_split
                and coverage[assignments[group]][class_id] > 1
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda group: (
                    sum(
                        coverage[assignments[group]][other_class] <= 1
                        for other_class in group_classes[group]
                    ),
                    -coverage[assignments[group]][class_id],
                    len(group_classes[group]),
                    group,
                )
            )
            group = candidates[0]
            source_split = assignments[group]
            assignments[group] = target_split
            changes.append(
                {
                    "class_id": class_id,
                    "group": group,
                    "from": source_split,
                    "to": target_split,
                }
            )
    return changes, impossible


def _resolve_image(image_root: Path, file_name: str) -> Path:
    direct = image_root / file_name
    if direct.is_file():
        return direct
    basename = image_root / Path(file_name).name
    return basename


def _safe_output_name(image: dict[str, Any]) -> str:
    source_name = str(image.get("file_name", image.get("url", image["id"])))
    original = Path(source_name)
    digest = hashlib.sha1(source_name.encode()).hexdigest()[:10]
    return f"ego_{image['id']}_{digest}{original.suffix.lower() or '.jpg'}"


def _source_name(image: dict[str, Any]) -> str:
    value = image.get("file_name", image.get("url"))
    if not value:
        raise ValueError(f"Image {image.get('id')} has neither file_name nor url")
    return str(value)


def convert_egoobjects(
    annotation_paths: list[Path],
    image_root: Path,
    output_root: Path,
    class_config: Path = PROJECT_ROOT / "configs/classes.yaml",
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    names, alias_to_id = load_class_mapping(class_config)
    images: dict[int, dict[str, Any]] = {}
    annotations: list[dict[str, Any]] = []
    source_categories: dict[int, str] = {}

    for annotation_path in annotation_paths:
        with annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for category in data.get("categories", []):
            source_categories[int(category["id"])] = str(category["name"])
        for image in data.get("images", []):
            image_id = int(image["id"])
            if image_id in images and images[image_id]["file_name"] != image["file_name"]:
                raise ValueError(f"Conflicting image id {image_id}")
            images[image_id] = image
        annotations.extend(data.get("annotations", []))

    category_remap = {
        source_id: alias_to_id[normalize_category(name)]
        for source_id, name in source_categories.items()
        if normalize_category(name) in alias_to_id
    }
    if not category_remap:
        available = sorted(normalize_category(name) for name in source_categories.values())
        raise ValueError(f"No target categories matched. First source categories: {available[:30]}")

    relevant_image_ids = {
        int(annotation["image_id"])
        for annotation in annotations
        if annotation.get("category_id") is not None
        and int(annotation["category_id"]) in category_remap
    }
    selected_images = {image_id: images[image_id] for image_id in relevant_image_ids if image_id in images}
    all_groups = {_group_key(image) for image in selected_images.values()}
    explicit_group_splits: dict[str, set[str]] = defaultdict(set)
    for image in selected_images.values():
        if image.get("subset") in {"train", "val", "test"}:
            explicit_group_splits[_group_key(image)].add(str(image["subset"]))
    has_source_splits = any(
        image.get("subset") in {"train", "val", "test"} for image in selected_images.values()
    )
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if annotation.get("category_id") is None:
            continue
        if int(annotation["category_id"]) in category_remap:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    # EgoObjects' provided val/test frame split shares most `group_id` values.
    # Re-split every complete group to satisfy the plan's no-sequence-leakage rule,
    # then minimally move whole groups so rare classes remain measurable in every split.
    group_assignments = _assign_splits(all_groups, split_ratios, seed)
    group_classes: dict[str, set[int]] = defaultdict(set)
    for image_id, image in selected_images.items():
        for annotation in annotations_by_image[image_id]:
            group_classes[_group_key(image)].add(category_remap[int(annotation["category_id"])])
    coverage_moves, insufficient_group_classes = _ensure_class_coverage(
        group_assignments, group_classes
    )

    stats = ConversionStats(
        missing_category_ids=sum(annotation.get("category_id") is None for annotation in annotations),
        skipped_categories=sum(
            annotation.get("category_id") is not None
            and int(annotation["category_id"]) not in category_remap
            for annotation in annotations
        ),
    )
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter(group_assignments.values())
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id, image in sorted(selected_images.items()):
        source = _resolve_image(image_root, _source_name(image))
        if not source.is_file():
            stats.missing_images += 1
            continue
        width, height = int(image.get("width", 0)), int(image.get("height", 0))
        if width <= 0 or height <= 0:
            stats.invalid_boxes += len(annotations_by_image[image_id])
            continue
        split = group_assignments[_group_key(image)]
        output_name = _safe_output_name(image)
        destination = output_root / "images" / split / output_name
        if not destination.exists():
            if link_mode == "symlink":
                destination.symlink_to(source.resolve())
            elif link_mode == "hardlink":
                os.link(source, destination)
            else:
                raise ValueError("link_mode must be symlink or hardlink")

        yolo_lines: list[str] = []
        for annotation in annotations_by_image[image_id]:
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4:
                stats.invalid_boxes += 1
                continue
            x, y, box_width, box_height = (float(value) for value in bbox)
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            if x2 <= x1 or y2 <= y1:
                stats.invalid_boxes += 1
                continue
            class_id = category_remap[int(annotation["category_id"])]
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height
            yolo_lines.append(
                f"{class_id} {cx:.8f} {cy:.8f} {normalized_width:.8f} {normalized_height:.8f}"
            )
            stats.annotations += 1
            class_counts[names[class_id]] += 1
        if not yolo_lines:
            destination.unlink(missing_ok=True)
            continue
        label_path = output_root / "labels" / split / f"{Path(output_name).stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        stats.images += 1
        split_counts[split] += 1

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False, allow_unicode=True)
    report = {
        "stats": asdict(stats),
        "class_counts": dict(sorted(class_counts.items())),
        "split_image_counts": dict(split_counts),
        "split_group_counts": dict(group_counts),
        "source_category_matches": {
            source_categories[source_id]: names[target_id]
            for source_id, target_id in sorted(category_remap.items())
        },
        "grouping_fields": ["group_id", "video_uid", "sequence_id", "clip_id", "video_id", "file_parent"],
        "source_splits_present": has_source_splits,
        "split_policy": "sequence_safe_group_resplit_with_class_coverage",
        "class_coverage_group_moves": coverage_moves,
        "classes_with_fewer_than_three_groups": [
            names[class_id] for class_id in insufficient_group_classes
        ],
        "source_split_conflicts": {
            group: {
                "source_splits": sorted(splits),
                "resolved_split": group_assignments[group],
            }
            for group, splits in sorted(explicit_group_splits.items())
            if len(splits) > 1
        },
        "seed": seed,
    }
    (output_root / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
