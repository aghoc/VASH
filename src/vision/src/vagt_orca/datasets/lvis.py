#!/usr/bin/env python3
"""Build a project-class supplement from official LVIS annotations and COCO images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.egoobjects import load_class_mapping

LVIS_TO_PROJECT = {
    "ball": "ball",
    "bottle": "bottle",
    "box": "box",
    "card": "card",
    "business_card": "card",
    "credit_card": "card",
    "identity_card": "card",
    "keycard": "card",
    "can": "can",
    "beer_can": "can",
    "cellular_telephone": "phone",
    "coin": "coin",
    "penny_(coin)": "coin",
    "cup": "cup",
    "fork": "fork",
    "key": "key",
    "Lego": "small_block",
    "cube": "small_block",
    "pen": "pen",
    "remote_control": "remote",
    "spoon": "spoon",
    "toothbrush": "toothbrush",
}


def _filename(image: dict[str, Any]) -> str:
    source_name = image.get("file_name", image.get("coco_url"))
    if not source_name:
        raise ValueError(f"LVIS image {image.get('id')} lacks file_name/coco_url")
    return f"lvis_{image['id']}_{Path(str(source_name)).name}"


def _download(url: str, destination: Path, retries: int = 4) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            if url.startswith(("http://images.cocodataset.org/", "https://images.cocodataset.org/")):
                relative_path = url.split("images.cocodataset.org/", 1)[1]
                url = f"https://s3.amazonaws.com/images.cocodataset.org/{relative_path}"
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def _load_lvis(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    category_names = {int(item["id"]): str(item["name"]) for item in data["categories"]}
    wanted_ids = {
        category_id for category_id, name in category_names.items() if name in LVIS_TO_PROJECT
    }
    images = {int(item["id"]): item for item in data["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        if int(annotation["category_id"]) in wanted_ids:
            enriched = dict(annotation)
            enriched["category_name"] = category_names[int(annotation["category_id"])]
            annotations[int(annotation["image_id"])].append(enriched)
    return images, annotations


def prepare(
    train_json: Path,
    val_json: Path,
    output_root: Path,
    per_class_train: int,
    per_class_eval: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    names, _ = load_class_mapping(PROJECT_ROOT / "configs/classes.yaml")
    project_ids = {name: index for index, name in enumerate(names)}
    rng = random.Random(seed)
    selected: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []

    for source_split, path, limit in (
        ("train", train_json, per_class_train),
        ("eval", val_json, per_class_eval),
    ):
        images, annotations = _load_lvis(path)
        by_target: dict[str, list[int]] = defaultdict(list)
        for image_id, records in annotations.items():
            for target in {LVIS_TO_PROJECT[item["category_name"]] for item in records}:
                by_target[target].append(image_id)
        chosen_ids: set[int] = set()
        for target, image_ids in by_target.items():
            rng.shuffle(image_ids)
            chosen_ids.update(image_ids[:limit])
        for image_id in sorted(chosen_ids):
            split = source_split
            if source_split == "eval":
                digest = int(hashlib.sha1(str(image_id).encode()).hexdigest(), 16)
                split = "val" if digest % 2 == 0 else "test"
            selected.append((split, images[image_id], annotations[image_id]))

    downloads: list[tuple[str, Path]] = []
    for split, image, _ in selected:
        filename = _filename(image)
        destination = output_root / "images" / split / filename
        downloads.append((str(image["coco_url"]), destination))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda item: _download(*item), downloads))

    counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for split, image, annotations in selected:
        filename = _filename(image)
        width, height = float(image["width"]), float(image["height"])
        lines: list[str] = []
        for annotation in annotations:
            target = LVIS_TO_PROJECT[annotation["category_name"]]
            class_id = project_ids[target]
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x1 = min(max(x, 0.0), width)
            y1 = min(max(y, 0.0), height)
            x2 = min(max(x + box_width, 0.0), width)
            y2 = min(max(y + box_height, 0.0), height)
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 0.0 or box_height <= 0.0:
                continue
            lines.append(
                f"{class_id} {(x1 + box_width / 2) / width:.8f} "
                f"{(y1 + box_height / 2) / height:.8f} {box_width / width:.8f} "
                f"{box_height / height:.8f}"
            )
            counts[target] += 1
        label = output_root / "labels" / split / f"{Path(filename).stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        split_counts[split] += 1

    report = {
        "source": "LVIS v1 / COCO images",
        "purpose": "class-balanced public-data supplement for all available project classes",
        "mapping": LVIS_TO_PROJECT,
        "images": len(selected),
        "annotations": dict(sorted(counts.items())),
        "split_images": dict(sorted(split_counts.items())),
        "seed": seed,
    }
    (output_root / "lvis_supplement_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class-train", type=int, default=500)
    parser.add_argument("--per-class-eval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(
        args.train_json,
        args.val_json,
        args.output,
        args.per_class_train,
        args.per_class_eval,
        args.workers,
        args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
