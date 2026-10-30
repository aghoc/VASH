#!/usr/bin/env python3
"""Prepare traceable Open Images / RF100 supplements without sequence leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from vagt_orca.datasets.lvis import _download

OI_MAPPING = {
    "Tin can": 1,
    "Coin": 13,
    "Bottle": 0,
    "Mug": 2,
    "Box": 3,
    "Ball": 4,
    "Cricket ball": 4,
    "Tennis ball": 4,
    "Volleyball (Ball)": 4,
    "Mobile phone": 5,
    "Remote control": 6,
    "Pen": 7,
    "Spoon": 8,
    "Fork": 9,
    "Toothbrush": 10,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def box_line(class_id: int, xyxy: list[float]) -> str:
    x1, y1, x2, y2 = [min(1.0, max(0.0, x)) for x in xyxy]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box: {xyxy}")
    return f"{class_id} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}"


def rf_group(filename: str) -> tuple[str, str]:
    original = filename.split(".rf.")[0]
    if re.match(r"^20220422\d+", original):
        return "train", "rf_video_20220422"
    match = re.match(r"IMG_(\d{8})_(\d{4})", original)
    if match:
        group = f"rf_burst_{match[1]}_{match[2]}"
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        return ("val" if bucket == 0 else "test" if bucket == 1 else "train"), group
    match = re.match(r"IMG_(\d{4})_", original)
    if match:
        # Keep the independently photographed stills out of training; nearby file numbers stay together.
        group = f"rf_stills_{int(match[1]) // 30}"
        return ("val" if int(match[1]) // 30 % 2 == 0 else "test"), group
    raise ValueError(f"Unrecognized RF100 sequence: {filename}")


def prepare_rf(raw: Path, root: Path) -> list[dict]:
    extracted = raw / "rf100"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(raw / "coins.tar.gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = member.name.split("/coins-1apki/", 1)[-1]
            relative = Path(suffix)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive member")
            if relative.suffix.lower() not in {".json", ".jpg", ".txt"}:
                continue
            target = extracted / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    records = []
    sampled_video = set()
    for source_split in ("train", "valid", "test"):
        data = json.loads((extracted / source_split / "_annotations.coco.json").read_text())
        category = {c["id"]: c["name"] for c in data["categories"]}
        by_image = defaultdict(list)
        for annotation in data["annotations"]:
            if category[annotation["category_id"]] == "coin":
                by_image[annotation["image_id"]].append(annotation)
            elif category[annotation["category_id"]] not in {"nail", "nut", "screw"}:
                raise ValueError("Unexpected annotated RF class")
        for image in sorted(data["images"], key=lambda x: x["file_name"]):
            if Path(image["file_name"]).name != image["file_name"]:
                raise ValueError("RF100 file_name must be a basename")
            split, group = rf_group(image["file_name"])
            if group == "rf_video_20220422":
                # Original test/valid frames overlap the training video. Do not treat them as holdouts.
                if source_split != "train":
                    continue
                timestamp = re.match(r"^(\d{17})", image["file_name"])[1]
                bucket = timestamp[:14] + str(int(timestamp[14:]) // 100)
                if bucket in sampled_video:
                    continue
                sampled_video.add(bucket)
            width, height = image["width"], image["height"]
            lines = []
            for annotation in by_image[image["id"]]:
                x, y, w, h = annotation["bbox"]
                if w > 0 and h > 0:
                    lines.append(
                        box_line(13, [x / width, y / height, (x + w) / width, (y + h) / height])
                    )
            filename = "rf100_" + image["file_name"]
            dest = root / "images" / split / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(
                    os.path.relpath(extracted / source_split / image["file_name"], dest.parent)
                )
            label = root / "labels" / split / (Path(filename).stem + ".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""))
            records.append(
                {
                    "source": "RF100 coins-1apki",
                    "original_split": source_split,
                    "split": split,
                    "group": group,
                    "image": str(dest),
                    "labels": lines,
                    "license": "CC BY 4.0",
                    "origin": "https://universe.roboflow.com/roboflow-100/coins-1apki",
                }
            )
    print("RF100 prepared", dict(Counter(r["split"] for r in records)), flush=True)
    return records


def prepare_oi(raw: Path, root: Path, class_csv: Path, workers: int) -> list[dict]:
    mids = {
        mid: OI_MAPPING[name] for mid, name in csv.reader(class_csv.open()) if name in OI_MAPPING
    }
    records = []
    for source_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        cache = raw / f"oi_selected_{split}.json"
        if cache.exists():
            selected = json.loads(cache.read_text())
        else:
            annotations = defaultdict(list)
            excluded = set()
            with (raw / f"oid_{split}.csv").open() as f:
                header = next(f).strip().split(",")
                index = {name: i for i, name in enumerate(header)}
                for line in f:
                    values = line.rstrip().split(",")
                    mid = values[index["LabelName"]]
                    if mid not in mids:
                        continue
                    image_id = values[index["ImageID"]]
                    if values[index["IsGroupOf"]] == "1" or values[index["IsDepiction"]] == "1":
                        excluded.add(image_id)
                        continue
                    xyxy = [float(values[index[name]]) for name in ("XMin", "YMin", "XMax", "YMax")]
                    annotations[image_id].append(box_line(mids[mid], xyxy))
            chosen = set()
            for class_id, limit in ((1, 2400), (13, 1600)):
                candidates = sorted(
                    i
                    for i, lines in annotations.items()
                    if i not in excluded and any(int(line.split()[0]) == class_id for line in lines)
                )
                random.Random(42 + class_id).shuffle(candidates)
                chosen.update(candidates[:limit] if split == "train" else candidates)
            selected = {i: annotations[i] for i in sorted(chosen)}
            cache.write_text(json.dumps(selected))
        jobs = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for image_id in selected:
                dest = root / "images" / split / f"oi_{image_id}.jpg"
                url = f"https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg"
                jobs[executor.submit(_download, url, dest)] = (image_id, dest, url)
            failed = []
            for n, future in enumerate(as_completed(jobs), 1):
                image_id, dest, url = jobs[future]
                try:
                    future.result()
                except (OSError, urllib.error.URLError) as error:
                    failed.append({"image_id": image_id, "error": str(error)})
                    continue
                lines = selected[image_id]
                label = root / "labels" / split / f"oi_{image_id}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("\n".join(lines) + "\n")
                records.append(
                    {
                        "source": "Open Images",
                        "split": split,
                        "group": f"oi_{image_id}",
                        "image": str(dest),
                        "labels": lines,
                        "image_id": image_id,
                        "original_split": source_split,
                        "url": url,
                        "license": "Images: CC BY 2.0, annotations: CC BY 4.0",
                        "origin": "https://storage.googleapis.com/openimages/web/download_v7.html",
                    }
                )
                if n % 100 == 0:
                    print(f"Open Images {split}: {n}/{len(jobs)}", flush=True)
            (raw / f"oi_download_failures_{split}.json").write_text(json.dumps(failed, indent=2))
        print("Open Images prepared", split, len(selected), "failed", len(failed), flush=True)
    return records


def perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    coefficients = cv2.dct(cv2.resize(image, (32, 32)).astype(np.float32))[:8, :8].flatten()
    threshold = np.median(coefficients[1:])
    value = 0
    for bit in coefficients > threshold:
        value = (value << 1) | int(bit)
    return value


def deduplicate(records: list[dict], frozen: Path) -> tuple[list[dict], list[dict]]:
    # Prefer frozen holdouts, then new test, then val, then train. Exact decoded pixels catch different encodings.
    exact = {}
    holdout_hashes = []
    for split in ("val", "test"):
        for image in (frozen / "images" / split).iterdir():
            frame = cv2.imread(str(image))
            if frame is not None:
                exact[hashlib.sha256(frame.tobytes()).hexdigest()] = f"frozen/{split}"
    kept, removed = [], []
    for record in sorted(
        records, key=lambda r: ({"test": 0, "val": 1, "train": 2}[r["split"]], r["image"])
    ):
        path = Path(record["image"])
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Corrupt image {path}")
        pixel_sha = hashlib.sha256(frame.tobytes()).hexdigest()
        phash = perceptual_hash(path)
        reason = exact.get(pixel_sha)
        if not reason and record["source"].startswith("RF100"):
            # Remove near-duplicate RF stills across partitions as well as filename/video grouping.
            reason = next(
                (
                    other
                    for value, split, other in holdout_hashes
                    if split != record["split"] and (value ^ phash).bit_count() <= 4
                ),
                None,
            )
        if reason:
            removed.append({"image": str(path), "split": record["split"], "matches": reason})
            path.unlink()
            (path.parent.parent.parent / "labels" / record["split"] / (path.stem + ".txt")).unlink()
            continue
        exact[pixel_sha] = str(path)
        if record["split"] != "train" and record["source"].startswith("RF100"):
            holdout_hashes.append((phash, record["split"], str(path)))
        record.update(sha256=digest(path), pixel_sha256=pixel_sha, phash=f"{phash:016x}")
        kept.append(record)
    groups = defaultdict(set)
    for record in kept:
        groups[record["group"]].add(record["split"])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Source-group split leakage")
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--class-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    raw, root = args.raw.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = prepare_rf(raw, root) + prepare_oi(raw, root, args.class_csv, args.workers)
    records, removed = deduplicate(records, args.frozen)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load((args.frozen / "data.yaml").read_text())["names"]
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
        )
    )
    with (root / "manifest.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    report = {
        "images": dict(Counter(r["split"] for r in records)),
        "sources": dict(Counter(r["source"] for r in records)),
        "boxes": {
            s: dict(
                Counter(line.split()[0] for r in records if r["split"] == s for line in r["labels"])
            )
            for s in ("train", "val", "test")
        },
        "removed_duplicates": removed,
        "archive_sha256": digest(raw / "coins.tar.gz"),
        "rf_revision": "ec1591c4dd8fd3f93a2060a5a5a2a181c8d4e190",
        "rf_split_policy": "Same-video frames only train (100ms sampling); burst minutes grouped; nearby still numbers held out",
        "oi_mapping": OI_MAPPING,
    }
    (root / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "removed_duplicates"}, indent=2))


if __name__ == "__main__":
    main()
