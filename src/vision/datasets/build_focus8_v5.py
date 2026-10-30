"""Build a traceable eight-class replay supplement, preserving frozen evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.lvis import LVIS_TO_PROJECT, _download, _load_lvis
from vagt_orca.datasets.validate import validate_yolo_dataset

ROOT = PROJECT_ROOT
BASE = ROOT / "data/processed/public_v4_can_coin_retention_15"
EGO = ROOT / "data/processed/egoobjects_15"
RAW = ROOT / "data/raw/coco2017"
OUT = ROOT / "data/processed/public_v5_focus8_15"
RUN = ROOT / "runs/focus8_20261015"
TARGETS = ["bottle", "cup", "phone", "remote", "spoon", "fork", "toothbrush", "pen"]
COCO_MAPPING = {
    "bottle": "bottle", "cup": "cup", "cell phone": "phone", "remote": "remote",
    "spoon": "spoon", "fork": "fork", "toothbrush": "toothbrush", "sports ball": "ball",
}


def read_rows(path):
    return [line for line in path.read_text().splitlines() if line.strip()]


def digest_pixels(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def link(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve() == source.resolve():
            return
        raise ValueError(f"Conflicting link {dest}")
    if dest.exists():
        raise ValueError(f"Refusing overwrite {dest}")
    dest.symlink_to(os.path.relpath(source.resolve(), dest.parent.resolve()))


def normalized_box(cid, bbox, width, height):
    x, y, w, h = bbox
    x1, y1 = max(0., x), max(0., y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        f"{cid} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} "
        f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"
    )


def iou(a, b):
    def corners(row):
        _, x, y, w, h = map(float, row.split())
        return x-w/2, y-h/2, x+w/2, y+h/2
    ax, ay, az, at = corners(a)
    bx, by, bz, bt = corners(b)
    intersection = max(0., min(az, bz)-max(ax, bx))*max(0., min(at, bt)-max(ay, by))
    return intersection / max(1e-12, (az-ax)*(at-ay)+(bz-bx)*(bt-by)-intersection)


def merge_rows(existing, extra, threshold=.9):
    rows = list(existing)
    for line in extra:
        if not any(line.split()[0] == old.split()[0] and iou(line, old) >= threshold for old in rows):
            rows.append(line)
    return rows


def ego_candidates(ids):
    manifest = ROOT / "runs/open_data_all15_20261014/egoobjects_remaining_frames.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    retained_frames = defaultdict(list)
    for image in (BASE / "images/train").iterdir():
        raw = image.resolve()
        if "/egoobjects/" in str(raw):
            group, video, frame = raw.stem.rsplit("_", 2)
            retained_frames[f"{group}:{video}"].append(int(frame))
    quotas = {"bottle": 1400, "cup": 1400, "phone": 1300, "remote": 1500,
              "spoon": 1500, "fork": 1000, "pen": 1800}
    selected = {}
    for name, quota in quotas.items():
        buckets = defaultdict(list)
        for row in records:
            if str(ids[name]) not in row["boxes"]:
                continue
            frame = int(Path(row["raw_name"]).stem.rsplit("_", 1)[1])
            old = retained_frames[row["video"]]
            distance = min((abs(frame-f) for f in old), default=100000)
            buckets[row["group"]].append((distance, row))
        groups = sorted(buckets)
        random.Random(20261015 + ids[name]).shuffle(groups)
        for group in groups:
            buckets[group].sort(key=lambda item: (-item[0], item[1]["raw_name"]))
        offsets = Counter()
        chosen = 0
        while chosen < quota:
            added = False
            for group in groups:
                offset = offsets[group]
                if offset < len(buckets[group]):
                    row = buckets[group][offset][1]
                    selected[row["image"]] = row
                    offsets[group] += 1
                    chosen += 1
                    added = True
                    if chosen == quota:
                        break
            if not added:
                break
    result = []
    for filename, row in sorted(selected.items()):
        result.append({
            "source": "EgoObjects replay", "split": "train", "group": row["group"],
            "video": row["video"], "image": str(EGO / "images/train" / filename),
            "labels": read_rows(EGO / "labels/train" / (Path(filename).stem + ".txt")),
            "output_name": "replay_" + filename, "license": "MIT (EgoObjects)",
        })
    print("Selected EgoObjects replay", len(result), flush=True)
    return result


def coco_candidates(ids):
    all_existing = set()
    for dataset in (ROOT / "data/processed").iterdir():
        if dataset == OUT:
            continue
        for split in ("train", "val", "test"):
            directory = dataset / "images" / split
            if not directory.exists():
                continue
            for image in directory.iterdir():
                match = re.search(r"lvis_(\d+)_", image.name)
                if match:
                    all_existing.add(int(match[1]))
    selected = []
    raw_selected = RUN / "coco_selection.json"
    if raw_selected.exists():
        return json.loads(raw_selected.read_text())
    for split in ("train", "val"):
        data = json.loads((RAW / f"annotations/instances_{split}2017.json").read_text())
        remap = {c["id"]: ids[COCO_MAPPING[c["name"]]] for c in data["categories"] if c["name"] in COCO_MAPPING}
        licenses = {item["id"]: item for item in data["licenses"]}
        images = {image["id"]: image for image in data["images"]}
        annotations = defaultdict(list)
        crowded = set()
        for annotation in data["annotations"]:
            if annotation["category_id"] in remap:
                if annotation.get("iscrowd"):
                    crowded.add(annotation["image_id"])
                else:
                    annotations[annotation["image_id"]].append(annotation)
        chosen = set()
        for name in TARGETS:
            if name == "pen":
                continue
            candidates = [
                image_id for image_id, rows in annotations.items()
                if image_id not in all_existing and image_id not in crowded
                and images[image_id]["license"] in {4, 5, 7, 8}
                and any(remap[a["category_id"]] == ids[name] for a in rows)
            ]
            random.Random(20261015 + ids[name]).shuffle(candidates)
            limit = (100000 if name == "toothbrush" else 600) if split == "train" else 120
            chosen.update(candidates[:limit])
            print("COCO candidates", split, name, len(candidates), "selected", min(limit, len(candidates)), flush=True)
        for image_id in sorted(chosen):
            image = images[image_id]
            image_name = f"coco_{split}2017_{image_id:012d}.jpg"
            final_split = "train" if split == "train" else (
                "val" if int(hashlib.sha256(str(image_id).encode()).hexdigest()[:8], 16) % 2 else "test"
            )
            rows = [normalized_box(remap[a["category_id"]], a["bbox"], image["width"], image["height"])
                    for a in annotations[image_id]]
            selected.append({
                "source": "COCO 2017", "split": final_split, "original_split": split,
                "group": f"coco_{image_id}", "image_id": image_id,
                "image": str(RAW / "selected_images" / image_name), "output_name": image_name,
                "labels": [row for row in rows if row], "width": image["width"], "height": image["height"],
                "url": image["coco_url"], "flickr_url": image.get("flickr_url"),
                "license": licenses[image["license"]],
            })
    print("Loading LVIS to preserve compatible companion labels", flush=True)
    _, lvis_annotations = _load_lvis(ROOT / "data/raw/lvis/lvis_v1_train.json")
    for row in selected:
        extra = [normalized_box(ids[LVIS_TO_PROJECT[a["category_name"]]], a["bbox"], row["width"], row["height"])
                 for a in lvis_annotations.get(row["image_id"], [])]
        row["labels"] = merge_rows(row["labels"], [line for line in extra if line])
    raw_selected.write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def download_selected(records):
    jobs = {}
    failed = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for row in records:
            jobs[pool.submit(_download, row["url"], Path(row["image"]), 3)] = row
        for n, future in enumerate(as_completed(jobs), 1):
            row = jobs[future]
            try:
                future.result()
            except OSError as exc:
                failed.append({"image_id": row["image_id"], "error": str(exc)})
            if n % 200 == 0:
                print("COCO download", n, "/", len(jobs), flush=True)
    (RUN / "download_failures.json").write_text(json.dumps(failed, indent=2))
    failed_ids = {item["image_id"] for item in failed}
    return [row for row in records if row["image_id"] not in failed_ids]


def main():
    cv2.setNumThreads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load((BASE / "data.yaml").read_text())["names"]
    ids = {name: int(cid) for cid, name in names.items()}
    new_records = ego_candidates(ids) + download_selected(coco_candidates(ids))
    frozen = []
    base_records = []
    for split in ("train", "val", "test"):
        for image in sorted((BASE / "images" / split).iterdir()):
            row = {
                "source": "Existing 15-class retention", "split": split,
                "image": str(image), "output_name": image.name,
                "labels": read_rows(BASE / "labels" / split / (image.stem + ".txt")),
            }
            (base_records if split == "train" else frozen).append(row)
    all_records = frozen + sorted(new_records, key=lambda row: {"test": 0, "val": 1, "train": 2}[row["split"]]) + base_records
    paths = sorted({Path(row["image"]).resolve() for row in all_records})
    cache_path = RUN / "pixel_hashes.json"
    hashes = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [path for path in paths if str(path) not in hashes]
    print("Hashing source images", len(todo), "cached", len(hashes), flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, (path, value) in enumerate(zip(todo, pool.map(digest_pixels, todo), strict=True), 1):
            hashes[str(path)] = value
            if n % 5000 == 0:
                print("Pixel hashes", n, "/", len(todo), flush=True)
    cache_path.write_text(json.dumps(hashes))
    exact = {}
    removed = []
    kept = []
    for row in all_records:
        sha = hashes[str(Path(row["image"]).resolve())]
        if sha in exact:
            old = exact[sha]
            if old in frozen and row in frozen:
                # Preserve historical benchmark exactly, even if it already contained a duplicate.
                kept.append({**row, "pixel_sha256": sha})
                continue
            if old["split"] == "train" and row["split"] == "train":
                old["labels"] = merge_rows(old["labels"], row["labels"])
            removed.append({"image": row["image"], "matches": old["image"], "split": row["split"]})
            continue
        row["pixel_sha256"] = sha
        exact[sha] = row
        kept.append(row)
    # Keep the exact existing validation/test files, and write independent COCO evaluations separately.
    for row in kept:
        split = row["split"]
        destination = OUT
        if row["source"] == "COCO 2017" and split != "train":
            destination = RUN / "external_coco"
        link(Path(row["image"]), destination / "images" / split / row["output_name"])
        label = destination / "labels" / split / (Path(row["output_name"]).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(row["labels"]) + ("\n" if row["labels"] else ""))
    for destination in (OUT, RUN / "external_coco"):
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                (destination / kind / split).mkdir(parents=True, exist_ok=True)
        (destination / "data.yaml").write_text(yaml.safe_dump({
            "path": str(destination), "train": "images/train", "val": "images/val", "test": "images/test", "names": names,
        }, sort_keys=False))
    with (OUT / "manifest.jsonl").open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    integrity = validate_yolo_dataset(OUT / "data.yaml")
    if not integrity["valid"] or integrity["missing_classes"]:
        raise ValueError(integrity)
    summary = {
        "targets": TARGETS, "seed": 20261015,
        "sources_and_splits": dict(Counter(f"{r['source']}/{r['split']}" for r in kept)),
        "deduplicated_images": len(removed), "removed": removed,
        "integrity": integrity,
        "labels_policy": "All compatible 15-class labels preserved; same-class overlapping COCO/LVIS annotations merged at IoU>=0.9.",
        "coco_image_license_ids_allowed": [4, 5, 7, 8],
        "ego_selection": "Class/group round-robin, preferring frames far from existing frames of the same video.",
        "frozen_evaluations": "Existing val/test image and label rows preserved; COCO holdouts separate.",
    }
    (OUT / "build_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "removed"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
