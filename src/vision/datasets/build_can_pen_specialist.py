"""Build a separate two-class public dataset without modifying prior splits."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.validate import validate_yolo_dataset

ROOT = PROJECT_ROOT
OUT = ROOT / "data/processed/can_pen_specialist_20261016"
SOURCE = ROOT / "data/processed/public_v6_focus8_coco_oi_15"
REMAP = {1: 0, 7: 1}


def order(row):
    return hashlib.sha256(("can_pen_20261016" + row["pixel_sha256"]).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "manifest.jsonl").exists(), "Dataset already exists; review before replacing"
    all_rows = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
    unique = {}
    for row in all_rows:
        sha = row["pixel_sha256"]
        image = Path(row["image"]).resolve()
        target = [f"{REMAP[int(line.split()[0])]} " + line.split(" ", 1)[1]
                  for line in row["labels"] if int(line.split()[0]) in REMAP]
        if sha in unique:
            assert unique[sha]["split"] == row["split"], "Pixel split leakage in parent"
            unique[sha]["labels"] = sorted(set(unique[sha]["labels"] + target))
            continue
        ego = "/egoobjects/images/" in str(image)
        group = image.stem.rsplit("_", 2)[0] if ego else image.stem
        unique[sha] = {
            "split": row["split"], "image": str(image), "pixel_sha256": sha,
            "group": group, "ego": ego, "labels": target,
            "parent_labels": row["labels"], "parent_source": row["source"],
            "parent_output_name": row["output_name"],
            "source_manifest": str(SOURCE / "manifest.jsonl"),
        }
    group_splits = defaultdict(set)
    for row in unique.values():
        if row["ego"]:
            group_splits[row["group"]].add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    selected = []
    for split in ("train", "val", "test"):
        pool = [row for row in unique.values() if row["split"] == split]
        positives = sorted((r for r in pool if r["labels"]), key=order)
        negatives = sorted((r for r in pool if not r["labels"]), key=order)
        if split == "train":
            group_counts = Counter()
            retained = []
            for row in positives:
                if row["ego"] and group_counts[row["group"]] >= 30:
                    continue
                retained.append(row)
                group_counts[row["group"]] += 1
            positives = retained
            # Lookalikes such as cutlery, toothbrushes, bottles and remotes are
            # retained as annotation-negative examples. Incomplete public labels
            # remain a limitation; these are not manually verified negatives.
            hard = [r for r in negatives if any(
                int(line.split()[0]) in {0, 2, 5, 6, 8, 9, 10} for line in r["parent_labels"])]
            hard_hashes = {r["pixel_sha256"] for r in hard}
            other = [r for r in negatives if r["pixel_sha256"] not in hard_hashes]
            negatives = hard[:2500] + other[:1000]
        elif split == "val":
            negatives = negatives[:1800]
        # Keep every test entry, including images with no target annotations.
        selected.extend(positives + negatives)
    for row in selected:
        stem = row["pixel_sha256"][:20]
        image = OUT / "images" / row["split"] / (stem + ".jpg")
        label = OUT / "labels" / row["split"] / (stem + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.symlink_to(os.path.relpath(row["image"], image.parent))
        label.write_text("\n".join(row["labels"]) + ("\n" if row["labels"] else ""))
        row["output_image"] = str(image)
    (OUT / "data.yaml").write_text(yaml.safe_dump({
        "path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "can", 1: "pen"},
    }, sort_keys=False))
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    integrity = validate_yolo_dataset(OUT / "data.yaml")
    assert integrity["valid"] and not any(integrity["missing_classes_by_split"].values())
    report = {
        "parent": str(SOURCE), "classes": {0: "can", 1: "pen"}, "new_camera_frames": 0,
        "split_policy": "Preserve every parent's train/val/test assignment; no new split",
        "parent_labels_unchanged": True, "pixel_train_holdout_intersection": 0,
        "ego_group_split_overlap": 0, "train_positive_ego_group_cap": 30,
        "counts": {split: {
            "images": sum(r["split"] == split for r in selected),
            "negatives": sum(r["split"] == split and not r["labels"] for r in selected),
            "positive_images": dict(Counter(
                cid for r in selected if r["split"] == split
                for cid in {int(line.split()[0]) for line in r["labels"]})),
            "sources": dict(Counter(r["parent_source"] for r in selected if r["split"] == split)),
        } for split in ("train", "val", "test")},
        "negative_scope": "No can/pen annotation, not manually verified object absence",
        "integrity": integrity,
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "integrity"}, indent=2))


if __name__ == "__main__":
    main()
