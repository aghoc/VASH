"""Combine corrected COCO/EgoObjects replay with new, attributed Open Images photos."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.validate import validate_yolo_dataset

ROOT = PROJECT_ROOT
SOURCE = ROOT / "data/processed/public_v5_focus8_clean_15"
OUT = ROOT / "data/processed/public_v6_focus8_coco_oi_15"
EXTERNAL = ROOT / "runs/focus8_20261015/external_openimages"
RAW = ROOT / "data/raw/focus8_openimages"
helpers = runpy.run_path(str(ROOT / "datasets/build_focus8_v5.py"), run_name="helpers")
link = helpers["link"]
digest_pixels = helpers["digest_pixels"]


def flickr_id(url):
    match = re.search(r"/(\d+)(?:_|/(?:in|sizes)/|/?$)", url or "")
    return match[1] if match else None


def main():
    records = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
    incoming = [json.loads(line) for line in (RAW / "manifest.jsonl").read_text().splitlines()]
    # Cross-corpus copies can have different JPEG encoding/resolution. Exclude shared Flickr
    # photograph IDs as well as exact decoded pixel matches against every existing split.
    coco_ids = set()
    for split in ("train", "val"):
        data = json.loads((ROOT / f"data/raw/coco2017/annotations/instances_{split}2017.json").read_text())
        coco_ids.update(fid for image in data["images"] if (fid := flickr_id(image.get("flickr_url"))))
    previous_authors = json.loads((ROOT / "data/raw/can_coin_open_v1/oi_image_attribution.json").read_text())
    old_md5 = {row.get("OriginalMD5") for row in previous_authors.values() if row.get("OriginalMD5")}
    exact = {row["pixel_sha256"]: row for row in records}
    accepted, removed = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(digest_pixels, [Path(row["image"]) for row in incoming]))
    # Holdouts take precedence when a new photo appears more than once.
    pairs = sorted(zip(incoming, hashes, strict=True), key=lambda pair: {"val": 0, "test": 1, "train": 2}[pair[0]["split"]])
    for row, sha in pairs:
        author = row["attribution"]
        fid = flickr_id(author.get("OriginalURL")) or flickr_id(author.get("OriginalLandingURL"))
        reason = None
        if fid and fid in coco_ids:
            reason = "Photograph also occurs in COCO (Flickr ID); excluded across corpora"
        elif author.get("OriginalMD5") and author["OriginalMD5"] in old_md5:
            reason = "Original image MD5 already in previous Open Images supplement"
        elif sha in exact:
            reason = f"Decoded pixels already present in {exact[sha]['source']}/{exact[sha]['split']}"
        if reason:
            removed.append({"image_id": row["image_id"], "split": row["split"], "reason": reason})
            continue
        row["pixel_sha256"] = sha
        exact[sha] = row
        accepted.append(row)
    combined = records + accepted
    names = yaml.safe_load((SOURCE / "data.yaml").read_text())["names"]
    for row in combined:
        split = row["split"]
        if row["source"] == "COCO 2017" and split != "train":
            continue  # Already preserved in external_coco_clean.
        destination = EXTERNAL if row["source"] == "Open Images focus8" and split != "train" else OUT
        link(Path(row["image"]), destination / "images" / split / row["output_name"])
        label = destination / "labels" / split / (Path(row["output_name"]).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(row["labels"]) + ("\n" if row["labels"] else ""))
    for destination in (OUT, EXTERNAL):
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                (destination / kind / split).mkdir(parents=True, exist_ok=True)
        (destination / "data.yaml").write_text(yaml.safe_dump({
            "path": str(destination), "train": "images/train", "val": "images/val", "test": "images/test", "names": names,
        }, sort_keys=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in combined))
    (EXTERNAL / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in accepted if row["split"] != "train"))
    # Verify frozen labels and identities against their original dataset, not just counts.
    frozen_digest = hashlib.sha256()
    original = ROOT / "data/processed/public_v4_can_coin_retention_15"
    for split in ("val", "test"):
        old = sorted(path.name for path in (original / "images" / split).iterdir())
        new = sorted(path.name for path in (OUT / "images" / split).iterdir())
        assert old == new
        for name in old:
            assert (original / "images" / split / name).resolve() == (OUT / "images" / split / name).resolve()
            labelname = Path(name).stem + ".txt"
            label = (OUT / "labels" / split / labelname).read_bytes()
            assert label == (original / "labels" / split / labelname).read_bytes()
            frozen_digest.update(f"{split}/{name}".encode() + label)
    hashes_by_split = defaultdict(set)
    for row in combined:
        hashes_by_split[row["split"]].add(row["pixel_sha256"])
    assert not hashes_by_split["train"] & (hashes_by_split["val"] | hashes_by_split["test"])
    integrity = validate_yolo_dataset(OUT / "data.yaml")
    assert integrity["valid"] and not any(integrity["missing_classes_by_split"].values())
    report = {
        "targets": helpers["TARGETS"], "parent_dataset": str(SOURCE), "seed": 20261015,
        "sources_and_splits": dict(Counter(f"{r['source']}/{r['split']}" for r in combined)),
        "openimages_class_positive_images": {split: dict(Counter(
            cid for row in accepted if row["split"] == split
            for cid in {int(line.split()[0]) for line in row["labels"]})) for split in ("train", "val", "test")},
        "openimages_excluded": removed, "openimages_accepted": len(accepted),
        "decoded_pixel_train_holdout_intersections": 0,
        "frozen_image_paths_and_labels_unchanged": True, "frozen_manifest_sha256": frozen_digest.hexdigest(),
        "labels_policy": "COCO authoritative for mapped COCO classes; LVIS companion labels. Open Images preserves all compatible mapped boxes, excluding group/depiction images.",
        "integrity": integrity,
        "external_evaluations": [str(ROOT / "runs/focus8_20261015/external_coco_clean/data.yaml"), str(EXTERNAL / "data.yaml")],
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "openimages_excluded"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
