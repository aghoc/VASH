"""Select new eight-class Open Images photographs with official attribution and splits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import runpy
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.lvis import _download

ROOT = PROJECT_ROOT
RAW = ROOT / "data/raw/focus8_openimages"
RUN = ROOT / "runs/focus8_20261015"
OLD_RAW = ROOT / "data/raw/can_coin_open_v1"
helpers = runpy.run_path(str(ROOT / "datasets/prepare_can_coin_open.py"), run_name="helpers")
MAPPING = helpers["OI_MAPPING"]
box_line = helpers["box_line"]
QUOTAS = {0: 500, 2: 500, 5: 600, 6: 100000, 7: 900, 8: 500, 9: 500, 10: 100000}
META = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "val": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}


def select():
    cache = RAW / "selection.json"
    if cache.exists():
        return json.loads(cache.read_text())
    mids = {mid: MAPPING[name] for mid, name in csv.reader(
        (ROOT / "runs/open_data_can_coin_20261014/openimages_boxable.txt").open()) if name in MAPPING}
    existing = set()
    for dataset in (ROOT / "data/processed").iterdir():
        for split in ("train", "val", "test"):
            directory = dataset / "images" / split
            if directory.exists():
                for image in directory.iterdir():
                    stem = image.stem
                    if "oi_" in stem:
                        existing.add(stem.split("oi_", 1)[1])
    result = []
    for split in ("train", "val", "test"):
        annotations = defaultdict(list)
        excluded = set()
        with (OLD_RAW / f"oid_{split}.csv").open() as handle:
            header = {name: i for i, name in enumerate(next(handle).strip().split(","))}
            for line in handle:
                values = line.rstrip().split(",")
                mid = values[header["LabelName"]]
                if mid not in mids:
                    continue
                image_id = values[header["ImageID"]]
                if image_id in existing:
                    continue
                if values[header["IsGroupOf"]] == "1" or values[header["IsDepiction"]] == "1":
                    excluded.add(image_id)
                    continue
                xyxy = [float(values[header[key]]) for key in ("XMin", "YMin", "XMax", "YMax")]
                annotations[image_id].append(box_line(mids[mid], xyxy))
        chosen = set()
        for cid, limit in QUOTAS.items():
            candidates = sorted(image_id for image_id, rows in annotations.items()
                                if image_id not in excluded and any(int(row.split()[0]) == cid for row in rows))
            random.Random(20261015 + cid).shuffle(candidates)
            cap = limit if split == "train" else 50
            chosen.update(candidates[:cap])
            print("Open Images eligible", split, cid, len(candidates), "selected", min(cap, len(candidates)), flush=True)
        for image_id in sorted(chosen):
            original_split = "validation" if split == "val" else split
            name = f"oi_focus8_{image_id}.jpg"
            result.append({
                "source": "Open Images focus8", "split": split, "original_split": original_split,
                "image_id": image_id, "group": f"oi_{image_id}", "output_name": name,
                "image": str(RAW / "images" / name), "labels": sorted(set(annotations[image_id])),
                "url": f"https://open-images-dataset.s3.amazonaws.com/{original_split}/{image_id}.jpg",
                "origin": "https://storage.googleapis.com/openimages/web/download_v7.html",
                "annotation_license": "CC BY 4.0",
            })
    cache.write_text(json.dumps(result, indent=2))
    return result


def attribution(selected):
    output = RAW / "image_attribution.json"
    if output.exists():
        return json.loads(output.read_text())
    result = {}
    for split, url in META.items():
        wanted = {row["image_id"] for row in selected if row["split"] == split}
        # Stream the official metadata instead of retaining its large, mostly unrelated CSV.
        with urllib.request.urlopen(url, timeout=180) as response:
            for row in csv.DictReader(io.TextIOWrapper(response, encoding="utf-8")):
                if row["ImageID"] in wanted:
                    result[row["ImageID"]] = row
        missing = wanted - result.keys()
        if missing:
            raise ValueError(f"Missing official attribution for {split}: {sorted(missing)}")
        print("Official image attribution", split, len(wanted), flush=True)
    output.write_text(json.dumps(result, indent=2))
    (RAW / "metadata_sources.json").write_text(json.dumps(META, indent=2))
    return result


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    selected = select()
    authors = attribution(selected)
    allowed, rejected = [], []
    for row in selected:
        author = authors[row["image_id"]]
        license_url = author["License"].replace("https://", "http://").rstrip("/")
        if license_url not in {"http://creativecommons.org/licenses/by/2.0", "http://creativecommons.org/licenses/by/4.0"}:
            rejected.append({"image_id": row["image_id"], "license": author["License"]})
            continue
        row["attribution"] = author
        row["license"] = author["License"]
        allowed.append(row)
    failures = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        jobs = {pool.submit(_download, row["url"], Path(row["image"])): row for row in allowed}
        kept = []
        for n, future in enumerate(as_completed(jobs), 1):
            row = jobs[future]
            try:
                future.result()
                row["file_sha256"] = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
                kept.append(row)
            except (OSError, ValueError) as error:
                failures.append({"image_id": row["image_id"], "error": str(error)})
            if n % 250 == 0:
                print("Downloaded Open Images", n, "/", len(jobs), flush=True)
    kept.sort(key=lambda row: (row["split"], row["image_id"]))
    (RAW / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))
    report = {"splits": dict(Counter(row["split"] for row in kept)), "download_failures": failures,
              "license_rejected": rejected, "selection_seed": 20261015,
              "class_positive_images": {split: dict(Counter(
                  cid for row in kept if row["split"] == split
                  for cid in {int(line.split()[0]) for line in row["labels"]}))
                  for split in ("train", "val", "test")},
              "annotation_policy": "Preserve every compatible 15-class box; exclude images with mapped group/depiction annotations.",
              "existing_image_ids_excluded": True}
    (RAW / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
