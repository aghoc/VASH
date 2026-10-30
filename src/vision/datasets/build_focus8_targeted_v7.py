"""Balance eight target classes and add contextual crops from training images only."""

from __future__ import annotations

import hashlib
import json
import random
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.validate import validate_yolo_dataset

ROOT = PROJECT_ROOT
SOURCE = ROOT / "data/processed/public_v6_focus8_coco_oi_15"
OUT = ROOT / "data/processed/public_v7_focus8_targeted_15"
CROPS = ROOT / "data/raw/focus8_targeted_crops"
helpers = runpy.run_path(str(ROOT / "datasets/build_focus8_v5.py"), run_name="helpers")
link = helpers["link"]
QUOTAS = {0: 1200, 2: 1200, 5: 1200, 6: 1400, 8: 1400, 9: 1400, 10: 1800, 7: 1800}
REPLAY = {1: 800, 13: 1000, 3: 250, 4: 250, 11: 100000, 12: 250, 14: 100000}


def classes(row):
    return {int(line.split()[0]) for line in row["labels"]}


def group(row):
    if "group" in row:
        return row["group"]
    raw = Path(row["image"]).resolve()
    return raw.stem.rsplit("_", 2)[0] if "/egoobjects/" in str(raw) else raw.stem


def choose(rows, cid, count, chosen, public_first):
    buckets = defaultdict(list)
    for row in rows:
        if cid not in row["class_ids"]:
            continue
        public = row["source"] in {"COCO 2017", "Open Images focus8"}
        buckets[(0 if public and public_first else 1, group(row))].append(row)
    keys = sorted(buckets)
    random.Random(20261015+cid).shuffle(keys)
    if public_first:
        keys.sort(key=lambda key: key[0])
    for key, bucket in buckets.items():
        random.Random(str(key)+str(cid)).shuffle(bucket)
    observed = sum(cid in row["class_ids"] for row in chosen.values())
    offset = 0
    while observed < count:
        added = False
        for key in keys:
            if offset >= len(buckets[key]):
                continue
            row = buckets[key][offset]
            if row["pixel_sha256"] not in chosen:
                chosen[row["pixel_sha256"]] = row
                observed += 1
                added = True
            if observed >= count:
                break
        if not added and all(offset >= len(bucket)-1 for bucket in buckets.values()):
            break
        offset += 1


def crop_examples(rows, holdout_hashes):
    result = []
    seen = set()
    for cid in QUOTAS:
        limit = 150 if cid in {7, 10} else 80
        eligible = [row for row in rows if cid in row["class_ids"]]
        random.Random(20261015+cid).shuffle(eligible)
        count = 0
        for row in eligible:
            image = cv2.imread(str(Path(row["image"])))
            if image is None:
                raise ValueError(row["image"])
            height, width = image.shape[:2]
            boxes = [list(map(float, line.split())) for line in row["labels"]]
            target = next((box for box in boxes if int(box[0]) == cid
                           and (box[3]*width*box[4]*height)**.5*640/max(width, height) < 48), None)
            if target is None:
                continue
            _, x, y, w, h = target
            side = min(min(width, height), max(max(w*width, h*height)*2.5, min(width, height)*.35))
            if side >= min(width, height)*.95:
                continue
            x1 = max(0, min(width-int(side), int(x*width-side/2)))
            y1 = max(0, min(height-int(side), int(y*height-side/2)))
            x2, y2 = x1+int(side), y1+int(side)
            labels = []
            reject = False
            target_visible = False
            for box in boxes:
                bc, bx, by, bw, bh = box
                ax, ay, az, at = (bx-bw/2)*width, (by-bh/2)*height, (bx+bw/2)*width, (by+bh/2)*height
                cx, cy, cz, ct = max(ax, x1), max(ay, y1), min(az, x2), min(at, y2)
                area = max(0, cz-cx)*max(0, ct-cy)
                if area == 0:
                    continue
                # Reject fragments of annotated neighboring objects rather than turning them into background.
                if area/max(1e-9, (az-ax)*(at-ay)) < .7 or min(cz-cx, ct-cy) < 2:
                    reject = True
                    break
                size = int(side)
                labels.append(f"{int(bc)} {((cx+cz)/2-x1)/size:.8f} {((cy+ct)/2-y1)/size:.8f} {(cz-cx)/size:.8f} {(ct-cy)/size:.8f}")
                target_visible |= int(bc) == cid
            if reject or not target_visible:
                continue
            cropped = image[y1:y2, x1:x2]
            sha = hashlib.sha256(str(cropped.shape).encode()+cropped.tobytes()).hexdigest()
            if sha in seen or sha in holdout_hashes:
                continue
            name = f"targetcrop_{cid}_{Path(row['output_name']).stem}.png"
            path = CROPS / name
            CROPS.mkdir(parents=True, exist_ok=True)
            if not path.exists() and not cv2.imwrite(str(path), cropped):
                raise OSError(path)
            seen.add(sha)
            result.append({"source": "Targeted contextual crop", "split": "train", "image": str(path),
                           "output_name": name, "labels": labels, "pixel_sha256": sha,
                           "group": group(row), "parent_image": row["image"], "parent_source": row["source"],
                           "parent_pixel_sha256": row["pixel_sha256"], "crop_xyxy": [x1, y1, x2, y2],
                           "license": row.get("license", "See parent manifest"), "focus_class": cid})
            count += 1
            if count >= limit:
                break
        print("Contextual crops", cid, count, flush=True)
    return result


def main():
    rows = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
    train = [row for row in rows if row["split"] == "train"]
    frozen = [row for row in rows if row["source"] == "Existing 15-class retention" and row["split"] != "train"]
    holdout_hashes = {row["pixel_sha256"] for row in rows if row["split"] != "train"}
    for row in train:
        row["class_ids"] = classes(row)
    chosen = {}
    for cid, quota in QUOTAS.items():
        choose(train, cid, quota, chosen, public_first=True)
    for cid, quota in REPLAY.items():
        choose(train, cid, quota, chosen, public_first=False)
    crops = crop_examples(list(chosen.values()), holdout_hashes)
    final = list(chosen.values())+crops+frozen
    assert not {row["pixel_sha256"] for row in final if row["split"] == "train"} & holdout_hashes
    for row in final:
        row.pop("class_ids", None)
        link(Path(row["image"]), OUT / "images" / row["split"] / row["output_name"])
        label = OUT / "labels" / row["split"] / (Path(row["output_name"]).stem+".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(row["labels"])+("\n" if row["labels"] else ""))
    names = yaml.safe_load((SOURCE / "data.yaml").read_text())["names"]
    (OUT / "data.yaml").write_text(yaml.safe_dump({"path": str(OUT), "train": "images/train",
                                                "val": "images/val", "test": "images/test", "names": names}, sort_keys=False))
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in final))
    integrity = validate_yolo_dataset(OUT / "data.yaml")
    assert integrity["valid"] and not any(integrity["missing_classes_by_split"].values())
    report = {"parent_dataset": str(SOURCE), "focus_class_quotas": QUOTAS, "retention_quotas": REPLAY,
              "source_counts": dict(Counter(row["source"] for row in final if row["split"] == "train")),
              "positive_train_images": dict(Counter(cid for row in final if row["split"] == "train" for cid in classes(row))),
              "parent_train_photographs": len(chosen), "derived_crops": len(crops),
              "train_holdout_pixel_intersections": 0, "all_compatible_class_labels_preserved": True,
              "selection": "Class/group round robin; COCO/Open Images groups first within each round. Keep all available card/block photos and can/coin replay.",
              "integrity": integrity}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
