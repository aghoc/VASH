"""Use authoritative COCO labels for its categories and LVIS for companion categories."""

import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.validate import validate_yolo_dataset

ROOT = PROJECT_ROOT
SOURCE = ROOT / "data/processed/public_v5_focus8_15"
OUTPUT = ROOT / "data/processed/public_v5_focus8_clean_15"
RUN = ROOT / "runs/focus8_20261015"
EXTERNAL = RUN / "external_coco_clean"
helpers = runpy.run_path(str(ROOT / "datasets/build_focus8_v5.py"), run_name="helpers")
norm = helpers["normalized_box"]
link = helpers["link"]
names = yaml.safe_load((SOURCE / "data.yaml").read_text())["names"]
ids = {name: cid for cid, name in names.items()}
records = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
coco_records = {row["image_id"]: row for row in records if row["source"] == "COCO 2017"}
mapping_names = helpers["COCO_MAPPING"]
coco_classes = {ids[name] for name in mapping_names.values()}
original = defaultdict(list)
for split in ("train", "val"):
    raw = json.loads((ROOT / f"data/raw/coco2017/annotations/instances_{split}2017.json").read_text())
    mapping = {row["id"]: ids[mapping_names[row["name"]]]
               for row in raw["categories"] if row["name"] in mapping_names}
    for annotation in raw["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in coco_records or annotation["category_id"] not in mapping or annotation.get("iscrowd"):
            continue
        row = coco_records[image_id]
        line = norm(mapping[annotation["category_id"]], annotation["bbox"], row["width"], row["height"])
        if line:
            original[image_id].append(line)
changed = []
for row in records:
    if row["source"] == "COCO 2017":
        old = row["labels"]
        row["labels"] = original[row["image_id"]] + [line for line in old if int(line.split()[0]) not in coco_classes]
        row["annotation_policy"] = "COCO authoritative for its 8 mapped categories; LVIS only for other project categories"
        if old != row["labels"]:
            changed.append({"image_id": row["image_id"], "split": row["split"],
                            "before_boxes": len(old), "after_boxes": len(row["labels"])})
    destination = EXTERNAL if row["source"] == "COCO 2017" and row["split"] != "train" else OUTPUT
    name = row["output_name"]
    link(Path(row["image"]), destination / "images" / row["split"] / name)
    label = destination / "labels" / row["split"] / (Path(name).stem + ".txt")
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("\n".join(row["labels"]) + ("\n" if row["labels"] else ""))

for destination in (OUTPUT, EXTERNAL):
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (destination / kind / split).mkdir(parents=True, exist_ok=True)
    (destination / "data.yaml").write_text(yaml.safe_dump({
        "path": str(destination), "train": "images/train", "val": "images/val", "test": "images/test", "names": names,
    }, sort_keys=False))
(OUTPUT / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
integrity = validate_yolo_dataset(OUTPUT / "data.yaml")
if not integrity["valid"] or any(integrity["missing_classes_by_split"].values()):
    raise ValueError(integrity)
summary = {
    "targets": helpers["TARGETS"], "parent_dataset": str(SOURCE), "seed": 20261015,
    "sources_and_splits": dict(Counter(f"{row['source']}/{row['split']}" for row in records)),
    "labels_policy": "COCO authoritative for bottle/cup/phone/remote/spoon/fork/toothbrush/ball; preserve LVIS companion categories and all original COCO overlapping instances.",
    "changed_images": changed,
    "coco_extra_boxes_removed": sum(row["before_boxes"]-row["after_boxes"] for row in changed),
    "integrity": integrity, "image_content_and_split_changes": 0,
    "frozen_evaluations": "Existing val/test files and labels unchanged; new COCO holdouts use the same corrected label policy.",
}
(OUTPUT / "build_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "changed_images"}, indent=2))
