"""Increase toothbrush sampling explicitly without counting repeats as new photographs."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.validate import validate_yolo_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/public_v5_focus8_15")
root = parser.parse_args().data_root.resolve()
records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
added = []
for record in records:
    if record["split"] != "train" or not any(line.split()[0] == "10" for line in record["labels"]):
        continue
    name = record["output_name"]
    for kind, filename in (("images", name), ("labels", name.rsplit(".", 1)[0] + ".txt")):
        source = root / kind / "train" / filename
        destination = source.with_name("tooth_weight2_" + filename)
        if not destination.is_symlink():
            destination.symlink_to(os.path.relpath(source, destination.parent))
    added.append({"source_name": name, "repeated_name": "tooth_weight2_" + name,
                  "pixel_sha256": record["pixel_sha256"], "weight": 2})

integrity = validate_yolo_dataset(root / "data.yaml")
if not integrity["valid"] or any(integrity["missing_classes_by_split"].values()):
    raise ValueError(integrity)
report = {
    "purpose": "Explicit sampling weight for requested toothbrush class; repeats are not new photographs.",
    "toothbrush_positive_weight": 2,
    "added_training_entries": len(added),
    "added_entries": added,
    "source_counts_before_repeats": dict(Counter(r["source"] for r in records if r["split"] == "train")),
    "integrity_after_weighting": integrity,
}
(root / "sampling_weights.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "added_entries"}, indent=2))
