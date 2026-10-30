"""Calibrate a can/pen specialist on validation and report held-out operating points."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import yaml

from vagt_orca.deployment import sha256
from vagt_orca.gpu import gpu_summary, require_cuda

NAMES = ("can", "pen")


def overlap(box, targets):
    if not len(targets):
        return np.zeros(0)
    intersection = np.maximum(0, np.minimum(box[2:4], targets[:, 2:4])
                              - np.maximum(box[:2], targets[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:4] - box[:2]).prod()
    target_area = np.maximum(0, targets[:, 2:4] - targets[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + target_area - intersection, 1e-9)


def collect(model_path, data_path, split, output, batch):
    from ultralytics import YOLO

    fingerprint = sha256(model_path)
    data = yaml.safe_load(data_path.read_text())
    root = Path(data["path"])
    paths = sorted((root / data[split]).glob("*.jpg"))
    model = YOLO(str(model_path))
    model_names = model.names
    remap = {cid: NAMES.index(name) for cid, name in model_names.items() if name in NAMES}
    assert set(remap.values()) == {0, 1}
    batch = 1 if model_path.suffix == ".engine" else batch
    records = []
    for start in range(0, len(paths), batch):
        subset = paths[start:start + batch]
        results = model.predict(
            source=[str(p) for p in subset], device=0, imgsz=640, rect=False,
            end2end=True, conf=0.001, quantize=16 if model_path.suffix == ".engine" else 32,
            verbose=False, batch=batch,
        )
        for path, result in zip(subset, results, strict=True):
            height, width = result.orig_shape
            targets = []
            for line in (root / "labels" / split / (path.stem + ".txt")).read_text().splitlines():
                cid, x, y, w, h = map(float, line.split())
                targets.append([(x-w/2)*width, (y-h/2)*height,
                                (x+w/2)*width, (y+h/2)*height, int(cid)])
            predictions = []
            for box in result.boxes.data.cpu().tolist():
                if int(box[5]) in remap:
                    predictions.append([*box[:5], remap[int(box[5])]])
            records.append({"image": path.name, "targets": targets, "predictions": predictions})
        if start % (batch * 40) == 0:
            print(f"{split}: {min(start+batch,len(paths))}/{len(paths)}", flush=True)
    assert sha256(model_path) == fingerprint
    payload = {"model": str(model_path.resolve()), "model_sha256": fingerprint,
               "data": str(data_path.resolve()), "split": split, "records": records}
    with gzip.open(output, "wt") as handle:
        json.dump(payload, handle)
    return payload


def measure(records, cid, threshold):
    tp = fp = fn = negatives = negative_fp = 0
    for row in records:
        targets = np.array([b[:4] for b in row["targets"] if int(b[4]) == cid]).reshape(-1, 4)
        predicted = sorted((b for b in row["predictions"] if int(b[5]) == cid and b[4] >= threshold),
                           key=lambda b: -b[4])
        negatives += int(not len(targets))
        negative_fp += int(not len(targets) and bool(predicted))
        matched = set()
        for box in predicted:
            iou = overlap(np.asarray(box[:4]), targets)
            if matched:
                iou[list(matched)] = -1
            index = int(iou.argmax()) if len(iou) else -1
            if index >= 0 and iou[index] >= 0.5:
                matched.add(index)
                tp += 1
            else:
                fp += 1
        fn += len(targets) - len(matched)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "images": len(records), "negative_images": negatives,
            "negative_fp_images": negative_fp,
            "negative_image_fp_rate": negative_fp / max(1, negatives)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    require_cuda(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.with_suffix(".predictions.json.gz")
    if cache.exists():
        with gzip.open(cache, "rt") as handle:
            payload = json.load(handle)
        assert payload["model_sha256"] == sha256(args.model)
        assert payload["split"] == args.split and payload["data"] == str(args.data.resolve())
    else:
        payload = collect(args.model, args.data, args.split, cache, args.batch)
    classes = {}
    selected = json.loads(args.calibration.read_text())["classes"] if args.calibration else None
    for cid, name in enumerate(NAMES):
        fixed = measure(payload["records"], cid, 0.35)
        if args.split == "val":
            curve = [measure(payload["records"], cid, round(float(threshold), 3))
                     for threshold in [*np.arange(0.1, 1.0, 0.05), 0.975, 0.99]]
            feasible = [r for r in curve if r["precision"] >= 0.85
                        and r["negative_image_fp_rate"] <= 0.01 and r["tp"] >= 10]
            chosen = max(feasible, key=lambda r: r["f1"]) if feasible else None
            classes[name] = {"fixed_035": fixed, "selected": chosen, "curve": curve,
                             "gate": {"minimum_precision": 0.85, "maximum_negative_fp_rate": 0.01,
                                      "minimum_true_positives": 10}}
        else:
            chosen = selected[name]["selected"] if selected else None
            operating = measure(payload["records"], cid, chosen["threshold"]) if chosen else None
            classes[name] = {"fixed_035": fixed, "validation_selected": operating}
    report = {"model": payload["model"], "model_sha256": payload["model_sha256"],
              "data": payload["data"], "split": args.split, "gpu": gpu_summary(0),
              "match_iou": 0.5, "classes": classes, "new_camera_frames": 0,
              "scope": "Public annotation-based measurements; missing annotations may count as FP",
              "threshold_selection": "Validation only; held-out test never sets thresholds"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "classes": {
        k: {key: value for key, value in v.items() if key != "curve"}
        for k, v in classes.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
