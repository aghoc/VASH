#!/usr/bin/env python3
"""Interpolate aligned fine-tuned weights; accept only through separate validation."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch

from vagt_orca.deployment import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-classifier-class", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    from ultralytics import YOLO

    reference = YOLO(str(args.reference)).model.float().cpu()
    candidate = YOLO(str(args.candidate)).model.float().cpu()
    if reference.names != candidate.names:
        raise ValueError("Class names differ")
    old, new = reference.state_dict(), candidate.state_dict()
    if old.keys() != new.keys() or any(old[k].shape != new[k].shape for k in old):
        raise ValueError("Checkpoint architectures differ")
    merged = {
        key: ((1.0 - args.alpha) * old[key] + args.alpha * new[key])
        if old[key].is_floating_point()
        else new[key]
        for key in old
    }
    copied = []
    names = {value: key for key, value in candidate.names.items()}
    for name in args.copy_classifier_class:
        if name not in names:
            raise ValueError(f"Unknown classifier class {name}")
        class_id = names[name]
        keys = [
            key for key in merged if re.search(r"\.(?:one2one_)?cv3\.\d+\.2\.(?:weight|bias)$", key)
        ]
        if len(keys) != 12:
            raise ValueError(f"Unexpected YOLO26 classifier tensors: {keys}")
        for key in keys:
            merged[key][class_id] = new[key][class_id]
        copied.append({"class": name, "class_id": class_id, "tensors": keys})
    candidate.load_state_dict(merged, strict=True)
    # These are project-owned training artifacts, not arbitrary downloaded pickle files.
    checkpoint = torch.load(args.candidate, map_location="cpu", weights_only=False)
    checkpoint["model"] = copy.deepcopy(candidate).half()
    checkpoint["ema"] = None
    checkpoint["optimizer"] = None
    checkpoint["epoch"] = -1
    checkpoint["updates"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "method": "aligned parameter interpolation, validation required",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "candidate_fraction": args.alpha,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "classes": candidate.names,
        "copied_classifier_classes": copied,
    }
    args.output.with_suffix(".blend.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
