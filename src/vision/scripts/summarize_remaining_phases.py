#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT, load_yaml


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    domain_root = PROJECT_ROOT / "data/domain/ar_glass"
    manifest = domain_root / "manifest.jsonl"
    records = []
    if manifest.is_file():
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    domain_config = load_yaml(PROJECT_ROOT / "configs/domain_capture.yaml")
    required_classes = set(domain_config["required_classes"])
    captured_classes = {str(record["class_name"]) for record in records}
    phase8_checks = {
        "ar_glass_device_or_recording_available": bool(records),
        "all_15_classes_captured": required_classes <= captured_classes,
        "bbox_annotations_present": any((domain_root / "labels").rglob("*.txt"))
        if (domain_root / "labels").exists()
        else False,
        "domain_finetune_completed": (PROJECT_ROOT / "reports/phase8_training.json").is_file(),
    }
    phase8_complete = all(phase8_checks.values())
    write_report(
        PROJECT_ROOT / "reports/phase8_status.json",
        {
            "phase": 8,
            "status": "complete" if phase8_complete else "hardware_pending",
            "accepted": phase8_complete,
            "records": len(records),
            "captured_classes": sorted(captured_classes),
            "checks": phase8_checks,
            "blocker": None if phase8_complete else "No AR Glass camera or recorded domain data is available.",
        },
    )

    observations = PROJECT_ROOT / "calibration/phase9/opening_observations.csv"
    with observations.open(encoding="utf-8", newline="") as stream:
        observation_rows = list(csv.DictReader(stream))
    pose_checks = PROJECT_ROOT / "calibration/phase9/pose_checks.csv"
    with pose_checks.open(encoding="utf-8", newline="") as stream:
        pose_rows = list(csv.DictReader(stream))
    pose_complete = bool(pose_rows) and all(
        row["pose_passed"].strip().lower() in {"true", "1", "yes"} for row in pose_rows
    )
    lateral_rows = [row for row in pose_rows if row["grip"] == "LATERAL_PINCH"]
    lateral_complete = bool(lateral_rows) and all(
        row["lateral_contact_passed"].strip().lower() in {"true", "1", "yes"}
        for row in lateral_rows
    )
    opening_report = PROJECT_ROOT / "reports/phase9_opening_calibration.json"
    opening_accepted = False
    if opening_report.is_file():
        opening_accepted = bool(json.loads(opening_report.read_text(encoding="utf-8"))["accepted"])
    phase9_checks = {
        "physical_opening_observations_present": bool(observation_rows),
        "opening_mae_at_most_0_10": opening_accepted,
        "four_grip_poses_verified": pose_complete,
        "lateral_pinch_contact_verified": lateral_complete,
    }
    phase9_complete = all(phase9_checks.values())
    write_report(
        PROJECT_ROOT / "reports/phase9_status.json",
        {
            "phase": 9,
            "status": "complete" if phase9_complete else "hardware_pending",
            "accepted": phase9_complete,
            "opening_observations": len(observation_rows),
            "checks": phase9_checks,
            "blocker": None if phase9_complete else "ORCA Hand physical calibration data is not available.",
        },
    )


if __name__ == "__main__":
    main()
