#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.deployment import sha256
from vagt_orca.precollection import build_plan, plan_checks


def artifact_checks() -> dict[str, bool]:
    model = PROJECT_ROOT / "artifacts/models/yolo26s_best.pt"
    onnx = PROJECT_ROOT / "artifacts/deployment/onnx/yolo26s_best.onnx"
    onnx_report = PROJECT_ROOT / "reports/yolo26s_phase6_onnx.json"
    if not all(path.is_file() for path in (model, onnx, onnx_report)):
        return {
            "current_checkpoint_present": model.is_file(),
            "onnx_matches_checkpoint": False,
            "deployment_precision_is_fp16": False,
        }
    onnx_data = json.loads(onnx_report.read_text(encoding="utf-8"))
    return {
        "current_checkpoint_present": True,
        "onnx_matches_checkpoint": onnx_data.get("model_sha256") == sha256(model)
        and onnx_data.get("onnx_sha256") == sha256(onnx),
        "deployment_precision_is_fp16": onnx_data.get("model_family") == "YOLO26s",
    }


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, but do not start, the mounted AR-glass collection"
    )
    parser.add_argument("--source", default="pixy")
    parser.add_argument(
        "--output", type=Path, default=Path("configs/ar_glass_collection_plan.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/phase8_precollection.json")
    )
    args = parser.parse_args()

    config = load_yaml(PROJECT_ROOT / "configs/domain_capture.yaml")
    rows = build_plan(config, args.source)
    write_csv(args.output, rows)
    checks = {
        **plan_checks(config, rows),
        **artifact_checks(),
        "recorder_present": (PROJECT_ROOT / "tools/record_ar_glass.py").is_file(),
        "pixy_bridge_build_script_present": (
            PROJECT_ROOT / "scripts/build_pixy_bridge.sh"
        ).is_file(),
    }
    bridge = PROJECT_ROOT / "artifacts/deployment/pixy/libpixybridge.so"
    model = PROJECT_ROOT / "artifacts/models/yolo26s_best.pt"
    target_report_path = PROJECT_ROOT / "reports/yolo26s_phase6_tensorrt.json"
    target_report = (
        json.loads(target_report_path.read_text(encoding="utf-8"))
        if target_report_path.is_file()
        else {}
    )
    current_target_engine = bool(target_report.get("accepted_target_build")) and (
        target_report.get("model_sha256") == sha256(model)
    )
    manifest = PROJECT_ROOT / "data/domain/ar_glass/manifest.jsonl"
    existing_records = (
        len(manifest.read_text(encoding="utf-8").splitlines()) if manifest.is_file() else 0
    )
    report = {
        "status": "hardware_connection_pending" if all(checks.values()) else "not_ready",
        "non_hardware_preparation_complete": all(checks.values()),
        "actual_collection_started": existing_records > 0,
        "source": args.source,
        "plan": str(args.output.resolve()),
        "planned_sessions": len(rows),
        "planned_images": sum(int(row["max_images"]) for row in rows),
        "classes": len(config["required_classes"]),
        "model_sha256": sha256(model),
        "checks": checks,
        "hardware_checks_deferred": {
            "current_jetson_engine_built": current_target_engine,
            "pixy_connected_on_collection_device": False,
            "pixy_bridge_built_on_collection_device": bridge.is_file(),
            "mounted_view_and_focus_confirmed": False,
        },
        "next_command_after_hardware_connection": rows[0]["command"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
