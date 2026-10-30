#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagt_orca.deployment import platform_summary, sha256


def read_optional(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 6 acceptance evidence")
    parser.add_argument("--report", type=Path, default=Path("reports/yolo26s_phase6_status.json"))
    args = parser.parse_args()
    onnx = read_optional(Path("reports/yolo26s_phase6_onnx.json"))
    target_engine = read_optional(Path("reports/yolo26s_phase6_tensorrt.json"))
    target_benchmark = read_optional(Path("reports/yolo26s_phase6_benchmark.json"))
    sustained_benchmark = read_optional(
        Path("reports/yolo26s_phase6_benchmark_sustained.json")
    )
    target_validation = read_optional(
        Path("artifacts/deployment/validation/jetson/val/metrics_summary.json")
    )
    demo = read_optional(Path("reports/yolo26s_phase6_demo_result.json"))
    bundle = read_optional(Path("reports/yolo26s_phase6_jetson_bundle.json"))
    model_path = Path("artifacts/models/yolo26s_best.pt")
    model_sha = sha256(model_path) if model_path.is_file() else None
    current_platform = platform_summary()
    checks = {
        "onnx_export_and_cuda_smoke": bool(
            onnx
            and onnx.get("cuda_smoke_test", {}).get("passed")
            and onnx.get("model_sha256") == model_sha
        ),
        "long_sequence_demo_verified": bool(
            demo and demo.get("accepted") and demo.get("model_sha256") == model_sha
        ),
        "jetson_transfer_bundle_ready": bool(
            (
                bundle
                and onnx
                and not bundle.get("contains_engine", True)
                and bundle.get("model_sha256") == model_sha
                and bundle.get("onnx_sha256") == onnx.get("onnx_sha256")
            )
            or (current_platform["is_jetson"] and onnx and demo)
        ),
        "jetson_fp16_engine_built": bool(
            target_engine
            and target_engine.get("accepted_target_build")
            and target_engine.get("precision") == "fp16"
            and target_engine.get("model_sha256") == model_sha
        ),
        "jetson_fp16_accuracy_recorded": bool(
            target_validation
            and target_engine
            and target_validation.get("precision") == "fp16"
            and target_validation.get("model_sha256") == target_engine.get("engine_sha256")
        ),
        "jetson_end_to_end_benchmark_recorded": bool(
            target_benchmark
            and target_engine
            and target_benchmark.get("deployment_accepted")
            and target_benchmark.get("model_sha256") == target_engine.get("engine_sha256")
            and target_engine.get("model_sha256") == model_sha
        ),
    }
    complete = all(checks.values())
    report = {
        "phase": 6,
        "status": "complete" if complete else "target_hardware_pending",
        "accepted": complete,
        "checks": checks,
        "current_platform": current_platform,
        "model_family": "YOLO26s",
        "deployment_precision": "fp16",
        "target_evidence": {
            "engine": target_engine,
            "validation": target_validation,
            "benchmark": target_benchmark,
            "sustained_benchmark": sustained_benchmark,
        },
        "blocker": (
            None
            if complete
            else (
                "Jetson Orin Nano is not connected; build the FP16 engine, validate the frozen "
                "15-class set, and benchmark on target."
            )
        ),
        "resume": (
            None
            if complete
            else "Follow deployment/JETSON_RUNBOOK.md on the Jetson, then rerun this script."
        ),
        "desktop_development_reports": [
            "reports/yolo26s_phase6_tensorrt_desktop.json",
            "reports/yolo26s_phase6_benchmark_onnx_desktop.json",
            "reports/yolo26s_phase6_benchmark_tensorrt_desktop.json",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
