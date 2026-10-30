#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from vagt_orca.deployment import platform_summary, require_jetson
from vagt_orca.gpu import gpu_summary, require_cuda


def run(command: list[str], completed: list[dict]) -> None:
    started = time.monotonic()
    result = subprocess.run(command, check=False)
    completed.append(
        {
            "command": command,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and record all target-side Phase 6 gates")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase6_jetson_run.json")
    )
    args = parser.parse_args()

    require_jetson()
    require_cuda(args.device)
    import tensorrt as trt

    python = sys.executable
    commands = [
        [python, "deployment/export_tensorrt.py"],
    ]
    validation_data = Path("data/processed/egoobjects_15/data.yaml")
    if not args.skip_validation:
        if not validation_data.is_file():
            raise FileNotFoundError(
                f"Validation dataset missing: {validation_data}. Copy it or pass --skip-validation."
            )
        commands.append(
            [
                python,
                "training/validate.py",
                "--model",
                "artifacts/deployment/tensorrt/yolo26s_best_fp16.engine",
                "--data",
                str(validation_data),
                "--split",
                "val",
                "--batch",
                "4",
                "--project",
                "artifacts/deployment/validation/jetson",
            ]
        )
    commands.extend(
        [
            [
                python,
                "deployment/benchmark.py",
                "--model",
                "artifacts/deployment/tensorrt/yolo26s_best_fp16.engine",
                "--report",
                "reports/yolo26s_phase6_benchmark.json",
            ],
            [python, "deployment/summarize_phase6.py"],
        ]
    )
    completed: list[dict] = []
    status = "complete"
    error = None
    try:
        for command in commands:
            run(command, completed)
    except Exception as caught:
        status = "failed"
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        report = {
            "status": status,
            "error": error,
            "platform": platform_summary(),
            "gpu": gpu_summary(args.device),
            "tensorrt_version": trt.__version__,
            "model_family": "YOLO26s",
            "deployment_precision": "fp16",
            "validation_skipped": args.skip_validation,
            "commands": completed,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
