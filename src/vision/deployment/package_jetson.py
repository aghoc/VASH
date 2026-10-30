#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.deployment import sha256

INCLUDE_PATHS = (
    Path("data/benchmark/yolo26s_phase6"),
    Path("calibration"),
    Path("configs"),
    Path("deployment"),
    Path("docs"),
    Path("reports/yolo26s_phase6_onnx.json"),
    Path("reports/yolo26s_phase6_benchmark_dataset.json"),
    Path("reports/yolo26s_phase6_demo_result.json"),
    Path("reports/CAN_COIN_OPEN_DATA_RESULT.md"),
    Path("reports/can_coin_open_data_result.json"),
    Path("reports/phase7_uart.json"),
    Path("reports/phase8_status.json"),
    Path("reports/phase8_precollection.json"),
    Path("reports/phase9_status.json"),
    Path("scripts"),
    Path("src"),
    Path("training"),
    Path("tools"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)

# Historical INT8 tooling stays in the repository for auditability but is not part of the
# active YOLO26s/FP16 target workflow.
EXCLUDE_PATHS = {Path("deployment/build_calibration_dataset.py")}


def included_files() -> list[Path]:
    files: list[Path] = []
    model = Path(load_yaml(PROJECT_ROOT / "configs/deployment.yaml")["model"])
    onnx = Path("artifacts/deployment/onnx") / f"{model.stem}.onnx"
    for relative in (model, onnx, *INCLUDE_PATHS):
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            files.append(path)
        else:
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix not in {".pyc", ".pyo"}
            )
    files = sorted(
        path
        for path in set(files)
        if path.relative_to(PROJECT_ROOT) not in EXCLUDE_PATHS
    )
    if any(path.suffix == ".engine" for path in files):
        raise RuntimeError("A platform-specific TensorRT engine must never enter the Jetson bundle")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Create portable inputs for target-side Jetson build")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/deployment/jetson_bundle/vagt_orca_yolo26s_fp16_inputs.tar.gz"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase6_jetson_bundle.json")
    )
    args = parser.parse_args()

    files = included_files()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("wb") as raw_output,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in files:
            info = archive.gettarinfo(path, arcname=path.relative_to(PROJECT_ROOT))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    members = []
    with tarfile.open(args.output, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                members.append({"path": member.name, "bytes": member.size})
    report = {
        "bundle": str(args.output.resolve()),
        "bundle_sha256": sha256(args.output),
        "bundle_bytes": args.output.stat().st_size,
        "model_sha256": sha256(
            PROJECT_ROOT / load_yaml(PROJECT_ROOT / "configs/deployment.yaml")["model"]
        ),
        "onnx_sha256": sha256(
            PROJECT_ROOT / "artifacts/deployment/onnx"
            / (Path(load_yaml(PROJECT_ROOT / "configs/deployment.yaml")["model"]).stem + ".onnx")
        ),
        "files": len(members),
        "uncompressed_bytes": sum(item["bytes"] for item in members),
        "contains_engine": any(item["path"].endswith(".engine") for item in members),
        "target_build_required": True,
        "model_family": "YOLO26s",
        "deployment_precision": "fp16",
        "instructions": str((PROJECT_ROOT / "deployment/JETSON_RUNBOOK.md").resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
