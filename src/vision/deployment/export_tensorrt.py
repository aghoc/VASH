#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vagt_orca.config import load_yaml
from vagt_orca.deployment import (
    platform_summary,
    require_jetson,
    sha256,
)
from vagt_orca.gpu import gpu_summary, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-specific YOLO26 TensorRT FP16 engine")
    parser.add_argument("--config", type=Path, default=Path("configs/deployment.yaml"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/yolo26s_phase6_tensorrt.json")
    )
    parser.add_argument(
        "--development-host",
        action="store_true",
        help="Build a non-deployable engine to test the export path on a non-Jetson NVIDIA GPU",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_path = args.model or Path(config["model"])
    platform = platform_summary() if args.development_host else require_jetson()
    if args.output is None:
        suffix = (
            "_fp16_RTX_DEVELOPMENT_ONLY_NOT_FOR_JETSON"
            if args.development_host and not platform["is_jetson"]
            else "_fp16"
        )
        args.output = Path("artifacts/deployment/tensorrt") / f"{model_path.stem}{suffix}.engine"
    if args.development_host and not platform["is_jetson"]:
        if args.output == Path("artifacts/deployment/tensorrt/yolo26s_best_fp16.engine"):
            args.output = Path(
                "artifacts/deployment/tensorrt/"
                "yolo26s_best_fp16_RTX_DEVELOPMENT_ONLY_NOT_FOR_JETSON.engine"
            )
        if args.report == Path("reports/yolo26s_phase6_tensorrt.json"):
            args.report = Path("reports/yolo26s_phase6_tensorrt_desktop.json")
    device = int(config["device"])
    tensorrt_config = config["tensorrt"]
    require_cuda(device)
    import tensorrt as trt
    from ultralytics import YOLO

    if str(tensorrt_config["precision"]).lower() != "fp16":
        raise ValueError("The project deployment precision is fixed to TensorRT FP16")
    requested_dynamic = bool(tensorrt_config["dynamic"])
    requested_batch = int(tensorrt_config["batch"])
    effective_dynamic = requested_dynamic
    effective_batch = requested_batch

    model = YOLO(str(model_path))
    end2end = bool(getattr(model.model, "end2end", False))
    if not end2end:
        raise RuntimeError("YOLO26 end-to-end/NMS-free head is not active")
    exported = Path(
        model.export(
            format="engine",
            imgsz=int(config["image_size"]),
            device=device,
            batch=effective_batch,
            dynamic=effective_dynamic,
            opset=int(config["onnx"]["opset"]),
            workspace=float(tensorrt_config["workspace_gib"]),
            quantize=16,
            nms=False,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != args.output.resolve():
        shutil.move(str(exported), args.output)
    report = {
        "accepted_target_build": bool(platform["is_jetson"]),
        "development_only": not bool(platform["is_jetson"]),
        "portability_warning": (
            None
            if platform["is_jetson"]
            else "This engine is GPU/platform-specific and MUST NOT be deployed to Jetson."
        ),
        "platform": platform,
        "gpu": gpu_summary(device),
        "tensorrt_version": trt.__version__,
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "engine": str(args.output.resolve()),
        "engine_bytes": args.output.stat().st_size,
        "engine_sha256": sha256(args.output),
        "model_family": "YOLO26" + str(model.model.yaml.get("scale", "s")),
        "precision": "fp16",
        "end2end_nms_free": end2end,
        "requested_settings": tensorrt_config,
        "settings": {
            **tensorrt_config,
            "dynamic": effective_dynamic,
            "batch": effective_batch,
            "opset": int(config["onnx"]["opset"]),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
