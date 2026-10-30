#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from vagt_orca.config import load_yaml
from vagt_orca.deployment import platform_summary, sha256
from vagt_orca.gpu import gpu_summary, prepare_onnxruntime_cuda, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and smoke-test YOLO ONNX on CUDA")
    parser.add_argument("--config", type=Path, default=Path("configs/deployment.yaml"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reports/yolo26s_phase6_onnx.json"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_path = args.model or Path(config["model"])
    args.output = args.output or Path("artifacts/deployment/onnx") / f"{model_path.stem}.onnx"
    device = int(config["device"])
    onnx_config = config["onnx"]
    require_cuda(device)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    end2end = bool(getattr(model.model, "end2end", False))
    if not end2end:
        raise RuntimeError("YOLO26 end-to-end/NMS-free head is not active")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=int(config["image_size"]),
            device=device,
            opset=int(onnx_config["opset"]),
            simplify=bool(onnx_config["simplify"]),
            dynamic=bool(onnx_config["dynamic"]),
            batch=int(onnx_config["batch"]),
            quantize=32,
            nms=False,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != args.output.resolve():
        shutil.move(str(exported), args.output)

    import onnx
    import onnxruntime as ort

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph, full_check=True)
    available = prepare_onnxruntime_cuda()
    session = ort.InferenceSession(
        str(args.output), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX Runtime did not activate CUDA first; providers={active}")
    input_metadata = session.get_inputs()[0]
    shape = [1 if not isinstance(value, int) else value for value in input_metadata.shape]
    outputs = session.run(None, {input_metadata.name: np.zeros(shape, dtype=np.float32)})
    if not outputs or outputs[0].size == 0:
        raise RuntimeError("ONNX CUDA smoke test returned no model output")

    report = {
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "onnx": str(args.output.resolve()),
        "onnx_bytes": args.output.stat().st_size,
        "onnx_sha256": sha256(args.output),
        "opsets": {entry.domain or "ai.onnx": entry.version for entry in graph.opset_import},
        "graph": {
            "nodes": len(graph.graph.node),
            "inputs": [value.name for value in graph.graph.input],
            "outputs": [value.name for value in graph.graph.output],
        },
        "cuda_smoke_test": {
            "available_providers": available,
            "active_providers": active,
            "input_name": input_metadata.name,
            "input_shape": shape,
            "output_shapes": [list(output.shape) for output in outputs],
            "passed": True,
        },
        "gpu": gpu_summary(device),
        "platform": platform_summary(),
        "model_family": "YOLO26s",
        "end2end_nms_free": end2end,
        "precision": "fp32",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
