#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagt_orca.deployment import sha256
from vagt_orca.gpu import gpu_summary, prepare_onnxruntime_cuda, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained checkpoint on CUDA")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, help="Override validation image size, e.g. 960")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--project", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args()
    require_cuda(args.device)
    if args.model.suffix.lower() == ".onnx":
        prepare_onnxruntime_cuda()
    print(f"GPU: {gpu_summary(args.device)}")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    image_options = {} if args.imgsz is None else {"imgsz": args.imgsz}
    metrics = model.val(
        data=str(args.data),
        device=args.device,
        split=args.split,
        batch=args.batch,
        amp=False,
        quantize=32,
        plots=True,
        save_json=True,
        project=str(args.project.resolve()),
        name=args.split,
        **image_options,
    )
    summary = {
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "model_format": args.model.suffix.lower(),
        "precision": "fp16" if args.model.suffix.lower() == ".engine" else "fp32",
        "data": str(args.data.resolve()),
        "split": args.split,
        "batch": args.batch,
        "image_size": args.imgsz or model.overrides.get("imgsz", 640),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class_map": {
            str(metrics.names[int(index)]): float(metrics.box.ap[row])
            for row, index in enumerate(metrics.box.ap_class_index)
        },
        "per_class": {
            str(metrics.names[int(index)]): {
                "map50_95": float(metrics.box.ap[row]),
                "map50": float(metrics.box.ap50[row]),
                "precision": float(metrics.box.p[row]),
                "recall": float(metrics.box.r[row]),
                "instances": int(metrics.nt_per_class[int(index)]),
            }
            for row, index in enumerate(metrics.box.ap_class_index)
        },
        "speed_ms_per_image": {
            str(key): float(value) for key, value in metrics.speed.items()
        },
        "gpu": gpu_summary(args.device),
    }
    output = Path(metrics.save_dir) / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
