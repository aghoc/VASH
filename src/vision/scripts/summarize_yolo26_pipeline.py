#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from vagt_orca.deployment import sha256

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(relative: str) -> dict[str, float]:
    report = read(relative)
    return {
        "precision": report["precision_mean"],
        "recall": report["recall_mean"],
        "map50": report["map50"],
        "map50_95": report["map50_95"],
    }


def delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: current[key] - baseline[key] for key in current}


def main() -> None:
    phase3_val = metrics("artifacts/validation_yolo26s/val/metrics_summary.json")
    phase3_test = metrics("artifacts/validation_yolo26s/test/metrics_summary.json")
    public_val = metrics("artifacts/validation_yolo26s_public_v2/val/metrics_summary.json")
    public_test = metrics("artifacts/validation_yolo26s_public_v2/test/metrics_summary.json")
    final_val = metrics("artifacts/validation_yolo26s_adl_v3/val/metrics_summary.json")
    final_test = metrics("artifacts/validation_yolo26s_adl_v3/test/metrics_summary.json")
    fp16_val = metrics(
        "artifacts/validation_yolo26s_tensorrt_fp16_desktop/val/metrics_summary.json"
    )
    model = ROOT / "artifacts/models/yolo26s_best.pt"
    report = {
        "model_family": "YOLO26s",
        "training_precision": "fp32",
        "deployment_precision": "TensorRT FP16",
        "final_model": str(model),
        "final_model_bytes": model.stat().st_size,
        "final_model_sha256": sha256(model),
        "stages": {
            "phase3_egoobjects": {"val": phase3_val, "test": phase3_test},
            "public_v2": {
                "val": public_val,
                "test": public_test,
                "delta_from_phase3": {
                    "val": delta(public_val, phase3_val),
                    "test": delta(public_test, phase3_test),
                },
            },
            "adl_v3_final": {
                "val": final_val,
                "test": final_test,
                "delta_from_public_v2": {
                    "val": delta(final_val, public_val),
                    "test": delta(final_test, public_test),
                },
                "delta_from_phase3": {
                    "val": delta(final_val, phase3_val),
                    "test": delta(final_test, phase3_test),
                },
            },
            "desktop_tensorrt_fp16_reference": {
                "val": fp16_val,
                "delta_from_final_fp32": delta(fp16_val, final_val),
                "accepted_as_jetson_evidence": False,
            },
        },
        "selection": {
            "checkpoint": "adl_v3 epoch 1 best",
            "reason": (
                "The ADL stage improves frozen-test mAP50-95 over public_v2 while adding "
                "egocentric-domain exposure; Jetson FP16 validation remains pending."
            ),
        },
    }
    output = ROOT / "reports/yolo26s_training_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
