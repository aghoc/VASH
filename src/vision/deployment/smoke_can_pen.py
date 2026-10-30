"""Check native supplement isolation on public images without opening the camera."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from vagt_orca.config import load_yaml
from vagt_orca.deployment import sha256
from vagt_orca.gpu import gpu_summary
from vagt_orca.vision.detector import YoloDetector
from vagt_orca.vision.supplement import with_supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_yaml(args.runtime)
    fingerprint = sha256(args.model)
    primary = YoloDetector(
        args.model, confidence=runtime["confidence_threshold"],
        iou=runtime["iou_threshold"], image_size=runtime["image_size"], device=0,
        fp16=True, detection_profile=runtime["detection_profile"],
    )
    combined = with_supplement(primary, runtime["supplement_profile"])
    paths = sorted(args.images.glob("*.jpg"))
    paths = [paths[i * (len(paths) - 1) // 31] for i in range(32)]
    primary_count = supplement_count = 0
    for path in paths:
        frame = cv2.imread(str(path))
        assert frame is not None
        before = primary.predict(frame)
        combined.enabled = True
        active = combined.predict(frame)
        assert active[:len(before)] == before
        assert all(d.class_name in {"can", "pen"} for d in active[len(before):])
        combined.enabled = False
        assert combined.predict(frame) == before
        primary_count += len(before)
        supplement_count += len(active) - len(before)
    assert sha256(args.model) == fingerprint
    assert len(combined.model.names) == 80
    assert set(combined.supplement.model.names.values()) == {"can", "pen"}
    report = {
        "public_images": len(paths), "primary_class_count": len(combined.model.names),
        "supplement_classes": combined.supplement.model.names,
        "primary_detections": primary_count, "supplement_detections": supplement_count,
        "enabled_primary_outputs_exact_match": True, "disabled_outputs_exact_match": True,
        "primary_sha256_before_after": fingerprint, "gpu": gpu_summary(0),
        "runtime": runtime, "camera_frames_saved": 0,
        "scope": "Functional isolation check, not an accuracy or latency benchmark",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
