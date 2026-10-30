from __future__ import annotations

import json

import cv2
import numpy as np

from vagt_orca.types import Detection
from vagt_orca.vision.pipeline import FrameResult


def _same_visual_target(a: Detection | None, b: Detection) -> bool:
    if a is None:
        return False
    if a.track_id is not None and b.track_id is not None:
        return a.track_id == b.track_id
    return a.class_id == b.class_id and a.bbox == b.bbox


def render(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    cv2.drawMarker(output, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    for detection in result.detections:
        color = (0, 200, 0)
        if _same_visual_target(result.candidate, detection):
            color = (0, 220, 255)
        if _same_visual_target(result.target, detection):
            color = (0, 0, 255)
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if result.plan.proposal:
        text = json.dumps(result.plan.proposal.as_dict(), ensure_ascii=False)
        cv2.putText(output, text, (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    return output
