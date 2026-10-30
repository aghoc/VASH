"""Optional can/pen detector; preserve all original primary detections."""
from __future__ import annotations

from vagt_orca.vision.detector import YoloDetector


class SupplementedDetector:
    def __init__(self, primary, supplement):
        if set(supplement.model.names.values()) != {"can", "pen"}:
            raise ValueError("The can/pen supplement must have exactly two output classes")
        self.primary = primary
        self.supplement = supplement
        self.model = primary.model
        self.enabled = True

    def predict(self, frame):
        original = self.primary.predict(frame)
        return original + self.supplement.predict(frame) if self.enabled else original


def with_supplement(primary, profile):
    if not profile:
        return primary
    thresholds = profile["class_confidence_thresholds"]
    if set(thresholds) != {"can", "pen"}:
        raise ValueError("Supplement requires validated can and pen confidence thresholds")
    extra = YoloDetector(
        profile["model"], confidence=primary.confidence, iou=primary.iou,
        image_size=primary.image_size, device=int(str(primary.device).split(":")[-1]),
        fp16=primary.quantize == 16,
        class_confidence_thresholds=thresholds,
        detection_profile={"allowed_classes": ["can", "pen"],
                           "project_class_ids": {"can": 1, "pen": 7},
                           "rect": False, "end2end": True},
    )
    return SupplementedDetector(primary, extra)


def grip_thresholds(runtime):
    return {
        **(runtime.get("class_confidence_thresholds") or {}),
        **((runtime.get("supplement_profile") or {}).get("class_confidence_thresholds") or {}),
    }
