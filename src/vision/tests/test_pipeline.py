import numpy as np

from vagt_orca.grip.planner import GripPlanner
from vagt_orca.types import Detection
from vagt_orca.vision.pipeline import VisionGripPipeline
from vagt_orca.vision.target_stabilizer import TargetStabilizer


class FakeDetector:
    def predict(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(0, "bottle", 0.91, (200, 100, 440, 400), track_id=7)]


class VanishingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.calls <= 4:
            return [Detection(0, "bottle", 0.91, (200, 100, 440, 400), track_id=7)]
        return []


def test_end_to_end_result_shape_after_debounce() -> None:
    pipeline = VisionGripPipeline(FakeDetector(), TargetStabilizer(), GripPlanner())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [pipeline.process(frame) for _ in range(4)]
    assert results[-1].target is not None
    proposal = results[-1].plan.proposal
    assert proposal is not None
    assert proposal.as_dict() == {
        "class": "bottle",
        "grip": "POWER",
        "opening": 0.76375,
        "confidence": 0.91,
    }


def test_no_detection_retains_lkg_without_new_command() -> None:
    pipeline = VisionGripPipeline(VanishingDetector(), TargetStabilizer(), GripPlanner())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [pipeline.process(frame) for _ in range(5)]
    assert results[4].target is None
    assert results[4].plan.proposal == results[3].plan.proposal
    assert not results[4].plan.updated
    assert results[4].plan.reason == "no_target"


def test_background_and_unmapped_objects_cannot_steal_grip_target() -> None:
    class Scene:
        def __init__(self):
            self.detections = [
                Detection(160, "dining table", 0.99, (0, 0, 640, 480)),
                Detection(173, "book", 0.95, (280, 200, 360, 280)),
                Detection(0, "bottle", 0.90, (200, 100, 440, 400)),
            ]

        def predict(self, frame):
            return self.detections

    scene = Scene()
    pipeline = VisionGripPipeline(
        scene, TargetStabilizer(), GripPlanner(), grip_supported_targets_only=True,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(4):
        result = pipeline.process(frame)
    assert len(result.detections) == 3
    assert result.candidate.class_name == result.target.class_name == "bottle"
    assert result.plan.proposal.class_name == "bottle" and result.plan.updated
    scene.detections = scene.detections[:2]
    result = pipeline.process(frame)
    assert result.candidate is None and result.target is None
    assert result.plan.reason == "no_target" and not result.plan.updated
