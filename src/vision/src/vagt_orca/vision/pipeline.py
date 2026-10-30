from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vagt_orca.grip.planner import GripPlanner, PlanResult
from vagt_orca.types import Detection
from vagt_orca.vision.target_selector import CenterBiasSelector
from vagt_orca.vision.target_stabilizer import TargetStabilizer


@dataclass(frozen=True, slots=True)
class FrameResult:
    detections: list[Detection]
    candidate: Detection | None
    target: Detection | None
    plan: PlanResult


class VisionGripPipeline:
    def __init__(
        self, detector, stabilizer: TargetStabilizer, planner: GripPlanner,
        *, grip_supported_targets_only: bool = False,
    ) -> None:
        self.detector = detector
        self.selector = CenterBiasSelector()
        self.stabilizer = stabilizer
        self.planner = planner
        self.grip_supported_targets_only = grip_supported_targets_only

    def process(self, frame: np.ndarray) -> FrameResult:
        height, width = frame.shape[:2]
        detections = self.detector.predict(frame)
        eligible = (
            [item for item in detections if item.class_name in self.planner.grip_map]
            if self.grip_supported_targets_only else detections
        )
        candidate = self.selector.select(eligible, (width, height))
        target = self.stabilizer.update(candidate, eligible, (width, height))
        plan = self.planner.plan(target, width)
        return FrameResult(detections, candidate, target, plan)
