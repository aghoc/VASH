from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.types import Detection, GripProposal

SUPPORTED_GRIPS = {"POWER", "TRIPOD", "PINCH", "LATERAL_PINCH"}


@dataclass(frozen=True, slots=True)
class PlanResult:
    proposal: GripProposal | None
    updated: bool
    reason: str


class GripPlanner:
    def __init__(
        self,
        confidence_threshold: float = 0.35,
        class_config: str | Path = PROJECT_ROOT / "configs/classes.yaml",
        grip_config: str | Path = PROJECT_ROOT / "configs/grip_map.yaml",
        opening_config: str | Path = PROJECT_ROOT / "configs/opening.yaml",
        class_confidence_thresholds: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold
        class_names = {str(value) for value in load_yaml(class_config)["names"].values()}
        self.class_confidence_thresholds = dict(class_confidence_thresholds or {})
        if set(self.class_confidence_thresholds) - class_names:
            raise ValueError("Unknown class confidence threshold")
        if any(not 0.0 <= value <= 1.0 for value in self.class_confidence_thresholds.values()):
            raise ValueError("class confidence thresholds must be in [0, 1]")
        self.grip_map = load_yaml(grip_config)["grips"]
        self.calibration = load_yaml(opening_config)["calibration"]
        if set(self.grip_map) != class_names:
            raise ValueError(
                "Class/grip config mismatch: "
                f"missing_grips={sorted(class_names - set(self.grip_map))}, "
                f"unknown_grips={sorted(set(self.grip_map) - class_names)}"
            )
        if set(self.grip_map) != set(self.calibration):
            missing_grips = sorted(set(self.calibration) - set(self.grip_map))
            missing_calibration = sorted(set(self.grip_map) - set(self.calibration))
            raise ValueError(
                "Grip/opening class mismatch: "
                f"missing_grips={missing_grips}, missing_calibration={missing_calibration}"
            )
        invalid_grips = sorted(set(self.grip_map.values()) - SUPPORTED_GRIPS)
        if invalid_grips:
            raise ValueError(f"Unsupported grip names: {invalid_grips}")
        for class_name, params in self.calibration.items():
            if not isinstance(params, dict) or "alpha" not in params or "beta" not in params:
                raise ValueError(f"Invalid opening calibration for {class_name!r}")
        self.last_known_good: GripProposal | None = None

    def estimate_opening(self, class_name: str, bbox_width: float, image_width: int) -> float:
        if image_width <= 0:
            raise ValueError("image_width must be positive")
        if class_name not in self.calibration:
            raise KeyError(f"No opening calibration for class {class_name!r}")
        params = self.calibration[class_name]
        opening = float(params["alpha"]) * max(0.0, bbox_width) / image_width + float(
            params["beta"]
        )
        return min(1.0, max(0.0, opening))

    def plan(self, target: Detection | None, image_width: int) -> PlanResult:
        if target is None:
            return PlanResult(self.last_known_good, False, "no_target")
        if target.confidence < self.class_confidence_thresholds.get(
            target.class_name, self.confidence_threshold
        ):
            return PlanResult(self.last_known_good, False, "below_confidence_threshold")
        if target.class_name not in self.grip_map:
            return PlanResult(self.last_known_good, False, "unknown_class")
        proposal = GripProposal(
            class_name=target.class_name,
            grip=str(self.grip_map[target.class_name]),
            opening=self.estimate_opening(target.class_name, target.width, image_width),
            confidence=target.confidence,
        )
        self.last_known_good = proposal
        return PlanResult(proposal, True, "updated")
