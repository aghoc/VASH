#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

from vagt_orca.grip.planner import GripPlanner
from vagt_orca.types import Detection
from vagt_orca.vision.target_selector import CenterBiasSelector
from vagt_orca.vision.target_stabilizer import TargetStabilizer

EXPECTED_GRIPS = {
    "bottle": "POWER",
    "can": "POWER",
    "cup": "POWER",
    "box": "POWER",
    "ball": "POWER",
    "phone": "POWER",
    "remote": "POWER",
    "pen": "TRIPOD",
    "spoon": "TRIPOD",
    "fork": "TRIPOD",
    "toothbrush": "TRIPOD",
    "card": "LATERAL_PINCH",
    "key": "LATERAL_PINCH",
    "coin": "PINCH",
    "small_block": "PINCH",
}


def detection(
    class_id: int, name: str, center_x: float, center_y: float, track_id: int
) -> Detection:
    return Detection(
        class_id,
        name,
        0.9,
        (center_x - 10, center_y - 10, center_x + 10, center_y + 10),
        track_id,
    )


def main() -> None:
    rng = random.Random(42)
    frame_size = (640, 480)
    selector = CenterBiasSelector()
    center_correct = 0
    debounce_correct = 0
    scenarios = 100

    for scenario in range(scenarios):
        target = detection(
            0,
            "bottle",
            320 + rng.uniform(-8, 8),
            240 + rng.uniform(-8, 8),
            scenario * 10 + 1,
        )
        distractors = [
            detection(1, "can", rng.uniform(30, 180), rng.uniform(30, 450), scenario * 10 + 2),
            detection(
                3, "box", rng.uniform(460, 610), rng.uniform(30, 450), scenario * 10 + 3
            ),
        ]
        if selector.select([*distractors, target], frame_size) == target:
            center_correct += 1

        stabilizer = TargetStabilizer()
        sequence = [target, target, distractors[0], target, target]
        result = None
        for candidate in sequence:
            result = stabilizer.update(candidate, [candidate], frame_size)
        if result == target:
            debounce_correct += 1

    planner = GripPlanner()
    grip_correct = 0
    for class_id, (class_name, expected_grip) in enumerate(EXPECTED_GRIPS.items()):
        result = planner.plan(detection(class_id, class_name, 320, 240, class_id), frame_size[0])
        if result.proposal is not None and result.proposal.grip == expected_grip:
            grip_correct += 1

    no_target = planner.plan(None, frame_size[0])
    report = {
        "seed": 42,
        "center_bias": {
            "scenarios": scenarios,
            "correct": center_correct,
            "accuracy": center_correct / scenarios,
        },
        "debounce_5_frame_4_vote": {
            "scenarios": scenarios,
            "correct": debounce_correct,
            "accuracy": debounce_correct / scenarios,
        },
        "grip_mapping": {
            "classes": len(EXPECTED_GRIPS),
            "correct": grip_correct,
            "accuracy": grip_correct / len(EXPECTED_GRIPS),
        },
        "no_target_lkg": {
            "proposal_retained": no_target.proposal is not None,
            "updated": no_target.updated,
            "reason": no_target.reason,
        },
    }
    output = Path("reports/phase45_acceptance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
