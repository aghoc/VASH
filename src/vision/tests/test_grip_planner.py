import pytest

from vagt_orca.grip.planner import GripPlanner
from vagt_orca.types import Detection


def target(name: str, confidence: float = 0.9, width: float = 200) -> Detection:
    names = [
        "bottle",
        "can",
        "cup",
        "box",
        "ball",
        "phone",
        "remote",
        "pen",
        "spoon",
        "fork",
        "toothbrush",
        "card",
        "key",
        "coin",
        "small_block",
    ]
    return Detection(names.index(name), name, confidence, (0, 0, width, 100))


@pytest.mark.parametrize(
    ("name", "grip"),
    [
        ("bottle", "POWER"),
        ("can", "POWER"),
        ("cup", "POWER"),
        ("box", "POWER"),
        ("ball", "POWER"),
        ("phone", "POWER"),
        ("remote", "POWER"),
        ("pen", "TRIPOD"),
        ("spoon", "TRIPOD"),
        ("fork", "TRIPOD"),
        ("toothbrush", "TRIPOD"),
        ("card", "LATERAL_PINCH"),
        ("key", "LATERAL_PINCH"),
        ("coin", "PINCH"),
        ("small_block", "PINCH"),
    ],
)
def test_all_fifteen_classes_have_deterministic_grip(name: str, grip: str) -> None:
    result = GripPlanner().plan(target(name), 640)
    assert result.updated
    assert result.proposal is not None
    assert result.proposal.grip == grip


def test_opening_is_clamped() -> None:
    planner = GripPlanner()
    assert planner.estimate_opening("bottle", 10_000, 640) == 1.0
    assert 0.0 <= planner.estimate_opening("coin", 0, 640) <= 1.0


def test_low_confidence_retains_last_known_good() -> None:
    planner = GripPlanner(confidence_threshold=0.5)
    good = planner.plan(target("bottle", 0.9), 640)
    low = planner.plan(target("pen", 0.49), 640)
    assert good.proposal is not None
    assert low.proposal == good.proposal
    assert not low.updated
    assert low.reason == "below_confidence_threshold"


def test_no_target_retains_last_known_good() -> None:
    planner = GripPlanner()
    assert planner.plan(None, 640).proposal is None
    good = planner.plan(target("key"), 640)
    assert planner.plan(None, 640).proposal == good.proposal


def test_invalid_confidence_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        GripPlanner(confidence_threshold=1.1)


def test_class_threshold_preserves_raw_score_and_other_class_gate() -> None:
    planner = GripPlanner(class_confidence_thresholds={"coin": 0.15})
    good = planner.plan(target("coin", 0.2), 640)
    assert good.updated
    assert good.proposal.confidence == 0.2
    assert good.proposal.grip == "PINCH"
    below = planner.plan(target("coin", 0.14), 640)
    assert not below.updated
    assert below.proposal == good.proposal
    assert not planner.plan(target("phone", 0.34), 640).updated


@pytest.mark.parametrize("thresholds", [{"unknown": 0.1}, {"coin": -0.1}, {"coin": 1.1}])
def test_invalid_class_threshold_is_rejected(thresholds) -> None:
    with pytest.raises(ValueError):
        GripPlanner(class_confidence_thresholds=thresholds)
