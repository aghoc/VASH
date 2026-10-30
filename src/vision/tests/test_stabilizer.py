from vagt_orca.types import Detection
from vagt_orca.vision.target_stabilizer import TargetStabilizer


def det(class_id: int, center_x: float, track_id: int) -> Detection:
    return Detection(class_id, f"class_{class_id}", 0.9, (center_x - 5, 45, center_x + 5, 55), track_id)


def test_locks_after_four_of_five_votes() -> None:
    stabilizer = TargetStabilizer()
    target = det(0, 50, 1)
    distractor = det(1, 20, 2)
    sequence = [target, target, distractor, target, target]
    results = [stabilizer.update(item, [item], (100, 100)) for item in sequence]
    assert results[:4] == [None, None, None, None]
    assert results[4] == target


def test_hysteresis_rejects_ten_percent_advantage() -> None:
    stabilizer = TargetStabilizer()
    locked = det(0, 70, 1)  # distance 20
    for _ in range(4):
        stabilizer.update(locked, [locked], (100, 100))
    challenger = det(1, 68, 2)  # distance 18: only 10% better
    for _ in range(5):
        result = stabilizer.update(challenger, [locked, challenger], (100, 100))
    assert result is not None
    assert result.track_id == 1


def test_hysteresis_switches_after_strong_challenger_debounce() -> None:
    stabilizer = TargetStabilizer()
    locked = det(0, 70, 1)
    for _ in range(4):
        stabilizer.update(locked, [locked], (100, 100))
    challenger = det(1, 60, 2)  # 50% closer to center
    results = [
        stabilizer.update(challenger, [locked, challenger], (100, 100)) for _ in range(4)
    ]
    assert results[-1] is not None
    assert results[-1].track_id == 2


def test_absent_locked_target_is_not_returned_as_current_detection() -> None:
    stabilizer = TargetStabilizer()
    locked = det(0, 50, 1)
    for _ in range(4):
        stabilizer.update(locked, [locked], (100, 100))
    assert stabilizer.update(None, [], (100, 100)) is None
    assert stabilizer.locked == locked


def test_brief_missing_target_does_not_switch_to_challenger() -> None:
    stabilizer = TargetStabilizer(missing_grace_frames=3)
    remote = det(6, 50, 1)
    bottle = det(0, 50, 2)
    for _ in range(4):
        stabilizer.update(remote, [remote], (100, 100))

    assert stabilizer.update(bottle, [bottle], (100, 100)) is None
    assert stabilizer.update(bottle, [bottle], (100, 100)) is None
    reacquired = stabilizer.update(remote, [remote], (100, 100))
    assert reacquired is not None
    assert reacquired.track_id == remote.track_id


def test_sustained_missing_target_allows_centered_challenger_after_grace() -> None:
    stabilizer = TargetStabilizer(missing_grace_frames=3)
    remote = det(6, 50, 1)
    bottle = det(0, 50, 2)
    for _ in range(4):
        stabilizer.update(remote, [remote], (100, 100))

    results = [stabilizer.update(bottle, [bottle], (100, 100)) for _ in range(4)]
    assert results[:3] == [None, None, None]
    assert results[3] is not None
    assert results[3].track_id == bottle.track_id
