from vagt_orca.types import Detection
from vagt_orca.vision.target_selector import CenterBiasSelector


def detection(name: str, class_id: int, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(class_id, name, 0.9, bbox)


def test_selects_closest_box_center() -> None:
    selector = CenterBiasSelector()
    edge = detection("bottle", 0, (0, 0, 20, 20))
    center = detection("cup", 2, (45, 45, 55, 55))
    assert selector.select([edge, center], (100, 100)) == center


def test_no_detections_returns_none() -> None:
    assert CenterBiasSelector().select([], (640, 480)) is None


def test_rejects_detection_that_does_not_contain_crosshair() -> None:
    off_center = detection("bottle", 0, (5, 5, 200, 200))
    assert CenterBiasSelector().select([off_center], (640, 480)) is None


def test_selects_closest_center_when_boxes_overlap_crosshair() -> None:
    wide = detection("bottle", 0, (100, 100, 500, 400))
    aimed = detection("remote", 6, (300, 220, 350, 270))
    assert CenterBiasSelector().select([wide, aimed], (640, 480)) == aimed
