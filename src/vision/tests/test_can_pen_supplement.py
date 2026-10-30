from types import SimpleNamespace

import pytest

from vagt_orca.types import Detection
from vagt_orca.vision import supplement as module
from vagt_orca.vision.supplement import SupplementedDetector, grip_thresholds, with_supplement


class FakeDetector:
    def __init__(self, names, detections):
        self.model = SimpleNamespace(names=names)
        self.detections = detections
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return self.detections


def test_supplement_preserves_primary_objects_and_disable_skips_second_inference():
    bottle = Detection(0, "bottle", 0.91, (10, 10, 30, 30))
    pen = Detection(7, "pen", 0.92, (50, 10, 60, 30))
    original = FakeDetector({39: "bottle", 67: "cell phone"}, [bottle])
    extra = FakeDetector({0: "can", 1: "pen"}, [pen])
    detector = SupplementedDetector(original, extra)
    combined = detector.predict(None)
    assert combined == [bottle, pen] and combined[0] is bottle
    assert detector.model is original.model
    detector.enabled = False
    assert detector.predict(None) is original.detections
    assert extra.calls == 1 and original.calls == 2


def test_non_two_class_supplement_is_rejected():
    with pytest.raises(ValueError, match="exactly two"):
        SupplementedDetector(FakeDetector({}, []), FakeDetector({0: "bottle", 1: "pen"}, []))


def test_baseline_config_is_unchanged_and_planner_gets_supplement_thresholds():
    original = object()
    assert with_supplement(original, None) is original
    runtime = {"class_confidence_thresholds": {"bottle": 0.4},
               "supplement_profile": {"class_confidence_thresholds": {"can": 0.8, "pen": 0.9}}}
    assert grip_thresholds(runtime) == {"bottle": 0.4, "can": 0.8, "pen": 0.9}
    assert runtime["class_confidence_thresholds"] == {"bottle": 0.4}
    assert grip_thresholds({"supplement_profile": None}) == {}


@pytest.mark.parametrize("device", ["0", "cuda:0", 0])
def test_supplement_reuses_primary_cuda_index(monkeypatch, device):
    primary = SimpleNamespace(confidence=0.35, iou=0.45, image_size=640,
                              device=device, quantize=16, model=object())
    calls = []

    def detector(*args, **kwargs):
        calls.append(kwargs)
        return FakeDetector({0: "can", 1: "pen"}, [])

    monkeypatch.setattr(module, "YoloDetector", detector)
    with_supplement(primary, {"model": "can_pen.engine",
                             "class_confidence_thresholds": {"can": 0.8, "pen": 0.8}})
    assert calls[0]["device"] == 0
    assert calls[0]["fp16"] is True
