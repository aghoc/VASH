import sys
from types import SimpleNamespace

import numpy as np

from vagt_orca.vision import detector as module


class Array:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_class_threshold_filters_before_target_selection(monkeypatch):
    names = {0: "bottle", 1: "can", 13: "coin"}
    calls = []

    class FakeYolo:
        def __init__(self, path):
            self.names = names

        def predict(self, **kwargs):
            calls.append(kwargs)
            boxes = SimpleNamespace(
                xyxy=Array([[0, 0, 20, 20]] * 3),
                cls=Array([0, 1, 13]),
                conf=Array([0.34, 0.34, 0.2]),
                id=None,
            )
            return [SimpleNamespace(names=names, boxes=boxes)]

    monkeypatch.setattr(module, "require_cuda", lambda device: device)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    detector = module.YoloDetector(
        "test.pt", confidence=0.35, class_confidence_thresholds={"coin": 0.15}
    )
    result = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert calls[0]["conf"] == 0.15
    assert [(item.class_name, item.confidence) for item in result] == [("coin", 0.2)]
