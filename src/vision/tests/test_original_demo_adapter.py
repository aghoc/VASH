import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vagt_orca.vision import detector as module
from vagt_orca.vision.target_selector import CenterBiasSelector


class Array:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def install_fake(monkeypatch):
    names = {0: "person", 32: "sports ball", 39: "bottle", 42: "fork",
             65: "remote", 67: "cell phone"}
    calls = []

    class FakeYolo:
        def __init__(self, path):
            self.names = names

        def predict(self, **kwargs):
            calls.append(kwargs)
            boxes = SimpleNamespace(
                xyxy=Array([[0, 0, 32, 32]] * len(names)),
                cls=Array(list(names)), conf=Array([0.99] * len(names)), id=None,
            )
            return [SimpleNamespace(names=names, boxes=boxes)]

    monkeypatch.setattr(module, "require_cuda", lambda device: device)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    return calls


def profile():
    return {
        "allowed_classes": ["fork", "remote", "ball", "phone"],
        "aliases": {"sports ball": "ball", "cell phone": "phone"},
        "project_class_ids": {"fork": 9, "remote": 6, "ball": 4, "phone": 5},
        "end2end": True, "rect": False,
    }


def test_native_coco_classes_are_filtered_and_mapped_before_target_selection(monkeypatch):
    calls = install_fake(monkeypatch)
    detector = module.YoloDetector(
        "original.pt", detection_profile=profile(),
        class_confidence_thresholds={"phone": 0.5},
    )
    detections = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert calls[0]["classes"] == [32, 42, 65, 67]
    assert calls[0]["end2end"] is True and calls[0]["rect"] is False
    assert {(d.class_id, d.class_name) for d in detections} == {
        (4, "ball"), (9, "fork"), (6, "remote"), (5, "phone"),
    }
    assert CenterBiasSelector().select(detections, (32, 32)).class_name != "person"
    assert detector.model.names[67] == "cell phone"
    assert detector.model.names[32] == "sports ball"


def test_all_native_classes_display_with_project_aliases_and_distinct_ids(monkeypatch):
    calls = install_fake(monkeypatch)
    config = profile()
    del config["allowed_classes"]
    config["project_class_ids"]["bottle"] = 0
    config["unmapped_class_id_offset"] = 100
    detector = module.YoloDetector("original.pt", detection_profile=config)
    detections = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert "classes" not in calls[0]
    assert len(detections) == len(detector.model.names)
    by_name = {item.class_name: item.class_id for item in detections}
    assert by_name["person"] == 100 and by_name["bottle"] == 0
    assert by_name["phone"] == 5 and by_name["ball"] == 4
    assert len(set(by_name.values())) == len(by_name)
    assert detector.model.names[67] == "cell phone"


def test_unrestricted_mapping_cannot_confuse_person_and_bottle_identity(monkeypatch):
    install_fake(monkeypatch)
    with pytest.raises(ValueError, match="collide"):
        module.YoloDetector("original.pt", detection_profile={"project_class_ids": {"bottle": 0}})


def test_background_exclusion_keeps_other_objects_even_if_backend_ignores_filter(monkeypatch):
    calls = install_fake(monkeypatch)
    config = profile()
    del config["allowed_classes"]
    config.update(excluded_classes=["person"], unmapped_class_id_offset=100)
    detections = module.YoloDetector("original.pt", detection_profile=config).predict(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert calls[0]["classes"] == [32, 39, 42, 65, 67]
    assert {item.class_name for item in detections} == {"bottle", "ball", "fork", "remote", "phone"}


@pytest.mark.parametrize("change", [
    {"allowed_classes": ["pen"]},
    {"project_class_ids": {"fork": 9}},
    {"project_class_ids": {"fork": 4, "remote": 6, "ball": 4, "phone": 5}},
])
def test_invalid_original_mapping_is_rejected(monkeypatch, change):
    install_fake(monkeypatch)
    with pytest.raises(ValueError):
        module.YoloDetector("original.pt", detection_profile=profile() | change)
