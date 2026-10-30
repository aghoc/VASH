import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def test_static_engine_calibration_uses_single_frame_batches(tmp_path):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "training/calibrate_can_coin.py"))
    for kind in ("images", "labels"):
        (tmp_path / kind / "val").mkdir(parents=True)
    for i in range(3):
        (tmp_path / "images" / "val" / f"{i}.jpg").touch()
        (tmp_path / "labels" / "val" / f"{i}.txt").write_text("")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(tmp_path), "val": "images/val"}))
    calls = []

    class Data:
        def cpu(self):
            return self

        def numpy(self):
            return np.empty((0, 6))

    class StaticEngine:
        model = "model.engine"

        def predict(self, *, source, batch, **kwargs):
            assert len(source) == batch == 1
            assert kwargs["rect"] is False
            calls.append(source)
            return [SimpleNamespace(orig_shape=(32, 32), boxes=SimpleNamespace(data=Data()))]

    records = namespace["load_predictions"](StaticEngine(), config, True)
    assert len(records) == len(calls) == 3
