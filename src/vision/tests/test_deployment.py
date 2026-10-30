from pathlib import Path

import pytest

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.deployment import percentile, sha256
from vagt_orca.precollection import build_plan, plan_checks


def test_sha256(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"phase6")
    assert sha256(sample) == "80e80cdf47a41f3ae0d34a0816593b6f723dbc586cb8459cdcb26cf323287154"


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_active_deployment_uses_supported_yolo26_checkpoint_and_static_fp16() -> None:
    model = load_yaml(PROJECT_ROOT / "configs/model.yaml")
    deployment = load_yaml(PROJECT_ROOT / "configs/deployment.yaml")
    runtime = load_yaml(PROJECT_ROOT / "configs/runtime.yaml")

    assert model["base_model"].endswith("yolo26s.pt")
    stem = Path(deployment["model"]).stem
    assert stem in {"yolo26s", "yolo26m"} or stem.startswith("yolo26s_")
    assert Path(deployment["model"]).suffix == ".pt"
    assert deployment["model"] == runtime["model"]
    assert deployment["tensorrt"]["precision"] == "fp16"
    assert deployment["tensorrt"]["batch"] == 1
    assert "calibration" not in deployment
    assert runtime["half"] is True


def test_ar_glass_plan_covers_every_class_and_variation() -> None:
    config = {
        "required_classes": ["bottle", "coin"],
        "required_variations": {
            "lighting": ["bright", "dark"],
            "distance": ["near", "far"],
        },
        "minimum_images_per_class": 100,
        "minimum_sequences_per_class": 3,
    }
    rows = build_plan(config, "pixy")

    assert len(rows) == 6
    assert all(plan_checks(config, rows).values())
    assert sum(int(row["max_images"]) for row in rows) == 600
    assert all("--source pixy" in str(row["command"]) for row in rows)
