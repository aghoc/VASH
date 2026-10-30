import csv

import pytest

from vagt_orca.calibration.opening import fit_opening_calibration


def write_observations(path, classes: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "class_name",
                "bbox_width_px",
                "image_width_px",
                "required_opening_normalized",
            ),
        )
        writer.writeheader()
        for name in classes:
            for bbox in (64, 128, 192, 256, 320):
                writer.writerow(
                    {
                        "class_name": name,
                        "bbox_width_px": bbox,
                        "image_width_px": 640,
                        "required_opening_normalized": 0.5 * bbox / 640 + 0.2,
                    }
                )


def test_fit_opening_calibration_recovers_linear_model(tmp_path) -> None:
    observations = tmp_path / "observations.csv"
    write_observations(observations, ["bottle", "pen"])
    calibration, metrics = fit_opening_calibration(observations, {"bottle", "pen"})
    assert calibration["bottle"]["alpha"] == pytest.approx(0.5)
    assert calibration["bottle"]["beta"] == pytest.approx(0.2)
    assert metrics["pen"]["mae"] < 1e-12


def test_fit_requires_every_class(tmp_path) -> None:
    observations = tmp_path / "observations.csv"
    write_observations(observations, ["bottle"])
    with pytest.raises(ValueError, match="Missing calibration classes"):
        fit_opening_calibration(observations, {"bottle", "pen"})
