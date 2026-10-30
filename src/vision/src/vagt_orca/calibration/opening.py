from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def fit_opening_calibration(
    observations: str | Path, required_classes: set[str], minimum_samples: int = 5
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float | int]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with Path(observations).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["class_name"].strip()
            if name not in required_classes:
                raise ValueError(f"Unknown class in observations: {name!r}")
            image_width = float(row["image_width_px"])
            bbox_width = float(row["bbox_width_px"])
            opening = float(row["required_opening_normalized"])
            if image_width <= 0 or bbox_width < 0 or not 0 <= opening <= 1:
                raise ValueError(f"Invalid calibration row for {name!r}: {row}")
            grouped[name].append((bbox_width / image_width, opening))
    missing = sorted(required_classes - grouped.keys())
    if missing:
        raise ValueError(f"Missing calibration classes: {missing}")

    calibration = {}
    metrics = {}
    for name in sorted(required_classes):
        samples = grouped[name]
        if len(samples) < minimum_samples:
            raise ValueError(f"{name!r} has {len(samples)} samples; need {minimum_samples}")
        x = np.asarray([item[0] for item in samples], dtype=np.float64)
        y = np.asarray([item[1] for item in samples], dtype=np.float64)
        design = np.column_stack((x, np.ones_like(x)))
        alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = np.clip(alpha * x + beta, 0.0, 1.0)
        calibration[name] = {"alpha": float(alpha), "beta": float(beta)}
        metrics[name] = {
            "samples": len(samples),
            "mae": float(np.mean(np.abs(prediction - y))),
            "max_error": float(np.max(np.abs(prediction - y))),
        }
    return calibration, metrics
