#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vagt_orca.calibration.opening import fit_opening_calibration
from vagt_orca.config import PROJECT_ROOT, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Phase 9 class opening calibration")
    parser.add_argument(
        "--observations",
        type=Path,
        default=PROJECT_ROOT / "calibration/phase9/opening_observations.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "calibration/phase9/opening_fitted.yaml"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports/phase9_opening_calibration.json"
    )
    parser.add_argument("--minimum-samples", type=int, default=5)
    args = parser.parse_args()
    names = {str(value) for value in load_yaml(PROJECT_ROOT / "configs/classes.yaml")["names"].values()}
    calibration, metrics = fit_opening_calibration(
        args.observations, names, args.minimum_samples
    )
    overall_mae = sum(float(item["mae"]) for item in metrics.values()) / len(metrics)
    accepted = overall_mae <= 0.10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump({"calibration": calibration}, sort_keys=False), encoding="utf-8"
    )
    report = {
        "phase": 9,
        "accepted": accepted,
        "opening_mae_target": 0.10,
        "overall_mae": overall_mae,
        "classes": metrics,
        "fitted_config": str(args.output.resolve()),
        "note": "Do not replace configs/opening.yaml until physical pose checks also pass.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
