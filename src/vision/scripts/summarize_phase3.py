#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_row(row: dict[str, str]) -> dict[str, int | float]:
    output: dict[str, int | float] = {}
    for key, value in row.items():
        output[key] = int(value) if key == "epoch" else float(value)
    return output


def read_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 3 result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed Phase 3 artifacts")
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/training/yolo26s_egoobjects_15")
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/yolo26s_phase3.pt"))
    parser.add_argument("--validation", type=Path, default=Path("artifacts/validation_yolo26s"))
    parser.add_argument(
        "--failures", type=Path, default=Path("artifacts/failures_yolo26s/failure_report.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/yolo26s_phase3_training_summary.json")
    )
    args = parser.parse_args()

    results_path = args.run / "results.csv"
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No completed epochs in {results_path}")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    metadata_path = args.model.with_suffix(".training.json")
    checkpoint_metadata = (
        read_json(metadata_path)
        if metadata_path.is_file()
        else {
            "run_directory": str(args.run.resolve()),
            "args": yaml.safe_load((args.run / "args.yaml").read_text(encoding="utf-8")),
            "completion": "manual_early_stop_after_5_epochs_without_validation_improvement",
        }
    )
    validation = {
        split: read_json(args.validation / split / "metrics_summary.json")
        for split in ("val", "test")
    }
    failure_report = read_json(args.failures)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "completed_epochs": len(rows),
            "best_epoch_by_map50_95": numeric_row(best),
            "final_epoch": numeric_row(rows[-1]),
            "metadata": checkpoint_metadata,
        },
        "checkpoint": {
            "path": str(args.model.resolve()),
            "bytes": args.model.stat().st_size,
            "sha256": sha256(args.model),
        },
        "validation": validation,
        "failure_analysis": {
            "split": failure_report["split"],
            "images_analyzed": failure_report["images_analyzed"],
            "sampling": failure_report["sampling"],
            "ground_truth_by_class": failure_report["ground_truth_by_class"],
            "misses_by_class": failure_report["misses_by_class"],
            "false_positives_by_class": failure_report["false_positives_by_class"],
            "saved_case_count": len(failure_report["saved_cases"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
