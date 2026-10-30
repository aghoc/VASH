#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagt_orca.datasets.validate import validate_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset")
    parser.add_argument("data_yaml", type=Path)
    args = parser.parse_args()
    report = validate_yolo_dataset(args.data_yaml)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
