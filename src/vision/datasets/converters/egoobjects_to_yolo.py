#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagt_orca.datasets.egoobjects import convert_egoobjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EgoObjects LVIS JSON to YOLO format")
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink"), default="symlink")
    args = parser.parse_args()
    report = convert_egoobjects(
        args.annotations, args.images, args.output, args.classes, tuple(args.split), args.seed, args.link_mode
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
