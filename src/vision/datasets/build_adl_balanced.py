#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a class-balanced ADL YOLO supplement")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--temporal-step", type=int, default=2)
    args = parser.parse_args()

    candidates_by_participant: dict[int, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for label in sorted((args.source / "labels" / "train").glob("*.txt")):
        participant = label.stem.split("_")[2]
        class_ids = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        for class_id in class_ids:
            candidates_by_participant[class_id][participant].append(label)

    selected: set[Path] = set()
    for class_id, participant_labels in sorted(candidates_by_participant.items()):
        labels = []
        for participant in sorted(participant_labels):
            labels.extend(participant_labels[participant][:: args.temporal_step])
        if len(labels) <= args.max_per_class:
            chosen = labels
        else:
            # Systematic sampling spans the full participant/time-ordered list.
            chosen = [
                labels[round(index * (len(labels) - 1) / (args.max_per_class - 1))]
                for index in range(args.max_per_class)
            ]
        selected.update(chosen)

    image_output = args.output / "images" / "train"
    label_output = args.output / "labels" / "train"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    counts: Counter[int] = Counter()
    participants: Counter[str] = Counter()
    for label in sorted(selected):
        image = args.source / "images" / "train" / f"{label.stem}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        for source, destination in (
            (image, image_output / image.name),
            (label, label_output / label.name),
        ):
            relative = os.path.relpath(source.resolve(), destination.parent.resolve())
            if not destination.exists():
                destination.symlink_to(relative)
        class_ids = [int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()]
        counts.update(class_ids)
        participants.update([label.stem.split("_")[2]])

    report = {
        "source": str(args.source.resolve()),
        "images": len(selected),
        "annotations_by_class_id": dict(sorted(counts.items())),
        "images_by_participant": dict(sorted(participants.items())),
        "max_per_class": args.max_per_class,
        "temporal_annotation_step": args.temporal_step,
    }
    (args.output / "adl_balanced_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
