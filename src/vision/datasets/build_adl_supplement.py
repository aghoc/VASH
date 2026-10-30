#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from vagt_orca.datasets.adl import load_adl_annotations, yolo_line


def extract_video(video: Path, annotations: Path, output: Path) -> tuple[int, Counter[int]]:
    frames = load_adl_annotations(annotations)
    wanted = sorted(frames)
    if not wanted:
        return 0, Counter()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (1280, 960):
        raise RuntimeError(f"Unexpected ADL video dimensions {width}x{height}: {video}")
    # Official boxes are recorded on the dataset's 640x480 annotation canvas while
    # the downloadable videos are 1280x960. Normalized coordinates are identical
    # only when the annotation canvas dimensions are used here.
    annotation_width, annotation_height = 640, 480
    images = output / "images" / "train"
    labels = output / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    wanted_index = 0
    frame_index = 0
    counts: Counter[int] = Counter()
    written = 0
    while wanted_index < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        target = wanted[wanted_index]
        if frame_index == target:
            stem = f"adl_{video.stem}_{target:08d}"
            lines = [
                yolo_line(box, annotation_width, annotation_height) for box in frames[target]
            ]
            lines = [line for line in lines if line is not None]
            if lines:
                if not cv2.imwrite(str(images / f"{stem}.jpg"), frame):
                    raise RuntimeError(f"Failed to write frame {stem}")
                (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                counts.update(box.class_id for box in frames[target])
                written += 1
            wanted_index += 1
        frame_index += 1
    capture.release()
    if wanted_index != len(wanted):
        raise RuntimeError(f"Video ended before annotated frame {wanted[wanted_index]}: {video}")
    return written, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert trustworthy ADL classes to YOLO train data")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    total_images = 0
    total_boxes: Counter[int] = Counter()
    participants: dict[str, int] = {}
    for participant in range(1, 21):
        name = f"P_{participant:02d}"
        images, boxes = extract_video(
            args.videos / f"{name}.MP4",
            args.annotations / f"object_annot_{name}.txt",
            args.output,
        )
        participants[name] = images
        total_images += images
        total_boxes.update(boxes)
        print(f"{name}: {images} frames", flush=True)
    report = {
        "source": "ADL Dataset official videos and object annotations",
        "images": total_images,
        "annotations_by_class_id": dict(sorted(total_boxes.items())),
        "participants": participants,
        "excluded_label": "knife/spoon/fork (ambiguous merged source label)",
    }
    (args.output / "adl_supplement_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
