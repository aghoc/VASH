from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ADL_TO_TARGET = {
    "bottle": 0,
    "mug/cup": 2,
    "cell_phone": 5,
    "cell": 5,
    "tv_remote": 6,
    "tooth_brush": 10,
}


@dataclass(frozen=True)
class AdlBox:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


def load_adl_annotations(path: Path) -> dict[int, list[AdlBox]]:
    frames: dict[int, list[AdlBox]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"Malformed ADL annotation at {path}:{line_number}")
        class_id = ADL_TO_TARGET.get(fields[-1])
        if class_id is None:
            continue
        x1, y1, x2, y2 = map(int, fields[1:5])
        frame_index = int(fields[5])
        if x2 <= x1 or y2 <= y1:
            continue
        frames[frame_index].append(AdlBox(class_id, x1, y1, x2, y2))
    return dict(frames)


def yolo_line(box: AdlBox, width: int, height: int) -> str | None:
    x1 = max(0, min(width, box.x1))
    y1 = max(0, min(height, box.y1))
    x2 = max(0, min(width, box.x2))
    y2 = max(0, min(height, box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        f"{box.class_id} {((x1 + x2) / 2) / width:.8f} "
        f"{((y1 + y2) / 2) / height:.8f} {(x2 - x1) / width:.8f} "
        f"{(y2 - y1) / height:.8f}"
    )
