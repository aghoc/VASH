from __future__ import annotations

import math

from vagt_orca.types import Detection


class CenterBiasSelector:
    @staticmethod
    def contains_frame_center(detection: Detection, frame_size: tuple[int, int]) -> bool:
        width, height = frame_size
        x1, y1, x2, y2 = detection.bbox
        center_x, center_y = width / 2.0, height / 2.0
        return x1 <= center_x <= x2 and y1 <= center_y <= y2

    @staticmethod
    def distance(detection: Detection, frame_size: tuple[int, int]) -> float:
        width, height = frame_size
        cx, cy = detection.center
        return math.hypot(cx - width / 2.0, cy - height / 2.0)

    def select(
        self, detections: list[Detection], frame_size: tuple[int, int]
    ) -> Detection | None:
        centered = [
            item for item in detections if self.contains_frame_center(item, frame_size)
        ]
        if not centered:
            return None
        return min(centered, key=lambda item: self.distance(item, frame_size))
