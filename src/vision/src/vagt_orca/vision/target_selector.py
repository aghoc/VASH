from __future__ import annotations

import math

from vagt_orca.types import Detection


class CenterBiasSelector:
    """Choose the box center nearest to the frame center, including off-center boxes."""

    @staticmethod
    def distance(detection: Detection, frame_size: tuple[int, int]) -> float:
        width, height = frame_size
        cx, cy = detection.center
        return math.hypot(cx - width / 2.0, cy - height / 2.0)

    def select(
        self, detections: list[Detection], frame_size: tuple[int, int]
    ) -> Detection | None:
        return min(
            detections, key=lambda item: self.distance(item, frame_size), default=None
        )
