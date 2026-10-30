from __future__ import annotations

from collections import deque

from vagt_orca.types import Detection
from vagt_orca.vision.target_selector import CenterBiasSelector


def bbox_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TargetStabilizer:
    """Debounce centered intent while resisting brief target occlusions."""

    def __init__(
        self,
        window_size: int = 5,
        required_votes: int = 4,
        switch_margin: float = 0.20,
        identity_iou_threshold: float = 0.30,
        missing_grace_frames: int = 8,
    ) -> None:
        if not 1 <= required_votes <= window_size:
            raise ValueError("required_votes must be in [1, window_size]")
        if not 0 <= switch_margin < 1:
            raise ValueError("switch_margin must be in [0, 1)")
        if missing_grace_frames < 0:
            raise ValueError("missing_grace_frames must be non-negative")
        self.history: deque[Detection | None] = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.switch_margin = switch_margin
        self.identity_iou_threshold = identity_iou_threshold
        self.missing_grace_frames = missing_grace_frames
        self.missing_locked_frames = 0
        self.locked: Detection | None = None

    def _same(self, a: Detection, b: Detection) -> bool:
        if a.track_id is not None and b.track_id is not None:
            return a.track_id == b.track_id
        return a.class_id == b.class_id and bbox_iou(a, b) >= self.identity_iou_threshold

    def _find_locked(self, detections: list[Detection]) -> Detection | None:
        if self.locked is None:
            return None
        matches = [item for item in detections if self._same(self.locked, item)]
        return max(matches, key=lambda item: bbox_iou(self.locked, item), default=None)

    def update(
        self,
        candidate: Detection | None,
        detections: list[Detection],
        frame_size: tuple[int, int],
    ) -> Detection | None:
        if candidate is None:
            self.history.append(None)
            current_locked = self._find_locked(detections)
            if current_locked is not None:
                self.locked = current_locked
                self.missing_locked_frames = 0
            elif self.locked is not None:
                self.missing_locked_frames += 1
            # Keep the internal lock for reacquisition, but do not present a stale
            # detection to the planner as if it were visible in the current frame.
            return None

        selector = CenterBiasSelector()
        current_locked = self._find_locked(detections)
        if self.locked is not None and current_locked is None:
            self.missing_locked_frames += 1
            self.history.append(candidate)
            if self.missing_locked_frames <= self.missing_grace_frames:
                return None
        else:
            self.missing_locked_frames = 0
        admitted = candidate
        if current_locked is not None and not self._same(current_locked, candidate):
            candidate_distance = selector.distance(candidate, frame_size)
            locked_distance = selector.distance(current_locked, frame_size)
            # The challenger must be at least 20% closer to the screen center.
            if candidate_distance > locked_distance * (1.0 - self.switch_margin):
                admitted = current_locked

        self.history.append(admitted)
        votes = sum(
            item is not None and self._same(admitted, item) for item in self.history
        )
        if votes >= self.required_votes:
            self.locked = admitted
            self.missing_locked_frames = 0
            return self.locked
        elif current_locked is not None:
            # Refresh geometry/confidence for the already locked target.
            self.locked = current_locked
            return self.locked
        # The previous lock is currently absent and a challenger is still being
        # debounced. Preserve the lock internally while withholding a new command.
        return None

    def reset(self) -> None:
        self.history.clear()
        self.locked = None
        self.missing_locked_frames = 0
