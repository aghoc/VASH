from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])


@dataclass(frozen=True, slots=True)
class GripProposal:
    class_name: str
    grip: str
    opening: float
    confidence: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "class": self.class_name,
            "grip": self.grip,
            "opening": round(self.opening, 6),
            "confidence": round(self.confidence, 6),
        }
