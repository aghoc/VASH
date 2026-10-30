from __future__ import annotations

import hashlib
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jetson_model() -> str | None:
    for path in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            model = path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
        except OSError:
            continue
        if "jetson" in model.lower():
            return model
    return None


def platform_summary() -> dict[str, str | bool | None]:
    model = jetson_model()
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "jetson_model": model,
        "is_jetson": platform.machine() == "aarch64" and model is not None,
    }


def require_jetson() -> dict[str, str | bool | None]:
    summary = platform_summary()
    if not summary["is_jetson"]:
        raise RuntimeError(
            "TensorRT deployment artifacts must be built and benchmarked on the target Jetson. "
            f"Current platform: machine={summary['machine']}, model={summary['jetson_model']!r}."
        )
    return summary


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
