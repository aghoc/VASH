from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Protocol

import numpy as np

# Pixy2 streams a fixed 316x208 raw sensor frame; see PIXY2_RAW_FRAME_* in
# libpixyusb2.h. The bridge reports the same values and is checked at open time.
PIXY_FRAME_WIDTH = 316
PIXY_FRAME_HEIGHT = 208

# The sensor is BGGR, which OpenCV names BayerBG. Overridable because the
# effective pattern depends on the firmware's row order; pixy_benchmark.py
# writes a color reference frame so this can be confirmed against a known target.
PIXY_BAYER_PATTERN = os.environ.get("VAGT_PIXY_BAYER", "BayerBG2BGR")

# Pixy2 processes at 60 fps internally, but the raw-frame path is a stop/grab
# request-response, not a stream. Treated as nominal metadata for video writers.
PIXY_NOMINAL_FPS = 20.0


class FrameSource(Protocol):
    """Frame producer for VisionGripPipeline, decoupled from cv2.VideoCapture."""

    @property
    def fps(self) -> float: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class OpenCVFrameSource:
    """Video file, stream URL, or V4L2/UVC camera index."""

    def __init__(self, source: str | int) -> None:
        import cv2

        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source {source!r}")
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def fps(self) -> float:
        return self._fps if self._fps > 0 else 30.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.read()
        return (ok, frame if ok else None)

    def release(self) -> None:
        self._capture.release()


class UVCFrameSource(OpenCVFrameSource):
    """Negotiate a Linux UVC camera mode before starting capture."""

    def __init__(
        self,
        device: str | int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 60,
        fourcc: str = "MJPG",
        buffer_size: int = 1,
        power_line_hz: int | None = None,
    ) -> None:
        import cv2

        if min(width, height, fps, buffer_size) <= 0 or len(fourcc) != 4:
            raise ValueError(
                "Camera dimensions, fps and buffer_size must be positive; fourcc needs 4 characters"
            )
        if power_line_hz is not None:
            import fcntl
            import struct

            if power_line_hz not in (0, 50, 60):
                raise ValueError("power_line_hz must be 0 (disabled), 50 or 60")
            path = f"/dev/video{device}" if isinstance(device, int) else device
            fd = os.open(path, os.O_RDWR)
            try:
                # V4L2_CID_POWER_LINE_FREQUENCY / VIDIOC_S_CTRL, Linux UVC ABI.
                value = {0: 0, 50: 1, 60: 2}[power_line_hz]
                fcntl.ioctl(fd, 0xC008561C, bytearray(struct.pack("Ii", 0x00980918, value)))
            finally:
                os.close(fd)
        self._capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(
                f"Unable to open UVC camera {device!r}; check USB and video group access"
            )
        try:
            for prop, value in (
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
                (cv2.CAP_PROP_FRAME_WIDTH, width),
                (cv2.CAP_PROP_FRAME_HEIGHT, height),
                (cv2.CAP_PROP_FPS, fps),
                (cv2.CAP_PROP_BUFFERSIZE, buffer_size),
            ):
                self._capture.set(prop, value)
            code = int(self._capture.get(cv2.CAP_PROP_FOURCC))
            self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            self.mode = {
                "device": str(device),
                "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self._fps,
                "fourcc": "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)),
            }
            if (
                (self.mode["width"], self.mode["height"]) != (width, height)
                or self.mode["fourcc"] != fourcc
                or abs(self._fps - fps) > 0.5
            ):
                raise RuntimeError(
                    f"UVC mode negotiation failed: requested {width}x{height}@{fps} {fourcc}; "
                    f"received {self.mode}"
                )
        except BaseException:
            self._capture.release()
            raise


class PixyBridgeError(RuntimeError):
    pass


def _load_bridge(library: str | Path | None = None) -> ctypes.CDLL:
    path = library or os.environ.get("VAGT_PIXY_BRIDGE")
    if path is None:
        raise PixyBridgeError(
            "Pixy bridge library not found. Build it with scripts/build_pixy_bridge.sh "
            "and set VAGT_PIXY_BRIDGE to the resulting libpixybridge.so."
        )
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise PixyBridgeError(f"Pixy bridge library does not exist: {resolved}")

    bridge = ctypes.CDLL(str(resolved))
    bridge.pixy_bridge_frame_width.restype = ctypes.c_int
    bridge.pixy_bridge_frame_height.restype = ctypes.c_int
    bridge.pixy_bridge_frame_bytes.restype = ctypes.c_int
    bridge.pixy_bridge_open.restype = ctypes.c_int
    bridge.pixy_bridge_stop.restype = ctypes.c_int
    bridge.pixy_bridge_resume.restype = ctypes.c_int
    bridge.pixy_bridge_grab.restype = ctypes.c_int
    bridge.pixy_bridge_grab.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    bridge.pixy_bridge_set_lamp.restype = ctypes.c_int
    bridge.pixy_bridge_set_lamp.argtypes = [ctypes.c_int, ctypes.c_int]
    bridge.pixy_bridge_close.restype = None
    return bridge


class PixyFrameSource:
    """Pixy2/Pixy2.1 raw-frame capture over USB via libpixyusb2.

    Pixy is not a UVC device, so cv2.VideoCapture cannot open it. Frames arrive
    as BGGR Bayer at 316x208 and are demosaiced to BGR here.

    cycle=False follows the official libpixyusb2 sequence: stop once, then call
    getRawFrame repeatedly. cycle=True is retained only as a diagnostic fallback.
    """

    def __init__(
        self,
        library: str | Path | None = None,
        cycle: bool = False,
        settle_us: int = 20_000,
        lamp: tuple[int, int] | None = None,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._bridge = _load_bridge(library)
        self._cycle = 1 if cycle else 0
        self._settle_us = int(settle_us)

        width = self._bridge.pixy_bridge_frame_width()
        height = self._bridge.pixy_bridge_frame_height()
        if (width, height) != (PIXY_FRAME_WIDTH, PIXY_FRAME_HEIGHT):
            raise PixyBridgeError(
                f"Bridge reports {width}x{height}; expected "
                f"{PIXY_FRAME_WIDTH}x{PIXY_FRAME_HEIGHT}. Rebuild against libpixyusb2."
            )
        self._width = width
        self._height = height
        self._nbytes = self._bridge.pixy_bridge_frame_bytes()

        rc = self._bridge.pixy_bridge_open()
        if rc < 0:
            raise PixyBridgeError(
                f"pixy_bridge_open failed with code {rc}. Check the USB cable, that "
                f"lsusb lists the Charmed Labs device, and that udev grants access "
                f"(otherwise the process needs root)."
            )
        self._closed = False

        if lamp is not None:
            self._bridge.pixy_bridge_set_lamp(int(lamp[0]), int(lamp[1]))
        if not self._cycle:
            stopped = self._bridge.pixy_bridge_stop()
            if stopped < 0:
                self.release()
                raise PixyBridgeError(f"pixy_bridge_stop failed with code {stopped}")

        self._buffer = np.zeros(self._nbytes, dtype=np.uint8)
        self._buffer_ptr = self._buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._conversion = getattr(cv2, f"COLOR_{PIXY_BAYER_PATTERN}")

    @property
    def fps(self) -> float:
        return PIXY_NOMINAL_FPS

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._closed:
            return (False, None)
        rc = self._bridge.pixy_bridge_grab(
            self._buffer_ptr, self._nbytes, self._cycle, self._settle_us
        )
        if rc < 0:
            return (False, None)
        bayer = self._buffer.reshape(self._height, self._width)
        return (True, self._cv2.cvtColor(bayer, self._conversion))

    def set_lamp(self, upper: int, lower: int) -> int:
        return self._bridge.pixy_bridge_set_lamp(int(upper), int(lower))

    def release(self) -> None:
        if not self._closed:
            self._bridge.pixy_bridge_close()
            self._closed = True


def open_frame_source(source: str | int) -> FrameSource:
    """Resolve a --source value to a frame source.

    "elp" selects the measured ELP IMX291 UVC profile, "pixy" selects the
    Pixy2 USB bridge, and other values preserve the OpenCV path/index behavior.
    """
    text = str(source)
    if text.lower() == "elp":
        from vagt_orca.config import PROJECT_ROOT, load_yaml

        return UVCFrameSource(**load_yaml(PROJECT_ROOT / "configs/camera_elp_imx291.yaml"))
    if text.lower() == "pixy":
        return PixyFrameSource()
    if text.isdigit():
        return OpenCVFrameSource(int(text))
    return OpenCVFrameSource(text)
