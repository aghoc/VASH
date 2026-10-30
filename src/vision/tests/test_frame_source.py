import ctypes

import numpy as np
import pytest

from vagt_orca.vision import frame_source
from vagt_orca.vision.frame_source import (
    PIXY_FRAME_HEIGHT,
    PIXY_FRAME_WIDTH,
    OpenCVFrameSource,
    PixyBridgeError,
    PixyFrameSource,
    UVCFrameSource,
    open_frame_source,
)


class FakeBridge:
    """Stands in for libpixybridge.so so the Pixy path is testable off-hardware."""

    def __init__(self, width=PIXY_FRAME_WIDTH, height=PIXY_FRAME_HEIGHT, open_code=0) -> None:
        self._width = width
        self._height = height
        self._open_code = open_code
        self.grabs: list[tuple[int, int]] = []
        self.lamp: tuple[int, int] | None = None
        self.stopped = False
        self.closed = False

    def _stub(self, value):
        def stub(*args):
            return value

        # Mimic the attributes ctypes function objects carry.
        stub.restype = None
        stub.argtypes = []
        return stub

    def __getattr__(self, name):
        table = {
            "pixy_bridge_frame_width": self._width,
            "pixy_bridge_frame_height": self._height,
            "pixy_bridge_frame_bytes": self._width * self._height,
            "pixy_bridge_open": self._open_code,
            "pixy_bridge_resume": 0,
        }
        if name in table:
            return self._stub(table[name])
        raise AttributeError(name)

    def pixy_bridge_stop(self, *args):
        self.stopped = True
        return 0

    def pixy_bridge_grab(self, pointer, size, cycle, settle_us):
        self.grabs.append((cycle, settle_us))
        # Fill the caller's buffer with a deterministic ramp.
        buffer = (ctypes.c_uint8 * size).from_address(ctypes.addressof(pointer.contents))
        for index in range(size):
            buffer[index] = index % 256
        return 0

    def pixy_bridge_set_lamp(self, upper, lower):
        self.lamp = (upper, lower)
        return 0

    def pixy_bridge_close(self):
        self.closed = True


@pytest.fixture
def fake_bridge(monkeypatch):
    bridge = FakeBridge()
    monkeypatch.setattr(frame_source, "_load_bridge", lambda library=None: bridge)
    return bridge


def test_pixy_source_demosaics_to_three_channel_bgr(fake_bridge) -> None:
    source = PixyFrameSource()
    try:
        ok, frame = source.read()
    finally:
        source.release()
    assert ok
    assert frame is not None
    assert frame.shape == (PIXY_FRAME_HEIGHT, PIXY_FRAME_WIDTH, 3)
    assert frame.dtype == np.uint8
    assert fake_bridge.closed


def test_cycled_mode_passes_settle_and_skips_persistent_stop(fake_bridge) -> None:
    source = PixyFrameSource(cycle=True, settle_us=1234)
    try:
        source.read()
    finally:
        source.release()
    assert fake_bridge.grabs == [(1, 1234)]
    assert not fake_bridge.stopped


def test_persistent_mode_stops_once_at_open(fake_bridge) -> None:
    source = PixyFrameSource(cycle=False)
    try:
        source.read()
        source.read()
    finally:
        source.release()
    assert fake_bridge.stopped
    assert [cycle for cycle, _ in fake_bridge.grabs] == [0, 0]


def test_lamp_is_applied_at_open(fake_bridge) -> None:
    source = PixyFrameSource(lamp=(1, 1))
    source.release()
    assert fake_bridge.lamp == (1, 1)


def test_read_after_release_reports_failure(fake_bridge) -> None:
    source = PixyFrameSource()
    source.release()
    assert source.read() == (False, None)


def test_unexpected_frame_geometry_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(frame_source, "_load_bridge", lambda library=None: FakeBridge(width=320))
    with pytest.raises(PixyBridgeError, match="Rebuild against libpixyusb2"):
        PixyFrameSource()


def test_open_failure_surfaces_actionable_message(monkeypatch) -> None:
    monkeypatch.setattr(frame_source, "_load_bridge", lambda library=None: FakeBridge(open_code=-1))
    with pytest.raises(PixyBridgeError, match="lsusb"):
        PixyFrameSource()


def test_missing_bridge_library_is_reported(monkeypatch) -> None:
    monkeypatch.delenv("VAGT_PIXY_BRIDGE", raising=False)
    with pytest.raises(PixyBridgeError, match="build_pixy_bridge.sh"):
        frame_source._load_bridge()


def test_factory_routes_pixy_and_opencv_sources(monkeypatch) -> None:
    seen: list[object] = []

    class Recorder:
        def __init__(self, source):
            seen.append(source)

    monkeypatch.setattr(frame_source, "PixyFrameSource", lambda: seen.append("pixy"))
    monkeypatch.setattr(frame_source, "OpenCVFrameSource", Recorder)
    open_frame_source("pixy")
    open_frame_source("0")
    open_frame_source("clip.mp4")
    assert seen == ["pixy", 0, "clip.mp4"]


def test_opencv_source_rejects_unopenable_input() -> None:
    with pytest.raises(RuntimeError, match="Unable to open video source"):
        OpenCVFrameSource("does_not_exist.mp4")


def test_uvc_rejects_silent_mode_fallback_and_releases_camera(monkeypatch) -> None:
    import cv2

    class Camera:
        released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            return False  # Device ignores the requested mode.

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 5,
                cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUYV"),
            }[prop]

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    with pytest.raises(RuntimeError, match="negotiation failed"):
        UVCFrameSource()
    assert camera.released


def test_elp_factory_loads_camera_profile(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(frame_source, "UVCFrameSource", lambda **kwargs: seen.update(kwargs))
    open_frame_source("elp")
    assert seen["device"].endswith("video-index0")
    assert (seen["width"], seen["height"], seen["fps"]) == (1280, 720, 60)
    assert seen["buffer_size"] == 1
