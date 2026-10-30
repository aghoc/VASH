#!/usr/bin/env python3
"""Serve a low-latency Pixy2 + YOLO intent view over HTTP."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.grip.planner import GripPlanner
from vagt_orca.vision.detector import YoloDetector
from vagt_orca.vision.frame_source import PixyFrameSource
from vagt_orca.vision.pipeline import FrameResult, VisionGripPipeline
from vagt_orca.vision.target_stabilizer import TargetStabilizer
from vagt_orca.visualization import render

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VASH Pixy Live Intent View</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
h2{margin:12px 0 4px}.legend{margin:0 0 10px;color:#ccc}
img{width:min(96vw,948px);height:auto;border:2px solid #555;background:#000}
.white{color:white}.yellow{color:#00dcff}.red{color:#ff5555}
</style></head><body>
<h2>VASH Pixy Live Intent View</h2>
<p class="legend"><span class="white">+ 시선 중앙</span> · <span class="yellow">노란색 candidate</span> · <span class="red">빨간색 target</span></p>
<img src="/stream.mjpg" alt="Pixy live stream">
</body></html>""".encode()


class LiveState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.status: dict[str, object] = {"ready": False}

    def update(self, jpeg: bytes, status: dict[str, object]) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.status = status
            self.sequence += 1
            self.condition.notify_all()


class LiveServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: LiveState) -> None:
        super().__init__(address, LiveHandler)
        self.state = state


class LiveHandler(BaseHTTPRequestHandler):
    server: LiveServer

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML)))
            self.end_headers()
            self.wfile.write(HTML)
            return
        if self.path == "/status.json":
            payload = json.dumps(self.server.state.status, ensure_ascii=False).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path != "/stream.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        sequence = -1
        try:
            while True:
                with self.server.state.condition:
                    if self.server.state.sequence == sequence:
                        self.server.state.condition.wait(timeout=5.0)
                    sequence = self.server.state.sequence
                    jpeg = self.server.state.jpeg
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        return


def annotate_intent(frame, result: FrameResult, fps: float):
    output = render(frame, result)
    height, width = output.shape[:2]
    center = (width // 2, height // 2)
    candidate = result.candidate
    if candidate is not None:
        cx, cy = (int(value) for value in candidate.center)
        distance = math.hypot(cx - center[0], cy - center[1])
        cv2.line(output, center, (cx, cy), (0, 220, 255), 1)
        cv2.circle(output, (cx, cy), 4, (0, 220, 255), -1)
        candidate_text = f"CANDIDATE: {candidate.class_name}  center_dist={distance:.1f}px"
    else:
        candidate_text = "CANDIDATE: none"
    target_text = f"TARGET: {result.target.class_name}" if result.target else "TARGET: acquiring"
    cv2.rectangle(output, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(output, candidate_text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    cv2.putText(output, target_text, (6, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
    cv2.putText(
        output,
        f"{fps:.1f} FPS",
        (width - 72, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    return output


def build_pipeline(runtime_path: Path, model_path: Path | None) -> VisionGripPipeline:
    runtime = load_yaml(runtime_path)
    device_text = str(runtime.get("device", "cuda:0"))
    if not device_text.startswith("cuda:"):
        raise ValueError("runtime.device must be cuda:N")
    detector = YoloDetector(
        model_path or runtime["model"],
        confidence=float(runtime["confidence_threshold"]),
        iou=float(runtime["iou_threshold"]),
        image_size=int(runtime["image_size"]),
        device=int(device_text.split(":", 1)[1]),
        fp16=bool(runtime.get("half", False)),
        class_confidence_thresholds=runtime.get("class_confidence_thresholds"),
    )
    target = runtime["target"]
    stabilizer = TargetStabilizer(
        int(target["debounce_window"]),
        int(target["required_votes"]),
        float(target["switch_margin"]),
        float(target["identity_iou_threshold"]),
        int(target.get("missing_grace_frames", 8)),
    )
    return VisionGripPipeline(
        detector,
        stabilizer,
        GripPlanner(
            float(runtime["confidence_threshold"]),
            class_confidence_thresholds=runtime.get("class_confidence_thresholds"),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    state = LiveState()
    server = LiveServer((args.host, args.port), state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Open http://0.0.0.0:{args.port} (replace 0.0.0.0 with the Jetson IP)", flush=True)

    pipeline = build_pipeline(args.runtime, args.model)
    source = PixyFrameSource(cycle=False)
    previous = time.perf_counter()
    smoothed_fps = 0.0
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                continue
            result = pipeline.process(frame)
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            smoothed_fps = (
                instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            )
            visualized = annotate_intent(frame, result, smoothed_fps)
            encoded, jpeg = cv2.imencode(
                ".jpg", visualized, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not encoded:
                continue
            state.update(
                jpeg.tobytes(),
                {
                    "ready": True,
                    "fps": round(smoothed_fps, 2),
                    "detections": len(result.detections),
                    "candidate": result.candidate.class_name if result.candidate else None,
                    "target": result.target.class_name if result.target else None,
                },
            )
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
