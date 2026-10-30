#!/usr/bin/env python3
"""Serve an ELP/UVC + GPU YOLO intent view on localhost."""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.grip.planner import GripPlanner
from vagt_orca.vision.detector import YoloDetector
from vagt_orca.vision.frame_source import open_frame_source
from vagt_orca.vision.pipeline import FrameResult, VisionGripPipeline
from vagt_orca.vision.supplement import SupplementedDetector, grip_thresholds, with_supplement
from vagt_orca.vision.target_stabilizer import TargetStabilizer
from vagt_orca.visualization import render

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VASH ELP IMX291 Live View</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
h2{margin:12px 0 4px}.legend{margin:0 0 10px;color:#ccc}
img{width:min(96vw,948px);height:auto;border:2px solid #555;background:#000}
.white{color:white}.yellow{color:#00dcff}.red{color:#ff5555}
button{font-size:16px;padding:8px 16px;margin:0 4px;cursor:pointer}
button[aria-pressed="true"]{background:#a9e8c5;color:#111;border:2px solid #58b783}
</style></head><body>
<h2>VASH ELP IMX291 Live View</h2>
<p class="legend"><span class="white">+ 시선 중앙</span> · <span class="yellow">노란색 candidate</span> · <span class="red">빨간색 target</span></p>
<p id="modes" hidden><button id="original" onclick="setMode('original')">원본</button>
<button id="can-pen" onclick="setMode('can-pen')">원본 + 캔·펜</button> <strong id="mode"></strong></p>
<p id="supplement-note" hidden>캔·펜 실험: 오탐을 줄인 설정으로, 놓치는 물체가 많을 수 있습니다.</p>
<img src="/stream.mjpg" alt="Camera live stream">
<p id="status">카메라 연결 중…</p>
<script>
async function setMode(mode){await fetch('/mode/'+mode,{method:'POST'});}
setInterval(async()=>{
  try {
    const s=await (await fetch('/status.json',{cache:'no-store'})).json();
    document.getElementById('modes').hidden=!s.supplement_available;
    document.getElementById('supplement-note').hidden=!s.supplement_enabled;
    document.getElementById('mode').textContent=s.supplement_enabled?'원본 + 캔·펜':'원본';
    document.getElementById('original').setAttribute('aria-pressed',!s.supplement_enabled);
    document.getElementById('can-pen').setAttribute('aria-pressed',!!s.supplement_enabled);
    document.getElementById('status').textContent=s.ready
      ? `${s.camera.width}×${s.camera.height} · ${s.fps} FPS · target: ${s.target || 'acquiring'}`
      : '카메라 연결 중…';
  } catch(e) { document.getElementById('status').textContent='연결 확인 중…'; }
},1000);
</script>
</body></html>""".encode()


class LiveState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.status: dict[str, object] = {"ready": False}
        self.supplement_enabled = True

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

    def do_POST(self) -> None:
        if self.path not in ("/mode/original", "/mode/can-pen"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self.server.state.status.get("supplement_available"):
            self.send_error(HTTPStatus.CONFLICT, "No can/pen supplement loaded")
            return
        with self.server.state.condition:
            self.server.state.supplement_enabled = self.path == "/mode/can-pen"
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

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
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/snapshot.jpg":
            with self.server.state.condition:
                jpeg = self.server.state.jpeg
            if jpeg is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
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
        focus_profile=runtime.get("focus_profile"),
        detection_profile=runtime.get("detection_profile"),
    )
    detector = with_supplement(detector, runtime.get("supplement_profile"))
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
            class_confidence_thresholds=grip_thresholds(runtime),
        ),
        grip_supported_targets_only=bool(target.get("grip_supported_only", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source", default="elp")
    parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--preview-width", type=int, default=960)
    args = parser.parse_args()
    if args.preview_width <= 0 or not 1 <= args.jpeg_quality <= 100:
        parser.error("preview-width must be positive and jpeg-quality must be 1..100")
    cv2.setNumThreads(2)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)

    state = LiveState()
    source = open_frame_source(args.source)
    server = None
    frames = 0
    failures = 0
    consecutive_failures = 0
    processing_times: deque[float] = deque(maxlen=120)
    started = time.perf_counter()
    previous = time.perf_counter()
    smoothed_fps = 0.0
    try:
        pipeline = build_pipeline(args.runtime, args.model)
        server = LiveServer((args.host, args.port), state)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"Open http://{args.host}:{args.port}; source={args.source}", flush=True)
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                failures += 1
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    raise RuntimeError("Camera stopped delivering frames; check the USB connection")
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            has_supplement = isinstance(pipeline.detector, SupplementedDetector)
            if has_supplement and pipeline.detector.enabled != state.supplement_enabled:
                pipeline.detector.enabled = state.supplement_enabled
                pipeline.stabilizer.reset()
                pipeline.planner.last_known_good = None
                processing_times.clear()
            process_start = time.perf_counter()
            result = pipeline.process(frame)
            processing_times.append((time.perf_counter() - process_start) * 1000)
            frames += 1
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            smoothed_fps = (
                instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            )
            visualized = annotate_intent(frame, result, smoothed_fps)
            if visualized.shape[1] > args.preview_width:
                scale = args.preview_width / visualized.shape[1]
                visualized = cv2.resize(visualized, None, fx=scale, fy=scale)
            encoded, jpeg = cv2.imencode(
                ".jpg", visualized, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not encoded:
                continue
            state.update(
                jpeg.tobytes(),
                {
                    "ready": True,
                    "supplement_available": has_supplement,
                    "supplement_enabled": has_supplement and pipeline.detector.enabled,
                    "model_mode": (
                        "original_plus_can_pen"
                        if has_supplement and pipeline.detector.enabled else "original"
                    ),
                    "source": args.source,
                    "camera": getattr(
                        source,
                        "mode",
                        {
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                            "fps": source.fps,
                        },
                    ),
                    "frames": frames,
                    "read_failures": failures,
                    "last_frame_unix": time.time(),
                    "uptime_seconds": round(now - started, 1),
                    "fps": round(smoothed_fps, 2),
                    "pipeline_mean_ms": round(statistics.mean(processing_times), 2),
                    "pipeline_p95_ms": round(
                        sorted(processing_times)[math.ceil(len(processing_times) * 0.95) - 1], 2
                    ),
                    "detections": len(result.detections),
                    "candidate": result.candidate.class_name if result.candidate else None,
                    "target": result.target.class_name if result.target else None,
                },
            )
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
