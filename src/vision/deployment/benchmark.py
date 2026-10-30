#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from vagt_orca.communication.uart import PacketPublisher
from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.deployment import percentile, platform_summary, sha256
from vagt_orca.gpu import gpu_summary, require_cuda
from vagt_orca.grip.planner import GripPlanner
from vagt_orca.vision.detector import YoloDetector
from vagt_orca.vision.pipeline import VisionGripPipeline
from vagt_orca.vision.supplement import grip_thresholds, with_supplement
from vagt_orca.vision.target_stabilizer import TargetStabilizer


class PacketSink:
    def __init__(self) -> None:
        self.frames = 0
        self.last_packet: bytes | None = None

    def write(self, data: bytes) -> None:
        self.frames += 1
        self.last_packet = data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full detection-to-grip pipeline on CUDA"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/benchmark/yolo26s_phase6/images"))
    parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    parser.add_argument("--config", type=Path, default=Path("configs/deployment.yaml"))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--hold-frames", type=int)
    parser.add_argument("--report", type=Path, default=Path("reports/phase6_benchmark.json"))
    parser.add_argument(
        "--serialize-uart",
        action="store_true",
        help="Include Jetson-to-ESP32 packet serialization in measured latency",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    runtime = load_yaml(args.runtime)
    device = int(str(runtime["device"]).split(":", 1)[1])
    require_cuda(device)
    benchmark_config = config["benchmark"]
    warmup = args.warmup or int(benchmark_config["warmup"])
    iterations = args.iterations or int(benchmark_config["iterations"])
    hold_frames = args.hold_frames or int(benchmark_config["hold_frames"])
    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No benchmark images in {args.images}")
    target = runtime["target"]
    pipeline = VisionGripPipeline(
        with_supplement(YoloDetector(
            args.model,
            confidence=float(runtime["confidence_threshold"]),
            iou=float(runtime["iou_threshold"]),
            image_size=int(runtime["image_size"]),
            device=device,
            # Exported TensorRT engines carry FP16 precision in the artifact itself.
            fp16=args.model.suffix.lower() != ".engine",
            class_confidence_thresholds=runtime.get("class_confidence_thresholds"),
            focus_profile=runtime.get("focus_profile"),
            detection_profile=runtime.get("detection_profile"),
        ), runtime.get("supplement_profile")),
        TargetStabilizer(
            int(target["debounce_window"]),
            int(target["required_votes"]),
            float(target["switch_margin"]),
            float(target["identity_iou_threshold"]),
            int(target.get("missing_grace_frames", 8)),
        ),
        GripPlanner(
            float(runtime["confidence_threshold"]),
            class_confidence_thresholds=grip_thresholds(runtime),
        ),
        grip_supported_targets_only=bool(target.get("grip_supported_only", False)),
    )
    packet_sink = PacketSink()
    publisher = PacketPublisher(packet_sink) if args.serialize_uart else None
    frames = [cv2.imread(str(path)) for path in image_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("A benchmark image failed to decode")
    for index in range(warmup):
        result = pipeline.process(frames[(index // hold_frames) % len(frames)])
        if publisher is not None:
            publisher.publish(result.plan)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    peak_device_memory = total_memory - free_memory
    latencies = []
    detections = 0
    proposal_frames = 0
    updated_frames = 0
    for index in range(iterations):
        frame = frames[(index // hold_frames) % len(frames)]
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        result = pipeline.process(frame)
        if publisher is not None:
            publisher.publish(result.plan)
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        detections += len(result.detections)
        proposal_frames += int(result.plan.proposal is not None)
        updated_frames += int(result.plan.updated)
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        peak_device_memory = max(peak_device_memory, total_memory - free_memory)
    total_seconds = sum(latencies) / 1000.0
    platform = platform_summary()
    specialist_path = (runtime.get("focus_profile") or {}).get("specialist_model")
    accepted = bool(platform["is_jetson"] and args.model.suffix == ".engine"
                    and (not specialist_path or Path(specialist_path).suffix == ".engine"))
    suffix = args.model.suffix.lower()
    precision = (
        "fp16" if suffix == ".engine" or (suffix == ".pt" and runtime.get("half")) else "fp32"
    )
    report = {
        "deployment_accepted": accepted,
        "acceptance_reason": (
            "Jetson TensorRT benchmark complete"
            if accepted
            else "Development benchmark only; requires .engine execution on target Jetson"
        ),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "model_format": args.model.suffix,
        "runtime": str(args.runtime.resolve()),
        "runtime_sha256": sha256(args.runtime),
        "focus_profile": runtime.get("focus_profile"),
        "detection_profile": runtime.get("detection_profile"),
        "supplement_profile": runtime.get("supplement_profile"),
        "specialist_sha256": sha256(Path(specialist_path)) if specialist_path else None,
        "precision": precision,
        "images": len(frames),
        "warmup": warmup,
        "iterations": iterations,
        "hold_frames_per_scene": hold_frames,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
        },
        "fps": iterations / total_seconds,
        "detections": detections,
        "frames_with_plan_proposal": proposal_frames,
        "frames_with_new_plan_update": updated_frames,
        "uart_packet_serialization": {
            "enabled": args.serialize_uart,
            "frame_bytes": len(packet_sink.last_packet) if packet_sink.last_packet else 0,
            "packets": packet_sink.frames,
            "included_in_latency": args.serialize_uart,
        },
        "peak_torch_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_device_memory_used_mib": peak_device_memory / 1024**2,
        "gpu": gpu_summary(device),
        "platform": platform,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
