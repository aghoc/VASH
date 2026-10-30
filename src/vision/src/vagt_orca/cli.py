from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT, load_yaml


def infer(args: argparse.Namespace) -> None:
    import cv2

    from vagt_orca.communication.uart import PacketPublisher, UartSender
    from vagt_orca.grip.planner import GripPlanner
    from vagt_orca.vision.detector import YoloDetector
    from vagt_orca.vision.frame_source import open_frame_source
    from vagt_orca.vision.pipeline import VisionGripPipeline
    from vagt_orca.vision.supplement import grip_thresholds, with_supplement
    from vagt_orca.vision.target_stabilizer import TargetStabilizer
    from vagt_orca.visualization import render

    runtime = load_yaml(args.runtime)
    device_text = str(runtime.get("device", "cuda:0"))
    if not device_text.startswith("cuda:"):
        raise ValueError("runtime.device must be cuda:N; CPU execution is prohibited")
    device = int(device_text.split(":", 1)[1])
    detector = YoloDetector(
        args.model or runtime["model"],
        confidence=float(runtime["confidence_threshold"]),
        iou=float(runtime["iou_threshold"]),
        image_size=int(runtime["image_size"]),
        device=device,
        fp16=bool(runtime.get("half", False)),
        class_confidence_thresholds=runtime.get("class_confidence_thresholds"),
        focus_profile=runtime.get("focus_profile"),
        detection_profile=runtime.get("detection_profile"),
    )
    target_config = runtime["target"]
    detector = with_supplement(detector, runtime.get("supplement_profile"))
    stabilizer = TargetStabilizer(
        int(target_config["debounce_window"]),
        int(target_config["required_votes"]),
        float(target_config["switch_margin"]),
        float(target_config["identity_iou_threshold"]),
        int(target_config.get("missing_grace_frames", 8)),
    )
    planner = GripPlanner(
        float(runtime["confidence_threshold"]),
        class_confidence_thresholds=grip_thresholds(runtime),
    )
    pipeline = VisionGripPipeline(
        detector, stabilizer, planner,
        grip_supported_targets_only=bool(target_config.get("grip_supported_only", False)),
    )
    publisher = (
        PacketPublisher(UartSender(args.uart, args.uart_baud)) if args.uart is not None else None
    )

    capture = open_frame_source(args.source)
    writer = None
    json_records = []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process(frame)
            uart_packet = publisher.publish(result.plan) if publisher is not None else None
            payload = {
                "frame": frame_number,
                "detections": len(result.detections),
                "candidate": result.candidate.class_name if result.candidate else None,
                "target": result.target.class_name if result.target else None,
                "plan": result.plan.proposal.as_dict() if result.plan.proposal else None,
                "plan_updated": result.plan.updated,
                "reason": result.plan.reason,
                "uart_sequence": uart_packet.sequence if uart_packet is not None else None,
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            json_records.append(payload)
            visualized = render(frame, result)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    fps = capture.fps
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to create output video {args.output}")
                writer.write(visualized)
            if args.show:
                cv2.imshow("VAGT-ORCA", visualized)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if publisher is not None:
            publisher.close()
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in json_records),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="vagt-orca")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer_parser = subparsers.add_parser("infer", help="Run GPU-only video/webcam inference")
    infer_parser.add_argument(
        "--source", required=True, help="video path, webcam index, 'elp' (IMX291), or 'pixy'"
    )
    infer_parser.add_argument("--model", type=Path)
    infer_parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    infer_parser.add_argument("--output", type=Path)
    infer_parser.add_argument("--json-output", type=Path)
    infer_parser.add_argument("--show", action="store_true")
    infer_parser.add_argument("--uart", type=Path, help="UART device, for example /dev/ttyTHS1")
    infer_parser.add_argument("--uart-baud", type=int, default=115200)
    infer_parser.set_defaults(func=infer)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
