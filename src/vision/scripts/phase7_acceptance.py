#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import select
import statistics
import time
from pathlib import Path

from vagt_orca.communication.packet import VisionPacket
from vagt_orca.communication.receiver import MockEsp32Receiver
from vagt_orca.communication.uart import PacketPublisher, UartSender
from vagt_orca.deployment import percentile, platform_summary
from vagt_orca.grip.planner import PlanResult
from vagt_orca.types import GripProposal


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def plan(name: str, grip: str, opening: float, updated: bool = True) -> PlanResult:
    return PlanResult(
        GripProposal(name, grip, opening, 0.91),
        updated,
        "updated" if updated else "no_target",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 UART mock acceptance")
    parser.add_argument("--report", type=Path, default=Path("reports/phase7_uart.json"))
    args = parser.parse_args()
    master_fd, slave_fd = os.openpty()
    slave_name = os.ttyname(slave_fd)
    os.close(slave_fd)
    receiver = MockEsp32Receiver()
    latencies_us = []
    sent = []
    received = []
    try:
        with UartSender(slave_name) as sender:
            publisher = PacketPublisher(sender, initial_sequence=100)
            plans = [
                plan("bottle", "POWER", 0.72),
                plan("pen", "TRIPOD", 0.31),
                plan("pen", "TRIPOD", 0.31, updated=False),
            ]
            for current in plans:
                started = time.perf_counter_ns()
                sent.append(publisher.publish(current))
                ready, _, _ = select.select([master_fd], [], [], 1.0)
                if not ready:
                    raise TimeoutError("PTY mock receiver timed out")
                received.extend(receiver.feed(os.read(master_fd, 1024)))
                latencies_us.append((time.perf_counter_ns() - started) / 1000.0)
    finally:
        os.close(master_fd)

    fake_clock = FakeClock()
    loss_receiver = MockEsp32Receiver(timeout_ms=250, clock=fake_clock)
    loss_receiver.feed(sent[0].encode())
    jumped = VisionPacket(
        sent[0].flags,
        sent[0].class_id,
        sent[0].grip_id,
        sent[0].confidence_pct,
        sent[0].opening_permille,
        sent[0].sequence + 3,
    )
    loss_receiver.feed(jumped.encode())
    active_before_timeout = loss_receiver.vision_assistance_active
    fake_clock.now = 0.251
    fallback_after_timeout = loss_receiver.emg_only_fallback

    corrupted = bytearray(sent[0].encode())
    corrupted[8] ^= 0x80
    loss_receiver.feed(bytes(corrupted))
    checks = {
        "pty_round_trip": received == sent,
        "target_change_changes_packet": (
            sent[0].class_id,
            sent[0].grip_id,
        )
        != (sent[1].class_id, sent[1].grip_id),
        "lkg_flagged_without_update": sent[2].last_known_good and not sent[2].plan_updated,
        "packet_loss_reproduced": loss_receiver.dropped_packets == 2,
        "checksum_corruption_rejected": loss_receiver.decoder.checksum_errors == 1,
        "active_before_timeout": active_before_timeout,
        "stale_250ms_emg_fallback": fallback_after_timeout,
    }
    report = {
        "phase": 7,
        "status": "complete" if all(checks.values()) else "failed",
        "accepted": all(checks.values()),
        "protocol": {
            "frame_bytes": 16,
            "baudrate": 115200,
            "byte_order": "little-endian",
            "checksum": "CRC-16/CCITT-FALSE",
            "timeout_ms": 250,
        },
        "platform": platform_summary(),
        "checks": checks,
        "pty_packets": len(received),
        "uart_round_trip_latency_us": {
            "mean": statistics.fmean(latencies_us),
            "p95": percentile(latencies_us, 95),
            "max": max(latencies_us),
        },
        "mock_receiver": loss_receiver.status(),
    }
    output = args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
