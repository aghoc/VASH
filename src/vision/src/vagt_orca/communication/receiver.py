from __future__ import annotations

import time
from collections.abc import Callable

from vagt_orca.communication.packet import PacketStreamDecoder, VisionPacket


def sequence_is_newer(candidate: int, previous: int) -> bool:
    delta = (candidate - previous) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


class MockEsp32Receiver:
    def __init__(
        self, timeout_ms: int = 250, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.timeout_seconds = timeout_ms / 1000.0
        self.clock = clock
        self.decoder = PacketStreamDecoder()
        self.latest: VisionPacket | None = None
        self.last_received_at: float | None = None
        self.accepted_packets = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.dropped_packets = 0

    def feed(self, data: bytes) -> list[VisionPacket]:
        accepted = []
        for packet in self.decoder.feed(data):
            if self.latest is not None:
                if packet.sequence == self.latest.sequence:
                    self.duplicate_packets += 1
                    continue
                if not sequence_is_newer(packet.sequence, self.latest.sequence):
                    self.out_of_order_packets += 1
                    continue
                delta = (packet.sequence - self.latest.sequence) & 0xFFFFFFFF
                self.dropped_packets += max(0, delta - 1)
            self.latest = packet
            self.last_received_at = self.clock()
            self.accepted_packets += 1
            accepted.append(packet)
        return accepted

    @property
    def link_fresh(self) -> bool:
        return bool(
            self.last_received_at is not None
            and self.clock() - self.last_received_at <= self.timeout_seconds
        )

    @property
    def vision_assistance_active(self) -> bool:
        return bool(self.link_fresh and self.latest and self.latest.target_valid)

    @property
    def emg_only_fallback(self) -> bool:
        return not self.vision_assistance_active

    def status(self) -> dict[str, int | bool | None]:
        return {
            "latest_sequence": self.latest.sequence if self.latest else None,
            "link_fresh": self.link_fresh,
            "vision_assistance_active": self.vision_assistance_active,
            "emg_only_fallback": self.emg_only_fallback,
            "accepted_packets": self.accepted_packets,
            "duplicate_packets": self.duplicate_packets,
            "out_of_order_packets": self.out_of_order_packets,
            "dropped_packets": self.dropped_packets,
            "checksum_errors": self.decoder.checksum_errors,
            "frame_errors": self.decoder.frame_errors,
            "discarded_bytes": self.decoder.discarded_bytes,
        }
