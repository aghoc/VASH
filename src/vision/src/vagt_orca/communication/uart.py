from __future__ import annotations

import os
import termios
from pathlib import Path
from types import TracebackType

from typing_extensions import Self

from vagt_orca.communication.packet import (
    FLAG_LAST_KNOWN_GOOD,
    FLAG_PLAN_UPDATED,
    FLAG_TARGET_VALID,
    GRIP_IDS,
    NO_CLASS,
    VisionPacket,
)
from vagt_orca.config import PROJECT_ROOT, load_yaml
from vagt_orca.grip.planner import PlanResult

_BAUDRATES = {
    rate: getattr(termios, f"B{rate}")
    for rate in (9600, 57600, 115200, 230400, 460800, 921600)
    if hasattr(termios, f"B{rate}")
}


class UartSender:
    def __init__(self, device: str | Path, baudrate: int = 115200) -> None:
        if baudrate not in _BAUDRATES:
            raise ValueError(f"Unsupported baudrate {baudrate}; choose from {sorted(_BAUDRATES)}")
        self.device = str(device)
        self.baudrate = baudrate
        self.fd = os.open(self.device, os.O_WRONLY | os.O_NOCTTY)
        attributes = termios.tcgetattr(self.fd)
        attributes[0] = 0
        attributes[1] = 0
        attributes[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attributes[3] = 0
        attributes[4] = _BAUDRATES[baudrate]
        attributes[5] = _BAUDRATES[baudrate]
        termios.tcsetattr(self.fd, termios.TCSANOW, attributes)

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            view = view[written:]
        termios.tcdrain(self.fd)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class PacketPublisher:
    def __init__(
        self,
        transport,
        initial_sequence: int = 0,
        class_config: str | Path = PROJECT_ROOT / "configs/classes.yaml",
    ) -> None:
        if not 0 <= initial_sequence <= 0xFFFFFFFF:
            raise ValueError("initial_sequence must fit uint32")
        names = load_yaml(class_config)["names"]
        self.class_ids = {str(name): int(class_id) for class_id, name in names.items()}
        self.transport = transport
        self.next_sequence = initial_sequence

    def packet_for(self, plan: PlanResult) -> VisionPacket:
        proposal = plan.proposal
        if proposal is None:
            return VisionPacket(0, NO_CLASS, 0, 0, 0, self.next_sequence)
        flags = FLAG_TARGET_VALID | (
            FLAG_PLAN_UPDATED if plan.updated else FLAG_LAST_KNOWN_GOOD
        )
        return VisionPacket(
            flags=flags,
            class_id=self.class_ids[proposal.class_name],
            grip_id=GRIP_IDS[proposal.grip],
            confidence_pct=round(min(1.0, max(0.0, proposal.confidence)) * 100),
            opening_permille=round(min(1.0, max(0.0, proposal.opening)) * 1000),
            sequence=self.next_sequence,
        )

    def publish(self, plan: PlanResult) -> VisionPacket:
        packet = self.packet_for(plan)
        self.transport.write(packet.encode())
        self.next_sequence = (self.next_sequence + 1) & 0xFFFFFFFF
        return packet

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()
