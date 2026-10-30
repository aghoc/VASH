from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"\xA5\x5A"
VERSION = 1
PACKET_SIZE = 16
NO_CLASS = 0xFF

FLAG_TARGET_VALID = 0x01
FLAG_PLAN_UPDATED = 0x02
FLAG_LAST_KNOWN_GOOD = 0x04
KNOWN_FLAGS = FLAG_TARGET_VALID | FLAG_PLAN_UPDATED | FLAG_LAST_KNOWN_GOOD

GRIP_IDS = {"POWER": 1, "TRIPOD": 2, "PINCH": 3, "LATERAL_PINCH": 4}
GRIP_NAMES = {value: key for key, value in GRIP_IDS.items()}

_PAYLOAD = struct.Struct("<2sBBBBBBHI")
_FRAME = struct.Struct("<2sBBBBBBHIH")


class PacketError(ValueError):
    pass


class ChecksumError(PacketError):
    pass


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True, slots=True)
class VisionPacket:
    flags: int
    class_id: int
    grip_id: int
    confidence_pct: int
    opening_permille: int
    sequence: int

    @property
    def target_valid(self) -> bool:
        return bool(self.flags & FLAG_TARGET_VALID)

    @property
    def plan_updated(self) -> bool:
        return bool(self.flags & FLAG_PLAN_UPDATED)

    @property
    def last_known_good(self) -> bool:
        return bool(self.flags & FLAG_LAST_KNOWN_GOOD)

    def _validate(self) -> None:
        if self.flags & ~KNOWN_FLAGS:
            raise PacketError(f"Unknown flags: 0x{self.flags:02x}")
        if not 0 <= self.sequence <= 0xFFFFFFFF:
            raise PacketError("sequence must fit uint32")
        if not 0 <= self.confidence_pct <= 100:
            raise PacketError("confidence_pct must be in [0, 100]")
        if not 0 <= self.opening_permille <= 1000:
            raise PacketError("opening_permille must be in [0, 1000]")
        if self.target_valid:
            if not 0 <= self.class_id <= 14:
                raise PacketError("valid target class_id must be in [0, 14]")
            if self.grip_id not in GRIP_NAMES:
                raise PacketError("valid target grip_id must be in [1, 4]")
            if self.plan_updated == self.last_known_good:
                raise PacketError("valid target must be exactly one of updated or LKG")
        elif any((self.class_id != NO_CLASS, self.grip_id, self.confidence_pct, self.opening_permille)):
            raise PacketError("no-target packet must carry neutral class/grip/confidence/opening")

    def encode(self) -> bytes:
        self._validate()
        payload = _PAYLOAD.pack(
            MAGIC,
            VERSION,
            self.flags,
            self.class_id,
            self.grip_id,
            self.confidence_pct,
            0,
            self.opening_permille,
            self.sequence,
        )
        return payload + struct.pack("<H", crc16_ccitt(payload))

    @classmethod
    def decode(cls, frame: bytes) -> VisionPacket:
        if len(frame) != PACKET_SIZE:
            raise PacketError(f"Expected {PACKET_SIZE} bytes, got {len(frame)}")
        (
            magic,
            version,
            flags,
            class_id,
            grip_id,
            confidence_pct,
            reserved,
            opening_permille,
            sequence,
            checksum,
        ) = _FRAME.unpack(frame)
        if magic != MAGIC:
            raise PacketError("Invalid packet magic")
        if version != VERSION:
            raise PacketError(f"Unsupported protocol version: {version}")
        if reserved != 0:
            raise PacketError("Reserved byte must be zero")
        expected = crc16_ccitt(frame[:-2])
        if checksum != expected:
            raise ChecksumError(f"CRC mismatch: expected 0x{expected:04x}, got 0x{checksum:04x}")
        packet = cls(flags, class_id, grip_id, confidence_pct, opening_permille, sequence)
        packet._validate()
        return packet


class PacketStreamDecoder:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.checksum_errors = 0
        self.frame_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[VisionPacket]:
        self.buffer.extend(data)
        packets: list[VisionPacket] = []
        while True:
            position = self.buffer.find(MAGIC)
            if position < 0:
                keep = 1 if self.buffer.endswith(MAGIC[:1]) else 0
                self.discarded_bytes += len(self.buffer) - keep
                del self.buffer[: len(self.buffer) - keep]
                break
            if position:
                self.discarded_bytes += position
                del self.buffer[:position]
            if len(self.buffer) < PACKET_SIZE:
                break
            frame = bytes(self.buffer[:PACKET_SIZE])
            try:
                packet = VisionPacket.decode(frame)
            except ChecksumError:
                self.checksum_errors += 1
                del self.buffer[0]
                continue
            except PacketError:
                self.frame_errors += 1
                del self.buffer[0]
                continue
            packets.append(packet)
            del self.buffer[:PACKET_SIZE]
        return packets
