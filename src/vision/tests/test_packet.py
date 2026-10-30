import os
import select

import pytest

from vagt_orca.communication.packet import (
    FLAG_PLAN_UPDATED,
    FLAG_TARGET_VALID,
    ChecksumError,
    PacketStreamDecoder,
    VisionPacket,
    crc16_ccitt,
)
from vagt_orca.communication.receiver import MockEsp32Receiver, sequence_is_newer
from vagt_orca.communication.uart import PacketPublisher, UartSender
from vagt_orca.grip.planner import PlanResult
from vagt_orca.types import GripProposal


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class MemoryTransport:
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.frames.append(data)


def proposal(name: str, grip: str, opening: float = 0.5) -> PlanResult:
    return PlanResult(GripProposal(name, grip, opening, 0.91), True, "updated")


def packet(sequence: int) -> VisionPacket:
    return VisionPacket(
        FLAG_TARGET_VALID | FLAG_PLAN_UPDATED, 0, 1, 91, 500, sequence
    )


def test_crc16_standard_vector() -> None:
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_packet_round_trip_and_checksum_rejection() -> None:
    original = packet(42)
    encoded = original.encode()
    assert len(encoded) == 16
    assert VisionPacket.decode(encoded) == original
    corrupted = bytearray(encoded)
    corrupted[9] ^= 0x01
    with pytest.raises(ChecksumError):
        VisionPacket.decode(bytes(corrupted))


def test_stream_decoder_recovers_from_noise_and_bad_crc() -> None:
    decoder = PacketStreamDecoder()
    bad = bytearray(packet(1).encode())
    bad[-1] ^= 0xFF
    frames = decoder.feed(b"noise" + bytes(bad) + packet(2).encode())
    assert frames == [packet(2)]
    assert decoder.checksum_errors == 1
    assert decoder.discarded_bytes >= 5


def test_target_change_and_lkg_change_packet_fields() -> None:
    transport = MemoryTransport()
    publisher = PacketPublisher(transport)
    bottle = publisher.publish(proposal("bottle", "POWER", 0.7))
    pen = publisher.publish(proposal("pen", "TRIPOD", 0.3))
    lkg_plan = PlanResult(proposal("pen", "TRIPOD", 0.3).proposal, False, "no_target")
    lkg = publisher.publish(lkg_plan)
    assert (bottle.class_id, bottle.grip_id) != (pen.class_id, pen.grip_id)
    assert pen.plan_updated
    assert lkg.last_known_good and not lkg.plan_updated
    assert len(transport.frames) == 3


def test_receiver_detects_loss_duplicate_order_wrap_and_timeout() -> None:
    clock = FakeClock()
    receiver = MockEsp32Receiver(timeout_ms=250, clock=clock)
    receiver.feed(packet(0xFFFFFFFE).encode())
    receiver.feed(packet(0).encode())
    assert receiver.dropped_packets == 1
    assert sequence_is_newer(0, 0xFFFFFFFF)
    receiver.feed(packet(0).encode())
    receiver.feed(packet(0xFFFFFFFF).encode())
    assert receiver.duplicate_packets == 1
    assert receiver.out_of_order_packets == 1
    assert receiver.vision_assistance_active
    clock.now = 0.251
    assert receiver.emg_only_fallback
    assert not receiver.link_fresh


def test_real_pty_uart_sender_to_mock_receiver() -> None:
    master_fd, slave_fd = os.openpty()
    slave_name = os.ttyname(slave_fd)
    os.close(slave_fd)
    receiver = MockEsp32Receiver()
    try:
        with UartSender(slave_name, 115200) as sender:
            publisher = PacketPublisher(sender, initial_sequence=77)
            expected = publisher.publish(proposal("key", "LATERAL_PINCH", 0.2))
            readable, _, _ = select.select([master_fd], [], [], 1.0)
            assert readable
            received = receiver.feed(os.read(master_fd, 1024))
        assert received == [expected]
        assert receiver.vision_assistance_active
    finally:
        os.close(master_fd)
