"""비전 없이 비전 제안 흉내 (11-02) — VASH UART 규격 v1 16 바이트 패킷을 UDP 로 20 Hz 전송.
emg_bridge_guarded.py --vision 127.0.0.1:9100 과 함께 시험.

  python vision_test_sender.py            → 숫자 + Enter 로 제안 바꾸기
    0 = 물체 없음, 1 = POWER(감싸 쥐기), 2 = TRIPOD(세손가락), 3 = PINCH(집기), q = 끝
"""
import socket
import struct
import sys
import threading
import time

from emg_bridge_guarded import crc16_ccitt_false

DEST = ("127.0.0.1", 9100)


def packet(grip, seq, conf=80, opening=600):
    valid = grip in (1, 2, 3, 4)
    b = bytearray(b"\xa5\x5a\x01")
    b += bytes([0b011 if valid else 0, 3 if valid else 255, grip if valid else 0, conf if valid else 0, 0])
    b += struct.pack("<HI", opening if valid else 0, seq)
    b += struct.pack("<H", crc16_ccitt_false(bytes(b)))
    return bytes(b)


def main():
    state = {"grip": 0, "run": True}

    def keys():
        names = {0: "물체 없음", 1: "POWER → 감싸 쥐기", 2: "TRIPOD → 세손가락", 3: "PINCH → 집기"}
        while state["run"]:
            line = sys.stdin.readline().strip().lower()
            if line == "q" or line == "":
                state["run"] = False; return
            if line.isdigit() and int(line) in names:
                state["grip"] = int(line); print(f"  제안: {names[int(line)]}", flush=True)
    threading.Thread(target=keys, daemon=True).start()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); seq = 0
    print(f"{DEST} 로 전송 중. 0/1/2/3 + Enter, q = 끝", flush=True)
    while state["run"]:
        seq += 1; s.sendto(packet(state["grip"], seq), DEST); time.sleep(0.05)


if __name__ == "__main__":
    main()
