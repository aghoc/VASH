"""Hold the assembled XC330 hand for manual spool adjustment, without a sweep.

Run with the project's Python environment. Inspection is read-only unless
--hold is supplied. No operating-mode, EEPROM, or calibration writes are made.
"""

import argparse
import importlib
import math
from pathlib import Path
import select
import signal
import sys
import threading
import time

import dynamixel_sdk as sdk

from orca_core import detect_hand
from orca_core.hand_config import OrcaHandConfig
from orca_core.hardware.motor_factory import motor_client_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hold', action='store_true')
    parser.add_argument('--duration', type=float, default=600)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive and finite')
    stopping = threading.Event()
    # Let an in-flight SDK transaction finish before releasing torque.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())

    config = OrcaHandConfig.from_config_path(str(Path(__file__).with_name('config.yaml')))
    detected = detect_hand()
    if detected.motor_type != 'dynamixel' or not detected.motor_port:
        raise RuntimeError('U2D2(모터 연결)를 찾지 못했어요 — USB·전원 확인. 모터는 움직이지 않았어요')
    driver = motor_client_class(detected.motor_type)
    registers = importlib.import_module(driver.__module__)
    config = config.with_family_currents(driver)
    current = int(config.max_current)
    port = sdk.PortHandler(detected.motor_port)
    packet = sdk.PacketHandler(registers.PROTOCOL_VERSION)
    touched = False
    originals = {}

    def read(mid, address, size):
        method = {1: packet.read1ByteTxRx, 2: packet.read2ByteTxRx, 4: packet.read4ByteTxRx}[size]
        value, comm, error = method(port, mid, address)
        if comm != sdk.COMM_SUCCESS or error:
            raise RuntimeError(f'ID {mid}, read {address}: comm={comm}, error={error}')
        return value

    def write(mid, address, size, value):
        method = {1: packet.write1ByteTxRx, 2: packet.write2ByteTxRx, 4: packet.write4ByteTxRx}[size]
        comm, error = method(port, mid, address, value & ((1 << (8 * size)) - 1))
        if comm != sdk.COMM_SUCCESS or error:
            raise RuntimeError(f'ID {mid}, write {address}: comm={comm}, error={error}')
        if read(mid, address, size) != value & ((1 << (8 * size)) - 1):
            raise RuntimeError(f'ID {mid}, register {address} failed readback')

    def position(mid):
        value = read(mid, registers.ADDR_PRESENT_POSITION, 4)
        return value - 2**32 if value >= 2**31 else value

    try:
        if not port.openPort() or not port.setBaudRate(detected.motor_baudrate):
            raise RuntimeError('Cannot open motor bus')
        for mid in config.motor_ids:
            model, comm, error = packet.ping(port, mid)
            if comm != sdk.COMM_SUCCESS or error or model != 1220:
                raise RuntimeError(f'ID {mid}: expected XC330-T288-T, got {model}; comm={comm}, error={error}')
            if read(mid, registers.ADDR_OPERATING_MODE, 1) != 5:
                raise RuntimeError(f'ID {mid}: current-based position mode required; mode left unchanged')
            if read(mid, registers.ADDR_TORQUE_ENABLE, 1) != 0:
                raise RuntimeError(f'ID {mid}: 다른 프로그램이 모터를 쓰고 있어요 (토크가 켜져 있음) — 터미널의 다른 프로그램을 먼저 끄세요')
            if read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1):
                raise RuntimeError(f'ID {mid}: hardware fault')
            # XC330 Current Limit(38) is read only; Goal Current is RAM.
            if read(mid, 38, 2) < current:
                raise RuntimeError(f'ID {mid}: configured current exceeds hardware ceiling')
            originals[mid] = read(mid, registers.ADDR_GOAL_CURRENT, 2)
            print(f'ID {mid}: 준비됨, 위치={position(mid)}, 온도={read(mid, registers.ADDR_PRESENT_TEMPERATURE, 1)}C', flush=True)
        if not args.hold:
            print('상태 확인만 끝났어요. 모터는 움직이지 않았어요. (고정하려면 --hold)', flush=True)
            return 0

        touched = True
        for mid in config.motor_ids:
            write(mid, registers.ADDR_GOAL_CURRENT, 2, 0)
        for mid in config.motor_ids:
            if stopping.is_set():
                return 0
            write(mid, registers.ADDR_TORQUE_ENABLE, 1, 1)
            if read(mid, registers.ADDR_GOAL_CURRENT, 2) != 0:
                raise RuntimeError(f'ID {mid}: current changed during torque enable')
        # This hand refreshes Goal Position while torque is off. Keep the
        # current at zero until all post-enable targets are verified.
        targets = {mid: position(mid) for mid in config.motor_ids}
        for mid, target in targets.items():
            write(mid, registers.ADDR_GOAL_POSITION, 4, target)
        for step in range(1, 11):
            if stopping.is_set():
                return 0
            for mid in config.motor_ids:
                write(mid, registers.ADDR_GOAL_CURRENT, 2, current * step // 10)
            stopping.wait(0.1)
        print(f'고정 중: 모터 {len(targets)}개를 지금 위치에 고정했어요 (전류 상한 {current}mA). 스풀을 조이세요.\n'
              f'  ※ 다 조인 뒤에 아무것도 쓰지 말고 Enter 만 누르면 풀려요 (다른 글자는 무시). {args.duration:g}초 뒤 자동 해제.', flush=True)
        deadline = time.monotonic() + args.duration
        entered = threading.Event()          # Windows 에서는 select 로 stdin 을 볼 수 없음 → Enter 를 별도 스레드에서 기다림 (10-30 수정)

        def wait_enter():                    # 빈 Enter 만 해제 — 다음 명령을 붙여 넣어도 풀리지 않게 (10-31)
            while True:
                line = sys.stdin.readline()
                if line == "" or line.strip() == "":
                    entered.set(); return
                print(f"  ※ 입력 '{line.strip()[:40]}' 은(는) 무시했어요. 조이기가 끝나면 아무것도 쓰지 말고 Enter 만 누르세요.", flush=True)
        threading.Thread(target=wait_enter, daemon=True).start()
        while not stopping.is_set() and time.monotonic() < deadline:
            if entered.is_set():
                break
            for mid in config.motor_ids:
                if read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1):
                    raise RuntimeError(f'ID {mid}: 고정 중 모터 하드웨어 오류')
                if abs(position(mid) - targets[mid]) > 114:
                    raise RuntimeError(f'ID {mid}: 고정 중 모터가 10도 넘게 돌아서 안전하게 풀었어요 (스풀 돌릴 때 모터 축을 잡아 주세요)')
            stopping.wait(0.5)
        return 0
    finally:
        failed = []
        if touched:
            for mid in config.motor_ids:
                released = False
                for _ in range(3):
                    try:
                        write(mid, registers.ADDR_GOAL_CURRENT, 2, 0)
                    except Exception:
                        pass
                    try:
                        write(mid, registers.ADDR_TORQUE_ENABLE, 1, 0)
                        released = True
                        break
                    except Exception:
                        time.sleep(0.05)
                if not released:
                    failed.append(mid)
                else:
                    try:
                        write(mid, registers.ADDR_GOAL_CURRENT, 2, originals[mid])
                    except Exception as exc:
                        print(f'ID {mid}: torque off, current restore failed: {exc}', flush=True)
            print(f'풀림: 모터 {len(config.motor_ids) - len(failed)}/{len(config.motor_ids)}개 힘 풀림 확인 (이제 다음 명령을 입력하세요)', flush=True)
        port.closePort()
        if failed:
            raise RuntimeError(f'모터 {failed} 힘 풀림을 확인하지 못했어요 — 모터 전원을 끄세요')


if __name__ == '__main__':
    raise SystemExit(main())
