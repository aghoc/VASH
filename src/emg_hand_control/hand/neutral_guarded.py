"""Run ORCA's configured neutral-pose move with verified zero-current startup."""

import importlib
import json
import math
from pathlib import Path
import signal
import threading

import numpy as np

from orca_core import load_hand
from orca_core.constants import CURRENT_BASED_POSITION, MODE_MAP
from orca_core.maintenance.motor_reads import read_motor_pos_checked


def main(action=None):
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    hand = load_hand(
        config_path=str(Path(__file__).with_name('config.yaml')),
        engage_feedback=False, engage_sensors=False,
    )
    if not hand.calibrated:
        raise RuntimeError('Complete calibration is required before a neutral move')
    connected, message = hand.connect(interactive=False)
    print(message, flush=True)
    if not connected:
        raise RuntimeError(message)
    client = hand.motor_client
    registers = importlib.import_module(type(client).__module__)
    original_write = hand._set_motor_pos
    failed_release = []

    def read(mid, address, size):
        method = {1: client.packet_handler.read1ByteTxRx, 2: client.packet_handler.read2ByteTxRx, 4: client.packet_handler.read4ByteTxRx}[size]
        value, comm, error = method(client.port_handler, mid, address)
        if comm != client.dxl.COMM_SUCCESS or error:
            raise RuntimeError(f'ID {mid}, read {address}: comm={comm}, error={error}')
        return value

    def guarded_write(desired, rel_to_current=False):
        if stopped.is_set():
            raise RuntimeError('Neutral move stopped by operator')
        current = read_motor_pos_checked(hand)
        values = desired.items() if isinstance(desired, dict) else zip(hand.config.motor_ids, desired)
        targets = {}
        for mid, value in values:
            if value is None:
                continue
            target = float(value)
            if rel_to_current:
                target += current[hand.config.motor_id_to_idx_dict[mid]]
            low, high = hand.motor_limits_dict[mid]
            offset = hand._wrap_offsets_dict.get(mid, 0)
            if not math.isfinite(target) or not low + offset - .05 <= target <= high + offset + .05:
                raise RuntimeError(f'ID {mid}: neutral waypoint outside calibrated motor limits')
            targets[mid] = target
        original_write(targets, rel_to_current=False)

    try:
        if hand.config.motor_type != 'dynamixel':
            raise RuntimeError('This runner requires the inspected Dynamixel hand')
        for mid in hand.config.motor_ids:
            if read(mid, registers.ADDR_OPERATING_MODE, 1) != MODE_MAP[CURRENT_BASED_POSITION]:
                raise RuntimeError(f'ID {mid}: current-based position mode required')
            if read(mid, registers.ADDR_TORQUE_ENABLE, 1) != 0:
                raise RuntimeError(f'ID {mid}: torque must already be off')
            if read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1):
                raise RuntimeError(f'ID {mid}: hardware fault')
        hand._compute_wrap_offsets_dict()
        hand.set_max_current(0)
        for mid in hand.config.motor_ids:
            if read(mid, registers.ADDR_GOAL_CURRENT, 2) != 0:
                raise RuntimeError(f'ID {mid}: zero current failed readback')
        if stopped.is_set():
            return 130
        failed = hand.enable_torque()
        if failed:
            raise RuntimeError(f'Torque enable failed for {failed}')
        for mid in hand.config.motor_ids:
            if read(mid, registers.ADDR_GOAL_CURRENT, 2) != 0:
                raise RuntimeError(f'ID {mid}: current changed during enable')
            target = read(mid, registers.ADDR_PRESENT_POSITION, 4)
            comm, error = client.packet_handler.write4ByteTxRx(
                client.port_handler, mid, registers.ADDR_GOAL_POSITION, target,
            )
            if comm != client.dxl.COMM_SUCCESS or error:
                raise RuntimeError(f'ID {mid}: current-position target write failed')
            if read(mid, registers.ADDR_GOAL_POSITION, 4) != target:
                raise RuntimeError(f'ID {mid}: current-position target failed readback')
        for step in range(1, 11):
            if stopped.is_set():
                return 130
            hand.set_max_current(hand.config.max_current * step / 10)
            stopped.wait(.1)
        for mid in hand.config.motor_ids:
            if read(mid, registers.ADDR_GOAL_CURRENT, 2) != int(hand.config.max_current / client.current_scale_ma):
                raise RuntimeError(f'ID {mid}: current cap failed readback')
        hand._set_motor_pos = guarded_write
        if action is not None:
            return action(hand, stopped, read, registers)
        print(f'Moving to configured ORCA neutral pose: {len(hand.config.motor_ids)} motors, {hand.config.max_current}mA cap.', flush=True)
        # This is the same public motion method used by scripts/neutral.py.
        hand.set_neutral_position()
        # Capture feedback with torque still on, including on a timeout.
        arrival_error = None
        try:
            hand.wait_for_motion(timeout=5)
        except Exception as exc:
            arrival_error = exc
        actual = read_motor_pos_checked(hand)
        target = hand._joint_to_motor_pos(hand.config.neutral_position)
        error = np.abs(actual - np.asarray(target, dtype=float))
        rows = []
        for i, mid in enumerate(hand.config.motor_ids):
            rows.append({
                'motor': mid, 'joint': hand.config.motor_to_joint_dict[mid],
                'target_rad': float(target[i]), 'actual_rad': float(actual[i]),
                'motor_error_deg': float(np.rad2deg(error[i])),
                'moving_status': read(mid, registers.ADDR_MOVING_STATUS, 1),
                'goal_current_raw': read(mid, registers.ADDR_GOAL_CURRENT, 2),
                'hardware_error': read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1),
            })
        report = {
            'feedback_with_torque_enabled': True,
            'arrival_tolerance_rad': client.arrival_tolerance_rad,
            'arrival_error': str(arrival_error) if arrival_error else None,
            'motors': rows,
        }
        report_path = Path(__file__).with_name('neutral_result.json')
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        for row in rows:
            print(f"ID {row['motor']} ({row['joint']}): error={row['motor_error_deg']:.3f}deg, status={row['moving_status']}, current cap={row['goal_current_raw']}, fault={row['hardware_error']}", flush=True)
        print(f'Feedback saved: {report_path}', flush=True)
        if arrival_error is not None:
            raise arrival_error
        if not np.all(np.isfinite(error)) or np.any(error > client.arrival_tolerance_rad):
            raise RuntimeError(f'Neutral arrival not verified; max motor error={np.max(error):.4f}rad')
        for mid in hand.config.motor_ids:
            if read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1):
                raise RuntimeError(f'ID {mid}: hardware fault after move')
        print(f'NEUTRAL REACHED: motor-position error max={np.rad2deg(np.max(error)):.3f}deg.', flush=True)
        return 0
    finally:
        try:
            hand.set_max_current(0)
        except Exception:
            pass
        hand.disable_torque()
        for mid in hand.config.motor_ids:
            try:
                if read(mid, registers.ADDR_TORQUE_ENABLE, 1) != 0:
                    failed_release.append(mid)
            except Exception:
                failed_release.append(mid)
        hand.disconnect()
        print(f'RELEASED: torque off verified for {len(hand.config.motor_ids)-len(failed_release)}/{len(hand.config.motor_ids)} motors.', flush=True)
        if failed_release:
            raise RuntimeError(f'Torque release unconfirmed for {failed_release}; disconnect motor power')


if __name__ == '__main__':
    raise SystemExit(main())
