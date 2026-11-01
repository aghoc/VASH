"""Bounded cycles of the original ORCA stress-test poses with guarded startup."""

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import orca_core

from neutral_guarded import main as run_guarded
from orca_core.maintenance.motor_reads import read_motor_pos_checked


def original_stress_poses():
    path = Path(orca_core.__file__).resolve().parents[1] / 'scripts' / 'stress_test.py'
    spec = importlib.util.spec_from_file_location('original_orca_stress_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OPEN_FRACTIONS, module.CLOSE_FRACTIONS


def run_stress(hand, stopped, read, registers, *, cycles, num_steps, step_size, hold,
               poses=None, label='Stress test', report_name='stress_result.json'):
    if poses is None:
        open_fractions, close_fractions = original_stress_poses()
        poses = [('OPEN', hand.pose_from_fractions(open_fractions)),
                 ('CLOSE', hand.pose_from_fractions(close_fractions))]
    report = {'requested_cycles': cycles, 'completed_cycles': 0,
              'sequence': label,
              'current_cap_ma': hand.config.max_current,
              'max_temperature_c': hand.motor_client.max_operating_temp_c,
              'peak_temperature_c': None, 'poses': [], 'completed': False}
    next_check = 0
    temperatures = {}
    guarded_write = hand._set_motor_pos

    def health(force=False):
        nonlocal next_check, temperatures
        if stopped.is_set():
            raise RuntimeError('Stress test stopped by operator')
        if not force and time.monotonic() < next_check:
            return
        temperatures = {}
        for mid in hand.config.motor_ids:
            temperatures[mid] = read(mid, registers.ADDR_PRESENT_TEMPERATURE, 1)
            fault = read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1)
            if fault:
                raise RuntimeError(f'ID {mid}: hardware fault {fault}')
        peak = max(temperatures.values())
        report['peak_temperature_c'] = max(peak, report['peak_temperature_c'] or peak)
        if peak >= hand.motor_client.max_operating_temp_c:
            raise RuntimeError(f'Motor temperature reached operating limit: {peak}C')
        next_check = time.monotonic() + 2

    def checked_write(*args, **kwargs):
        health()
        return guarded_write(*args, **kwargs)

    hand._set_motor_pos = checked_write
    try:
        health(force=True)
        print(f'{label}: {cycles} cycle(s), {len(poses)} poses per cycle; {hand.config.max_current}mA cap; temperature cutoff={hand.motor_client.max_operating_temp_c:g}C.', flush=True)
        for cycle in range(1, cycles + 1):
            for name, pose in poses:
                health(force=True)
                print(f'Cycle {cycle}/{cycles}: {name}', flush=True)
                hand.set_joint_positions(pose, num_steps=num_steps, step_size=step_size)
                hold_deadline = time.monotonic() + hold
                while time.monotonic() < hold_deadline:
                    if stopped.wait(min(.25, max(0, hold_deadline - time.monotonic()))):
                        raise RuntimeError('Motion test stopped by operator')
                    health()
                health(force=True)
                actual = read_motor_pos_checked(hand)
                target = np.asarray(hand._joint_to_motor_pos(pose.as_dict()), dtype=float)
                errors = np.rad2deg(np.abs(actual - target))
                if not np.all(np.isfinite(errors)):
                    raise RuntimeError('Non-finite motor feedback')
                report['poses'].append({
                    'cycle': cycle, 'pose': name,
                    'temperatures_c': temperatures.copy(),
                    'motor_positions_rad': dict(zip(hand.config.motor_ids, map(float, actual))),
                    'motor_targets_rad': dict(zip(hand.config.motor_ids, map(float, target))),
                    'motor_errors_deg': dict(zip(hand.config.motor_ids, map(float, errors))),
                })
                print(f'  peak temperature={max(temperatures.values())}C; largest motor position error={max(errors):.2f}deg', flush=True)
            report['completed_cycles'] = cycle
        report['completed'] = True
        print(f'COMPLETE: {label}, {cycles}/{cycles} cycle(s); peak temperature={report["peak_temperature_c"]}C.', flush=True)
        return 0
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        path = Path(__file__).with_name(report_name)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Results saved: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--step-size', type=float, default=.02)
    parser.add_argument('--hold', type=float, default=2)
    args = parser.parse_args()
    if args.cycles < 1 or args.num_steps < 2 or not np.isfinite(args.step_size) or args.step_size <= 0 or not np.isfinite(args.hold) or args.hold < 0:
        parser.error('Positive cycles/step size, at least two steps, and a non-negative hold are required')
    return run_guarded(lambda hand, stopped, read, registers: run_stress(
        hand, stopped, read, registers, cycles=args.cycles,
        num_steps=args.num_steps, step_size=args.step_size, hold=args.hold,
    ))


if __name__ == '__main__':
    raise SystemExit(main())
