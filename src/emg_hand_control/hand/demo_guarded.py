"""Play the original packaged ORCA demos with guarded startup and monitoring."""

import argparse
import math

from neutral_guarded import main as run_guarded
from stress_guarded import run_stress
from orca_core.demo_poses import load_demo_poses
from orca_core.joint_position import OrcaJointPositions


def demo_sequence(hand, names):
    demos = load_demo_poses()
    poses = []
    for name in names:
        if name == 'pinch':
            demo = demos['main']
            for pose_name in ('open_hand', 'pinch', 'open_hand'):
                poses.append((f'pinch:{pose_name}', hand.pose_from_fractions(demo.pose_fractions[pose_name])))
            poses.append(('pinch:neutral', OrcaJointPositions.from_dict(hand.config.neutral_position)))
            continue
        demo = demos[name]
        for pose_name in demo.sequence:
            poses.append((f'{name}:{pose_name}', hand.pose_from_fractions(demo.pose_fractions[pose_name])))
        poses.append((f'{name}:neutral', OrcaJointPositions.from_dict(hand.config.neutral_position)))
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', choices=['main', 'abduction', 'both', 'pinch'], default='both')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--step-size', type=float, default=.02)
    parser.add_argument('--hold', type=float, default=2)
    args = parser.parse_args()
    if args.cycles < 1 or args.num_steps < 2 or not math.isfinite(args.step_size) or args.step_size <= 0 or not math.isfinite(args.hold) or args.hold < 0:
        parser.error('Positive cycles/step size, at least two steps, and a non-negative hold are required')
    names = ['main', 'abduction'] if args.demo == 'both' else [args.demo]
    return run_guarded(lambda hand, stopped, read, registers: run_stress(
        hand, stopped, read, registers, cycles=args.cycles, num_steps=args.num_steps, step_size=args.step_size,
        hold=args.hold, poses=demo_sequence(hand, names),
        label='Original ORCA demo poses: ' + ', '.join(names),
        report_name='pinch_result.json' if args.demo == 'pinch' else 'demo_result.json',
    ))


if __name__ == '__main__':
    raise SystemExit(main())
