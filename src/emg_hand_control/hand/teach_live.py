"""토크 켠 채로 자세 가르치기 (11-02) — 모터가 실제로 당기고 있는 상태(줄 장력·백래시가 쓸 때와 같음)에서 자세를 기록.
teach_pose.py(토크 끔)로 만든 집기·세손가락이 재생하면 손끝이 매번 틀어지던 문제 대응.

orca_core 폴더에서:
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/teach_live.py pinch
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/teach_live.py tripod --soft 50

진행:
  1) 팀 기동 절차(0 전류 → 토크 켬 → 한계 검사)로 시작, 지금 저장된 그 자세로 천천히 이동
  2) 전류 상한을 --soft (기본 60 mA) 로 낮춤 → 손가락을 손으로 밀면 모터가 약하게 버티며 따라옴 (민 자리에 머묾)
     화면: 관절 비율. 손끝이 맞닿도록 만든 뒤
       Enter        → 저장 (0.2 s 평균)
       r + Enter    → 대기 자세로 갔다가 다시 이 자세로 (재현 확인. 매번 같은 자리에서 만나는지 보기)
       q + Enter    → 저장 안 하고 끝
  3) 저장 = taught_poses.yaml (이전 파일은 backup/). 끝나면 토크 해제 확인
주의: 모터가 켜진 상태. 손가락을 범위 끝까지 세게 밀지 말 것. 이상하면 Ctrl+C (토크 해제)
"""

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from neutral_guarded import main as run_guarded
from emg_bridge_guarded import build_poses
import emg_bridge_guarded as B

HERE = Path(__file__).resolve().parent
OUT = HERE / "taught_poses.yaml"
ARGS = None


def run(hand, stopped, read, registers):
    B.ARGS = argparse.Namespace()                              # build_poses 는 ARGS 를 쓰지 않음 (안전용)
    poses, _ = build_poses(hand, str(HERE / "emg_grips.yaml"))
    name = ARGS.name
    roms = hand.config.joint_roms_dict; joints = list(hand.config.joint_ids)
    start = poses.get(name, poses["relax"])
    hand.set_max_current(float(ARGS.hold))
    print(f"'{name}' 지금 저장된 자세로 이동", flush=True)
    hand.set_joint_positions(poses["relax"], num_steps=40, step_size=0.02)
    hand.set_joint_positions(start, num_steps=60, step_size=0.02)
    goal = dict(start)
    hand.set_max_current(float(ARGS.soft))
    print(f"전류 {ARGS.soft:g} mA — 손가락을 밀면 따라옵니다. Enter = 저장, r + Enter = 재현 확인, q + Enter = 끝", flush=True)

    def actual():
        r = hand.get_motor_state()
        return hand._motor_to_joint_pos(r.position)

    cmd = {}
    def keys():
        while not stopped.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":
                return
            cmd["v"] = line.strip().lower()
    threading.Thread(target=keys, daemon=True).start()

    taught = (yaml.safe_load(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}) or {}
    while not stopped.is_set():
        a = actual()
        moved = False
        for j in joints:                                       # 민 관절만 목표를 실제 위치로 (데드밴드 1.5°) → 손을 떼면 그 자리에 머묾
            if a.get(j) is not None and abs(a[j] - goal[j]) > ARGS.deadband:
                goal[j] = float(a[j]); moved = True
        if moved:
            hand.set_joint_positions(goal)
        f = {j: (goal[j] - roms[j][0]) / (roms[j][1] - roms[j][0]) for j in joints}
        sys.stdout.write("\r" + "  ".join(f"{j.replace('_', '')[:9]}={f[j]:.2f}" for j in
                         ("thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip", "index_mcp", "index_pip", "middle_mcp", "middle_pip")) + "   ")
        sys.stdout.flush()
        v = cmd.pop("v", None)
        if v == "q":
            print("\n저장 안 하고 끝"); return 0
        if v == "r":
            print("\n재현 확인: 대기 → 이 자세", flush=True)
            hand.set_max_current(float(ARGS.hold))
            keep = dict(goal)
            hand.set_joint_positions(poses["relax"], num_steps=40, step_size=0.02); stopped.wait(0.5)
            hand.set_joint_positions(keep, num_steps=60, step_size=0.02); stopped.wait(0.5)
            goal = keep; hand.set_max_current(float(ARGS.soft))
            print("다시 조정 가능. Enter = 저장", flush=True)
        elif v is not None:                                    # Enter → 저장
            samples = []
            for _ in range(5):
                a = actual()
                if all(a.get(j) is not None for j in joints):
                    samples.append([a[j] for j in joints])
                time.sleep(0.04)
            q = np.mean(samples, 0) if samples else np.array([goal[j] for j in joints])
            frac = {j: float(np.clip((q[i] - roms[j][0]) / (roms[j][1] - roms[j][0]), 0.0, 0.97)) for i, j in enumerate(joints)}
            taught[name] = {j: round(v_, 3) for j, v_ in frac.items()}
            if OUT.exists():
                (HERE / "backup").mkdir(exist_ok=True)
                shutil.copy(OUT, HERE / "backup" / f"taught_poses_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
            OUT.write_text("# teach_pose.py / teach_live.py 로 손으로 만든 자세 (관절 ROM 비율 0~1). emg_bridge_guarded.py 가 emg_grips.yaml 보다 우선 사용\n"
                           + yaml.safe_dump(taught, sort_keys=False), encoding="utf-8")
            print(f"\n저장: {name} (토크 켠 상태) → {OUT.name}", flush=True)
            hand.set_max_current(float(ARGS.hold))
            hand.set_joint_positions(poses["relax"], num_steps=50, step_size=0.02)
            return 0
        stopped.wait(0.05)
    return 130


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="자세 이름 (pinch, tripod, wrap 등 emg_grips.yaml 의 이름)")
    ap.add_argument("--soft", type=float, default=60.0, help="가르치는 동안 전류 상한 mA (작을수록 손으로 밀기 쉬움)")
    ap.add_argument("--hold", type=float, default=200.0, help="이동·재현 확인 때 전류 상한 mA")
    ap.add_argument("--deadband", type=float, default=1.5, help="이 각도(°) 이상 밀린 관절만 목표를 따라 옮김")
    ARGS = ap.parse_args()
    return run_guarded(run)


if __name__ == "__main__":
    raise SystemExit(main())
