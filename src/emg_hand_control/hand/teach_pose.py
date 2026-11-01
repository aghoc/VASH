"""손으로 직접 자세를 만들어 저장 (토크 꺼진 상태로 읽기만 함 — 모터를 움직이지 않음).

orca_core 폴더에서:
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/teach_pose.py tripod
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/teach_pose.py open fist pinch tripod     여러 자세 차례로

동작:
  1) 연결 후 16개 모터 토크가 꺼져 있는지 확인 (켜져 있으면 끄고 다시 확인)
  2) 화면에 관절 비율(0~1)이 계속 갱신됨 → 손가락을 손으로 원하는 모양으로 만들기
  3) Enter → 그 순간 값을 0.2 s 동안 5번 읽어 평균, taught_poses.yaml 에 저장 (이전 파일은 backup/ 에 복사)
  4) emg_bridge_guarded.py 는 taught_poses.yaml 에 있는 자세를 emg_grips.yaml 보다 우선 사용 (16관절 모두 그대로, 약지·소지 예측 안 함)
주의: 줄(텐던) 구동이라 손가락을 너무 세게 꺾지 말 것. 관절 끝까지 밀면 범위 끝 값(0 또는 1)으로 저장됨
"""

import importlib
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from orca_core import load_hand

HERE = Path(__file__).resolve().parent
OUT = HERE / "taught_poses.yaml"


def main():
    names = sys.argv[1:] or ["tripod"]
    hand = load_hand(config_path=str(HERE / "config.yaml"), engage_feedback=False, engage_sensors=False)
    ok, msg = hand.connect(interactive=False)
    print(msg, flush=True)
    if not ok:
        raise SystemExit(msg)
    client = hand.motor_client
    reg = importlib.import_module(type(client).__module__)

    def read1(mid, addr):
        v, comm, err = client.packet_handler.read1ByteTxRx(client.port_handler, mid, addr)
        if comm != client.dxl.COMM_SUCCESS or err:
            raise RuntimeError(f"ID {mid} 읽기 실패")
        return v

    try:
        on = [m for m in hand.config.motor_ids if read1(m, reg.ADDR_TORQUE_ENABLE)]
        if on:
            print(f"토크 켜진 모터 {on} → 끄기", flush=True); hand.disable_torque()
            if [m for m in hand.config.motor_ids if read1(m, reg.ADDR_TORQUE_ENABLE)]:
                raise RuntimeError("토크 해제 확인 실패 — 모터 전원 확인")
        print("토크 꺼짐 확인 (16/16). 손가락을 손으로 움직여도 됩니다.", flush=True)
        roms = hand.config.joint_roms_dict; joints = list(hand.config.joint_ids)

        def frac():
            q = hand.get_joint_position()
            if not hand.last_read_ok:
                return None
            d = dict(q)
            return {j: (float(d[j]) - roms[j][0]) / (roms[j][1] - roms[j][0]) for j in joints if d.get(j) is not None}

        taught = yaml.safe_load(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        taught = taught or {}
        for name in names:
            print(f"\n=== '{name}' 자세를 손으로 만든 뒤 Enter (건너뛰려면 s + Enter) ===", flush=True)
            done = threading.Event(); key = {}

            def wait():
                key["v"] = input(); done.set()
            threading.Thread(target=wait, daemon=True).start()
            while not done.is_set():
                f = frac()
                if f:
                    line = "  ".join(f"{j.replace('_', '')[:9]}={f[j]:.2f}" for j in
                                     ("thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip", "index_mcp", "index_pip", "middle_mcp", "middle_pip"))
                    sys.stdout.write("\r" + line + "   "); sys.stdout.flush()
                time.sleep(0.15)
            if key.get("v", "").strip().lower() == "s":
                print("\n건너뜀"); continue
            samples = []
            for _ in range(5):
                f = frac()
                if f and len(f) == len(joints):
                    samples.append([f[j] for j in joints])
                time.sleep(0.04)
            if len(samples) < 3:
                print("\n읽기 실패가 많아 저장 안 함 — 다시 시도"); continue
            raw = np.mean(samples, 0)
            m = np.clip(raw, 0.0, 0.97)        # 범위 끝(1.0)까지 밀면 모터가 끝 멈춤을 계속 누르고, 범위 밖이면 안전 검사에 걸림 → 0.97 로
            edge = [j for j, v in zip(joints, raw) if v > 0.97 or v < 0.0]
            if edge:
                print(f"\n  ※ 범위 끝까지 민 관절 {edge} → 0.97/0 으로 저장. 저장 뒤 그 손가락은 손으로 살짝 되돌려 두기", flush=True)
            taught[name] = {j: round(float(v), 3) for j, v in zip(joints, m)}
            print(f"\n저장: {name} → " + ", ".join(f"{j}={taught[name][j]:.2f}" for j in joints), flush=True)
        if OUT.exists():
            (HERE / "backup").mkdir(exist_ok=True)
            shutil.copy(OUT, HERE / "backup" / f"taught_poses_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
        OUT.write_text("# teach_pose.py 로 손으로 만든 자세 (관절 ROM 비율 0~1). emg_bridge_guarded.py 가 emg_grips.yaml 보다 우선 사용\n"
                       + yaml.safe_dump(taught, sort_keys=False), encoding="utf-8")
        print(f"\n파일: {OUT}")
        return 0
    finally:
        try:
            hand.disable_torque()
        finally:
            hand.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
