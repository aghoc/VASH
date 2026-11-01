"""물체 접촉 측정 — 허공 vs 물체를 쥘 때 손가락별 모터 전류·위치를 기록 (전류로 '닿음'을 감지할 수 있는지 확인용)

orca_core 폴더에서:
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/contact_probe.py --pose fist
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/contact_probe.py --pose pinch --cap 250

진행 (반복):
  1) 손이 대기 자세(relax)로 감 → 화면에 '시도 이름' 입력 (예: 공기, 스펀지, 종이컵, 물병) 후 Enter.  q + Enter = 끝
  2) 물체를 손 안에 대고 있으면, 손이 기준점 경로를 따라 3 s 동안 천천히 닫힘 → 2 s 유지 (목표는 자세보다 조금 더 안쪽 = 닿으면 누름)
  3) 다시 대기 자세로 열림. 매 주기(50 Hz) 모터 16 개의 명령 위치·실제 위치·속도·전류를 CSV 로 저장
저장: hand-config/right-no-wrist/probe/<자세>_<이름>_<시각>.csv
팀 기동 절차(neutral_guarded: 0 전류 기동·한계 검사·종료 시 토크 해제) 그대로 사용. 전류 상한 --cap (기본 300 mA = 팀 설정)
"""

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import yaml

from neutral_guarded import main as run_guarded
from emg_bridge_guarded import build_poses, pose_at

HERE = Path(__file__).resolve().parent
ARGS = None
# 물체 목록 (화면 표시 한글, 파일 이름 영어)
OBJECTS = {
    "wrap":     [("공기(빈손)", "air"), ("빈 종이컵", "cup"), ("빈 페트병", "bottle_empty"), ("물 든 페트병", "bottle_full")],
    "cylinder": [("공기(빈손)", "air"), ("빈 종이컵", "cup"), ("빈 페트병", "bottle_empty"), ("물 든 페트병", "bottle_full")],
    "fist":     [("공기(빈손)", "air"), ("빈 종이컵", "cup"), ("빈 페트병", "bottle_empty"), ("물 든 페트병", "bottle_full")],
    "pinch":    [("공기(빈손)", "air"), ("작은 물체", "small")],
    "tripod":   [("공기(빈손)", "air"), ("작은 물체", "small")],
}


def run(hand, stopped, read, registers):
    poses, _ = build_poses(hand, str(HERE / "emg_grips.yaml"))
    key = ARGS.pose; o = poses["relax"]; g = poses[key]; roms = poses["_roms"]
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    m2j = {abs(v): (k, 1 if v > 0 else -1) for k, v in cfg["joint_to_motor_map"].items()}
    mids = list(hand.config.motor_ids)
    client = hand.motor_client
    out_dir = HERE / "probe"; out_dir.mkdir(exist_ok=True)
    hand.set_max_current(float(ARGS.cap))
    dt = 1.0 / 50

    def target(c):
        q = {}
        for j in o:
            q[j] = pose_at(poses, key, j, min(c, 1.0)) + max(0.0, c - 1.0) * (g[j] - o[j])   # c > 1 = 자세보다 더 안쪽 (누르기)
            q[j] = min(max(q[j], roms[j][0]), roms[j][1])
        return q

    print("대기 자세로 이동", flush=True)
    hand.set_joint_positions(o, num_steps=50, step_size=0.02)
    while not stopped.is_set():
        menu = OBJECTS.get(key, OBJECTS["pinch"])
        print(f"\n[{key}] 물체 번호를 입력하고 Enter: " + "  ".join(f"{i + 1}={ko}" for i, (ko, _) in enumerate(menu)) + "  q=끝", flush=True)
        raw = input("번호 → ").strip()                       # 한글 입력이 빈 값으로 들어오는 Windows 콘솔 문제 → 번호로 고름
        if raw.lower() == "q" or stopped.is_set():
            break
        if raw.isdigit() and 1 <= int(raw) <= len(menu):
            ko, name = menu[int(raw) - 1]
        elif raw:
            ko = name = raw
        else:
            print("  번호가 입력되지 않았어요. 다시 입력해 주세요.", flush=True); continue
        print(f"  선택: {ko}", flush=True)
        print("  3 s 동안 닫힘 → 2 s 유지 → 열림 ...", flush=True)
        rows = []; t0 = time.monotonic(); temp_i = 0
        T_close, T_hold = ARGS.close_s, ARGS.hold_s
        while not stopped.is_set():
            t = time.monotonic() - t0
            if t > T_close + T_hold:
                break
            c = min(1.0, t / T_close) + (ARGS.squeeze if t > T_close else ARGS.squeeze * max(0.0, (t / T_close - 0.8) / 0.2))
            q = target(c)
            hand.set_joint_positions(q)
            r = client.read_position_velocity_current()
            row = {"t": round(t, 3), "c": round(c, 3)}
            for k, mid in enumerate(mids):
                j, sgn = m2j.get(mid, (str(mid), 1))
                row[f"{j}_cmd"] = round(q.get(j, float("nan")), 2)
                row[f"{j}_pos"] = round(float(r.position[k]), 4)
                row[f"{j}_vel"] = round(float(r.velocity[k]), 4)
                row[f"{j}_cur"] = round(float(r.current[k]), 1)
            rows.append(row)
            if len(rows) % 25 == 0:                       # 0.5 s 마다 모터 하나 온도·오류
                mid = mids[temp_i % len(mids)]; temp_i += 1
                if read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1):
                    raise RuntimeError(f"ID {mid}: hardware fault")
                if read(mid, registers.ADDR_PRESENT_TEMPERATURE, 1) >= hand.motor_client.max_operating_temp_c:
                    raise RuntimeError(f"ID {mid}: 온도 한계")
            stopped.wait(max(0.0, dt - (time.monotonic() - t0 - t)))
        hand.set_joint_positions(o, num_steps=40, step_size=0.02)
        fn = out_dir / f"{key}_{name}_{time.strftime('%H%M%S')}.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        peak = {j: max(abs(rw[f'{j}_cur']) for rw in rows) for j in ("thumb_mcp", "index_mcp", "middle_mcp", "ring_mcp")}
        print(f"  저장 {fn.name}  ({len(rows)} 줄)  최대 전류 " + ", ".join(f"{j.split('_')[0]} {v:.0f}" for j, v in peak.items()) + " mA", flush=True)
    try:
        hand.set_max_current(float(hand.config.max_current))
    except Exception:
        pass
    return 0


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="wrap", choices=["wrap", "pinch", "tripod", "cylinder", "fist"])
    ap.add_argument("--cap", type=float, default=300.0, help="전류 상한 mA (팀 설정 300)")
    ap.add_argument("--close-s", type=float, default=3.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--squeeze", type=float, default=0.15, help="유지 때 자세보다 더 안쪽 목표 (비율)")
    ARGS = ap.parse_args()
    return run_guarded(run)


if __name__ == "__main__":
    raise SystemExit(main())
