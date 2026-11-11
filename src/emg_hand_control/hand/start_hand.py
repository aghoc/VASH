"""의수 시작 절차: 상태 확인 → 캘리브레이션 파일 검사 → 중립 → 손 제어

  cd C:\\ads_bringup\\hand\\VGAT_ORCA\\orca_core
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/start_hand.py [emg_bridge_guarded.py 옵션]

지금 기본은 캘리브레이션을 다시 하지 않음. 저장된 값의 이동량이 기준에서 벗어나면 손을 움직이기 전에 멈춤.
  --calibrate        장력 → 캘리브레이션까지 실행 (순서는 ORCA 문서 initial-tensioning-and-calibration.md)
  --skip-tension     --calibrate 때 장력 단계 건너뜀
  --update-baseline  줄을 새로 달아 기준이 바뀌었을 때: 지금 캘리브레이션 값을 새 기준으로 저장
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import yaml

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "calibration.yaml"
BASELINE = HERE / "travel_baseline.yaml"
NEUTRAL_LIMIT_DEG = 3.0                  # 중립 도착 한도 최소값 (모터 각도)
NEUTRAL_LIMIT_FRAC = 0.10                # 11-09: 모터별 한도 = 캘리브레이션 범위의 비율. 11-11: 0.08→0.10 (ID 14 가 4.18° 로 한도 4.16° 를 넘어 시작 실패 → 5.0°)
NEUTRAL_LIMIT_MAX_DEG = 5.0              #   단 5° 넘게는 안 늘림 (범위 큰 MCP 133° → 5°)
TRAVEL_MARGIN = 0.25                     # 11-02: 줄이 풀린 ID 14 는 평소 45~62° 가 8° 로 측정됨. 기준은 줄 교체 뒤 값 (travel_baseline.yaml)


STOP = threading.Event()                 # 통합 앱 정지(SIGBREAK): 진행 중인 단계가 안전하게 끝난 뒤 다음 단계로 안 넘어감


def run_step(title, *cmd):
    if STOP.is_set():
        return 130
    print(f"\n===== {title} =====", flush=True)
    code = subprocess.call([sys.executable, *map(str, cmd)], cwd=Path.cwd())
    return 130 if STOP.is_set() else code


def travel_problems(measured, baseline, margin=TRAVEL_MARGIN):
    """기준에서 margin 넘게 벗어난 관절 설명 목록. 빠진 관절도 문제로 봄"""
    out = []
    for joint, base in baseline.items():
        value = measured.get(joint)
        if value is None:
            out.append(f"{joint}: 측정값 없음 (기준 {base:.1f}°)")
        elif abs(value - base) > margin * base:
            out.append(f"{joint}: {value:.1f}° (기준 {base:.1f}°, 허용 ±{margin * 100:.0f} %)")
    return out


def measured_travel(path=CALIBRATION):
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("motor_travel_measured") or {}


def calibrate_and_check(update_baseline):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    previous = HERE / "backup" / f"calibration-before-start-{stamp}.yaml"
    shutil.copy2(CALIBRATION, previous)
    code = run_step("캘리브레이션 (관절마다 양쪽 끝까지 움직임, 손 주변을 비워 두세요)", HERE / "calibrate_guarded.py")
    if code:
        print(f"캘리브레이션 실패 (코드 {code}). 이전 값으로 되돌립니다.", flush=True)
        shutil.copy2(previous, CALIBRATION)
        return code
    measured = measured_travel()
    if update_baseline or not BASELINE.exists():
        BASELINE.write_text(yaml.safe_dump(measured, allow_unicode=True, sort_keys=True), encoding="utf-8")
        print(f"이번 측정값을 이동량 기준으로 저장: {BASELINE.name}", flush=True)
        return 0
    problems = travel_problems(measured, yaml.safe_load(BASELINE.read_text(encoding="utf-8")))
    if problems:
        failed = HERE / "backup" / f"calibration-failed-{stamp}.yaml"
        shutil.copy2(CALIBRATION, failed)
        shutil.copy2(previous, CALIBRATION)
        print("이동량 이상 — 줄 풀림·끊어짐을 확인하세요:", *problems, sep="\n  ", flush=True)
        print(f"이번 값은 {failed.name} 에 두고 이전 캘리브레이션으로 되돌렸습니다.\n"
              "줄을 새로 달아 기준이 바뀐 것이 맞으면 --update-baseline 으로 다시 실행하세요.", flush=True)
        return 1
    print("이동량 정상 (16 관절 기준 이내)", flush=True)
    return 0


def neutral_close_enough(since, limit_deg=NEUTRAL_LIMIT_DEG):
    """neutral_guarded 의 도착 허용치(약 1.1°)는 줄 장력에 따라 살짝 못 미칠 수 있음. 이번 결과가 오류 없이 limit_deg 안이면 계속 진행"""
    path = HERE / "neutral_result.json"
    if not path.exists() or path.stat().st_mtime < since:
        return False
    rows = json.loads(path.read_text())["motors"]
    lim = neutral_limits(limit_deg)
    faults = [r["motor"] for r in rows if r["hardware_error"]]
    far = [f"ID {r['motor']} {r['motor_error_deg']:.1f}° (한도 {lim.get(r['motor'], limit_deg):.1f}°)" for r in rows
           if r["motor_error_deg"] > lim.get(r["motor"], limit_deg)]
    if faults or far:
        print(f"중립 도착 실패 — 하드웨어 오류 {faults}, 많이 벗어남 {far}. 멈춥니다.", flush=True)
        return False
    worst = max(rows, key=lambda r: r["motor_error_deg"] / lim.get(r["motor"], limit_deg))
    print(f"※ 위 시간 초과 오류는 허용치를 조금 넘은 것입니다: 가장 큰 ID {worst['motor']} {worst['motor_error_deg']:.1f}° "
          f"(한도 {lim.get(worst['motor'], limit_deg):.1f}°) → 계속 진행", flush=True)
    return True


def neutral_limits(limit_deg=NEUTRAL_LIMIT_DEG):
    """모터별 중립 도착 한도 (11-09): 캘리브레이션 모터 범위의 NEUTRAL_LIMIT_FRAC, 최소 limit_deg, 최대 NEUTRAL_LIMIT_MAX_DEG.
    범위가 좁은 엄지 벌림(ID 14, 약 52°)은 같은 3° 가 범위의 6 % 라 다른 관절보다 훨씬 엄격했음"""
    try:
        L = yaml.safe_load(CALIBRATION.read_text(encoding="utf-8")).get("motor_limits") or {}
    except Exception:
        return {}
    out = {}
    for mid, v in L.items():
        if v and None not in v:
            span = abs(math.degrees(float(v[1]) - float(v[0])))
            out[int(mid)] = min(NEUTRAL_LIMIT_MAX_DEG, max(limit_deg, NEUTRAL_LIMIT_FRAC * span))
    return out


def check_saved_calibration(update_baseline):
    """손을 움직이기 전에 저장된 캘리브레이션 값만 검사 (모터 안 움직임)"""
    measured = measured_travel()
    if update_baseline:
        BASELINE.write_text(yaml.safe_dump(measured, allow_unicode=True, sort_keys=True), encoding="utf-8")
        print(f"지금 캘리브레이션 값을 이동량 기준으로 저장: {BASELINE.name}", flush=True)
        return 0
    problems = travel_problems(measured, yaml.safe_load(BASELINE.read_text(encoding="utf-8")))
    if problems:
        print(f"{CALIBRATION.name} 이동량 이상 — 손을 움직이지 않고 멈춥니다:", *problems, sep="\n  ", flush=True)
        print("다른 컴퓨터에서 캘리브레이션했다면 그 calibration.yaml 을 이 폴더로 복사하세요.\n"
              "새 줄 때문에 값이 달라진 것이 확실하면 --update-baseline, 여기서 다시 재려면 --calibrate.", flush=True)
        return 1
    print("캘리브레이션 파일 정상 (16 관절 기준 이내)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--skip-tension", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true", help="장력·캘리브레이션만 하고 손 제어는 시작하지 않음 (앱의 [장력·캘리브레이션])")
    a, bridge_args = ap.parse_known_args()
    # 정지 요청(Ctrl+C / 앱 정지)은 실행 중인 단계에도 같이 전달되어 그쪽이 손을 펴고 토크를 끔.
    # 여기서 KeyboardInterrupt 로 받으면 subprocess.call 이 0.25 s 뒤 그 단계를 강제로 죽이므로, 신호만 기록하고 단계가 끝나길 기다림
    # (11-07: 제슨/리눅스는 앱 정지가 SIGINT 로 옴)
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            signal.signal(sig, lambda *_: STOP.set())
    try:
        code = run_step("상태 확인 (모터 안 움직임)", HERE / "hold_tension.py")
        if code:
            return code
        if a.calibrate:
            if not a.skip_tension:
                code = run_step("장력 조절 (ORCA 라쳇으로 스풀을 팽팽할 때까지만. 끝나면 Enter)", HERE / "hold_tension.py", "--hold")
                if code:
                    return code
            code = calibrate_and_check(a.update_baseline)
            if code:
                return code
            if a.calibrate_only:
                print("장력·캘리브레이션 끝 (손 제어는 시작하지 않음)", flush=True)
                return 0
        else:
            code = check_saved_calibration(a.update_baseline)
            if code:
                return code
        t_neutral = time.time()
        code = run_step("중립 자세 도착 확인", HERE / "neutral_guarded.py")
        if code == 130 or (code and not neutral_close_enough(t_neutral)):
            return code
        return run_step("손 제어 (Ctrl+C = 손 펴고 토크 끔)", HERE / "emg_bridge_guarded.py", *bridge_args)
    except KeyboardInterrupt:                # Ctrl+C 는 실행 중인 단계에도 전달되어 그쪽이 먼저 안전하게 멈춤
        print("\n시작 절차를 멈췄습니다.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
