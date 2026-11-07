#!/usr/bin/env python3
"""
latency_live.py — 실제 의수 지연 측정 (11-02). 근전도 판단 기록과 의수 기록을 같은 PC 시계로 맞춰서
  근육 시작 → 판단 켜짐 → 확정 → 손이 보이게 움직임(5°) → 절반 도착 → 거의 도착  을 시도마다 계산.
시뮬레이션(분석_지연_1101.md: 손 10 % 341 ms · 50 % 501 ms)에 모터·통신 지연이 더해진 실제 값.

기록 (두 터미널, 같은 PC):
  터미널 B: uv run --frozen --no-dev python ../hand-config/right-no-wrist/emg_bridge_guarded.py --log C:\\ads_bringup\\pc_tools\\data\\S01\\LAT_hand.csv
  터미널 A: python grip_rt.py run --model data\\S01\\grip_all7.pkl --port COM6 --udp 127.0.0.1:9000 --log data\\S01\\LAT_emg.csv
  동작: 쉬기 3 s → 주먹 2 s → 완전히 힘 빼기, 10 번. 이어서 손 펴기로 10 번 (팔은 책상 위)
분석:
  python latency_live.py data\\S01\\LAT_emg.csv data\\S01\\LAT_hand.csv
"""
import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    em = pd.read_csv(sys.argv[1]); hd = pd.read_csv(sys.argv[2])
    if len(em) < 50 or len(hd) < 50:
        print(f"기록이 비어 있음 — 근전도 {len(em)} 줄, 의수 {len(hd)} 줄. grip_rt run 화면에 판단 줄(상태·세기)이 계속 바뀌었는지 확인 후 다시 측정"); return
    pos_cols = [c for c in hd.columns if c.endswith("_pos")]
    z = em["z"].to_numpy(float); tw = em["wall"].to_numpy(float)
    rest = em["state"] == "rest"
    zr = np.median(z[rest]) if rest.any() else np.percentile(z, 20)
    za = np.percentile(z, 90)
    thr = zr + 0.3 * (za - zr)                                     # 시뮬레이션과 같은 정의: 휴식↔동작 세기의 30 %
    above = z > thr
    starts = [i for i in range(1, len(z)) if above[i] and not above[i - 1]]
    onsets = []
    for i in starts:
        if onsets and tw[i] - tw[onsets[-1]] < 2.0: continue       # 2 s 안의 재교차는 같은 시도
        if (~above[max(0, i - 20):i]).all(): onsets.append(i)      # 앞 1 s 동안 쉬고 있었던 시작만
    rows = []
    ht = hd["wall"].to_numpy(float); P = hd[pos_cols].to_numpy(float)
    for i in onsets:
        t0 = tw[i]
        seg = em[(em["wall"] >= t0 - 0.5) & (em["wall"] <= t0 + 2.5)]
        on = seg[(seg["state"] == "active") & (seg["wall"] >= t0 - 0.3)]["wall"]
        cm = seg[(seg["committed"] == 1) & (seg["wall"] >= t0 - 0.3)]
        grip = cm["grip"].mode().iloc[0] if len(cm) else ""
        base_m = (ht >= t0 - 0.5) & (ht < t0)
        win = (ht >= t0) & (ht <= t0 + 2.5)
        if base_m.sum() < 3 or win.sum() < 10: continue
        base = np.nanmedian(P[base_m], 0)
        dev = np.nanmax(np.abs(P[win] - base), 1)                  # 가장 많이 움직인 관절의 이동 (°)
        full = np.nanmax(dev)
        if not np.isfinite(full) or full < 8: continue             # 손이 거의 안 움직인 시도 (오판·휴식) 제외
        tt = ht[win] - t0
        first = lambda thr_: float(tt[np.argmax(dev >= thr_)]) if (dev >= thr_).any() else np.nan
        rows.append(dict(grip=grip, on=float(on.iloc[0] - t0) if len(on) else np.nan,
                         commit=float(cm["wall"].iloc[0] - t0) if len(cm) else np.nan,
                         move5=first(5.0), half=first(0.5 * full), near=first(0.9 * full), travel=full))
    if not rows:
        print("시도를 찾지 못함 — 두 기록이 같은 시간대인지, 손이 움직였는지 확인"); return
    df = pd.DataFrame(rows)
    ms = lambda v: f"{np.nanmedian(v) * 1000:4.0f}"
    print(f"시도 {len(df)} 개 (근육 시작 = 세기 30 % 교차 기준, 중앙값 ms)")
    print(f"  판단 켜짐 {ms(df.on)}  확정 {ms(df.commit)}  손 움직임 시작(5°) {ms(df.move5)}  절반 {ms(df.half)}  거의 도착(90 %) {ms(df.near)}")
    for g, d in df.groupby("grip"):
        print(f"  {g or '(확정 없음)':16s} {len(d):2d} 번  켜짐 {ms(d.on)}  확정 {ms(d.commit)}  5° {ms(d.move5)}  절반 {ms(d.half)}  90 % {ms(d.near)}")
    print("참고 (모터 없는 시뮬레이션, 분석_지연_1101): 손 10 % 341 · 50 % 501 · 90 % 736 ms → 차이 = 모터·통신·장력 지연")
    out = sys.argv[1].replace("_emg.csv", "_결과.csv"); df.to_csv(out, index=False, encoding="utf-8-sig"); print(f"시도별 결과 {out}")


if __name__ == "__main__":
    main()
