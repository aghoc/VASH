#!/usr/bin/env python3
"""
finger_test.py — 엄지·검지·중지 전극 위치 검증 (기록 + 분리도 분석)

ads_monitor.py 와 같은 폴더에 두고 사용 (펌웨어 v1.5 이상).

1) 기록 — 화면 지시에 따라 손가락을 하나씩 움직이면 라벨이 붙은 CSV 저장
    python finger_test.py record --port COM6 --ch 1-8 --side flex   --out sessionA_flexor.csv
    python finger_test.py record --port COM6 --ch 1-8 --side extend --out sessionB_extensor.csv
    python finger_test.py record --port COM6 --ch 1-8 --side band   --out S01_R1.csv   ← 측정 기준 v1 (둘레 밴드, 구부리기)
    python finger_test.py record --port COM6 --side band --gestures "주먹,손 펴기,집기" --force-mix --reps 6 --out G1.csv
       ← 동작 목록 자유 (라벨 1,2,3…), --force-mix = 반복마다 약하게/세게
       옵션: --reps 5 (반복)  --hold 3 (힘주기 s)  --rest 3 (쉬기 s)

2) 분석 — 손가락 × 채널 반응표, 그룹 분리도(dB), 분류 정확도(교차검증)
    python finger_test.py analyze sessionA_flexor.csv
    python finger_test.py analyze sessionA_flexor.csv sessionB_extensor.csv     ← 두 세션 비교
       옵션: --groups "1-2;3-5;6-8"  (엄지;검지;중지 채널, 기본값)

3) 자가 점검 — 하드웨어 없이 가상 데이터로 분석 기능 검증
    python finger_test.py demo
"""
import argparse, sys, time, threading, os, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads_monitor import (open_serial, read_line, FrameParser, SOS, parse_channels,  # noqa: E402
                         NCH, C, auto_port)

FINGERS = ["엄지", "검지", "중지"]          # 라벨 1, 2, 3 (0 = 휴식)
DEFAULT_GROUPS = "1-2;3-5;6-8"
TRIM_HOLD = (0.5, 0.3)                       # 힘주기 구간 앞·뒤 제외 (반응 지연·풀림)
TRIM_REST = (1.0, 0.3)                       # 휴식 구간 앞·뒤 제외
WIN_S, STEP_S = 0.2, 0.1                     # 분류용 창 200 ms, 간격 100 ms


def parse_groups(txt):
    out = []
    for g in txt.split(";"):
        m = parse_channels(g)
        out.append([c for c in range(NCH) if (m >> c) & 1])
    if len(out) != 3:
        raise SystemExit("--groups 는 '엄지;검지;중지' 3개 그룹이어야 합니다. 예: 1-2;3-5;6-8")
    return out


# =========================== 기록 ===========================
def cmd_record(a):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK JP", "AppleGothic", "NanumGothic", "DejaVu Sans"]

    mask = parse_channels(a.ch)
    verb = "펴기" if a.side == "extend" else "구부리기"      # band(둘레 밴드, 측정 기준 v1) = 구부리기
    names = [g.strip() for g in a.gestures.split(",")] if a.gestures else [f"{f} {verb}" for f in FINGERS]
    if a.gestures:
        verb = ""
    port = a.port or auto_port()
    if not port:
        raise SystemExit("COM 포트를 찾지 못함 → --port 지정")
    s = open_serial(port, a.baud)
    s.write(b"X"); time.sleep(0.3); s.reset_input_buffer()
    s.write(f"K{mask:02X}".encode()); time.sleep(0.2)
    s.write(b"L")
    hdr = None; t0 = time.time()
    while time.time() - t0 < 6:
        line = read_line(s, 1)
        if line and line.startswith("#LIVE"):
            hdr = dict(kv.split("=") for kv in line.split(",")[1:]); break
    if not hdr:
        s.close(); raise SystemExit("스트리밍 시작 응답(#LIVE) 없음 — selftest 먼저 확인")
    if int(hdr.get("mask", "FF"), 16) != mask:
        print(f"{C['WARN']}경고: 펌웨어가 채널 선택을 지원하지 않음 (v1.5 이상 업로드 필요){C['END']}")
    uv = float(hdr["vref"]) / float(hdr["gain"]) / 2 ** 23 * 1e6

    parser = FrameParser(); stop = threading.Event()
    def reader():
        while not stop.is_set():
            d = s.read(max(4096, s.in_waiting))
            if d: parser.feed(d)
    threading.Thread(target=reader, daemon=True).start()

    # 시나리오: 처음 휴식 5 s → (반복마다 동작 순서 섞음) 힘주기 hold / 휴식 rest
    #   --chain N : 동작 N 개를 쉬지 않고 이어서 (전환 학습용), 그 뒤 휴식
    #   --rest-jitter J : 휴식 길이를 rest ± J 로 무작위 (다음 지시 시각을 예측해 미리 힘주는 것 방지)
    rng = random.Random(a.seed)
    plan = [(0, -1, 5.0)]
    for r in range(a.reps):
        order = list(range(1, len(names) + 1)); rng.shuffle(order)
        for k, f in enumerate(order):
            plan.append((f, r, a.hold))
            if (k + 1) % max(1, a.chain) == 0 or k == len(order) - 1:
                plan.append((0, r, max(1.0, a.rest + rng.uniform(-a.rest_jitter, a.rest_jitter))))
    total = sum(p[2] for p in plan)
    force_txt = lambda r: ("" if not a.force_mix or r < 0 else ("  (약하게)" if r % 2 == 0 else "  (세게)"))
    print(f"기록 시작: {a.reps}회 × {len(names)}동작 ({', '.join(names)}), 약 {total:.0f} s. 창의 지시를 따르세요. (창 닫으면 중단)")
    if a.force_mix:
        print("  힘: 짝수 반복(0,2,4…) = 약하게 (최대의 약 20~30 %), 홀수 반복 = 세게 (약 60~70 %)")

    fig = plt.figure(figsize=(8, 4.5)); fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
    t_main = ax.text(0.5, 0.6, "", ha="center", va="center", fontsize=54, weight="bold")
    t_sub = ax.text(0.5, 0.3, "", ha="center", va="center", fontsize=18)
    t_info = ax.text(0.5, 0.08, "", ha="center", va="center", fontsize=11, color="#555")
    state = {"run": True}
    fig.canvas.mpl_connect("close_event", lambda e: state.update(run=False))
    plt.ion(); plt.show()

    rows_codes, rows_lab, rows_rep, rows_tgt = [], [], [], []
    # 실시간 내 힘 (바이오피드백): 20–450 Hz 필터 후 최근 0.2 s, 8 채널 RMS
    from scipy.signal import sosfilt, sosfilt_zi
    fb = {"zi": None, "buf": np.zeros((int(0.2 * 2000), NCH)), "env": 0.0}
    def update_env(c):
        xx = c * uv
        if fb["zi"] is None:
            fb["zi"] = sosfilt_zi(SOS)[:, :, None] * xx[0][None, None, :]
        yy, fb["zi"] = sosfilt(SOS, xx, axis=0, zi=fb["zi"])
        n = min(len(yy), len(fb["buf"]))
        fb["buf"] = np.roll(fb["buf"], -n, 0); fb["buf"][-n:] = yy[-n:]
        fb["env"] = float(np.sqrt(np.mean(fb["buf"] ** 2)))
    def drain(lab, rep, tgt=0.0, keep=True):
        c, _ = parser.take()
        if len(c):
            update_env(c)
            if not keep: return
            rows_codes.append(c); rows_lab.append(np.full(len(c), lab, np.int8)); rows_rep.append(np.full(len(c), rep, np.int16))
            rows_tgt.append(np.full(len(c), tgt, np.float32))
    # --ramp: 동작 구간 동안 목표 세기 막대가 0 → peak → 0 (삼각형). 피험자는 막대 높이만큼 힘을 줌 → 비례 제어 학습용
    bar_bg = ax.add_patch(plt.Rectangle((0.90, 0.15), 0.05, 0.7, fc="#eeeeee", ec="#999999", visible=a.ramp))
    bar = ax.add_patch(plt.Rectangle((0.90, 0.15), 0.05, 0.0, fc="#ef6c00", visible=a.ramp))
    LV = [float(v) for v in a.levels.split(",")] if a.levels else None
    BEND = bool(a.bend and LV)
    if LV and not BEND:
        bar_bg.set_visible(True); bar.set_visible(True)
        for v in LV:                                   # 단계 눈금선
            ax.plot([0.885, 0.965], [0.15 + 0.7 * v, 0.15 + 0.7 * v], color="#bbbbbb", lw=0.8)
    # --bend: 목표 = 손가락 굽힘 정도. 오른쪽에 손가락 그림 (회색 = 펴짐, 주황 = 목표 굽힘)
    ASP = 4.5 / 8.0                                    # 그림 가로세로 비 보정 (axes 좌표)
    def finger_xy(f, extend=False):
        if extend:                                     # 손 펴기: 편하게 살짝 굽은 손(0) → 끝까지 쭉 편 손(1)
            m, p, d = (1 - f) * 25, (1 - f) * 35, (1 - f) * 20
        else:                                          # 쥐기: 펴짐(0) → MCP 80° · PIP 100° · DIP 60° (1)
            m, p, d = 80 * f, 100 * f, 60 * f
        x, y, ang = [0.93], [0.20], 90.0             # 손등이 오른쪽, 손바닥 쪽(왼쪽)으로 굽음
        for L, dj in ((0.26, m), (0.17, p), (0.12, d)):
            ang += dj; r = np.deg2rad(ang)
            x.append(x[-1] + L * ASP * np.cos(r)); y.append(y[-1] + L * np.sin(r))
        return x, y
    if BEND:
        xr, yr = finger_xy(0.0); ax.plot(xr, yr, color="#cccccc", lw=9, solid_capstyle="round")
        f_tgt, = ax.plot(*finger_xy(0.0), color="#ef6c00", lw=9, solid_capstyle="round", marker="o", ms=6, mfc="#444", mec="#444")
        ax.plot([0.80, 0.97], [0.20, 0.20], color="#555555", lw=3)     # 손바닥
        t_bend = ax.text(0.41, 0.45, "", ha="center", va="center", fontsize=18, weight="bold", color="#555555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)               # plot 이 축 범위를 바꾸지 않게 고정 (막대·글자 위치 유지)
    def target(lab, frac, rep=0, dur=1.0):
        if lab == 0: return 0.0
        if LV:                                         # --levels: 부드럽게 올라가(힘 0.6 s, 굽힘 1.0 s) 그 높이 유지, 끝 0.4 s 내려옴
            v = LV[step_level[(lab, rep)]]; t = frac * dur
            up = min(1.0, t / (1.0 if BEND else 0.6)); dn = min(1.0, (dur - t) / 0.4)
            s = min(up, dn); return v * s * s * (3 - 2 * s)
        if not a.ramp: return 0.0
        return a.ramp_peak * (1 - abs(2 * frac - 1))
    # 단계 배정: 동작마다 모든 단계를 같은 횟수, 순서는 반복마다 어긋나게 (라틴 방진) + 동작별로 섞음
    step_level = {}
    if LV:
        for f in range(1, len(names) + 1):         # 단계 수만큼씩 묶어 그 안에서 섞음 → 모든 단계 같은 횟수, 순서 예측 불가
            seq = []
            while len(seq) < a.reps:
                blk = list(range(len(LV))); rng.shuffle(blk); seq += blk
            for r in range(a.reps):
                step_level[(f, r)] = seq[r]

    palette = ["#c62828", "#1565c0", "#6a1b9a", "#ef6c00", "#00838f", "#5d4037", "#ad1457", "#455a64"]
    colors = {0: "#2e7d32", **{i + 1: palette[i % len(palette)] for i in range(len(names))}}

    # 내 힘 막대 (파랑, 목표 막대 왼쪽) + 안내 문구. 최대 힘(MVC)은 녹화 전 동작마다 3 s 측정 (저장 안 함)
    FB = (LV is not None or a.ramp) and not a.no_feedback and not BEND   # 굽힘 모드는 힘 피드백 안 씀 (힘이 아니라 모양을 맞춤)
    if BEND:
        for t_ in (t_main, t_sub, t_info, t_bend): t_.set_x(0.36)
    mvc, base = {}, 0.0
    if FB:                                             # 글자는 왼쪽으로 옮겨 막대와 안 겹치게
        for t_ in (t_main, t_sub, t_info): t_.set_x(0.41)
        ax.add_patch(plt.Rectangle((0.83, 0.15), 0.05, 0.7, fc="#eeeeee", ec="#999999"))
        me = ax.add_patch(plt.Rectangle((0.83, 0.15), 0.05, 0.0, fc="#1565c0"))
        ax.text(0.855, 0.11, "내 힘", ha="center", fontsize=10, color="#1565c0")
        ax.text(0.925, 0.11, "목표", ha="center", fontsize=10, color="#ef6c00")
        t_fb = ax.text(0.41, 0.45, "", ha="center", va="center", fontsize=20, weight="bold")

        def phase(text, color, dur, sub=""):
            vals = []; t0_ = time.time()
            t_main.set_text(text); t_main.set_color(color); t_main.set_fontsize(34)
            while state["run"] and time.time() - t0_ < dur:
                drain(0, -1, keep=False); el = time.time() - t0_
                if el > 0.8: vals.append(fb["env"])
                t_sub.set_text(f"{dur - el:4.1f} s   {sub}")
                t_info.set_text(f"최대 힘 측정 (저장 안 함)   수신 {parser.frames} 샘플")
                fig.canvas.draw_idle(); fig.canvas.flush_events(); plt.pause(0.05)
            return vals
        print("최대 힘 측정: 동작마다 3 s 최대로 세게 → 3 s 쉬기 (약 30 s)")
        base = float(np.median(phase("가만히 — 힘 빼기", colors[0], 4.0, "기준 잡는 중") or [0.0]))
        for k, nm in enumerate(names, 1):
            phase(f"다음: {nm}", "#777777", 2.0, "준비")
            v = phase(f"{nm} — 최대로 세게!", colors[k], 3.0, "끝까지 꽉")
            mvc[k] = max(1e-6, float(np.percentile(v, 80)) - base) if v else 1.0
            phase("힘 빼기", colors[0], 3.0)
        print("  기준 %.1f µV, 최대 힘 " % base + ", ".join(f"{names[k-1]} {mvc[k]:.0f} µV" for k in mvc))

    def show_fb(lab, tg):
        if not FB: return
        f = 0.0 if lab == 0 else max(0.0, (fb["env"] - base) / mvc[lab])
        me.set_height(0.7 * min(f, 1.15))
        if lab == 0 or tg < 0.05:
            t_fb.set_text(""); return
        if f < tg - 0.08: t_fb.set_text("▲ 조금 더 세게"); t_fb.set_color("#c62828")
        elif f > tg + 0.08: t_fb.set_text("▼ 조금 살살"); t_fb.set_color("#ef6c00")
        else: t_fb.set_text("● 좋아요, 유지"); t_fb.set_color("#2e7d32")

    try:
        drain(0, -1)                                   # 시작 전 쌓인 데이터 = 휴식
        for i, (lab, rep, dur) in enumerate(plan):
            if not state["run"]: break
            nxt = plan[i + 1] if i + 1 < len(plan) else None
            t_start = time.time(); t_end = t_start + dur
            is_ext = lab and names[lab - 1].replace(" ", "").startswith("손펴기")
            t_main.set_text(("휴식 — 완전히 힘 빼기" if a.relax else "휴식") if lab == 0 else names[lab - 1] + force_txt(rep)
                            + (f"  {LV[step_level[(lab, rep)]] * 100:.0f}%" + ((" 펴기" if is_ext else " 굽히기") if BEND else "") if LV and lab else ""))
            t_main.set_fontsize(28 if BEND else 34 if FB else (40 if lab == 0 and a.relax else 54))
            if BEND:
                t_bend.set_text("" if lab == 0 else "그림만큼 움직여서 멈추고, 그 모양 그대로 유지")
            t_main.set_color(colors[lab])
            nxt_txt = "" if not nxt else ("다음: 휴식" if nxt[0] == 0 else f"다음: {names[nxt[0]-1]}{force_txt(nxt[1])}")
            while state["run"] and time.time() < t_end:
                tg = target(lab, (time.time() - t_start) / dur, rep, dur)
                drain(lab, rep, tg)
                bar.set_height(0.7 * tg); show_fb(lab, tg)
                if BEND: f_tgt.set_data(*finger_xy(tg, bool(is_ext)))
                show_next = a.next_cue < 0 or (t_end - time.time()) <= a.next_cue
                t_sub.set_text(f"{t_end - time.time():4.1f} s     {nxt_txt if show_next else ''}")
                t_info.set_text(f"{i+1}/{len(plan)}   수신 {parser.frames} 샘플   누락 {parser.dropped}")
                fig.canvas.draw_idle(); fig.canvas.flush_events(); plt.pause(0.05)
            drain(lab, rep)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); time.sleep(0.2)
        try: s.write(b"X"); time.sleep(0.2); s.close()
        except Exception: pass
        plt.close(fig)

    if not rows_codes:
        raise SystemExit("데이터 없음")
    x = np.concatenate(rows_codes) * uv
    lab = np.concatenate(rows_lab); rep = np.concatenate(rows_rep); tgt = np.concatenate(rows_tgt)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# finger_test fs={hdr['fs']} mask={mask:02X} side={a.side} reps={a.reps} hold={a.hold} rest={a.rest} "
                f"dropped={parser.dropped} bad={parser.bad}"
                + (f" gestures={'|'.join(n.replace(' ', '_') for n in names)}" if a.gestures else "")
                + (" force=mix_even_weak" if a.force_mix else "")
                + (f" subject={a.subject}" if a.subject else "")
                + f" chain={a.chain} rest_jitter={a.rest_jitter} relax={int(a.relax)} next_cue={a.next_cue}"
                + (f" ramp=tri peak={a.ramp_peak}" if a.ramp and not LV else "")
                + (f" ramp={'bend' if BEND else 'step'} peak={1.0 if BEND else max(LV)} levels={'|'.join(str(v) for v in LV)}" if LV else "") + "\n")
        f.write("label,rep," + ",".join(f"ch{i+1}_uV" for i in range(NCH)) + (",target" if a.ramp or LV else "") + "\n")
        for l, r, row, tg in zip(lab, rep, x, tgt):
            f.write(f"{l},{r}," + ",".join(f"{v:.3f}" for v in row) + (f",{tg:.3f}" if a.ramp or LV else "") + "\n")
    print(f"저장: {a.out}  ({len(x)} 샘플, 누락 {parser.dropped}, 깨진 프레임 {parser.bad})")
    if parser.dropped:
        print(f"{C['WARN']}누락이 있습니다. 분석은 가능하지만 누락 위치에서 작은 잡음이 생길 수 있습니다.{C['END']}")
    print("다음: Claude 에게 '녹화 끝' 이라고 알려 주세요" if a.gestures else f"다음: python finger_test.py analyze {a.out}")


# =========================== 분석 ===========================
def load_csv(path):
    meta = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if first.startswith("#"):
            for kv in first[1:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1); meta[k] = v
    d = np.loadtxt(path, delimiter=",", comments="#", skiprows=2 if meta else 1, encoding="utf-8")
    lab = d[:, 0].astype(int); rep = d[:, 1].astype(int); x = d[:, 2:2 + NCH]
    fs = float(meta.get("fs", 2000)); mask = int(meta.get("mask", "FF"), 16)
    return x, lab, rep, fs, mask, meta


def segments(lab):
    """같은 라벨이 이어진 구간 (시작, 끝, 라벨)"""
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, edges]; ends = np.r_[edges, len(lab)]
    return [(int(s), int(e), int(lab[s])) for s, e in zip(starts, ends)]


def harmonic_sos(fs, freqs=(240, 300, 360, 420), q=30.0):
    """60 Hz 고조파 추가 노치 (emg_filters.h 에는 60/120/180 만 있음)"""
    from scipy import signal
    rows = []
    for f0 in freqs:
        b, a = signal.iirnotch(f0, q, fs)
        rows.append(np.r_[b, a])
    return np.array(rows)


HARMONIC_NOTCH = True     # analyze --no-harmonic 으로 끄면 기기 필터(60/120/180)만 사용


def analyze_one(path, groups):
    from scipy import signal
    x, lab, rep, fs, mask, meta = load_csv(path)
    active = [c for c in range(NCH) if (mask >> c) & 1]
    sos = np.vstack([SOS, harmonic_sos(fs)]) if HARMONIC_NOTCH else SOS
    y = np.asarray(signal.sosfilt(sos, x, axis=0))          # 20–450 Hz + 60 Hz 노치(고조파 포함)
    y[: int(fs * 1.0)] = 0                                   # 필터 시작 과도응답 제외
    segs = segments(lab)

    rest_rms, task = [], {1: [], 2: [], 3: []}
    win, step = int(WIN_S * fs), int(STEP_S * fs)
    Xw, Yw, Rw = [], [], []
    for s, e, l in segs:
        tr = TRIM_HOLD if l else TRIM_REST
        a0, a1 = s + int(tr[0] * fs), e - int(tr[1] * fs)
        if a1 - a0 < win or a0 < fs:      # 너무 짧거나 시작 1 s 이내
            continue
        seg = y[a0:a1]
        rms = np.sqrt(np.mean(seg ** 2, axis=0))
        if l == 0:
            rest_rms.append(rms)
        else:
            task[int(l)].append(rms)
            for k in range(a0, a1 - win + 1, step):
                w = y[k:k + win]
                Xw.append(np.log(np.sqrt(np.mean(w ** 2, axis=0))[active] + 1e-3))
                Yw.append(l); Rw.append(rep[k])
    if not rest_rms or any(len(task[f]) == 0 for f in (1, 2, 3)):
        raise SystemExit(f"{path}: 휴식/손가락 구간이 부족합니다 (기록을 끝까지 진행했는지 확인)")
    base = np.median(rest_rms, axis=0)
    near_sat = [c for c in active if np.mean(np.abs(x[:, c]) > 0.9 * 187500) > 0.01]   # Gain 24 범위의 90% 초과
    act_uv = np.array([np.mean(task[f], axis=0) for f in (1, 2, 3)])         # 3 × 8 (µV)
    act = act_uv / np.maximum(base, 1e-3)                                     # 휴식 대비 배수

    # 그룹 분리도: 해당 손가락 그룹 평균 vs 다른 그룹 중 최대
    sel = []
    for fi, g in enumerate(groups):
        g = [c for c in g if c in active]
        others = [[c for c in og if c in active] for j, og in enumerate(groups) if j != fi]
        tgt = act[fi, g].mean() if g else np.nan
        oth = max(act[fi, og].mean() for og in others if og) if any(others) else np.nan
        sel.append(20 * np.log10(tgt / oth) if g and oth > 0 else np.nan)

    # 분류 정확도: 정규화 LDA, 반복(rep) 단위 교차검증
    Xw = np.array(Xw); Yw = np.array(Yw); Rw = np.array(Rw)
    reps = sorted(set(Rw.tolist()))
    conf = np.zeros((3, 3), int)
    for r in reps:
        tr, te = Rw != r, Rw == r
        if tr.sum() < 10 or te.sum() == 0: continue
        mu = np.array([Xw[tr & (Yw == f)].mean(0) for f in (1, 2, 3)])
        Z = Xw[tr] - np.take(mu, Yw[tr] - 1, axis=0)
        S = np.cov(Z.T) + 1e-3 * np.eye(Xw.shape[1])
        Si = np.linalg.inv(S)
        W = mu @ Si; b = -0.5 * np.einsum("ij,ij->i", W, mu)
        pred = np.argmax(Xw[te] @ W.T + b, axis=1) + 1
        for t, p in zip(Yw[te], pred): conf[t - 1, p - 1] += 1
    acc = np.trace(conf) / max(1, conf.sum())
    return dict(path=path, meta=meta, active=active, base=base, near_sat=near_sat, act=act, act_uv=act_uv, sel=sel,
                conf=conf, acc=acc, nreps=len(reps))


def grade(db):
    if np.isnan(db): return "-"
    return "좋음" if db >= 6 else ("보통" if db >= 3 else "부족")


def report(r, groups):
    side = {"flex": "손바닥 쪽(굴근)", "extend": "손등 쪽(신근)", "band": "둘레 밴드(기준 v1)"}.get(r["meta"].get("side", ""), "")
    print(f"\n{C['B']}== {os.path.basename(r['path'])}  {side} =={C['END']}")
    if int(r["meta"].get("dropped", 0)):
        print(f"  (기록 중 누락 {r['meta']['dropped']} 샘플)")
    act = r["active"]
    if r.get("near_sat"):
        print(f"  {C['WARN']}주의: CH{', CH'.join(str(c+1) for c in r['near_sat'])} 원신호가 측정범위(±187.5 mV) 끝에 붙어 있음 "
              f"→ 전극 DC 오프셋 과다. 전극 교체 후 재측정 권장{C['END']}")
    if np.median(r["base"][act]) > 8:
        print(f"  {C['WARN']}주의: 휴식 중 근전도가 높음 (중앙값 {np.median(r['base'][act]):.1f} µV). "
              f"휴식 구간에 손에 힘이 들어가 있으면 반응 배수가 작게 나옵니다.{C['END']}")
    gname = {}
    for fi, g in enumerate(groups):
        for c in g: gname[c] = FINGERS[fi][0]
    print("  휴식 대비 반응 배수 (각 손가락 동작에서 채널별 RMS / 휴식 RMS)")
    print("          " + "".join(f"  CH{c+1}({gname.get(c,'-')})" for c in act))
    print("  휴식µV  " + "".join(f"{r['base'][c]:9.2f}" for c in act))
    for fi in range(3):
        row = r["act"][fi]
        cells = []
        for c in act:
            hl = c in groups[fi]
            cells.append(f"{C['PASS'] if hl else ''}{row[c]:9.1f}{C['END'] if hl else ''}")
        print(f"  {FINGERS[fi]}    " + "".join(cells))
    print("\n  그룹 분리도 (해당 그룹 평균 / 다른 그룹 최대, dB):  6 이상 좋음 · 3~6 보통 · 3 미만 부족")
    for fi in range(3):
        db = r["sel"][fi]; g = grade(db)
        col = C["PASS"] if g == "좋음" else (C["WARN"] if g == "보통" else C["FAIL"])
        print(f"   {FINGERS[fi]}: {db:6.1f} dB  {col}{g}{C['END']}")
    print(f"\n  분류 정확도 (반복 {r['nreps']}회 교차검증, 우연 수준 33%): {C['B']}{100*r['acc']:.1f}%{C['END']}")
    print("  혼동표 (행=실제, 열=예측):   " + "  ".join(FINGERS))
    for fi in range(3):
        print(f"     {FINGERS[fi]}            " + "  ".join(f"{v:4d}" for v in r["conf"][fi]))


def save_figure(results, groups, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK JP", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    from matplotlib.patches import Rectangle
    n = len(results)
    vmax = max(np.log10(np.maximum(r["act"][:, r["active"]], 1e-3)).max() for r in results)   # 두 세션 같은 색 기준
    fig, axs = plt.subplots(1, n, figsize=(6.5 * n, 3.6), squeeze=False)
    for ax, r in zip(axs[0], results):
        act = r["active"]
        M = r["act"][:, act]
        L = np.log10(np.maximum(M, 1e-3))
        im = ax.imshow(L, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(act))); ax.set_xticklabels([f"CH{c+1}" for c in act])
        ax.set_yticks(range(3)); ax.set_yticklabels(FINGERS)
        for i in range(3):
            for j, c in enumerate(act):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        color="k" if L[i, j] > 0.6 * vmax else "w", fontsize=9, weight="bold")
            cols = [j for j, c in enumerate(act) if c in groups[i]]   # 정답 칸(해당 손가락 그룹) 빨간 테두리
            if cols:
                ax.add_patch(Rectangle((min(cols) - 0.5, i - 0.5), len(cols), 1, fill=False, ec="red", lw=2.5))
        # 그룹 경계선
        pos = 0
        for g in groups[:-1]:
            pos += len([c for c in g if c in act]); ax.axvline(pos - 0.5, color="w", lw=2)
        side = {"flex": "굴근", "extend": "신근", "band": "둘레 밴드"}.get(r["meta"].get("side", ""), "")
        ax.set_title(f"{os.path.basename(r['path'])} {side}\n분류 {100*r['acc']:.0f}%  "
                     + "  ".join(f"{FINGERS[i]} {r['sel'][i]:.1f}dB" for i in range(3)), fontsize=9)
        fig.colorbar(im, ax=ax, label="log10(휴식 대비 배수)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def cmd_analyze(a):
    groups = parse_groups(a.groups)
    results = [analyze_one(p, groups) for p in a.csv]
    for r in results:
        report(r, groups)
    if len(results) == 2:
        A, B = results
        print(f"\n{C['B']}== 두 세션 비교 =={C['END']}")
        print("            " + f"{os.path.basename(A['path'])[:18]:>20s}  {os.path.basename(B['path'])[:18]:>20s}   더 나은 쪽")
        for fi in range(3):
            a_, b_ = A["sel"][fi], B["sel"][fi]
            best = os.path.basename(A["path"]) if a_ >= b_ else os.path.basename(B["path"])
            print(f"  {FINGERS[fi]} 분리도 {a_:18.1f} dB  {b_:18.1f} dB   {best}")
        print(f"  분류 정확도 {100*A['acc']:17.1f} %  {100*B['acc']:18.1f} %   "
              f"{os.path.basename(A['path']) if A['acc'] >= B['acc'] else os.path.basename(B['path'])}")
    png = os.path.splitext(a.csv[0])[0] + ("_compare.png" if len(results) == 2 else "_analysis.png")
    save_figure(results, groups, png)
    print(f"\n그림 저장: {png}")


# =========================== 자가 점검 ===========================
def cmd_demo(a):
    """가상 세션 2개(분리 잘 됨 / 잘 안 됨)를 만들어 분석 결과가 기대대로 나오는지 확인"""
    rng = np.random.default_rng(0)
    fs = 2000; groups = parse_groups(DEFAULT_GROUPS)
    def make(path, leak, side):
        plan = [(0, -1, 5.0)]
        for r in range(5):
            o = [1, 2, 3]; random.Random(r).shuffle(o)
            for f in o: plan += [(f, r, 3.0), (0, r, 3.0)]
        labs, reps, xs = [], [], []
        for l, r, d in plan:
            n = int(d * fs); amp = np.full(NCH, 2.0)
            trial_gain = rng.uniform(0.6, 1.4, NCH)        # 시행마다 힘·접촉 변동 (실제 데이터처럼)
            if l:
                amp += 30 * leak
                for c in groups[l - 1]: amp[c] = 60.0
                amp *= trial_gain
            xs.append(rng.standard_normal((n, NCH)) * amp + 3 * np.sin(2 * np.pi * 60 * np.arange(n) / fs)[:, None])
            labs.append(np.full(n, l)); reps.append(np.full(n, r))
        x = np.concatenate(xs); lab = np.concatenate(labs); rep = np.concatenate(reps)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# finger_test fs=2000 mask=FF side={side} reps=5 hold=3 rest=3 dropped=0 bad=0\n")
            f.write("label,rep," + ",".join(f"ch{i+1}_uV" for i in range(NCH)) + "\n")
            for l, r, row in zip(lab, rep, x):
                f.write(f"{l},{r}," + ",".join(f"{v:.3f}" for v in row) + "\n")
    import tempfile
    d = tempfile.mkdtemp()
    pa, pb = os.path.join(d, "demo_good.csv"), os.path.join(d, "demo_poor.csv")
    make(pa, 0.1, "flex"); make(pb, 1.6, "extend")
    ra, rb = analyze_one(pa, groups), analyze_one(pb, groups)
    report(ra, groups); report(rb, groups)
    png = save_figure([ra, rb], groups, os.path.join(d, "demo_compare.png"))
    print(f"그림 저장 확인: {png} ({os.path.getsize(png)} bytes)")
    ok = (min(ra["sel"]) > 6 and ra["acc"] > 0.95 and max(rb["sel"]) < min(ra["sel"]) and rb["acc"] <= ra["acc"])
    print("\n자가 점검:", f"{C['PASS']}PASS{C['END']}" if ok else f"{C['FAIL']}FAIL{C['END']}",
          "(분리 잘 되는 세션이 dB·정확도 모두 높게 나와야 함)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="엄지·검지·중지 전극 위치 검증")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--port", default=None); p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--ch", default="1-8"); p.add_argument("--side", choices=["flex", "extend", "band"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=5); p.add_argument("--hold", type=float, default=3.0)
    p.add_argument("--rest", type=float, default=3.0)
    p.add_argument("--gestures", default=None, help='손가락 대신 동작 목록. 예: "주먹,손 펴기,집기"  (라벨 1,2,3… 순서)')
    p.add_argument("--force-mix", action="store_true", help="반복마다 약하게/세게 번갈아 지시")
    p.add_argument("--subject", default=None, help="피험자 코드 (예: S02) — 머리말에 기록")
    p.add_argument("--chain", type=int, default=1, help="동작 N 개를 쉬지 않고 연속 (전환 학습)")
    p.add_argument("--rest-jitter", type=float, default=0.0, help="휴식 길이 ± 무작위 (s)")
    p.add_argument("--relax", action="store_true", help="휴식 화면에 '완전히 힘 빼기' 표시")
    p.add_argument("--next-cue", type=float, default=-1, help="다음 동작 안내를 끝나기 N 초 전에만 표시 (−1 = 항상)")
    p.add_argument("--ramp", action="store_true", help="동작 구간에 목표 세기 막대 (0→peak→0), CSV 에 target 열 추가")
    p.add_argument("--levels", default=None, help='세기 단계 녹화: 막대가 이 높이(최대 힘 비율)로 올라가 유지. 예: "0.2,0.4,0.6,0.8" (reps 는 단계 수의 배수로)')
    p.add_argument("--bend", action="store_true", help="--levels 를 '굽힘 정도'로: 손가락 그림만큼 굽혀 유지 (힘 아님)")
    p.add_argument("--no-feedback", action="store_true", help="--levels/--ramp 에서 내 힘 막대·최대 힘 측정 끄기")
    p.add_argument("--ramp-peak", type=float, default=0.7, help="막대 최고 높이 = 최대 힘의 비율")
    p.add_argument("--seed", type=int, default=1234, help="동작 순서 무작위 시드 (녹화마다 바꾸면 순서가 달라짐)")
    p = sub.add_parser("analyze")
    p.add_argument("csv", nargs="+"); p.add_argument("--groups", default=DEFAULT_GROUPS)
    p.add_argument("--no-harmonic", action="store_true", help="240~420 Hz 고조파 노치 끄기 (기기 필터와 동일 조건)")
    sub.add_parser("demo")
    a = ap.parse_args()
    if a.cmd == "record": cmd_record(a)
    elif a.cmd == "analyze":
        global HARMONIC_NOTCH
        HARMONIC_NOTCH = not a.no_harmonic
        print("필터: 20–450 Hz + 60/120/180 Hz 노치" + (" + 240/300/360/420 Hz 고조파 노치" if HARMONIC_NOTCH else " (기기와 동일)"))
        if len(a.csv) > 2: raise SystemExit("세션은 최대 2개까지 비교")
        cmd_analyze(a)
    else: sys.exit(0 if cmd_demo(a) else 1)


if __name__ == "__main__":
    main()
