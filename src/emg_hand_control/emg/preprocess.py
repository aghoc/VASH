#!/usr/bin/env python3
"""
preprocess.py — 근전도 녹화 표준 전처리 (데이터 표준 v1, 11-02). 원본 CSV 는 건드리지 않음.

  python preprocess.py all                      # pc_tools/data 의 모든 녹화 (이미 한 것은 건너뜀, --force 로 다시)
  python preprocess.py one data\\S01\\GR3.csv     # 새 녹화 한 개
  python preprocess.py index                    # recordings.csv · qc_report.md 다시 만들기

출력 (C:\\ads_bringup\\dataset):
  derivatives/emg_v1/sub-<피험자>/ses-<세션>/<녹화ID>_emg.npz     필터된 근전도(float32) + 지시 라벨 + 근육 기준 라벨 + 반복
  derivatives/emg_v1/sub-<피험자>/ses-<세션>/<녹화ID>_events.csv   시도별 표: 지시·실제 근육 시작/끝, 반응 시간, 세기, 잡음, 통과 여부
  recordings.csv   녹화 목록 + 메타데이터 + 품질(QC) 지표 + 사용 구분(train/test/practice/exclude)
  qc_report.md     품질 요약
처리:
  ① 헤더 해석 → 동작 이름을 표준 코드로 (gestures.csv, 별칭·[살살]/[꽉] 수식어 분리)
  ② 필터 20–450 Hz + 60 Hz·고조파 노치 — 실시간(grip_rt)과 같은 인과 필터, 첫 1 s 는 필터 과도 구간이라 0
  ③ 채널 품질: 휴식 RMS, 60 Hz 비율(원신호), 끊김(평탄), 잡음 창 비율(창 RMS > 400 µV), 신호 대 휴식 비
  ④ 시도마다 실제 근육 시작·끝 (25 ms 세기가 휴식↔유지의 50 % 를 넘는 때) → 반응 시간, 근육 기준 라벨 (경계 ±는 −1)
"""
import argparse, csv, json, os, re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "dataset"
DER = OUT / "derivatives" / "emg_v1"
VERSION = "emg_v1"
LINE_FLAG = 0.9        # 원신호 20–450 Hz 전력 중 60 Hz(±1.5) 비율이 이보다 크면 표시 (노치로 제거되지만 전극 접촉·충전기 확인 신호)

# ---------------- 표준 동작 코드 (gestures.csv 와 같음) ----------------
GESTURES = [
    # code, 표준 한글 이름, 별칭(정확히 같은 이름, 수식어 제외)
    ("rest", "휴식", []),
    ("open", "손 펴기", []),
    ("fist", "주먹", ["바위"]),
    ("pinch", "집기 (중지 펴기)", []),
    ("tripod", "세손가락 (중지 굽힘)", []),
    ("pinch_v0", "집기 (중지 지시 없음, 10-24 이전 정의)", ["집기"]),
    ("tripod_v0", "세손가락 (중지 지시 없음, 10-24 이전 정의)", ["세손가락"]),
    ("lateral", "옆집기", []),
    ("precision_v0", "정밀 쥐기 (10-27 연습, 폐기된 정의)", ["정밀 쥐기"]),
    ("scissors", "가위", ["V"]),
    ("index_ext", "검지 펴기", []),
    ("thumb_ext", "엄지 펴기", []),
    ("middle_ext", "중지 펴기", []),
    ("thumb_flex", "엄지 굽히기", ["엄지", "엄지 구부리기"]),
    ("index_flex", "검지 굽히기", ["검지", "검지 구부리기"]),
    ("middle_flex", "중지 굽히기", ["중지", "중지 구부리기"]),
]
_ALIAS = {}
for code, name, al in GESTURES:
    for n in [name] + al:
        _ALIAS[re.sub(r"\s+", "", n)] = code


def canon(raw):
    """'주먹 [꽉→살살]' → ('fist', '꽉→살살', ''), '휴식 [팔 움직이기]' → ('rest', '', '팔 움직이기'), '주먹 (최대의 50~70 %)' → ('fist', '', '최대의 50~70 %')"""
    n = raw.replace("_", " ").strip()
    mods = re.findall(r"\[(.*?)\]", n); n2 = re.sub(r"\[.*?\]", "", n).strip()
    force = next((m for m in mods if any(k in m for k in ("살살", "꽉", "힘빼기"))), "")
    cond = ", ".join(m for m in mods if m != force)
    key = re.sub(r"\s+", "", n2)
    if key in _ALIAS:
        return _ALIAS[key], force, cond
    base = re.sub(r"\(.*?\)", "", n2).strip()                    # '주먹 (최대의 50~70 %)' → 주먹 + 조건
    paren = ", ".join(re.findall(r"\((.*?)\)", n2))
    k2 = re.sub(r"\s+", "", base)
    if k2 in _ALIAS:
        return _ALIAS[k2], force, ", ".join(x for x in (cond, paren) if x)
    if n2.startswith("휴식"):
        return "rest", force, cond
    return "unknown:" + n2, force, cond


# ---------------- 녹화 → 피험자·세션·과제 ----------------
SUBJECT_GUESS = {"data": "S01", "pc_tools": "S01"}
# 바로잡기 (근거: 10-30 대화 "오늘 찬 사람은 나야", 녹화가이드_손가락.md "10-30 BEND 녹화도 data\S01 로 복사")
#   10-30 BEND·LV 는 S01 이 찼는데 헤더에 subject=S02 로 기록됨. S02 폴더의 같은 파일은 복사본(내용 동일) → 한 번만 처리
SUBJECT_FIX = {r"data/S01/(BEND|LV)": ("S01", "헤더 subject=S02 오기 — 10-30 착용자 S01")}     # 10-24~10-27 녹화는 헤더에 subject 가 없음 → 분석 문서 기준 S01
TASKS = [  # 파일 이름 앞부분 → 과제 코드, 설명
    (r"^GR_FWD", "GRfwd", "쥐기 4동작 살살/꽉, 팔 앞으로"), (r"^GR", "GR", "쥐기 4동작 살살/꽉"),
    (r"^GE_FWD", "GEfwd", "동작 5종, 팔 앞으로"), (r"^GE", "GE", "동작 5종 (펴기·주먹·가위·검지·엄지 펴기)"),
    (r"^ARM", "ARM", "휴식 + 팔 움직이기"), (r"^RELEASE", "REL", "꽉→살살→힘빼기 (놓기 확인)"), (r"^OS", "OS", "손 펴기·가위"),
    (r"^FG", "FG", "손가락별 + 쥐기"), (r"^BEND", "BEND", "굽힘 정도 막대"), (r"^LV", "LV", "세기 계단"), (r"^RAMP", "RAMP", "세기 삼각"),
    (r"^RPS", "RPS", "쥐기 + 손가락 + 가위"), (r"^CH", "CH", "동작 이어 하기 (chain)"), (r"^EXT", "EXT", "손가락 펴기"),
    (r"^HOLD", "HOLD", "꽉→살살 유지"), (r"^B\d", "B", "쥐기 4동작"), (r"^F1_", "FAT1", "피로 실험 F1 (손가락)"),
    (r"^F2_", "FAT2", "피로 실험 F2 (쥐기)"), (r"^G\d", "G", "쥐기 동작 세트"), (r"^20261024_S01_R", "FIN", "손가락 굽히기 (10-24)"),
    (r"^20261024_G", "G", "쥐기 5종 (10-24)"), (r"^20261024_poster", "POSTER", "포스터용"), (r"^A1_", "ELEC", "전극 시험 굽힘근"),
    (r"^B\d_extensor", "ELEC", "전극 시험 폄근"), (r"^practice", "PRAC", "연습"),
]


def describe(path):
    p = Path(path); stem = p.stem
    sub_dir = p.parent.name
    head = p.open(encoding="utf-8").readline()
    meta = {k: v for k, v in (kv.split("=", 1) for kv in head[1:].split() if "=" in kv)}
    sub = meta.get("subject") or (sub_dir if re.match(r"^S\d+$", sub_dir) else SUBJECT_GUESS.get(sub_dir, "S01"))
    note = "" if meta.get("subject") else "헤더에 subject 없음 — 분석 문서 기준 S01"
    rel = str(p.relative_to(HERE)).replace("\\", "/")
    for pat, (sv, nt) in SUBJECT_FIX.items():
        if re.search(pat, rel):
            sub, note = sv, nt
    mt = time.localtime(p.stat().st_mtime)
    dm = re.match(r"^(\d{8})_", stem)
    date = dm.group(1) if dm else (meta["recorded"][:8] if meta.get("recorded") else time.strftime("%Y%m%d", mt))   # 표준 v1: 헤더 recorded= 우선
    hh = mt.tm_hour if not dm else 12
    task, tdesc = "OTHER", ""
    name_for_task = stem
    for pat, code, d in TASKS:
        if re.search(pat, name_for_task):
            task, tdesc = code, d; break
    if meta.get("task"):                                          # 표준 v1: 헤더 task= 우선
        task = meta["task"]
    if stem.startswith("B") and "extensor" in stem:
        task, tdesc = "ELEC", "전극 시험 폄근"
    practice = "practice" in stem.lower()
    run = "prac" if practice else (re.findall(r"(\d+)$", stem) or ["1"])[-1].zfill(2)
    if task in ("FAT1", "FAT2"):
        run = stem.split("_", 1)[1]
    if task == "ELEC":
        run = stem.replace("_", "")
    return dict(file=str(p.relative_to(HERE.parent)).replace("\\", "/"), stem=stem, subject=sub, date=date, hour=hh, task=task,
                task_desc=tdesc, run=run, practice=practice, meta=meta, note=note)


def session_of(d, all_desc):
    if d["meta"].get("session"):                                  # 표준 v1: 녹화 때 적은 착용 번호 그대로
        return d["meta"]["session"]
    """같은 피험자·날짜에서 녹화 사이 2 시간 넘게 비면 다른 착용으로 보고 a/b 로 나눔 (착용 기록이 없어서 추정)"""
    same = sorted([x for x in all_desc if x["subject"] == d["subject"] and x["date"] == d["date"]], key=lambda x: x["mtime"])
    block, last, lab = 0, None, {}
    for x in same:
        if last is not None and x["mtime"] - last > 2 * 3600:
            block += 1
        lab[x["file"]] = block; last = x["mtime"]
    n = max(lab.values()) + 1 if lab else 1
    return d["date"] + ("" if n == 1 else "abcdefgh"[lab[d["file"]]])


# ---------------- 처리 ----------------
def load(path):
    try:
        import pandas as pd
    except ImportError:                                           # pandas 없으면 numpy 로 (느림)
        with open(path, encoding="utf-8") as fh:
            fh.readline(); cols = fh.readline().strip().split(",")
        d = np.loadtxt(path, delimiter=",", skiprows=2, encoding="utf-8")
        x = d[:, 2:10]; lab = d[:, 0].astype(np.int64); rep = d[:, 1].astype(np.int64)
        tgt = d[:, cols.index("target")].astype(np.float32) if "target" in cols else None
        return x, lab, rep, tgt
    df = pd.read_csv(path, skiprows=1, encoding="utf-8")
    x = df[[f"ch{i}_uV" for i in range(1, 9)]].to_numpy(np.float64)
    lab = df["label"].to_numpy(np.int64); rep = df["rep"].to_numpy(np.int64)
    tgt = df["target"].to_numpy(np.float32) if "target" in df.columns else None
    return x, lab, rep, tgt


def z_envelope(y, win, ref):
    c = np.cumsum(np.r_[np.zeros((1, y.shape[1])), y ** 2], 0)
    ms = np.zeros_like(y); ms[win:] = (c[win + 1:] - c[1:-win]) / win; ms[:win] = ms[win]
    return 0.5 * np.log(np.mean(ms / ref ** 2, 1) + 1e-12)


def process(path, force=False):
    from scipy import signal
    d = describe(path); d["mtime"] = Path(path).stat().st_mtime
    x, lab, rep, tgt = load(path)
    fs = int(d["meta"].get("fs", 2000)); n = len(x)
    gest_raw = E.file_classes(d["meta"])
    codes = [canon(g) for g in gest_raw]
    y = signal.sosfilt(E.full_sos(fs), x, axis=0); y[:fs] = 0
    t = np.arange(n) / fs
    # ---- 채널 품질 ----
    restm = (lab == 0) & (t > 2.0)
    actm = (lab > 0) & (t > 2.0)
    rest_rms = np.sqrt(np.mean(y[restm] ** 2, 0)) if restm.any() else np.full(8, np.nan)
    act_rms = np.sqrt(np.mean(y[actm] ** 2, 0)) if actm.any() else np.full(8, np.nan)
    seg = x[fs:min(n, fs * 61)] - np.mean(x[fs:min(n, fs * 61)], 0)
    f, P = signal.welch(seg, fs=fs, nperseg=fs, axis=0)
    band = (f >= 20) & (f <= 450); line = (np.abs(f - 60) <= 1.5)
    line_ratio = P[line].sum(0) / (P[band].sum(0) + 1e-12)
    flat = np.std(x[fs:], 0) < 1.0
    w = int(0.2 * fs); nw = (n - fs) // w
    wr = np.sqrt(np.mean(y[fs:fs + nw * w].reshape(nw, w, 8) ** 2, 1))
    art_win = (wr > E.ART_UV).any(1)
    art_frac = float(art_win.mean()) if nw else 0.0
    # ---- 시도 (실제 근육 시작·끝) ----
    ref = np.sqrt(np.percentile(y[restm] ** 2, 20, 0)) + 1e-3 if restm.any() else np.ones(8)
    z5 = z_envelope(y, int(0.025 * fs), ref)
    L = np.full(n, -1, np.int8); L[lab == 0] = 0
    events = []; k = 0
    segs = E.segments(lab)
    rest_seg = np.zeros(n, bool)                                  # '휴식 [팔 움직이기]' 처럼 지시 라벨은 동작이지만 실제로는 휴식인 구간
    for s, e, c in segs:
        if c <= 0: continue
        k += 1
        pre = z5[max(fs, s - fs):max(fs + 1, s - fs // 5)]
        hold = z5[s + min(fs, (e - s) // 3):e]
        post = z5[e + int(1.6 * fs):e + int(2.4 * fs)] if e + int(2.4 * fs) < n else pre
        zb0 = float(np.median(pre)) if len(pre) else np.nan; zh = float(np.median(hold)) if len(hold) else np.nan
        zb1 = float(np.median(post)) if len(post) else zb0
        amp = zh - max(zb0, zb1) if np.isfinite(zh) else np.nan
        on = off = None
        if np.isfinite(amp) and amp >= 0.3:
            a0, a1 = s - fs // 2, min(n, e + int(1.8 * fs))
            up = np.flatnonzero(z5[a0:a1] > zb0 + 0.5 * (zh - zb0))
            dn = np.flatnonzero(z5[e - fs // 2:a1] < zh - 0.5 * (zh - zb1))
            if len(up):
                on = a0 + int(up[0]); off = e - fs // 2 + (int(dn[0]) if len(dn) else fs // 2)
        art_t = float((np.sqrt(np.mean(y[s:e] ** 2, 0)) > E.ART_UV).any()) if e > s else 0.0
        code, frc, cond = codes[c - 1] if c - 1 < len(codes) else (f"label{c}", "", "")
        chained = s > 0 and lab[s - 1] > 0                        # 이어 하기(chain): 앞 동작에서 바로 넘어옴 → 휴식 기준이 없어 시작 시각은 지시 기준
        if code == "rest":
            rest_seg[s:e] = True; on = off = None
            ok = art_t == 0
        elif chained:
            on, off = s, e; ok = art_t == 0
        else:
            ok = on is not None and art_t == 0 and (on - s) / fs < 1.5 and off > on
        if on is not None:
            L[on:off] = c
        events.append(dict(trial=k, rep=int(rep[s]), label=int(c), gesture=code, gesture_raw=gest_raw[c - 1] if c - 1 < len(gest_raw) else "",
                           force=frc, condition=cond, cue_on_s=round(s / fs, 3), cue_off_s=round(e / fs, 3),
                           muscle_on_s=round(on / fs, 3) if on is not None else "", muscle_off_s=round(off / fs, 3) if on is not None else "",
                           rt_ms=round((on - s) / fs * 1000) if on is not None else "", hold_s=round((off - on) / fs, 2) if on is not None else "",
                           amp_z=round(amp, 2) if np.isfinite(amp) else "", artifact=int(art_t), ok=int(ok)))
    # 근육 기준 휴식: 끝 + 0.4 s 부터 다음 시작 − 0.3 s 까지만 확실한 휴식, 나머지 경계 −1
    Lm = np.full(n, -1, np.int8)
    act = L > 0
    Lm[act] = L[act]
    quiet = (lab == 0) | rest_seg
    if act.any():
        from scipy.ndimage import binary_dilation
        guard = binary_dilation(act, structure=np.ones(int(0.4 * fs) * 2 + 1, bool))
        quiet &= ~guard
    Lm[quiet] = 0
    Lm[:fs] = -1
    # ---- 저장 ----
    sub, ses = d["subject"], d["session"]
    rid = f"sub-{sub}_ses-{ses}_task-{d['task']}_run-{d['run']}"
    od = DER / f"sub-{sub}" / f"ses-{ses}"; od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"{rid}_emg.npz", emg=y.astype(np.float32), label_cue=lab.astype(np.int8), label_muscle=Lm, rep=rep.astype(np.int16),
                        target=(tgt if tgt is not None else np.zeros(0, np.float32)), fs=fs,
                        gestures=np.array([c[0] for c in codes]), gestures_raw=np.array(gest_raw), version=VERSION,
                        source=d["file"])
    with open(od / f"{rid}_events.csv", "w", newline="", encoding="utf-8-sig") as fh:
        cols = ["trial", "rep", "label", "gesture", "gesture_raw", "force", "condition", "cue_on_s", "cue_off_s", "muscle_on_s",
                "muscle_off_s", "rt_ms", "hold_s", "amp_z", "artifact", "ok"]
        wcsv = csv.DictWriter(fh, fieldnames=cols); wcsv.writeheader(); wcsv.writerows(events)
    rt = [e["rt_ms"] for e in events if e["rt_ms"] != ""]
    snr = act_rms / (rest_rms + 1e-9)
    qc = dict(duration_s=round(n / fs, 1), n_trials=len(events), trials_ok=sum(e["ok"] for e in events),
              rt_median_ms=int(np.median(rt)) if rt else "", rest_rms_uV=";".join(f"{v:.1f}" for v in rest_rms),
              snr=";".join(f"{v:.1f}" for v in snr), line60_ratio_max=round(float(np.max(line_ratio)), 3),
              flat_channels=";".join(str(i + 1) for i in np.flatnonzero(flat)), artifact_frac=round(art_frac, 3),
              dropped=d["meta"].get("dropped", ""), bad=d["meta"].get("bad", ""))
    return rid, qc, events


def flags(row):
    f = []
    if row["practice"] == "True": f.append("연습")
    if row["flat_channels"]: f.append("끊긴 채널 " + row["flat_channels"])
    if float(row["artifact_frac"] or 0) > 0.02: f.append(f"잡음 창 {float(row['artifact_frac'])*100:.0f}%")
    if float(row["line60_ratio_max"] or 0) > LINE_FLAG: f.append("60 Hz 큼")
    if row["dropped"] not in ("", "0"): f.append("누락 " + row["dropped"])
    nt, ok = int(row["n_trials"] or 0), int(row["trials_ok"] or 0)
    if nt and ok / nt < 0.8: f.append(f"통과 시도 {ok}/{nt}")
    if row["unknown_gestures"]: f.append("모르는 동작 " + row["unknown_gestures"])
    return f


def use_of(row, fl):
    if row["task"] == "NOISE": return "test"                   # 전원 잡음 시험 (power_noise_test.py) — 학습에 안 씀
    if row["practice"] == "True" or row["task"] in ("ELEC", "POSTER"): return "practice"
    if any(x.startswith(("끊긴", "잡음", "누락")) for x in fl): return "check"
    return "ok"


def run_all(files, force=False, budget=None):
    t0 = time.time()
    descs = []
    for p in files:
        d = describe(p); d["mtime"] = Path(p).stat().st_mtime; descs.append(d)
    for d in descs:
        d["session"] = session_of(d, descs)
    idx_path = OUT / "recordings.csv"
    rows = {}
    if idx_path.exists():
        with open(idx_path, encoding="utf-8-sig") as fh:
            rows = {r["source"]: r for r in csv.DictReader(fh)}
    done = 0
    for p, d in zip(files, descs):
        if not force and d["file"] in rows:
            continue
        if budget and time.time() - t0 > budget:
            print(f"시간 한도 — 나머지는 다시 실행하면 이어서 함"); break
        rid, qc, ev = process_with(p, d)
        unknown = sorted({e["gesture"] for e in ev if str(e["gesture"]).startswith("unknown")})
        row = dict(recording_id=rid, source=d["file"], subject=d["subject"], session=d["session"], date=d["date"], task=d["task"],
                   task_desc=d["task_desc"], run=d["run"], practice=str(d["practice"]),
                   gestures="|".join(sorted({e["gesture"] for e in ev})), force_levels="|".join(sorted({e["force"] for e in ev if e["force"]})),
                   reps=d["meta"].get("reps", ""), hold_s=d["meta"].get("hold", ""), rest_s=d["meta"].get("rest", ""),
                   side=d["meta"].get("side", ""), ramp=d["meta"].get("ramp", ""), chain=d["meta"].get("chain", ""),
                   unknown_gestures=";".join(unknown), note=d.get("note", ""),
                   duplicates=";".join(k for k, v in DUPES.items() if v == d["file"]), **qc)
        fl = flags(row); row["flags"] = "; ".join(fl); row["use"] = use_of(row, fl)
        rows[d["file"]] = row; done += 1
        print(f"  {rid}  {qc['duration_s']} s  시도 {qc['trials_ok']}/{qc['n_trials']}  {row['use']} {row['flags']}", flush=True)
        write_index(rows)
    write_index(rows)
    return done


def process_with(p, d):
    global describe
    orig = describe
    describe = lambda _p: d  # noqa: E731  — 세션 정보(전체 녹화를 봐야 정해짐)를 넣은 설명 사용
    try:
        return process(p)
    finally:
        describe = orig


def write_index(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    rs = sorted(rows.values(), key=lambda r: (r["subject"], r["session"], r["task"], r["run"]))
    if not rs: return
    cols = list(rs[0].keys())
    with open(OUT / "recordings.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rs)
    with open(OUT / "gestures.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["code", "name", "aliases"])
        for c, nme, al in GESTURES: w.writerow([c, nme, "|".join(al)])
    # QC 보고서
    lines = ["# 품질(QC) 보고서 — " + VERSION, "", f"녹화 {len(rs)} 개, 총 {sum(float(r['duration_s']) for r in rs)/60:.0f} 분", "",
             "| 녹화 ID | 길이 s | 시도 통과 | 반응 ms | 휴식 RMS µV (ch1~8) | 60 Hz 최대 | 잡음 창 | 구분 | 표시 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        lines.append(f"| {r['recording_id']} | {r['duration_s']} | {r['trials_ok']}/{r['n_trials']} | {r['rt_median_ms']} | {r['rest_rms_uV'].replace(';', ' ')} | "
                     f"{r['line60_ratio_max']} | {float(r['artifact_frac'])*100:.1f}% | {r['use']} | {r['flags']} |")
    (OUT / "qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def all_files():
    """내용이 같은 파일(복사본)은 하나만 — S01 폴더 쪽을 남김"""
    import hashlib
    fs = sorted(DATA.glob("S01/*.csv")) + sorted(DATA.glob("S0[2-9]/*.csv")) + sorted(DATA.glob("*.csv")) + sorted(HERE.glob("*_flexor.csv")) + sorted(HERE.glob("*_extensor.csv"))
    seen, out = {}, []
    for p in fs:
        with open(p, encoding="utf-8", errors="replace") as fh:
            if not fh.readline().startswith("# finger_test"):     # 녹화 파일만 (LAT_*·결과 CSV 등 제외)
                continue
        h = hashlib.md5()
        with open(p, "rb") as fh:
            h.update(fh.read(4 << 20)); fh.seek(0, 2); h.update(str(fh.tell()).encode())
        k = h.hexdigest()
        if k in seen:
            DUPES[str(p.relative_to(HERE.parent)).replace("\\", "/")] = seen[k]; continue
        seen[k] = str(p.relative_to(HERE.parent)).replace("\\", "/"); out.append(str(p))
    return out


DUPES = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("all"); p.add_argument("--force", action="store_true"); p.add_argument("--budget", type=float, default=None)
    p = sub.add_parser("one"); p.add_argument("csv"); p.add_argument("--force", action="store_true")
    sub.add_parser("index")
    a = ap.parse_args()
    if a.cmd == "all":
        n = run_all(all_files(), a.force, a.budget); print(f"처리 {n} 개 → {OUT}")
    elif a.cmd == "one":
        files = all_files(); tgt = str(Path(a.csv).resolve())
        if tgt not in [str(Path(f).resolve()) for f in files]: files.append(tgt)
        n = run_all([f for f in files], a.force)
        print(f"처리 {n} 개")
    else:
        rows = {}
        with open(OUT / "recordings.csv", encoding="utf-8-sig") as fh:
            rows = {r["source"]: r for r in csv.DictReader(fh)}
        for r in rows.values():                               # 표시·구분 다시 계산 (기준 바꿨을 때)
            fl = flags(r); r["flags"] = "; ".join(fl); r["use"] = use_of(r, fl)
        write_index(rows); print("다시 만듦")


if __name__ == "__main__":
    main()
