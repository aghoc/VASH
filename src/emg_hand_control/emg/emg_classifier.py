#!/usr/bin/env python3
"""
emg_classifier.py — 둘레형 8채널 패치용 실시간 손가락 동작 분류기 (PC/Jetson 공용)

구조 (실시간 1회 판단 = 50 ms 마다):
  2000 Hz 원신호 ─▶ 필터(20–450 Hz + 60·120·…·420 Hz 노치, 상태 유지)
                 ─▶ 최근 200 ms 창 ─▶ 특징(채널별 log-RMS · WL · ZC · SSC = 32개)
                 ─▶ 정규화 ─▶ LDA(공유 공분산, 수축) ─▶ 확률
                 ─▶ 다수결 평활(최근 3회) ─▶ 동작
                 ─▶ 서보 제어 단계: 동작별 RMS 패턴 투영 세기 → 2.1 Hz 평활 → 히스테리시스(켜짐 0.12/꺼짐 0.06)
                    → 데드존 0.08 → 0~90° → 속도 제한 16°/50 ms ─▶ 화면 / UDP(서보 제어 프로세스)

사용법 (ads_monitor.py, finger_test.py 와 같은 폴더):
  1) 학습 데이터 기록 (기존 도구 그대로, 휴식 구간도 학습에 사용)
       python finger_test.py record --port COM6 --ch 1-8 --side flex --out cal_flex.csv
  2) 학습 + 교차검증 정확도
       python emg_classifier.py train cal_flex.csv --out model.npz
  3a) 녹화 파일로 실시간 재생 (하드웨어 없이 '실시간처럼' 확인, 정답과 비교)
       python emg_classifier.py run --model model.npz --replay cal_flex.csv
  3b) 실제 실시간
       python emg_classifier.py run --model model.npz --port COM6
       옵션: --udp 127.0.0.1:9000  (판단 결과를 JSON 으로 전송 → Jetson 서보 프로세스)
  4) 재착용(다른 세션) 시험 — 저장된 모델을 손대지 않고 다른 녹화에 적용
       python emg_classifier.py eval --model model.npz S01_R2.csv
  5) 분류기 비교 (LDA vs SVM, 같은 특징·같은 반복 단위 교차검증, scikit-learn 필요)
       python emg_classifier.py compare S01_R1.csv
  5b) 서보 제어 단계 평가 (반복 3·4 를 학습에서 빼고 재생 → 기존/새 제어기 각도 비교)
       python emg_classifier.py evalctrl S01_R1.csv --plot ctrl.png
  6) 자가 점검 (가상 데이터)
       python emg_classifier.py demo
"""
import argparse, sys, os, time, json, socket, threading, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads_monitor import SOS, NCH, C, FrameParser, open_serial, read_line, auto_port, stop_and_report  # noqa: E402

CLASSES = ["휴식", "엄지", "검지", "중지"]     # finger_test.py 라벨 0,1,2,3 과 동일
FS = 2000
WIN = int(0.200 * FS)            # 200 ms 창
STEP = int(0.050 * FS)           # 50 ms 마다 판단
TRIM_HOLD = (0.5, 0.3)           # 학습 시 힘주기 구간 앞·뒤 제외
TRIM_REST = (1.0, 0.3)
VOTE = 3                         # 다수결 평활 길이 (3 × 50 ms)
SHRINK = 0.05                    # LDA 공분산 수축

# ---- 서보 제어 단계 (emg_filters.h 의 EMG_* 값과 같은 기준) ----
def ema_alpha(fc, dt):
    """1차 지수 평활 계수 α = 1 − exp(−2π·fc·dt)"""
    return float(1 - np.exp(-2 * np.pi * fc * dt))


CTRL_FC = 2.12                   # 세기 평활 차단 주파수 (Hz). 판단 간격 50 ms 에서 α = 0.4866 (예전 값 그대로)
CTRL_ALPHA = ema_alpha(CTRL_FC, STEP / FS)   # ※ 예전 주석의 "5 Hz" 는 20 ms 간격(펌웨어 50 Hz) 기준이었음. 5 Hz 로 바꾸면 α 0.792 — R2→R3 에서 지표 거의 같고 떨림만 증가해 유지
DZ_ON, DZ_OFF = 0.12, 0.06       # 손가락 켜짐/꺼짐 문턱 (히스테리시스, EMG_DEADZONE 0.08 을 가운데에 둠)
DEADZONE = 0.08                  # 각도 0 이 되는 세기 = EMG_DEADZONE
GAMMA = 1.0                      # 세기 → 각도 곡선 = EMG_GAMMA
THETA_MAX = 90.0                 # 최대 굽힘 각도 (°) = EMG_THETA_MAX
DELTA_MAX = 16.0                 # 판단 1회(50 ms)당 최대 각도 변화 (°) = EMG_DELTA_MAX → 320 °/s
P_MIN = 0.55                     # 이보다 확신이 낮으면 손가락을 바꾸지 않고 이전 상태 유지
CONFIRM = 1                      # 휴식 → 손가락 켜짐: 같은 후보 연속 N회(×50 ms) (다수결 3회가 이미 걸려 있어 1)
SWITCH_CONFIRM = 4               # 손가락 A 가 켜진 동안 B 로 바꾸려면 B 가 연속 N회(200 ms) 나와야 함
                                 #   → 힘 뺄 때 잠깐 생기는 다른 손가락 오판(100~150 ms)은 무시, 처음 오판은 200 ms 뒤 바로잡음
LEVEL_CLIP = 1.5
AUG_SCALES = (0.6, 0.8)       # 세기 증강: s·창 + √(1−s²)·휴식창 → 잡음 바닥은 그대로, 근전도 신호만 s배 (약한 수축 모사)
                                 #   (약하게 쥘 때 중지가 엄지로 보이는 문제 대책. S01 R2→R3: 맞는 손가락 89.7→93.4 %, 휴식 오작동 +3 %p)


# =========================== 신호 처리 ===========================
def full_sos(fs=FS):
    from scipy import signal
    rows = [SOS]
    for f0 in (240, 300, 360, 420):                      # 60 Hz 고조파 추가 노치
        b, a = signal.iirnotch(f0, 30.0, fs); rows.append(np.r_[b, a][None, :])
    return np.vstack(rows)


def features(w, zc_th):
    """w: (WIN, nch) 필터된 창 → (4*nch,) [logRMS, logWL, ZC비율, SSC비율]"""
    rms = np.sqrt(np.mean(w * w, axis=0)) + 1e-3
    d = np.diff(w, axis=0)
    wl = np.mean(np.abs(d), axis=0) + 1e-3
    s0, s1 = w[:-1], w[1:]
    zc = np.mean((s0 * s1 < 0) & (np.abs(s0 - s1) > zc_th), axis=0)
    d0, d1 = d[:-1], d[1:]
    ssc = np.mean((d0 * d1 < 0) & ((np.abs(d0) > zc_th) | (np.abs(d1) > zc_th)), axis=0)
    return np.concatenate([np.log(rms), np.log(wl), zc, ssc])


ART_UV = 400.0      # 한 채널이라도 창 RMS 가 이 값(µV)을 넘으면 잡음(전극 들뜸)으로 보고 학습·실시간 판단에서 제외
CLASSES = None      # 통합 동작 목록 (예: ["손 펴기", "주먹", ..., "엄지", "검지", "중지"]) — 설정하면 녹화마다 다른 라벨 번호를 이름으로 맞춤


def class_key(name):
    """동작 이름 → 비교용 키. '엄지 구부리기'·'엄지 굽히기'·'엄지' 는 같은 손가락 동작으로"""
    import re
    n = re.sub(r"\[.*?\]", "", name.replace("_", " ")).strip()    # [대괄호] = 화면 안내용, 동작 이름에서 제외 (예: '주먹 [꽉→살살]' = 주먹)
    for fg in ("엄지", "검지", "중지"):
        if n.startswith(fg) and "(" not in n:
            return fg + "펴기" if "펴" in n else fg          # '엄지 펴기' 는 굽히기('엄지')와 다른 동작
    return n.replace(" ", "")


def file_classes(meta):
    if meta.get("gestures"):
        return meta["gestures"].replace("_", " ").split("|")
    return ["엄지", "검지", "중지"]                     # finger_test 기본 (손가락 구부리기)


def load_csv(path):
    meta = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if first.startswith("#"):
            for kv in first[1:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1); meta[k] = v
    d = np.loadtxt(path, delimiter=",", comments="#", skiprows=2 if meta else 1, encoding="utf-8")   # 헤더 한글(동작 이름) → Windows 기본 cp949 로 읽으면 오류
    lab = d[:, 0].astype(int)
    if CLASSES is not None:                            # 이름으로 라벨 다시 매김, 목록에 없는 동작 = −1 (학습에서 제외)
        # '쥐기=주먹|집기 (중지 펴기)' 처럼 쓰면 여러 녹화 동작을 한 동작으로 합침 (10-31: 쥐기 종류는 비전이 고르는 방식 시험)
        want = {}
        for i, c in enumerate(CLASSES):
            for alt in (c.split("=", 1)[1] if "=" in c else c).split("|"):
                want[class_key(alt)] = i + 1
        m = {0: 0}
        for i, c in enumerate(file_classes(meta)):
            # '휴식 (팔 움직이기)' 처럼 휴식으로 시작하는 동작 = 휴식(0) 으로 학습 → 팔을 움직여도 손이 오작동하지 않게
            m[i + 1] = 0 if class_key(c).startswith("휴식") else want.get(class_key(c), -1)
        lab = np.array([m.get(int(v), -1) for v in lab])
    return d[:, 2:2 + NCH], lab, d[:, 1].astype(int), meta


def segments(lab):
    e = np.flatnonzero(np.diff(lab)) + 1
    return [(int(s), int(t), int(lab[s])) for s, t in zip(np.r_[0, e], np.r_[e, len(lab)])]


# =========================== 학습 ===========================
def build_dataset(paths, active, reps=None):
    from scipy import signal
    sos = full_sos()
    X, Y, R = [], [], []
    rep_offset = 0
    for p in paths:
        x, lab, rep, meta = load_csv(p)
        y = np.asarray(signal.sosfilt(sos, x, axis=0))[:, active]
        y[:FS] = 0
        for s, t, l in segments(lab):
            if l < 0:                                  # CLASSES 에 없는 동작
                continue
            tr = TRIM_HOLD if l else TRIM_REST
            a0, a1 = s + int(tr[0] * FS), t - int(tr[1] * FS)
            if a1 - a0 < WIN or a0 < FS:
                continue
            for k in range(a0, a1 - WIN + 1, STEP):
                if reps is not None and int(rep[k]) not in reps:
                    continue
                if np.sqrt(np.mean(y[k:k + WIN] ** 2, 0)).max() > ART_UV:     # 전극 들뜸·케이블 흔들림 (10-30 RPS2 끝 900~1200 µV) → 학습 제외
                    continue
                X.append(y[k:k + WIN]); Y.append(l); R.append(rep[k] + rep_offset)
        rep_offset += 1000
    return X, np.array(Y), np.array(R)


def fit_lda(F, Y, ncls):
    mu = np.array([F[Y == c].mean(0) for c in range(ncls)])
    Z = F - mu[Y]
    S = np.cov(Z.T)
    S = (1 - SHRINK) * S + SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    Si = np.linalg.inv(S)
    W = mu @ Si
    b = -0.5 * np.einsum("ij,ij->i", W, mu)
    return W, b


def lda_proba(F, W, b):
    g = F @ W.T + b
    g -= g.max(axis=-1, keepdims=True)
    p = np.exp(g)
    return p / p.sum(axis=-1, keepdims=True)


def augment_force(Xw, Y, R, scales=AUG_SCALES, seed=0):
    """동작 창 w → s·w + √(1−s²)·(무작위 휴식 창). 휴식 잡음 전력이 같다면 잡음 = s²·Pn + (1−s²)·Pn = Pn 유지, 신호만 s배.
    같은 반복 번호를 유지해 교차검증 누출 없음"""
    if not scales:
        return Xw, Y, R
    rng = np.random.default_rng(seed)
    Xw = np.asarray(Xw); rest = Xw[Y == 0]; act = np.flatnonzero(Y > 0)
    xs, ys, rs = [Xw], [Y], [R]
    for s in scales:
        xs.append(np.array([Xw[i] * s + rest[rng.integers(len(rest))] * np.sqrt(1 - s * s) for i in act]))
        ys.append(Y[act]); rs.append(R[act])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)


def prepare(paths, active, reps=None):
    """녹화 → 창 → 특징(정규화). 반환: F(원 특징), Fn(정규화), Y, R, zc_th, fmu, fsd"""
    Xw, Y, R = build_dataset(paths, active, reps)
    present = sorted(set(Y.tolist()))
    if present != [0, 1, 2, 3]:
        raise SystemExit(f"학습 데이터에 휴식/엄지/검지/중지가 모두 있어야 합니다 (있음: {present})")
    # ZC/SSC 임계값: 휴식 구간 잡음의 표준편차 × 1.0 (채널별)
    rest = np.concatenate([w for w, l in zip(Xw, Y) if l == 0])
    zc_th = rest.std(axis=0) * 1.0
    F = np.array([features(w, zc_th) for w in Xw])
    fmu, fsd = F.mean(0), F.std(0) + 1e-6
    return F, (F - fmu) / fsd, Y, R, zc_th, fmu, fsd


def rep_cv(Fn, Y, R, fit_predict):
    """반복(rep) 하나를 통째로 빼고 학습 → 그 반복으로 시험. 반환: 정확도, 혼동표"""
    conf = np.zeros((4, 4), int)
    for r in sorted(set(R.tolist())):
        if r % 1000 == 999:        # rep=-1 (첫 휴식)은 시험에서 제외
            continue
        tr, te = R != r, R == r
        if te.sum() == 0: continue
        pred = fit_predict(Fn[tr], Y[tr], Fn[te])
        for t, p in zip(Y[te], pred): conf[t, p] += 1
    return np.trace(conf) / max(1, conf.sum()), conf


def lda_fit_predict(Ftr, Ytr, Fte):
    return lda_proba(Fte, *fit_lda(Ftr, Ytr, 4)).argmax(1)


def print_conf(conf):
    print("혼동표 (행=실제, 열=예측):  " + "  ".join(f"{c:>4s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        rec = conf[i, i] / max(1, conf[i].sum())
        print(f"   {c:>4s}   " + "  ".join(f"{v:5d}" for v in conf[i]) + f"    재현율 {100*rec:5.1f}%")


def fit_model(F, Y, active, zc_th, fmu=None, fsd=None):
    """특징 → 모델 사전 (LDA + 세기 기준). 파일로 저장하거나 RealtimeEngine 에 바로 넣을 수 있음"""
    n = len(active)
    if fmu is None:
        fmu, fsd = F.mean(0), F.std(0) + 1e-6
    W, b = fit_lda((F - fmu) / fsd, Y, 4)
    # (구) 세기 기준: 창의 평균 log-RMS 를 휴식 중앙값 → 동작 중앙값 사이로 정규화
    E = F[:, :n].mean(1)
    e_rest = float(np.median(E[Y == 0]))
    e_cls = np.array([float(np.median(E[Y == c])) if c else e_rest for c in range(4)])
    # (신) 채널별 RMS 패턴: 휴식 / 동작별 중앙값 → 세기 = 현재 RMS 를 그 동작 패턴 방향으로 투영
    rms = np.exp(F[:, :n]) - 1e-3
    rms_cls = np.array([np.median(rms[Y == c], axis=0) for c in range(4)])
    return dict(W=W, b=b, fmu=fmu, fsd=fsd, zc_th=zc_th, active=np.array(active),
                e_rest=e_rest, e_cls=e_cls, rms_rest=rms_cls[0], rms_cls=rms_cls,
                classes=np.array(CLASSES), win=WIN, step=STEP, fs=FS)


def cmd_train(a):
    active = [c for c in range(NCH) if (int(a.mask, 16) >> c) & 1]
    F, Fn, Y, R, zc_th, fmu, fsd = prepare(a.csv, active)
    scales = () if a.no_augment else AUG_SCALES
    if scales:
        Xw, Y0, R0 = build_dataset(a.csv, active)
        Xa, Ya, Ra = augment_force(Xw, Y0, R0, scales)
        Fa = np.array([features(w, zc_th) for w in Xa])
        orig = np.zeros(len(Ya), bool); orig[:len(Y0)] = True
        Fan = (Fa - fmu) / fsd
        conf = np.zeros((4, 4), int)                  # 시험은 원래 창만, 학습은 증강 포함
        for r in sorted(set(R0.tolist())):
            if r % 1000 == 999:
                continue
            tr = Ra != r; te = (Ra == r) & orig
            for t, p in zip(Ya[te], lda_fit_predict(Fan[tr], Ya[tr], Fan[te])): conf[t, p] += 1
        acc = np.trace(conf) / max(1, conf.sum())
        model = fit_model(Fa, Ya, active, zc_th, fmu, fsd)
        o = fit_model(F, Y, active, zc_th, fmu, fsd)  # 세기 기준(rms·e)은 원래 창으로
        for k in ("e_rest", "e_cls", "rms_rest", "rms_cls"): model[k] = o[k]
        np.savez(a.out, **model)
        print(f"세기 증강 학습: 동작 창을 ×{', ×'.join(map(str, scales))} 로 줄인 사본 추가 (끄기: --no-augment)")
    else:
        acc, conf = rep_cv(Fn, Y, R, lda_fit_predict)
        np.savez(a.out, **fit_model(F, Y, active, zc_th, fmu, fsd))

    print(f"{C['B']}학습 완료: {a.out}{C['END']}  (창 {len(Y)}개, 채널 {[c+1 for c in active]}, 특징 {F.shape[1]}개)")
    print(f"교차검증 정확도 (반복 단위, 4클래스, 우연 25%): {C['B']}{100*acc:.1f}%{C['END']}")
    print_conf(conf)
    return acc


def cmd_eval(a):
    """저장된 모델(정규화·임계값 포함)을 그대로 다른 녹화에 적용 — 재착용·다른 날 성능"""
    m = np.load(a.model, allow_pickle=False)
    active = [int(c) for c in m["active"]]
    W, b, fmu, fsd, zc_th = m["W"], m["b"], m["fmu"], m["fsd"], m["zc_th"]
    for path in a.csv:
        Xw, Y, _ = build_dataset([path], active)
        if len(Y) == 0:
            print(f"{path}: 쓸 수 있는 창 없음"); continue
        Fn = (np.array([features(w, zc_th) for w in Xw]) - fmu) / fsd
        pred = lda_proba(Fn, W, b).argmax(1)
        conf = np.zeros((4, 4), int)
        for t, p in zip(Y, pred): conf[t, p] += 1
        acc = np.trace(conf) / conf.sum()
        print(f"{C['B']}{os.path.basename(path)}{C['END']} ← 모델 {os.path.basename(a.model)} "
              f"(재학습 없음): 정확도 {C['B']}{100*acc:.1f}%{C['END']}  (창 {len(Y)}개, 다수결 평활 없음)")
        print_conf(conf)


def cmd_compare(a):
    """같은 특징·같은 반복 단위 교차검증으로 LDA 와 SVM 비교 (판단 기준 확인용)"""
    active = [c for c in range(NCH) if (int(a.mask, 16) >> c) & 1]
    _, Fn, Y, R, *_ = prepare(a.csv, active)
    rows = [("LDA (현재, 수축 0.05)", lda_fit_predict)]
    try:
        from sklearn.svm import SVC
        def svm(**kw):
            return lambda Ftr, Ytr, Fte: SVC(**kw).fit(Ftr, Ytr).predict(Fte)
        rows += [(f"SVM-RBF C={c}", svm(C=c, gamma="scale")) for c in (0.3, 1, 3, 10)]
        rows += [("SVM 선형 C=1", svm(kernel="linear", C=1))]
    except ImportError:
        print("scikit-learn 없음 → LDA 만 계산  (pip install scikit-learn)")
    print(f"창 {len(Y)}개, 반복 {len([r for r in set(R.tolist()) if r % 1000 != 999])}개, 다수결 평활 없음")
    for name, fp in rows:
        t0 = time.perf_counter(); acc, _ = rep_cv(Fn, Y, R, fp)
        print(f"  {name:22s} {100*acc:5.1f}%   ({time.perf_counter()-t0:.2f} s)")
    print("※ SVM 의 C 는 시험 결과를 보고 고른 것이 되므로 SVM 쪽이 실제보다 유리하게 나오는 비교")


# =========================== 실시간 엔진 ===========================
class RealtimeEngine:
    """샘플을 넣으면 50 ms 마다 판단을 돌려주는 상태 기계 (PC·Jetson 공용, 화면과 분리)"""
    def __init__(self, model_path, vote=VOTE):
        from scipy import signal
        m = model_path if isinstance(model_path, dict) else dict(np.load(model_path, allow_pickle=False))
        self.model = m
        self.W, self.b = m["W"], m["b"]
        self.fmu, self.fsd, self.zc_th = m["fmu"], m["fsd"], m["zc_th"]
        self.active = [int(c) for c in m["active"]]
        self.e_rest, self.e_cls = float(m["e_rest"]), m["e_cls"]
        self.sos = full_sos()
        self.zi = np.zeros((self.sos.shape[0], 2, len(self.active)))
        self.zi_init = False
        self.buf = np.zeros((WIN, len(self.active)))
        self.since = 0
        self.votes = collections.deque(maxlen=vote)
        self.intensity = 0.0
        self.last_rms = np.zeros(len(self.active))
        self._signal = signal

    def push(self, x_uv):
        """x_uv: (n, 8) µV. 반환: 이번 호출에서 나온 판단 리스트 [(class, prob, intensity, sample_idx_in_chunk)]"""
        x = x_uv[:, self.active]
        if not self.zi_init:
            zi1 = self._signal.sosfilt_zi(self.sos)
            self.zi = zi1[:, :, None] * x[0][None, None, :]
            self.zi_init = True
        y, self.zi = self._signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        out = []
        i = 0
        n = len(y)
        while i < n:
            take = min(STEP - self.since, n - i)
            self.buf = np.roll(self.buf, -take, axis=0); self.buf[-take:] = y[i:i + take]
            self.since += take; i += take
            if self.since >= STEP:
                self.since = 0
                f = (features(self.buf, self.zc_th) - self.fmu) / self.fsd
                p = lda_proba(f[None, :], self.W, self.b)[0]
                self.votes.append(int(p.argmax()))
                cls = collections.Counter(self.votes).most_common(1)[0][0]
                self.last_rms = np.sqrt(np.mean(self.buf ** 2, axis=0))
                e = float(np.log(self.last_rms + 1e-3).mean())
                tgt = self.e_cls[cls] if cls else max(self.e_cls[1:])
                lvl = np.clip((e - self.e_rest) / max(1e-6, tgt - self.e_rest), 0, 1.5) if cls else 0.0
                self.intensity = 0.7 * self.intensity + 0.3 * lvl          # 부드럽게
                out.append((cls, float(p[cls]), float(self.intensity), i - 1, p))
        return out


# =========================== 서보 제어 단계 ===========================
class ServoController:
    """분류 결과 + 채널 RMS → 손가락 3개 목표 각도 (50 ms 마다 1회)

    1) 세기: 채널 RMS 를 휴식값으로 나눈 증가분을 각 동작의 학습 패턴 방향으로 투영 → 0(휴식) ~ 1(학습 때 평소 힘)
       동작마다 따로 계산하므로 다른 손가락이 함께 켜져도 해당 손가락 근육 패턴만 반영
    2) 지수평활 (2.1 Hz, ema_alpha 로 판단 간격에서 계산): 200 ms 창 RMS 의 들쭉날쭉을 줄임
    3) 손가락 선택: 다수결 결과가 휴식이면 해제, 확신 P_MIN 미만이면 이전 손가락 유지(깜빡임 방지),
       켜질 때는 DZ_ON 이상이 CONFIRM 회 연속, 꺼질 때는 DZ_OFF 미만 (히스테리시스)
       켜진 손가락 → 다른 손가락 전환은 SWITCH_CONFIRM 회 연속일 때만 (힘 뺄 때 잠깐 생기는 오판 차단)
    4) 각도: 데드존 → 0~1 정규화 → 감마 → 0~THETA_MAX
    5) 속도 제한: 판단 1회당 DELTA_MAX° (서보 급가속·떨림 방지). 손가락이 바뀌면 이전 손가락은 같은 속도로 복귀
    """
    def __init__(self, eng, alpha=CTRL_ALPHA, dz_on=DZ_ON, dz_off=DZ_OFF, deadzone=DEADZONE, gamma=GAMMA,
                 theta_max=THETA_MAX, delta_max=DELTA_MAX, p_min=P_MIN, confirm=CONFIRM, switch_confirm=SWITCH_CONFIRM):
        self.eng = eng
        m = eng.model
        self.has_pattern = "rms_cls" in m
        if self.has_pattern:
            # 채널마다 휴식 RMS 로 나눈 "휴식 대비 증가분" 공간에서 투영 → 휴식 잡음이 큰 채널이 세기를 좌우하지 않음.
            # 동작 패턴의 음수 성분(힘줄 때 오히려 줄어든 채널 = 잡음)은 0 으로 둠
            rest = np.maximum(np.asarray(m["rms_rest"], float), 1e-3)
            D = np.clip(np.asarray(m["rms_cls"], float) / rest - 1.0, 0.0, None)
            self.rest, self.D, self.DD = rest, D, np.maximum((D * D).sum(1), 1e-9)
        self.alpha, self.dz_on, self.dz_off, self.dz = alpha, dz_on, dz_off, deadzone
        self.gamma, self.theta_max, self.delta_max, self.p_min = gamma, theta_max, delta_max, p_min
        self.confirm, self.switch_confirm = confirm, switch_confirm
        self.pending, self.pend_n = 0, 0
        self.level = np.zeros(4)
        self.active = 0
        self.angles = np.zeros(3)

    def raw_levels(self, rms):
        if not self.has_pattern:                     # 예전 모델 파일: 평균 세기만 있음
            lv = np.zeros(4); lv[1:] = self.eng.intensity; return lv
        lv = ((np.asarray(rms) / self.rest - 1.0) @ self.D.T) / self.DD
        lv[0] = 0.0
        return np.clip(lv, 0.0, LEVEL_CLIP)

    def update(self, cls, p, rms):
        self.level += self.alpha * (self.raw_levels(rms) - self.level)
        if cls == 0:
            cand = 0
        elif p is not None and p[cls] < self.p_min:
            cand = self.active                       # 확신 낮음 → 유지
        else:
            cand = cls
        if cand == 0:
            self.active = 0
        elif cand == self.active:
            if self.level[cand] < self.dz_off:
                self.active = 0
        if cand and cand != self.active and self.level[cand] >= self.dz_on:
            self.pend_n = self.pend_n + 1 if cand == self.pending else 1
            self.pending = cand
            if self.pend_n >= (self.switch_confirm if self.active else self.confirm):
                self.active, self.pend_n = cand, 0
        elif cand != self.pending or cand == self.active:
            self.pending, self.pend_n = 0, 0
        tgt = np.zeros(3)
        if self.active:
            u = np.clip((self.level[self.active] - self.dz) / (1.0 - self.dz), 0.0, 1.0) ** self.gamma
            tgt[self.active - 1] = u * self.theta_max
        self.angles += np.clip(tgt - self.angles, -self.delta_max, self.delta_max)
        return self.angles.copy()


class LegacyController:
    """v1.0 방식 (비교용): 다수결 동작에 평균 세기 × 90°, 지수평활 0.4, 속도 제한 없음"""
    def __init__(self, eng):
        self.eng, self.angles, self.active = eng, np.zeros(3), 0

    def update(self, cls, p, rms):
        tgt = np.zeros(3)
        if cls: tgt[cls - 1] = min(1.0, self.eng.intensity) * 90
        self.angles = 0.6 * self.angles + 0.4 * tgt
        self.active = cls
        return self.angles.copy()


def control_metrics(t, ang, truth, onsets, offsets, settle=0.5, thr=10.0):
    """판단 시각 t(s), 각도 ang(T×3), 정답 truth(T) → 제어 품질 지표"""
    d = {}
    since = np.zeros(len(t))
    last = -1e9; prev = None
    for i, (tt, tr) in enumerate(zip(t, truth)):
        if tr != prev: last, prev = tt, tr
        since[i] = tt - last
    rest = (truth == 0) & (since >= settle)
    hold = (truth > 0) & (since >= settle)
    on = ang > thr
    d["휴식 중 오작동 %"] = 100 * np.mean(on[rest].any(1)) if rest.any() else np.nan
    if hold.any():
        idx = np.flatnonzero(hold); f = truth[idx] - 1
        right = on[idx, f]
        others = np.array([np.delete(on[i], ff).any() for i, ff in zip(idx, f)])
        d["힘줄 때 맞는 손가락 %"] = 100 * np.mean(right & ~others)
        d["힘줄 때 다른 손가락 움직임 %"] = 100 * np.mean(others)
    steady = (truth > 0) & (since >= 1.0)
    if steady.any():
        idx = np.flatnonzero(steady); f = truth[idx] - 1
        dd = np.abs(np.diff(ang, axis=0, prepend=ang[:1]))
        d["유지 구간 떨림 (°/50ms)"] = float(np.mean(dd[idx, f]))
    dd_all = np.abs(np.diff(ang, axis=0)).max(1)
    d["최대 순간 변화 (°/50ms)"] = float(dd_all.max()) if len(dd_all) else 0.0
    lat_on, lat_off = [], []
    for (t0, f), t1 in zip(onsets, offsets):
        k = np.flatnonzero((t >= t0) & (t < t1 + 1.0) & (ang[:, f - 1] >= 30))
        if len(k): lat_on.append(t[k[0]] - t0)
        k = np.flatnonzero((t >= t1) & (ang[:, f - 1] < thr))
        if len(k): lat_off.append(t[k[0]] - t1)
    d["켜짐 지연 중앙값 (s, 30° 도달)"] = float(np.median(lat_on)) if lat_on else np.nan
    d["꺼짐 지연 중앙값 (s, 10° 미만)"] = float(np.median(lat_off)) if lat_off else np.nan
    return d


def cmd_evalctrl(a):
    """녹화 파일로 서보 제어 단계 비교 (v1.0 방식 vs 새 제어기). 시험 반복은 학습에서 빼고 평가"""
    active = [c for c in range(NCH) if (int(a.mask, 16) >> c) & 1]
    x, lab, rep, _ = load_csv(a.csv)
    test = {int(r) for r in a.test_reps.split(",")}
    allreps = sorted(set(int(r) for r in rep))
    train = [r for r in allreps if r not in test]
    F, _, Y, _, zc_th, _, _ = prepare([a.csv], active, reps=set(train))
    eng = RealtimeEngine(fit_model(F, Y, active, zc_th), vote=a.vote)
    ctrls = {"v1.0 (기존)": LegacyController(eng), "새 제어기": ServoController(eng)}
    T, CL, ANG = [], [], {k: [] for k in ctrls}
    for k0 in range(0, len(x), 40):                 # 20 ms 묶음 = 실제 USB 수신과 비슷
        for cls, prob, inten, idx, p in eng.push(x[k0:k0 + 40]):
            T.append((k0 + idx) / FS); CL.append(cls)
            for name, c in ctrls.items():
                ANG[name].append(c.update(cls, p, eng.last_rms))
    T = np.array(T); CL = np.array(CL)
    si = np.clip((T * FS).astype(int), 0, len(lab) - 1)
    truth = lab[si]; trep = rep[si]
    sel = np.isin(trep, list(test))
    segs = [(s0, s1, int(lab[s0])) for s0, s1, _ in segments(lab)]
    onsets = [(s0 / FS, l) for s0, s1, l in segs if l and int(rep[s0]) in test]
    offsets = [s1 / FS for s0, s1, l in segs if l and int(rep[s0]) in test]
    print(f"{C['B']}{os.path.basename(a.csv)}{C['END']}  학습 반복 {train} / 시험 반복 {sorted(test)}  "
          f"(판단 {sel.sum()}회, 손가락 동작 {len(onsets)}번)")
    res = {name: control_metrics(T[sel], np.array(A)[sel], truth[sel], onsets, offsets) for name, A in ANG.items()}
    keys = list(next(iter(res.values())).keys())
    print(f"  {'지표':34s}" + "".join(f"{n:>14s}" for n in res))
    for k in keys:
        print(f"  {k:34s}" + "".join(f"{res[n][k]:14.2f}" for n in res))
    print("  ※ 켜짐 지연에는 화면 지시 → 실제 힘주기까지의 사람 반응시간(약 0.2~0.4 s)이 포함됨")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        ts = T[sel]; t0 = ts[0]
        fig, axes = plt.subplots(len(res) + 1, 1, figsize=(13, 2.4 * (len(res) + 1)), sharex=True)
        cols = ["#c62828", "#1565c0", "#6a1b9a"]
        for f in range(3):
            axes[0].fill_between(ts - t0, 0, (truth[sel] == f + 1) * 1.0, color=cols[f], alpha=0.5, step="mid",
                                 label=CLASSES[f + 1])
        axes[0].set_yticks([]); axes[0].set_title("정답 (화면 지시)"); axes[0].legend(loc="upper right", ncol=3)
        for ax, (name, A) in zip(axes[1:], ANG.items()):
            A = np.array(A)[sel]
            for f in range(3): ax.plot(ts - t0, A[:, f], color=cols[f], lw=1.2)
            ax.set_ylim(-5, 100); ax.set_ylabel("각도 (°)"); ax.set_title(name)
        axes[-1].set_xlabel("시간 (s)")
        fig.tight_layout(); fig.savefig(a.plot, dpi=110); plt.close(fig)
        print(f"  그림: {a.plot}")
    return res


# =========================== 실시간 화면 ===========================
def cmd_run(a):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    eng = RealtimeEngine(a.model, vote=a.vote)
    udp = None
    if a.udp:
        host, port = a.udp.split(":"); udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))

    # ---- 데이터 원천: 재생 파일 또는 실제 보드 ----
    q = collections.deque(); qlock = threading.Lock(); stop = threading.Event()
    truth = None
    if a.replay:
        x_all, lab_all, _, _ = load_csv(a.replay)
        truth = lab_all
        def feeder():
            t0 = time.perf_counter(); k = 0; N = len(x_all)
            while not stop.is_set() and k < N:
                due = int((time.perf_counter() - t0) * FS * a.speed)
                if due > k:
                    j = min(N, due)
                    with qlock: q.append((x_all[k:j], k, time.perf_counter()))
                    k = j
                time.sleep(0.002)
            stop.set()
        src = "재생: " + os.path.basename(a.replay)
        s = None
    else:
        port = a.port or auto_port()
        if not port: raise SystemExit("COM 포트를 찾지 못함 → --port 지정")
        s = open_serial(port, a.baud)
        s.write(b"X"); time.sleep(0.3); s.reset_input_buffer()
        mask = sum(1 << c for c in eng.active)
        s.write(f"K{mask:02X}".encode()); time.sleep(0.2); s.write(b"L")
        hdr = None; t0 = time.time()
        while time.time() - t0 < 6:
            line = read_line(s, 1)
            if line and line.startswith("#LIVE"):
                hdr = dict(kv.split("=") for kv in line.split(",")[1:]); break
        if not hdr: raise SystemExit("스트리밍 시작 응답 없음 — selftest 먼저 확인")
        uv = float(hdr["vref"]) / float(hdr["gain"]) / 2 ** 23 * 1e6
        parser = FrameParser()
        def feeder():
            k = 0
            while not stop.is_set():
                d = s.read(max(4096, s.in_waiting))
                if d:
                    parser.feed(d)
                    c, _ = parser.take()
                    if len(c):
                        with qlock: q.append((c * uv, k, time.perf_counter()))
                        k += len(c)
        src = f"실시간: {port}"
    threading.Thread(target=feeder, daemon=True).start()

    # ---- 화면 ----
    fig = plt.figure(figsize=(13, 7.5))
    fig.suptitle(f"근전도 손가락 분류기 — {src}", fontsize=13)
    ax_big = fig.add_axes((0.04, 0.62, 0.40, 0.30)); ax_big.axis("off")
    t_cls = ax_big.text(0.5, 0.62, "", ha="center", va="center", fontsize=60, weight="bold")
    t_sub = ax_big.text(0.5, 0.12, "", ha="center", va="center", fontsize=12)
    ax_p = fig.add_axes((0.52, 0.62, 0.44, 0.28))
    bars_p = ax_p.bar(CLASSES, [0] * 4, color=["#9e9e9e", "#c62828", "#1565c0", "#6a1b9a"])
    ax_p.set_ylim(0, 1); ax_p.set_title("동작 확률 (LDA)", fontsize=10)
    ax_f = fig.add_axes((0.04, 0.30, 0.40, 0.25))
    bars_f = ax_f.barh(["중지", "검지", "엄지"], [0, 0, 0], color=["#6a1b9a", "#1565c0", "#c62828"])
    ax_f.set_xlim(0, 90); ax_f.set_xlabel("손가락 목표 각도 (°) — 서보 명령 (데드존·히스테리시스·속도 제한 적용)", fontsize=9)
    ax_h = fig.add_axes((0.52, 0.30, 0.44, 0.22))
    ax_h.set_title("최근 10 s 판단 (위) / 정답 (아래, 재생 시)", fontsize=10)
    ax_h.set_ylim(-1.2, 1.2); ax_h.set_yticks([]); ax_h.set_xlim(-10, 0)
    cmap = {0: "#bdbdbd", 1: "#c62828", 2: "#1565c0", 3: "#6a1b9a"}
    ax_t = fig.add_axes((0.04, 0.03, 0.92, 0.20)); ax_t.axis("off")
    t_info = ax_t.text(0, 1, "", va="top", fontsize=10, family=["Consolas", "DejaVu Sans Mono", "Malgun Gothic"])
    state = {"run": True}
    fig.canvas.mpl_connect("close_event", lambda e: state.update(run=False))
    fig.canvas.mpl_connect("key_press_event", lambda e: state.update(run=False) if getattr(e, "key", None) == "q" else None)
    plt.ion(); plt.show()

    # ---- 처리 스레드: 화면과 분리 (화면이 느려도 판단·UDP 출력은 제때) ----
    sh = {"hist": collections.deque(maxlen=200), "lat": collections.deque(maxlen=400),
          "ok": 0, "tot": 0, "last": (0, 0.0, 0.0, None), "angles": np.zeros(3), "pos": 0}
    shlock = threading.Lock()

    ctrl = ServoController(eng)

    def worker():
        angles = np.zeros(3)
        while not (stop.is_set() and not q):
            with qlock:
                chunks = list(q); q.clear()
            if not chunks:
                time.sleep(0.001); continue
            for x, k0, t_arr in chunks:
                for cls, prob, inten, idx, p in eng.push(x):
                    now = time.perf_counter()
                    tsamp = (k0 + idx) / FS
                    tr = int(truth[min(len(truth) - 1, k0 + idx)]) if truth is not None else None
                    angles = ctrl.update(cls, p, eng.last_rms)
                    if udp:
                        msg = {"t": round(tsamp, 3), "cls": CLASSES[cls], "id": cls, "p": round(prob, 3),
                               "intensity": round(inten, 3), "active": CLASSES[ctrl.active],
                               "level": round(float(ctrl.level[ctrl.active]), 3) if ctrl.active else 0.0,
                               "angles_deg": [round(v, 1) for v in angles]}
                        udp[0].sendto(json.dumps(msg, ensure_ascii=False).encode(), udp[1])
                    with shlock:
                        sh["lat"].append((now - t_arr) * 1000)
                        sh["hist"].append((tsamp, cls, tr))
                        if tr is not None:
                            sh["tot"] += 1; sh["ok"] += int(tr == cls)
                        sh["last"] = (cls, prob, inten, p); sh["angles"] = angles.copy()
                with shlock:
                    sh["pos"] = k0 + len(x)
    wth = threading.Thread(target=worker, daemon=True); wth.start()

    from matplotlib.colors import ListedColormap
    strip = np.full((2, 200), np.nan)
    im = ax_h.imshow(strip, aspect="auto", cmap=ListedColormap([cmap[i] for i in range(4)]), vmin=-0.5, vmax=3.5,
                     extent=(-10, 0, -1.2, 1.2), interpolation="nearest")
    t_start = time.perf_counter()
    try:
        while state["run"] and wth.is_alive():
            with shlock:
                cls, prob, inten, p = sh["last"]; angles = sh["angles"]
                hist = list(sh["hist"]); lat = list(sh["lat"]); ok, tot, pos = sh["ok"], sh["tot"], sh["pos"]
            t_cls.set_text(CLASSES[cls]); t_cls.set_color(cmap[cls])
            t_sub.set_text(f"확률 {prob:.2f}   세기 {min(inten, 1.5):.2f}")
            if p is not None:
                for bar, v in zip(bars_p, p): bar.set_height(v)
            for bar, v in zip(bars_f, angles[::-1]): bar.set_width(v)
            strip[:] = np.nan
            if hist:
                tn = hist[-1][0]
                for (tt, c, trv) in hist:
                    j = 199 - int(round((tn - tt) / 0.05))
                    if 0 <= j < 200:
                        strip[0, j] = c
                        if trv is not None: strip[1, j] = trv
            im.set_data(strip)
            el = time.perf_counter() - t_start
            lat_txt = f"{np.median(lat):5.1f} ms (최대 {np.max(lat):5.1f})" if lat else "-"
            acc_txt = f"   실시간 일치율 {100*ok/max(1,tot):5.1f}% (전환 구간 포함)" if truth is not None else ""
            t_info.set_text(
                f"처리 지연(샘플 도착→판단): {lat_txt}   |   판단 주기 50 ms, 창 200 ms, 다수결 {a.vote}회   |   "
                f"진행 {pos/FS:6.1f} s / 경과 {el:6.1f} s{acc_txt}\n"
                f"전체 반응 지연(이론) ≈ 창 중심 100 ms + 판단 주기 ≤50 ms + 다수결 ≈{(a.vote-1)//2*50 + 50} ms + 처리"
                + (f"\nUDP 출력 → {a.udp}" if a.udp else ""))
            fig.canvas.draw_idle(); fig.canvas.flush_events(); plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); time.sleep(0.2)
        if s is not None:
            stop_and_report(s)
        with shlock:
            ok, tot, lat = sh["ok"], sh["tot"], list(sh["lat"])
        if truth is not None and tot:
            print(f"재생 결과: 실시간 일치율 {100*ok/tot:.1f}% ({tot}회 판단, 동작 전환 직후 포함)")
        if lat:
            print(f"처리 지연 중앙값 {np.median(lat):.1f} ms, 95% {np.percentile(lat, 95):.1f} ms, 최대 {np.max(lat):.1f} ms")


# =========================== 자가 점검 ===========================
def cmd_demo(a):
    import tempfile, random
    rng = np.random.default_rng(1)
    d = tempfile.mkdtemp(); path = os.path.join(d, "demo_cal.csv")
    patt = {0: np.full(NCH, 2.0), 1: np.array([30, 25, 4, 6, 20, 25, 4, 3.]),
            2: np.array([4, 5, 8, 22, 6, 18, 20, 25.]), 3: np.array([5, 12, 40, 10, 4, 5, 28, 12.])}
    plan = [(0, -1, 5.0)]
    for r in range(5):
        o = [1, 2, 3]; random.Random(r).shuffle(o)
        for f in o: plan += [(f, r, 3.0), (0, r, 3.0)]
    xs, labs, reps = [], [], []
    for l, r, dur in plan:
        n = int(dur * FS); g = rng.uniform(0.7, 1.3, NCH)
        xs.append(rng.standard_normal((n, NCH)) * (patt[l] * (g if l else 1)) + 3 * np.sin(2 * np.pi * 60 * np.arange(n) / FS)[:, None])
        labs.append(np.full(n, l)); reps.append(np.full(n, r))
    x = np.concatenate(xs); lab = np.concatenate(labs); rep = np.concatenate(reps)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# finger_test fs=2000 mask=FF side=flex reps=5 hold=3 rest=3 dropped=0 bad=0\n")
        f.write("label,rep," + ",".join(f"ch{i+1}_uV" for i in range(NCH)) + "\n")
        for l, r, row in zip(lab, rep, x):
            f.write(f"{l},{r}," + ",".join(f"{v:.3f}" for v in row) + "\n")
    model = os.path.join(d, "demo_model.npz")
    acc = cmd_train(argparse.Namespace(csv=[path], out=model, mask="FF", no_augment=False))
    # 실시간 엔진을 시간 순서대로 돌려 판단 일치율·지연 측정 (화면 없이)
    eng = RealtimeEngine(model)
    ctrl = ServoController(eng)
    ok = tot = 0; t0 = time.perf_counter()
    TT, AA, TR = [], [], []
    for k in range(0, len(x), 100):
        for cls, prob, inten, idx, p in eng.push(x[k:k + 100]):
            tr = lab[min(len(lab) - 1, k + idx)]
            ang = ctrl.update(cls, p, eng.last_rms)
            if k + idx > 2 * FS:
                tot += 1; ok += int(tr == cls)
                TT.append((k + idx) / FS); AA.append(ang); TR.append(tr)
    dt = (time.perf_counter() - t0) / max(1, tot) * 1000
    print(f"실시간 엔진 일치율 {100*ok/tot:.1f}% (전환 포함), 판단 1회 처리시간 {dt:.2f} ms (제어 포함)")
    segs = [(s0 / FS, s1 / FS, int(lab[s0])) for s0, s1, _ in segments(lab) if lab[s0]]
    cm = control_metrics(np.array(TT), np.array(AA), np.array(TR), [(s0, l) for s0, _, l in segs], [s1 for _, s1, _ in segs])
    jump = cm["최대 순간 변화 (°/50ms)"]
    print(f"제어: 휴식 중 오작동 {cm['휴식 중 오작동 %']:.1f}%, 맞는 손가락 {cm['힘줄 때 맞는 손가락 %']:.1f}%, "
          f"최대 순간 변화 {jump:.1f}°/50ms (한계 {DELTA_MAX})")
    good = (acc > 0.95 and ok / tot > 0.85 and dt < 20 and np.isfinite(np.array(AA)).all()
            and cm["휴식 중 오작동 %"] < 5 and cm["힘줄 때 맞는 손가락 %"] > 85 and jump <= DELTA_MAX + 1e-6)
    print("자가 점검:", f"{C['PASS']}PASS{C['END']}" if good else f"{C['FAIL']}FAIL{C['END']}")
    return good


def main():
    ap = argparse.ArgumentParser(description="실시간 손가락 동작 분류기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("csv", nargs="+"); p.add_argument("--out", default="model.npz")
    p.add_argument("--no-augment", action="store_true", help="세기 증강 끄기")
    p.add_argument("--mask", default="FF", help="사용 채널 16진 마스크 (기본 FF = 8채널)")
    p = sub.add_parser("run"); p.add_argument("--model", default="model.npz")
    p.add_argument("--port", default=None); p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--replay", default=None); p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--vote", type=int, default=VOTE); p.add_argument("--udp", default=None)
    p = sub.add_parser("eval"); p.add_argument("--model", default="model.npz"); p.add_argument("csv", nargs="+")
    p = sub.add_parser("compare"); p.add_argument("csv", nargs="+"); p.add_argument("--mask", default="FF")
    p = sub.add_parser("evalctrl", help="녹화 파일로 서보 제어 단계 평가 (기존 vs 새 제어기)")
    p.add_argument("csv"); p.add_argument("--test-reps", default="3,4"); p.add_argument("--mask", default="FF")
    p.add_argument("--vote", type=int, default=VOTE); p.add_argument("--plot", default=None)
    sub.add_parser("demo")
    a = ap.parse_args()
    if a.cmd == "train": cmd_train(a)
    elif a.cmd == "run": cmd_run(a)
    elif a.cmd == "eval": cmd_eval(a)
    elif a.cmd == "compare": cmd_compare(a)
    elif a.cmd == "evalctrl": cmd_evalctrl(a)
    else: sys.exit(0 if cmd_demo(a) else 1)


if __name__ == "__main__":
    main()
