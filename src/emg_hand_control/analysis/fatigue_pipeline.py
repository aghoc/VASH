# (랩실용 피로적응_파이프라인/fatigue_pipeline.py 에서 2026-10-27 복사. 원본은 그대로 둠. 대회용 통합은 fatigue_experiment.py)
# 피로 조건부 리만 EMG 파이프라인 (FC-Riemann)
# 대회용 파이프라인을 바탕으로 수정, 보완하고 피로 보정 모듈을 추가함
# 구성
#   1. 신호 정리: StreamFilter (대회용과 같음)
#   2. 특징: TD 32개 + 휴식 재중심화 리만 특징 (선택: 시간지연 공분산)
#   3. 피로 상태 추정: FatigueMonitor (라벨 없음, 클래스별 MDF 비율)
#   4. 분류: FatigueConditionedLDA (클래스 평균이 피로 상태 phi 의 함수)
#   5. 적응: BoundedAdapter (피로 정도로 보정 폭을 묶음)
#   6. 신뢰도: 온도 보정 + MondrianConformal (피로 구간별 순응 예측)
#   7. 세기: IntensityComp (피로로 커진 RMS 를 나눠서 의도 세기로 되돌림)
#   8. 실시간 묶음: Pipeline (빠른 루프 50 ms, 느린 루프 2 s)
import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar

FS = 2000


# region 1. 신호 정리
def design_sos(fs=FS, hp=20.0, lp=450.0, hum=60.0, hum_max=420.0, q=30.0):
    """20-450 Hz 대역통과 + 60 Hz 배수 노치 (대회용과 같은 구성)"""
    secs = [signal.butter(2, hp, "highpass", fs=fs, output="sos"),
            signal.butter(2, lp, "lowpass", fs=fs, output="sos")]
    f0 = hum
    while f0 <= hum_max + 1e-9:
        b, a = signal.iirnotch(f0, q, fs)
        secs.append(signal.tf2sos(b, a))
        f0 += hum
    return np.vstack(secs)


class StreamFilter:
    """묶음 경계에서 필터 상태를 이어 받음 (zi)"""
    def __init__(self, n_ch, sos=None):
        self.sos = design_sos() if sos is None else sos
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))

    def __call__(self, x):
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y
# endregion


# region 2. 특징
def td_features(w, thr):
    """Hudgins 시간영역 특징. 채널별 log-RMS, log-파형길이, 영교차율, 기울기부호변화율"""
    n = len(w)
    rms = np.sqrt(np.mean(w ** 2, 0)) + 1e-12
    wl = np.sum(np.abs(np.diff(w, axis=0)), 0) / n + 1e-12
    a, b = w[:-1], w[1:]
    zc = np.sum((a * b < 0) & (np.abs(a - b) >= thr), 0) / n
    d1, d2 = w[1:-1] - w[:-2], w[1:-1] - w[2:]
    ssc = np.sum(d1 * d2 >= thr ** 2, 0) / n
    return np.concatenate([np.log(rms), np.log(wl), zc, ssc])


def window_cov(w, lag=0, eps=1e-6):
    """채널 공분산. lag > 0 이면 [x_t; x_(t-lag)] 의 2n x 2n 공분산 (스펙트럼 정보 포함)"""
    if lag:
        w = np.hstack([w[lag:], w[:-lag]])
    w = w - w.mean(0)
    c = w.T @ w / len(w)
    return c + eps * np.trace(c) / len(c) * np.eye(len(c))


def _eigfn(c, fn):
    e, v = np.linalg.eigh(c)
    return (v * fn(e)) @ v.T


class Tangent:
    """휴식 재중심화 후 행렬 로그. 비대각 성분 x sqrt(2)"""
    def __init__(self, n):
        self.iu = np.triu_indices(n)
        self.wt = np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0))
        self.ris = np.eye(n)

    def set_reference(self, rest_covs):
        # 기준 R = 휴식 공분산의 로그-유클리드 평균
        L = np.mean([_eigfn(c, np.log) for c in rest_covs], 0)
        R = _eigfn(L, np.exp)
        self.ris = _eigfn(R, lambda e: e ** -0.5)

    def __call__(self, c):
        s = _eigfn(self.ris @ c @ self.ris, np.log)
        return s[self.iu] * self.wt


class FeatureExtractor:
    def __init__(self, n_ch, lag=0):
        self.lag = lag
        self.tan = Tangent(n_ch * (2 if lag else 1))
        self.thr = np.zeros(n_ch)

    def calibrate_rest(self, rest_windows):
        # 영교차, 기울기 문턱 = 휴식 표준편차, 리만 기준 = 휴식 평균
        self.thr = np.mean([w.std(0) for w in rest_windows], 0)
        self.tan.set_reference([window_cov(w, self.lag) for w in rest_windows])

    def __call__(self, w):
        return np.concatenate([td_features(w, self.thr), self.tan(window_cov(w, self.lag))])
# endregion


# region 3. 피로 상태 추정
def median_freq(w, fs=FS, band=(20.0, 450.0), hum=60.0, guard=4.0):
    """Welch 스펙트럼에서 60 Hz 배수 +-guard 를 빼고 누적 면적 절반 지점을 보간"""
    nper = 512 if len(w) >= 512 else 256
    f, p = signal.welch(w, fs, nperseg=nper, noverlap=nper // 2, axis=0)
    m = (f >= band[0]) & (f <= band[1])
    h = hum
    while h <= band[1]:
        m &= np.abs(f - h) > guard
        h += hum
    f, p = f[m], p[m]
    c = np.cumsum(p, 0)
    return np.array([np.interp(c[-1, j] / 2, c[:, j], f) for j in range(p.shape[1])])


class FatigueMonitor:
    """라벨 없는 피로 지수 phi = MDF / 같은 동작의 처음 MDF
    - MDF 는 동작과 힘에 따라 다르므로 반드시 같은 클래스끼리 비교함 (대회용은 섞어서 비교했음)
    - 휴식 MDF 비율을 따로 추적. 휴식이 바뀌면 피로가 아닌 전극, 피부 변화로 판단함
    - CUSUM 으로 피로 시작을 검출함. 창 하나는 잡음이 커서 활성 창 n_blk 개 평균(블록)에 적용
    - 블록 잡음 sigma 는 피로 전 학습 데이터에서 잼 (calibrate_noise)
    - 문턱 h = ln(평균 오경보 간격 / 블록 간격). 기본 10분 / 약 1 s"""
    def __init__(self, mdf0, rest_class=0, rest_mdf0=None, lam=0.03,
                 phi_range=(0.6, 1.05), cusum_delta=0.04, cusum_sigma=0.03,
                 n_blk=20, arl0_blocks=600):
        self.mdf0 = mdf0                  # {클래스: 채널별 MDF}
        self.rest = rest_class
        self.rest_mdf0 = rest_mdf0
        self.lam, self.rng = lam, phi_range
        self.phi, self.rest_ratio = 1.0, 1.0
        self.d, self.s2, self.h = cusum_delta, cusum_sigma ** 2, np.log(arl0_blocks)
        self.n_blk, self.blk = n_blk, []
        self.S, self.alarm, self.n_active, self.alarm_at = 0.0, False, 0, None

    def window_ratio(self, w, p, classes):
        """창 하나의 MDF 비율 (동작 확률 가중). 활성 창이 아니면 None"""
        p = np.asarray(p)
        ir = int(np.searchsorted(classes, self.rest))
        if p[ir] > 0.2:
            return None
        mdf = median_freq(w)
        pw = np.mean(w ** 2, 0)
        num = den = 0.0
        for i, c in enumerate(classes):
            if c == self.rest or c not in self.mdf0:
                continue
            num += p[i] * np.sum(pw * mdf / self.mdf0[c])
            den += p[i] * np.sum(pw)
        return float(np.clip(num / den, *self.rng))

    def calibrate_noise(self, ratios):
        """피로 전 창별 비율 목록으로 블록 평균의 표준편차를 잼 (겹친 창의 상관이 자동 반영됨)"""
        r = np.asarray(ratios)
        nb = len(r) // self.n_blk
        if nb >= 3:
            self.s2 = float(np.var(r[:nb * self.n_blk].reshape(nb, -1).mean(1), ddof=1))
        return np.sqrt(self.s2)

    def update(self, w, p, classes):
        """p: 클래스 확률. 동작 구분이 불확실해도 휴식이냐 아니냐만 확실하면 갱신함
        - 휴식 확률 0.8 이상: 휴식 MDF 비율 갱신
        - 휴식 확률 0.2 이하: 동작 확률로 가중한 MDF 비율 sum_c p_c MDF / MDF0_c 로 phi 갱신
        - 피로로 분류가 흔들리는 바로 그때 phi 가 멈추는 문제를 피함"""
        p = np.asarray(p)
        ir = int(np.searchsorted(classes, self.rest))
        mdf = None
        if p[ir] >= 0.8:
            if self.rest_mdf0 is not None:
                r = np.median(median_freq(w) / self.rest_mdf0)
                self.rest_ratio = (1 - self.lam) * self.rest_ratio + self.lam * r
            return self.phi
        if p[ir] > 0.2:
            return self.phi
        r = self.window_ratio(w, p, classes)
        self.phi = (1 - self.lam) * self.phi + self.lam * r
        self.n_active += 1
        self.blk.append(r)
        if len(self.blk) == self.n_blk:
            z = 1 - np.mean(self.blk)
            self.blk = []
            # 아래 방향 CUSUM (정규 로그우도비). 정상 평균 0, 변화량 delta
            self.S = max(0.0, self.S + (self.d / self.s2) * (z - self.d / 2))
            if self.S > self.h and not self.alarm:
                self.alarm, self.alarm_at = True, self.n_active
        return self.phi

    @property
    def non_fatigue_drift(self):
        return abs(self.rest_ratio - 1) > 0.05
# endregion


# region 4. 분류
def softmax(g):
    g = g - g.max(1, keepdims=True)
    e = np.exp(g)
    return e / e.sum(1, keepdims=True)


class FatigueConditionedLDA:
    """수축 LDA. 클래스 평균 mu_c(phi) = mu_c0 + (1 - phi) v_c + delta_c
    - v_c: 이전 세션이나 F1 피로 실험에서 배운 피로 이동 방향 (없으면 0, 일반 LDA 와 같음)
    - delta_c: BoundedAdapter 가 채우는 잔여 보정"""
    def __init__(self, shrink=0.05):
        self.shrink, self.tau = shrink, 1.0

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mu0 = np.array([X[y == c].mean(0) for c in self.classes])
        Xc = X - self.mu0[np.searchsorted(self.classes, y)]
        S = Xc.T @ Xc / (len(X) - len(self.classes))
        n = S.shape[0]
        S = (1 - self.shrink) * S + self.shrink * np.trace(S) / n * np.eye(n)
        self.Si = np.linalg.inv(S)
        self.v = np.zeros_like(self.mu0)
        self.delta = np.zeros_like(self.mu0)
        return self

    def fit_fatigue(self, X, y, phi):
        """v_c = argmin sum ||x - mu_c0 - (1-phi) v_c||^2 (원점 통과 회귀)"""
        u = 1 - phi
        for i, c in enumerate(self.classes):
            m = (y == c) & (u > 1e-3)
            if m.sum() < 10:
                continue
            self.v[i] = (u[m, None] * (X[m] - self.mu0[i])).sum(0) / np.sum(u[m] ** 2)
        return self

    def means(self, phi):
        return self.mu0 + (1 - phi) * self.v + self.delta

    def decision(self, X, phi=1.0):
        M = self.means(phi)
        return X @ self.Si @ M.T - 0.5 * np.sum(M @ self.Si * M, 1)

    def proba(self, X, phi=1.0):
        return softmax(self.decision(np.atleast_2d(X), phi) / self.tau)

    def fit_temperature(self, X, y, phi):
        """tau 는 음의 로그우도를 최소화 (1차원 볼록 문제, log tau 로 탐색)"""
        g = np.vstack([self.decision(X[i:i + 1], phi[i]) for i in range(len(X))])
        idx = np.searchsorted(self.classes, y)
        def nll(lt):
            p = softmax(g / np.exp(lt))
            return -np.mean(np.log(p[np.arange(len(y)), idx] + 1e-12))
        self.tau = float(np.exp(minimize_scalar(nll, bounds=(-4, 6), method="bounded").x))
        return self
# endregion


# region 5. 적응
class BoundedAdapter:
    """라벨 없는 잔여 보정. 대회용 자가 적응 LDA 를 고친 것
    - 후보 집합이 하나인 창만 모음 (확신도 0.9 대신 순응 예측 사용)
    - 갱신 폭을 마할라노비스 거리로 b0 + b1 (1 - phi) 안에 묶음. 피로가 클수록 더 움직일 수 있음
    - 휴식 클래스는 따로. 휴식 이동은 피로가 아니라 재중심화로 처리함"""
    def __init__(self, model, eta=0.3, b0=0.3, b1=6.0, min_n=15, rest_class=0):
        self.m, self.eta, self.b0, self.b1, self.min_n = model, eta, b0, b1, min_n
        self.rest = rest_class
        self.buf = {c: [] for c in model.classes}

    def collect(self, x, cls, singleton):
        if singleton and cls != self.rest:
            self.buf[cls].append(x)

    def step(self, phi):
        m = self.m
        base = m.mu0 + (1 - phi) * m.v
        lim = self.b0 + self.b1 * (1 - phi)
        for i, c in enumerate(m.classes):
            if len(self.buf[c]) < self.min_n:
                continue
            target = np.mean(self.buf[c], 0) - base[i]
            d = (1 - self.eta) * m.delta[i] + self.eta * target
            nrm = np.sqrt(d @ m.Si @ d)
            if nrm > lim:
                d *= lim / nrm
            m.delta[i] = d
            self.buf[c] = []


class UnboundedSelfAdapt:
    """비교용: 대회용 방식 (확신도 0.9 이상 창 평균으로 mu <- 0.5 mu + 0.5 평균, 제한 없음)"""
    def __init__(self, model, conf_min=0.9, min_n=15):
        self.m, self.conf_min, self.min_n = model, conf_min, min_n
        self.buf = {c: [] for c in model.classes}

    def collect(self, x, cls, conf):
        if conf >= self.conf_min:
            self.buf[cls].append(x)

    def step(self, phi=1.0):
        for i, c in enumerate(self.m.classes):
            if len(self.buf[c]) >= self.min_n:
                cur = self.m.means(phi)[i]
                self.m.delta[i] += 0.5 * (np.mean(self.buf[c], 0) - cur)
                self.buf[c] = []
# endregion


# region 6. 신뢰도
class MondrianConformal:
    """피로 구간별 순응 예측
    - 일반 순응 예측은 교환가능성 가정이라 피로로 분포가 바뀌면 90% 보장이 깨짐
    - 피로 구간(phi 구간)마다 따로 임계값을 잡으면 구간 안에서는 보장이 유지됨 (Vovk 의 Mondrian 방식)
    - 보정 데이터는 피로 구간을 포함한 라벨 있는 세션에서 얻어야 함"""
    def __init__(self, alpha=0.1, edges=(0.95, 0.90), min_n=30):
        self.alpha, self.edges, self.min_n = alpha, edges, min_n

    def _bin(self, phi):
        return int(np.sum(np.asarray(phi)[..., None] < np.array(self.edges), -1))

    def _q(self, s):
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        return np.sort(s)[min(k, n) - 1]

    def fit(self, p_true, phi):
        s = 1 - np.asarray(p_true)
        b = np.array([self._bin(x) for x in phi])
        self.q_all = self._q(s)
        self.q = {}
        for i in range(len(self.edges) + 1):
            self.q[i] = self._q(s[b == i]) if np.sum(b == i) >= self.min_n else self.q_all
        return self

    def predict_set(self, p, phi=1.0, mondrian=True):
        q = self.q[self._bin(phi)] if mondrian else self.q_all
        return p >= 1 - q
# endregion


# region 7. 세기와 평활
class IntensityComp:
    """같은 힘에서 피로로 RMS 가 커짐 -> 세기를 과대추정함
    - g(phi) = 1 + gamma (1 - phi), gamma 는 힘 고정 피로 실험(F1)에서 RMS 비율로 맞춤
    - 보정 세기 a' = a / g(phi)"""
    def __init__(self, gamma=0.0):
        self.gamma = gamma

    def fit(self, phi, rms_ratio):
        u = 1 - np.asarray(phi)
        self.gamma = float(np.sum(u * (np.asarray(rms_ratio) - 1)) / np.sum(u ** 2))
        return self

    def __call__(self, a, phi):
        return a / (1 + self.gamma * (1 - phi))


def ema_alpha(fc, dt):
    """1차 지수 평활 계수. 정확한 관계 alpha = 1 - exp(-2 pi fc dt)
    - dt 50 ms 에서 alpha 0.4866 은 fc 2.1 Hz 임 (대회용 표의 5 Hz 는 dt 약 21 ms 일 때만 맞음)"""
    return 1 - np.exp(-2 * np.pi * fc * dt)


def ema_cutoff(alpha, dt):
    return -np.log(1 - alpha) / (2 * np.pi * dt)
# endregion


# region 8. 실시간 묶음
class Pipeline:
    """빠른 루프: 50 ms 마다 창 200 ms 로 분류
    느린 루프: 2 s 마다 잔여 보정 갱신
    phi 는 매 창 갱신되지만 시간상수가 수 초라 분류 지연에는 영향 없음"""
    def __init__(self, fe, model, monitor, conformal, adapter=None, intensity=None,
                 n_ch=8, win=400, step=100, slow_every=40, rest_class=0):
        self.filt = StreamFilter(n_ch)
        self.fe, self.m, self.mon, self.cp = fe, model, monitor, conformal
        self.ad, self.ic = adapter, intensity
        self.win, self.step, self.slow_every, self.rest = win, step, slow_every, rest_class
        self.ring = np.zeros((0, n_ch))
        self.k = 0
        self.last_cls = rest_class

    def process(self, block):
        """block: 새 샘플 (step x n_ch). 창이 차면 결과 dict, 아니면 None"""
        self.ring = np.vstack([self.ring, self.filt(block)])[-self.win:]
        if len(self.ring) < self.win:
            return None
        w = self.ring
        phi = self.mon.phi
        x = self.fe(w)
        p = self.m.proba(x, phi)[0]
        cset = self.cp.predict_set(p, phi)
        top = int(np.argmax(p))
        cls = self.m.classes[top]
        single = cset.sum() == 1 and cset[top]
        # 후보가 둘 이상이면 동작을 바꾸지 않음 (대회용 규칙 유지)
        out_cls = cls if single else self.last_cls
        self.last_cls = out_cls
        self.mon.update(w, p, self.m.classes)
        if self.ad is not None:
            self.ad.collect(x, cls, single)
        self.k += 1
        if self.ad is not None and self.k % self.slow_every == 0:
            self.ad.step(self.mon.phi)
        res = dict(cls=out_cls, p=p, set=cset, phi=self.mon.phi,
                   fatigue_alarm=self.mon.alarm, drift=self.mon.non_fatigue_drift)
        if self.ic is not None:
            res["intensity_gain"] = 1 / (1 + self.ic.gamma * (1 - self.mon.phi))
        return res
# endregion
