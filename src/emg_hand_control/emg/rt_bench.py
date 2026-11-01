#!/usr/bin/env python3
"""
rt_bench.py — 실시간 성능 측정 + '언제/무엇을' 분리 디코더 (녹화를 실시간처럼 재생, 인과적 처리만)

  python rt_bench.py [--train data\\F2_B1.csv] [--test data\\F2_B2.csv data\\F2_T1.csv ...] [--step-ms 50]

디코더
  D0 현재 방식   : 200 ms 창 LDA → 다수결 3 → 동작 중 다른 동작으로 바꾸려면 같은 후보 4회 연속 (SWITCH_CONFIRM)
  D1 분리 방식   : '언제' = 짧은 창(100 ms) 근육 세기 z 의 히스테리시스 (켜짐 θ_on, 꺼짐 θ_off) → 켜짐/꺼짐 즉시
                   '무엇을' = 켜진 뒤 창마다 log P 를 누적(순차 베이즈), 사후확률 ≥ 0.9 또는 0.4 s 경과 시 쥐기 확정,
                   확정 뒤 바꾸려면 최근 0.3 s 누적 증거가 현재 쥐기보다 log 10 이상 커야 함 (SPRT 형태)
                   확정 전에는 잠정 쥐기(누적 최대)를 출력 → 서보는 바로 닫기 시작 가능
측정 (근육 신호 시작/끝 기준 = 사람 반응 시간을 뺀 '시스템 지연')
  · 감지 지연     : 근육 시작 → 처음 '동작 중' 출력
  · 정답 도달 지연 : 근육 시작 → 정답 쥐기 출력이 150 ms 이상 유지되기 시작
  · 풀림 지연     : 근육 끝 → '휴식' 출력
  · 유지 중 오류 % : 근육 시작 0.5 s 뒤 ~ 지시 끝, 정답이 아닌 출력 비율
  · 휴식 오작동/분 : 근육 끝 0.5 s 뒤 ~ 다음 지시, 휴식 → 동작으로 바뀐 횟수
  · 첫 확정 오류 % : 시도마다 처음 확정한 쥐기가 틀린 비율 (D1)
근육 시작/끝: 시도마다 25 ms 창 세기 z 가 (지시 전 기준 ↔ 유지 중) 중간을 넘는/(유지 중 ↔ 다음 휴식) 중간 아래로 내려가는 첫 시각
  (이 녹화는 지시보다 0.3~0.7 s 늦게 시작·늦게 풀어서 지시 시각 대신 실제 근육 시각을 씀)
  D2 = '언제' 를 P(휴식) 히스테리시스로 (다수결 없음) + '무엇을' 새는 누적 증거
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ACT = list(range(E.NCH))


def filt(path):
    from scipy import signal
    x, lab, rep, meta = E.load_csv(path)
    y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
    return y, lab, rep, meta


def zfast(y, win, rms_rest):
    """win 샘플 창의 세기 z = ½·log(채널 평균 (RMS/휴식 RMS)²) — 켜진 채널이 지배 (패턴과 무관하게 '힘을 줬는지'), 매 샘플"""
    c = np.cumsum(np.r_[np.zeros((1, y.shape[1])), y ** 2], 0)
    ms = np.zeros_like(y); ms[win:] = (c[win + 1:] - c[1:-win]) / win; ms[:win] = ms[win]
    return 0.5 * np.log(np.mean(ms / rms_rest ** 2, 1) + 1e-9)


def trial_times(z5, lab):
    """시도마다 실제 근육 시작·끝 (샘플) — 지시 전/유지/다음 휴식 세기 중앙값의 중간 교차 (시도별 적응)"""
    out = []
    for s, e, c in E.segments(lab):
        if c == 0 or s < 2 * E.FS: continue
        zb0 = np.median(z5[s - E.FS:s - E.FS // 5]); zh = np.median(z5[s + E.FS:e])
        zb1 = np.median(z5[e + int(1.6 * E.FS):e + int(2.4 * E.FS)]) if e + int(2.4 * E.FS) < len(z5) else zb0
        if zh - max(zb0, zb1) < 0.3: continue
        a0, a1 = s - E.FS // 2, min(len(lab), e + int(1.8 * E.FS))
        up = np.flatnonzero(z5[a0:a1] > zb0 + 0.5 * (zh - zb0))
        dn = np.flatnonzero(z5[e - E.FS // 2:a1] < zh - 0.5 * (zh - zb1))
        if len(up): out.append((a0 + up[0], e - E.FS // 2 + (dn[0] if len(dn) else E.FS // 2), c))
    return out


def relabeled_dataset(path):
    """학습 라벨을 실제 근육 시각으로 다시 붙임: 동작 = [시작+0.1 s, 끝], 휴식 = 끝+0.4 s ~ 다음 시작−0.3 s.
    지시 기준 라벨은 반응 지연(0.3~0.7 s) 때문에 '힘이 남은 휴식'·'아직 쉬는 동작' 창이 섞임 → 경계가 흐려짐"""
    y, lab, rep, _ = filt(path)
    z5 = zfast(y, int(0.025 * E.FS), np.sqrt(np.percentile(y[2 * E.FS:] ** 2, 20, 0)) + 1e-3)
    tt = trial_times(z5, lab)
    L = np.full(len(lab), -1)
    prev = 2 * E.FS
    for k, (a, b, c) in enumerate(tt):
        L[prev:max(prev, a - int(0.3 * E.FS))] = 0
        L[a + int(0.1 * E.FS):b] = c
        prev = b + int(0.4 * E.FS)
    L[prev:] = 0
    X, Y, R = [], [], []
    for k in range(2 * E.FS + E.WIN, len(y), E.STEP):
        l = L[k - E.WIN // 2]
        if l < 0 or L[k - E.WIN] != L[k - 1]: continue
        X.append(y[k - E.WIN:k]); Y.append(int(l)); R.append(int(rep[k - 1]))
    return np.asarray(X), np.array(Y), np.array(R)


class Model:
    def __init__(self, paths, aug=True, relabel=False):
        paths = [paths] if isinstance(paths, str) else list(paths)
        if relabel:
            parts = [relabeled_dataset(p_) for p_ in paths]
            X = np.concatenate([q[0] for q in parts]); Y = np.concatenate([q[1] for q in parts])
            R = np.concatenate([q[2] + 1000 * i for i, q in enumerate(parts)])
        else:
            X, Y, R = E.build_dataset(paths, ACT); X = np.asarray(X)
        path = paths[0]
        Xa, Ya, Ra = E.augment_force(X, Y, R, E.AUG_SCALES if aug else ())
        self.m = M.MathModel(Xa, Ya, Ra, ACT, True); self.k = self.m.k
        ys = [filt(p_) for p_ in paths]
        y = np.concatenate([q[0][2 * E.FS:] for q in ys]); lab = np.concatenate([q[1][2 * E.FS:] for q in ys])
        segs = [(s_, t_, l_) for s_, t_, l_ in E.segments(lab) if s_ > E.FS]
        ridx = np.concatenate([np.arange(s_ + E.FS, t_ - E.FS // 4) for s_, t_, l_ in segs if l_ == 0 and t_ - s_ > 1.5 * E.FS])
        w = int(0.1 * E.FS); c = np.cumsum(np.r_[np.zeros((1, y.shape[1])), y ** 2], 0)
        ms = np.zeros_like(y); ms[w:] = (c[w + 1:] - c[1:-w]) / w
        self.rms_rest = np.sqrt(np.percentile(ms[ridx], 20, 0))        # 가장 조용한 휴식 수준 (휴식 구간에 남은 힘 제외)
        z = zfast(y, w, self.rms_rest)
        zm = np.concatenate([z[s_ + E.FS // 2:t_] for s_, t_, l_ in segs if l_ > 0])
        self.z_mu, self.z_sd, self.z_act = float(np.median(z[ridx])), float(z[ridx].std()), float(np.median(zm))
        # 히스테리시스: 켜짐 = 휴식 중앙값과 동작 중앙값 사이 60 %, 꺼짐 = 40 % (켜짐보다 낮아 깜빡임 방지)
        self.th_on = self.z_mu + 0.6 * (self.z_act - self.z_mu)
        self.th_off = self.z_mu + 0.4 * (self.z_act - self.z_mu)
        self.th_phys = self.z_mu + 0.5 * (self.z_act - self.z_mu)        # 평가용 '근육 시작/끝' (25 ms 창)

    def proba(self, W):
        m = self.m; Fn = (m.feat(W) - m.fmu) / m.fsd
        return M.softmax_t(Fn @ m.W.T + m.b, m.tau)


def decisions(model, y, step):
    ends = np.arange(E.FS + E.WIN, len(y), step)
    W = np.array([y[e - E.WIN:e] for e in ends])
    P = model.proba(W)
    zs = zfast(y, int(0.1 * E.FS), model.rms_rest)[ends - 1]
    return ends, P, zs


def dec_baseline(P, vote=3, confirm=4):
    out = np.zeros(len(P), int); votes = []; active = 0; run_c, run_n = 0, 0
    for i, p in enumerate(P):
        votes.append(int(p.argmax())); votes = votes[-vote:]
        c = max(set(votes), key=votes.count)
        if c == 0: active = 0; run_n = 0
        elif active == 0: active = c
        elif c != active:
            run_n = run_n + 1 if c == run_c else 1; run_c = c
            if run_n >= confirm: active = c; run_n = 0
        else: run_n = 0
        out[i] = active
    return out


def dec_split(P, zs, model, step, commit_p=0.9, commit_s=0.4, switch_s=0.3, switch_llr=np.log(10)):
    out = np.zeros(len(P), int); commit = np.zeros(len(P), int)
    on = False; L = None; t_on = 0; cls = 0; hist = []
    nsw = max(1, int(round(switch_s * E.FS / step)))
    for i, (p, z) in enumerate(zip(P, zs)):
        if not on and z > model.th_on:
            on, L, t_on, cls, hist = True, np.zeros(model.k - 1), i, 0, []
        elif on and z < model.th_off:
            on = False
        if not on:
            out[i] = 0; continue
        lp = np.log(np.maximum(p[1:], 1e-9)); lp -= lp.max()
        L += lp; hist.append(lp); hist = hist[-nsw:]
        post = np.exp(L - L.max()); post /= post.sum()
        if cls == 0:
            if post.max() >= commit_p or (i - t_on) * step / E.FS >= commit_s:
                cls = int(post.argmax()) + 1; commit[i] = cls
            out[i] = int(post.argmax()) + 1          # 잠정 쥐기
        else:
            H = np.sum(hist, 0)
            b = int(H.argmax()) + 1
            if b != cls and H[b - 1] - H[cls - 1] > switch_llr * len(hist) / nsw * 3:
                cls = b; L = H.copy(); commit[i] = cls
            out[i] = cls
    return out, commit


def dec_prob(P, step, tau_p=0.05, p_on=0.4, p_off=0.6, tau_L=0.5, commit_p=0.9, min_s=0.1, commit_s=0.4, switch_llr=np.log(20)):
    """D2: '언제' = 가볍게 평활한 P(휴식) 의 히스테리시스 (다수결 없음), '무엇을' = 새는 누적 log 증거 (시정수 tau_L)"""
    dt = step / E.FS; a = 1 - np.exp(-dt / tau_p); lam = np.exp(-dt / tau_L)
    out = np.zeros(len(P), int); commit = np.zeros(len(P), int)
    pr = P[0].copy(); on = False; L = None; t_on = 0; cls = 0
    for i, p in enumerate(P):
        pr += a * (p - pr)
        if not on and pr[0] < p_on:
            on, L, t_on, cls = True, np.zeros(len(p) - 1), i, 0
        elif on and pr[0] > p_off:
            on = False
        if not on:
            continue
        lp = np.log(np.maximum(p[1:], 1e-9)); lp -= lp.max()
        L = lam * L + lp
        post = np.exp(L - L.max()); post /= post.sum(); b = int(post.argmax()) + 1
        el = (i - t_on) * dt
        if cls == 0:
            if (post.max() >= commit_p and el >= min_s) or el >= commit_s:
                cls = b; commit[i] = cls
            out[i] = b
        else:
            if b != cls and L[b - 1] - L[cls - 1] > switch_llr:
                cls = b; commit[i] = cls
            out[i] = cls
    return out, commit


def dec_hybrid(P, zs, model, step, tau_p=0.025, p_on=0.3, p_off=0.6, gate=0.3, rel=0.45, tau_b=1.0, tau_h=0.3, n_low=3, p_agree=0.4,
               tau_L=0.5, commit_p=0.9, min_s=0.1, commit_s=0.4, switch_llr=np.log(20)):
    """D3: 켜짐 = P(휴식) 낮음 AND 세기가 직전 휴식 기준보다 gate·Δ 이상 (남은 힘·잔떨림 무시)
           꺼짐 = 세기가 '켜질 때 기준 ↔ 이번 수축 최고 세기' 의 rel 지점 아래로 2회 연속 (시도마다 상대 기준) 또는 P(휴식) 높음
           쥐기 = 새는 누적 log 증거 (D2 와 같음)"""
    dt = step / E.FS; a = 1 - np.exp(-dt / tau_p); ab = 1 - np.exp(-dt / tau_b); ah = 1 - np.exp(-dt / tau_h); lam = np.exp(-dt / tau_L)
    dz = model.z_act - model.z_mu
    out = np.zeros(len(P), int); commit = np.zeros(len(P), int)
    pr = P[0].copy(); zb = zs[0]; on = False; L = None; t_on = 0; cls = 0; zh = zpk = zb_on = 0.0; low = 0
    for i, (p, z) in enumerate(zip(P, zs)):
        pr += a * (p - pr)
        if not on:
            if pr[0] < p_on and z > zb + gate * dz:
                on, L, t_on, cls, zh, zpk, zb_on, low = True, np.zeros(len(p) - 1), i, 0, z, z, zb, 0
            else:
                zb += ab * (z - zb)
                continue
        zh += ah * (z - zh); zpk = max(zpk, zh)
        low = low + 1 if z < zb_on + rel * (zpk - zb_on) else 0
        if (low >= n_low and pr[0] >= p_agree) or pr[0] > p_off:
            on = False; zb = z; continue
        lp = np.log(np.maximum(p[1:], 1e-9)); lp -= lp.max()
        L = lam * L + lp
        post = np.exp(L - L.max()); post /= post.sum(); b = int(post.argmax()) + 1
        el = (i - t_on) * dt
        if cls == 0:
            if (post.max() >= commit_p and el >= min_s) or el >= commit_s:
                cls = b; commit[i] = cls
            out[i] = b
        else:
            if b != cls and L[b - 1] - L[cls - 1] > switch_llr:
                cls = b; commit[i] = cls
            out[i] = cls
    return out, commit


def evaluate(out, ends, lab, z5, model, commit=None, step=None):
    t = ends / E.FS; T = len(lab) / E.FS
    segs = [(s, e, l) for s, e, l in E.segments(lab) if s > E.FS]
    lat_det, lat_ok, lat_rel, hold_err, first_err = [], [], [], [], []
    rest_ev, rest_time = 0, 0.0
    thr = model.th_phys
    act = [(s, e, l) for s, e, l in segs if l > 0]
    for j, (s, e, c) in enumerate(act):
        a0, a1 = s - int(0.5 * E.FS), min(len(lab), e + int(1.8 * E.FS))
        zb0 = np.median(z5[s - int(1.0 * E.FS):s - int(0.2 * E.FS)])            # 지시 전 기준
        zh = np.median(z5[s + E.FS:e])                                           # 유지 중
        zb1 = np.median(z5[e + int(1.6 * E.FS):e + int(2.4 * E.FS)]) if e + int(2.4 * E.FS) < len(z5) else zb0
        if zh - max(zb0, zb1) < 0.3: continue                                    # 세기 차이가 작아 시작/끝을 못 정함
        above = np.flatnonzero(z5[a0:a1] > zb0 + 0.5 * (zh - zb0))
        if not len(above): continue
        ton = (a0 + above[0]) / E.FS
        after = np.flatnonzero(z5[e - E.FS // 2:a1] < zh - 0.5 * (zh - zb1))
        toff = (e - E.FS // 2 + (after[0] if len(after) else 0)) / E.FS
        k = np.flatnonzero((t >= ton) & (t <= e / E.FS + 1.0) & (out > 0))
        if len(k): lat_det.append(t[k[0]] - ton)
        run = int(round(0.15 * E.FS / (ends[1] - ends[0])))
        idx = np.flatnonzero((t >= ton) & (t <= e / E.FS + 1.0))
        ok = out[idx] == c
        for q in range(len(idx) - run + 1):
            if ok[q:q + run].all(): lat_ok.append(t[idx[q]] - ton); break
        k = np.flatnonzero((t >= toff - 0.2) & (out == 0))
        if len(k): lat_rel.append(max(0.0, t[k[0]] - toff))
        h = (t >= ton + 0.5) & (t < toff)
        if h.any(): hold_err.append(100 * np.mean(out[h] != c))
        if commit is not None:
            cm = commit[(t >= ton - 0.2) & (t <= e / E.FS)]
            cm = cm[cm > 0]
            if len(cm): first_err.append(100.0 * (cm[0] != c))
        nxt = act[j + 1][0] / E.FS if j + 1 < len(act) else T
        r = (t >= toff + 0.5) & (t < nxt - 0.3)
        if r.any():
            o = out[r]; rest_ev += int(np.sum((o[1:] > 0) & (o[:-1] == 0)) + (o[0] > 0)); rest_time += r.sum() * (ends[1] - ends[0]) / E.FS
    f = lambda v: float(np.median(v)) * 1000 if v else np.nan
    return dict(det=f(lat_det), ok=f(lat_ok), rel=f(lat_rel), hold=float(np.mean(hold_err)) if hold_err else np.nan,
                fa=rest_ev / max(rest_time / 60, 1e-9), first=float(np.mean(first_err)) if first_err else np.nan, n=len(lat_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[f"{D}/F2_B1.csv"])
    ap.add_argument("--test", nargs="+", default=[f"{D}/F2_{k}.csv" for k in ("B2", "T1", "T2", "T3", "recover")])
    ap.add_argument("--step-ms", type=float, default=50)
    ap.add_argument("--relabel", action="store_true", help="학습 라벨을 실제 근육 시작·끝으로 다시 붙임")
    a = ap.parse_args()
    step = int(a.step_ms * E.FS / 1000)
    model = Model(a.train, relabel=a.relabel)
    print(f"학습 {[os.path.basename(p_) for p_ in a.train]}  |  세기 문턱 z: 휴식 {model.z_mu:.2f}±{model.z_sd:.2f}, 동작 {model.z_act:.2f} → 켜짐 {model.th_on:.2f} / 꺼짐 {model.th_off:.2f}   판단 간격 {a.step_ms:g} ms")
    hdr = f"{'시험':12s} {'디코더':8s} {'감지(ms)':>9s} {'정답도달':>9s} {'풀림':>7s} {'유지오류%':>9s} {'휴식오작동/분':>12s} {'첫확정오류%':>11s}"
    print(hdr); agg = {"D0": [], "D1": [], "D2": [], "D3": []}
    for path in a.test:
        y, lab, _, _ = filt(path)
        ends, P, zs = decisions(model, y, step)
        z5 = zfast(y, int(0.025 * E.FS), model.rms_rest)
        o0 = dec_baseline(P, vote=3, confirm=max(1, int(round(4 * E.STEP / step))))
        o1, cm = dec_split(P, zs, model, step)
        o2, cm2 = dec_prob(P, step)
        o3, cm3 = dec_hybrid(P, zs, model, step)
        for name, o, c in (("D0", o0, None), ("D1", o1, cm), ("D2", o2, cm2), ("D3", o3, cm3)):
            r = evaluate(o, ends, lab, z5, model, c, step); agg[name].append(r)
            print(f"{os.path.basename(path)[:12]:12s} {name:8s} {r['det']:9.0f} {r['ok']:9.0f} {r['rel']:7.0f} {r['hold']:9.1f} {r['fa']:12.2f} {r['first']:11.1f}")
    print("-" * len(hdr))
    for name in agg:
        R = agg[name]; g = lambda k: np.nanmean([r[k] for r in R])
        print(f"{'평균':12s} {name:8s} {g('det'):9.0f} {g('ok'):9.0f} {g('rel'):7.0f} {g('hold'):9.1f} {g('fa'):12.2f} {g('first'):11.1f}")


if __name__ == "__main__":
    main()
