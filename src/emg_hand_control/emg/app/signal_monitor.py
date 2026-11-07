#!/usr/bin/env python3
"""
8 채널 실시간 신호 상태 (통합 앱용): COM6 스트림 → 0.25 s 마다 '#CH {json}' 한 줄.
  rms = 지금 세기, rest = 최근 20 s 하위 20 % (휴식 수준, 힘을 줘도 잘 안 변함), hum = 필터 전 60 Hz, peak = 최근 20 s 최대
기준은 지금까지 착용 기록으로 정한 잠정값 (grip_model.py 경고와 같은 60 Hz 100 µV)
"""
import argparse, collections, json, os, pickle, signal, sys, threading, time
import numpy as np
from scipy import signal as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import emg_classifier as E  # noqa: E402

HUM_BAD, HUM_WARN = 100.0, 50.0
REST_BAD, REST_WARN = 20.0, 8.0
REST_FLAT = 0.2                     # 입력 단락 잡음(약 0.4 µV) 수준 아래 = 신호 없음
RAW_LIMIT = 180000.0                # 게인 24 측정 범위 ±187.5 mV 근처


def judge(rest, hum, raw_peak):
    if raw_peak > RAW_LIMIT: return "bad", "측정 범위 초과: 전극 끊김 의심"
    if hum > HUM_BAD: return "bad", "60 Hz 큼: 전극 접촉 확인"
    if rest > REST_BAD: return "bad", "휴식 잡음 큼: 접촉 확인, 힘 빼기"
    if rest < REST_FLAT: return "bad", "신호 없음: 연결 확인"
    if hum > HUM_WARN: return "warn", "60 Hz 조금 큼"
    if rest > REST_WARN: return "warn", "휴식 잡음 조금 큼"
    return "ok", "좋음"


class Meter:
    def __init__(self, rest_n=80, rest_pct=20, hum_n=12):
        self.sos = E.full_sos(); self.hum_sos = sp.butter(2, [57, 63], "band", fs=E.FS, output="sos")
        self.zi = self.zh = None; self.y = np.zeros((E.FS, E.NCH)); self.h = np.zeros((E.FS, E.NCH)); self.raw = np.zeros((E.FS, E.NCH))
        self.rest_pct = rest_pct; self.hist = collections.deque(maxlen=rest_n); self.humh = collections.deque(maxlen=hum_n); self.n = 0

    def push(self, x):
        if self.zi is None:
            self.zi = sp.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]; self.zh = sp.sosfilt_zi(self.hum_sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sp.sosfilt(self.sos, x, axis=0, zi=self.zi); h, self.zh = sp.sosfilt(self.hum_sos, x, axis=0, zi=self.zh)
        k = min(len(x), E.FS)
        for buf, new in ((self.y, y), (self.h, h), (self.raw, x)):
            buf[:-k] = buf[k:]; buf[-k:] = new[-k:]
        self.n += len(x)
        return y

    def report(self):
        win = int(0.25 * E.FS)
        rms = np.sqrt((self.y[-win:] ** 2).mean(0)); self.hist.append(rms); self.humh.append(np.sqrt((self.h ** 2).mean(0)))
        settled = self.n >= 3 * E.FS
        H = np.array(self.hist); rest = np.percentile(H, self.rest_pct, 0); peak = H.max(0); hum = np.median(np.array(self.humh), 0)
        raw_peak = np.abs(self.raw - np.median(self.raw, 0)).max(0) + np.abs(np.median(self.raw, 0))
        ch = []
        for c in range(E.NCH):
            st, why = judge(rest[c], hum[c], raw_peak[c]) if settled else ("wait", "안정 중")
            ch.append(dict(n=c + 1, rms=round(float(rms[c]), 1), rest=round(float(rest[c]), 1), hum=round(float(hum[c]), 1), peak=round(float(peak[c]), 1), state=st, why=why))
        return dict(t=round(self.n / E.FS, 1), ch=ch)


class RestCal:
    """대기 보정: 모든 블록이 괜찮고 조용한 상태가 QUIET_S 이어지면 REC_S 측정, 끊기면 다시. 끝나면 모델·원신호 저장"""
    QUIET_S, REC_S, HOT_N = 3.0, 15.0, 4

    def __init__(self, out, base, raw_out):
        self.out, self.base, self.raw_out = out, base, raw_out
        self.stage = "wait"; self.quiet = 0.0; self.hot = 0; self.xs = []; self.ys = []; self.n = 0; self.windows = None; self.note = ""; self.thread = None

    def feed(self, x, y):
        if self.stage == "record":
            self.xs.append(x); self.ys.append(y); self.n += len(x)

    def check(self, rep):
        if self.stage not in ("wait", "record"): return
        ch = rep["ch"]; bad = [c["n"] for c in ch if c["state"] in ("bad", "wait") or c["rms"] > REST_BAD]
        self.hot = self.hot + 1 if bad else 0
        if self.stage == "wait" and bad or self.hot >= self.HOT_N:       # 짧은 튐은 계산에서 빼고, 1 s 넘게 이어지면 다시 잼
            if self.stage == "record": self.note = "CH" + ", CH".join(map(str, bad)) + " 신호가 커져 다시 잽니다"
            self.stage = "wait"; self.quiet = 0.0; self.xs, self.ys, self.n = [], [], 0
        elif bad:
            pass
        elif self.stage == "wait":
            self.quiet += 0.25
            if self.quiet >= self.QUIET_S:
                self.stage = "record"; self.note = ""
        elif self.n >= self.REC_S * E.FS:
            self.stage = "compute"; self.thread = threading.Thread(target=self._finish, daemon=True); self.thread.start()

    def _finish(self):
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import grip_model as G
            x, y = np.vstack(self.xs), np.vstack(self.ys)
            with open(self.base, "rb") as f:
                pk = G.calibrate_rest(pickle.load(f), y, os.path.basename(self.raw_out), max_uv=REST_BAD)
            with open(self.raw_out, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"# quick_rest fs={E.FS} recorded={time.strftime('%Y%m%dT%H%M%S')}\nlabel,rep," + ",".join(f"ch{c + 1}_uV" for c in range(E.NCH)) + "\n")
                np.savetxt(f, np.column_stack([np.zeros(len(x)), -np.ones(len(x)), x]), fmt=["%d", "%d"] + ["%.3f"] * E.NCH, delimiter=",")
            tmp = self.out + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(pk, f)
            os.replace(tmp, self.out)
            self.windows = pk["calibrated"]["windows"]; self.stage = "done"
        except Exception as e:
            self.note = str(e); self.stage = "fail"

    def view(self):
        left = max(0.0, self.REC_S - self.n / E.FS) if self.stage == "record" else None
        return dict(stage=self.stage, left=None if left is None else round(left, 1), quiet=round(self.quiet, 2), rec_s=self.REC_S, note=self.note,
                    model=os.path.basename(self.out) if self.stage == "done" else None, windows=self.windows)


def replay_chunks(path, speed):
    x = E.load_csv(path)[0]; k = 0; t0 = time.perf_counter()
    while k < len(x):
        due = min(len(x), int((time.perf_counter() - t0) * E.FS * speed))
        if due > k:
            yield x[k:due]; k = due
        time.sleep(0.005)


def live_chunks(a):
    from ads_monitor import open_serial, read_line, FrameParser, auto_port, stop_and_report
    s = open_serial(a.port or auto_port(), a.baud)
    try:
        s.write(b"X"); time.sleep(0.3); s.reset_input_buffer(); s.write(b"KFF"); time.sleep(0.2); s.write(b"L")
        hdr = None; t0 = time.time()
        while time.time() - t0 < 6:
            line = read_line(s, 1)
            if line and line.startswith("#LIVE"):
                hdr = dict(kv.split("=") for kv in line.split(",")[1:]); break
        if not hdr:
            raise SystemExit("스트리밍 시작 응답 없음: 보드 연결 확인")
        uv = float(hdr["vref"]) / float(hdr["gain"]) / 2 ** 23 * 1e6; parser = FrameParser()
        while True:
            d = s.read(max(30, s.in_waiting))
            if d:
                parser.feed(d); c, _ = parser.take()
                if len(c):
                    yield c * uv
    finally:
        stop_and_report(s)


def main():
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None); ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--replay", default=None, help="녹화 재생 (시험용)"); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--quick-out", default=None, help="대기 보정 모델 저장 경로 (주면 자동 측정)"); ap.add_argument("--quick-raw", default=None); ap.add_argument("--base", default=None)
    a = ap.parse_args()
    # 대기 보정 중에는 전극을 옮긴 결과가 바로 보이도록 최근 2 s 중앙값으로 판단
    m = Meter(8, 50, 4) if a.quick_out else Meter(); t_next = 0.0
    cal = RestCal(a.quick_out, a.base, a.quick_raw) if a.quick_out else None
    try:
        for x in (replay_chunks(a.replay, a.speed) if a.replay else live_chunks(a)):
            y = m.push(x)
            if cal: cal.feed(x, y)
            if time.perf_counter() >= t_next:
                t_next = time.perf_counter() + 0.25
                rep = m.report()
                print("#CH " + json.dumps(rep), flush=True)
                if cal:
                    cal.check(rep); print("#QC " + json.dumps(cal.view()), flush=True)
    except KeyboardInterrupt:
        pass
    if cal and cal.thread:
        cal.thread.join(); print("#QC " + json.dumps(cal.view()), flush=True)


if __name__ == "__main__":
    main()
