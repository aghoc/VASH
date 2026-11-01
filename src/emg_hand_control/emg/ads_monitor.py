#!/usr/bin/env python3
"""
ads_monitor.py — ADS1299 보드 초기 검진 PC 도구 (ads1299_selftest.ino 와 짝)

사용법 (Windows 명령 프롬프트, 이 파일이 있는 폴더에서):
    pip install pyserial numpy scipy matplotlib
    python ads_monitor.py ports                      # COM 포트 목록
    python ads_monitor.py selftest --port COM5        # 전체 검진 + 로그 저장
    python ads_monitor.py live --port COM5            # 실시간 EMG (원신호/필터 비교)
    python ads_monitor.py live --port COM5 --mode short   # 입력 단락 (잡음 바닥)
    python ads_monitor.py live --port COM5 --mode test    # 내부 테스트 신호
    python ads_monitor.py live --port COM5 --csv rec.csv  # 원신호 CSV 저장
    python ads_monitor.py filtercheck                 # emg_filters.h 계수 주파수 응답 검증
    python ads_monitor.py simulate                    # 하드웨어 없이 파서/필터 자가 점검

실시간 창 단축키: 1~8 = PSD 채널 선택, f = 원신호/필터 표시 전환, q = 종료
"""
import argparse, sys, time, threading, datetime, os, collections
import numpy as np

# ---------------- emg_filters.h 와 동일한 계수 (fs = 2000 Hz) ----------------
# {b0, b1, b2, a1, a2}  Direct Form II Transposed
EMG_FS = 2000.0
EMG_BIQUAD = np.array([
    [0.956543226, -1.913086451, 0.956543226, -1.911197067, 0.914975835],  # HP 20 Hz
    [0.248341079,  0.496682158, 0.248341079, -0.184213803, 0.177578119],  # LP 450 Hz
    [0.996868236, -1.958421917, 0.996868236, -1.958421917, 0.993736472],  # Notch 60
    [0.993755965, -1.847941858, 0.993755965, -1.847941858, 0.987511930],  # Notch 120
    [0.990662945, -1.672888779, 0.990662945, -1.672888779, 0.981325890],  # Notch 180
])
SOS = np.column_stack([EMG_BIQUAD[:, :3], np.ones(len(EMG_BIQUAD)), EMG_BIQUAD[:, 3:]])

FRAME_LEN = 30          # 0xA0 | seq | status(3) | 8ch*3 | 0xC0
NCH = 8

# ---------------- 콘솔 색 ----------------
os.system("")  # Windows 콘솔 ANSI 활성화
C = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m", "END": "\033[0m", "B": "\033[1m"}


def list_ports():
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    for p in ports:
        print(f"{p.device:8s}  {p.description}  [VID:PID={p.vid}:{p.pid}]")
    return ports


def auto_port():
    from serial.tools import list_ports as lp
    for p in lp.comports():
        d = (p.description or "").lower()
        if p.vid in (0x10C4, 0x1A86, 0x303A) or any(k in d for k in ("cp210", "ch34", "ch910", "usb serial", "jtag")):
            return p.device
    return None


def open_serial(port, baud):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 0.2
    s.dtr = False          # 포트 열 때 보드 자동 리셋 방지
    s.rts = False
    try:
        s.open()
    except serial.SerialException as e:
        msg = str(e)
        if "PermissionError" in msg or "액세스" in msg or "Access is denied" in msg:
            print(f"\n{port} 를 다른 프로그램이 사용 중입니다. 아래를 닫고 다시 실행하세요:\n"
                  "  · Arduino IDE 시리얼 모니터 / 시리얼 플로터\n"
                  "  · 이전에 띄운 ads_monitor 실시간 창 (창이 닫혔는데도 안 되면 터미널에서 Ctrl+C)\n"
                  "  · 다른 VS Code 터미널에서 실행 중인 python\n"
                  "  · 그래도 안 되면 USB 케이블을 뽑았다 다시 꽂기")
        else:
            print(f"\n{port} 열기 실패: {msg}\n  python ads_monitor.py ports 로 포트 번호를 확인하세요.")
        sys.exit(1)
    sbs = getattr(s, "set_buffer_size", None)     # Windows 전용: 수신 버퍼 1 MB (2 kSPS × 30 B 끊김 방지)
    if sbs:
        try:
            sbs(rx_size=1 << 20, tx_size=4096)
        except Exception:
            pass
    time.sleep(2.0)        # 혹시 리셋됐으면 부팅 대기
    s.reset_input_buffer()
    return s


def stop_and_report(s, close=True):
    """스트리밍 중지('X') → 펌웨어가 보내는 #STOPPED 통계 줄을 읽어 출력. 반환: 통계 dict (없으면 None)"""
    try:
        s.write(b"X")
    except Exception:
        return None
    t0 = time.time(); buf = b""; line = None
    while time.time() - t0 < 1.5:
        try:
            d = s.read(max(1, s.in_waiting))
        except Exception:
            break
        if d:
            buf += d
            i = buf.find(b"#STOPPED")
            if i >= 0 and b"\n" in buf[i:]:
                line = buf[i:buf.index(b"\n", i)].decode("ascii", "replace").strip(); break
    st = None
    if line:
        st = {}
        for kv in line.split(",")[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    st[k] = int(v)
                except ValueError:
                    st[k] = v
        print(f"보드 통계: 프레임 {st.get('frames')}  DRDY {st.get('drdy')}  ESP 쪽 놓침 {st.get('missed')}  "
              f"SPI 헤더 오류 {st.get('spi_hdr_bad')}  SPI {st.get('spi_khz')} kHz (자동 감속 {st.get('spi_downshift')}회)")
        if st.get("missed", 0) or st.get("spi_hdr_bad", 0):
            print("  → ESP32 쪽에서 샘플을 놓쳤거나 SPI 오류가 있었음 (점퍼선 길이·접촉 확인)")
    else:
        print("보드 통계 줄(#STOPPED)을 받지 못함 — 펌웨어 v1.10 이상인지 확인")
    if close:
        try:
            s.close()
        except Exception:
            pass
    return st


def read_line(s, timeout):
    t0 = time.time(); buf = b""
    while time.time() - t0 < timeout:
        ch = s.read(1)
        if not ch:
            continue
        if ch == b"\n":
            return buf.decode("utf-8", errors="replace").rstrip("\r")
        buf += ch
    return None


# =========================== 검진 ===========================
HINTS = {
    "T0": "DevKit 모듈 확인. 'V' 모델이면 RESET 배선을 다른 3.3V GPIO로 옮기고 펌웨어 PIN_RESET 수정",
    "T1": "멀티미터: ADS1299 DVDD(3.3V), 5V 점퍼, RESET/PWDN 핀 전압(3.3V), START 점퍼 확인",
    "T2": "J4.16 CS / J4.17 DIN / J4.18 SCLK / J4.19 DOUT 점퍼 순서·접촉 확인",
    "T3": "MOSI(J4.17↔GPIO11)·MISO(J4.19↔GPIO13) 점퍼 재삽입",
    "T4": "점퍼선을 짧게, SPI 선 사이에 GND 선 추가",
    "T5": "메시지에 표시된 점퍼(START 또는 DRDY) 재연결",
    "T6": "RESET 점퍼 J5.17 ↔ GPIO47",
    "T7": "PWDN 점퍼 J5.18 ↔ GPIO21",
    "T8a": "멀티미터: J4.21 5V → U2 출력(약 9V) → U3 출력(5.0V) 순서로 측정",
    "T8b": "3V3 점퍼 J4.1 ↔ DevKit 3V3",
    "T10": "이상 채널의 입력 필터 부품(R/C) 납땜 브리지 확인",
    "T11a": "AVDD 디커플링(C26~C30, C47, C48) 납땜, GND 점퍼 확인",
    "T11b": "AVDD 디커플링, USB 전원 잡음(노트북 충전기 분리 후 재시험)",
    "T12": "X 표시 전극 부착·납땜 확인, BIAS(J1 BI) 전극 반드시 부착",
}


def cmd_selftest(a):
    s = open_serial(a.port, a.baud)
    s.write(b"X"); time.sleep(0.2); s.reset_input_buffer()
    s.write(b"T")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logname = f"selftest_{stamp}.txt"
    results = []
    began = False
    with open(logname, "w", encoding="utf-8") as log:
        t_end = time.time() + 120
        while time.time() < t_end:
            line = read_line(s, 5)
            if line is None:
                continue
            log.write(line + "\n")
            if line.startswith("#BEGIN_SELFTEST"):
                began = True; continue
            if not began:
                continue
            if line.startswith("#END_SELFTEST"):
                break
            if line.startswith("RESULT,"):
                _, tid, lvl, msg = line.split(",", 3)
                results.append((tid, lvl, msg))
                print(f"  {C.get(lvl,'')}{C['B']}[{lvl}]{C['END']} {tid}: {msg}")
            else:
                print(line)
    s.close()
    if not results:
        print(f"{C['FAIL']}응답 없음 — 포트/보레이트(921600)/펌웨어 업로드/Arduino 시리얼 모니터 닫힘 여부 확인{C['END']}")
        return
    print("\n" + "=" * 70)
    n = collections.Counter(r[1] for r in results)
    print(f"{C['B']}요약: PASS {n['PASS']}  WARN {n['WARN']}  FAIL {n['FAIL']}{C['END']}   (로그: {logname})")
    bad = [r for r in results if r[1] != "PASS"]
    if bad:
        print("\n조치 순서 (위에서부터):")
        for tid, lvl, msg in bad:
            h = HINTS.get(tid, "")
            print(f"  {C[lvl]}{tid}{C['END']} → {h}")
    else:
        print(f"{C['PASS']}모든 항목 통과 — 다음: python ads_monitor.py live --port {a.port}{C['END']}")


# =========================== 프레임 파서 ===========================
class FrameParser:
    def __init__(self):
        self.buf = bytearray()
        self.frames = 0; self.bad = 0; self.dropped = 0; self.last_seq = None
        self.lock = threading.Lock()
        self.out_codes = []      # list of (n,8) int32
        self.out_status = []

    def feed(self, data):
        self.buf += data
        b = self.buf; i = 0; n = len(b)
        codes = []; stats = []
        while i + FRAME_LEN <= n:
            if b[i] != 0xA0:
                i += 1; continue
            if b[i + 29] != 0xC0 or (b[i + 2] & 0xF0) != 0xC0:
                self.bad += 1; i += 1; continue
            seq = b[i + 1]
            if self.last_seq is not None:
                d = (seq - self.last_seq) & 0xFF
                if d > 1:
                    self.dropped += d - 1
            self.last_seq = seq
            stats.append((b[i + 2] << 16) | (b[i + 3] << 8) | b[i + 4])
            codes.append(bytes(b[i + 5:i + 29]))   # 24 bytes
            i += FRAME_LEN
        del self.buf[:i]
        if codes:
            raw = np.frombuffer(b"".join(codes), dtype=np.uint8).reshape(-1, NCH, 3).astype(np.int32)
            v = (raw[:, :, 0] << 16) | (raw[:, :, 1] << 8) | raw[:, :, 2]
            v = np.where(v & 0x800000, v - (1 << 24), v)
            with self.lock:
                self.out_codes.append(v)
                self.out_status.append(np.array(stats, dtype=np.uint32))
                self.frames += len(codes)

    def take(self):
        with self.lock:
            if not self.out_codes:
                return np.zeros((0, NCH), np.int32), np.zeros(0, np.uint32)
            c = np.concatenate(self.out_codes); st = np.concatenate(self.out_status)
            self.out_codes.clear(); self.out_status.clear()
            return c, st


def parse_channels(txt):
    """'1' / '1,3' / '1-4' → 비트마스크 (bit0 = CH1)"""
    m = 0
    for part in str(txt).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-"))
            for c in range(lo, hi + 1): m |= 1 << (c - 1)
        else:
            m |= 1 << (int(part) - 1)
    m &= 0xFF
    return m or 0xFF


# =========================== 실시간 ===========================
def cmd_live(a):
    import matplotlib.pyplot as plt
    from scipy import signal

    s = open_serial(a.port, a.baud)
    s.write(b"X"); time.sleep(0.3); s.reset_input_buffer()
    mask = parse_channels(a.ch)
    if a.mode == "emg":
        s.write(f"K{mask:02X}".encode()); time.sleep(0.2)
    s.write({"emg": b"L", "short": b"S", "test": b"G"}[a.mode])
    hdr = None
    t0 = time.time()
    while time.time() - t0 < 6:
        line = read_line(s, 1)
        if line and line.startswith("#LIVE"):
            hdr = dict(kv.split("=") for kv in line.split(",")[1:]); break
    if not hdr:
        print("스트리밍 시작 응답(#LIVE) 없음 — 먼저 selftest 를 통과시키세요"); s.close(); return
    fs = float(hdr["fs"]); gain = float(hdr["gain"]); vref = float(hdr["vref"])
    active = int(hdr.get("mask", "FF"), 16)
    if a.mode == "emg" and active != mask:
        print("경고: 펌웨어가 채널 선택을 지원하지 않음 (v1.5 이상 필요). 8채널 모두 BIAS 에 포함됩니다.")
    uv_per_code = vref / gain / (2 ** 23) * 1e6
    print(f"스트리밍: fs={fs:.0f} Hz, gain={gain:.0f}, mode={hdr['mode']}, SPI={int(hdr['spi'])//1000} kHz")

    parser = FrameParser()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            d = s.read(4096)
            if d:
                parser.feed(d)
    th = threading.Thread(target=reader, daemon=True); th.start()

    WIN = int(fs * 4)                       # 화면 4 s
    raw_ring = np.zeros((WIN, NCH)); flt_ring = np.zeros((WIN, NCH))
    zi = np.stack([signal.sosfilt_zi(SOS) for _ in range(NCH)], axis=-1) * 0.0   # (nsec,2,NCH)
    first = True
    csvf = open(a.csv, "w", encoding="utf-8") if a.csv else None
    if csvf:
        csvf.write("sample," + ",".join(f"ch{i+1}_uV" for i in range(NCH)) + ",loff_p,loff_n\n")
    nsamp = 0
    state = {"sel": 0, "show_filt": hdr["mode"] == "emg", "run": True}

    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"   # 로그축 10^-n 지수 기호 깨짐 방지
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(NCH, 2, width_ratios=[2.2, 1], hspace=0.15)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(NCH)]
    ax_psd = fig.add_subplot(gs[0:4, 1]); ax_txt = fig.add_subplot(gs[5:8, 1]); ax_txt.axis("off")   # 한 칸 띄워 축 라벨과 겹침 방지
    tx = np.arange(WIN) / fs - WIN / fs
    DEC = 4
    lines = []
    for i, ax in enumerate(axs):
        ln, = ax.plot(tx[::DEC], np.zeros(WIN // DEC), lw=0.7)
        ax.set_ylabel(f"CH{i+1}\nµV", fontsize=8); ax.tick_params(labelsize=7)
        if i < NCH - 1: ax.set_xticklabels([])
        lines.append(ln)
    axs[-1].set_xlabel("시간 (s)")
    psd_raw, = ax_psd.semilogy([], [], lw=0.8, label="원신호")
    psd_flt, = ax_psd.semilogy([], [], lw=0.8, label="필터 후 (20–450 Hz + 노치)")
    for f0 in (60, 120, 180):
        ax_psd.axvline(f0, color="r", ls=":", lw=0.6)
    ax_psd.axvspan(20, 450, color="g", alpha=0.06)
    ax_psd.set_xlim(0, 600); ax_psd.set_xlabel("Hz"); ax_psd.set_ylabel("µV²/Hz"); ax_psd.legend(fontsize=8)
    txt = ax_txt.text(0, 1, "", va="top", fontsize=8.5,
                      fontfamily=["Consolas", "DejaVu Sans Mono", "Malgun Gothic", "AppleGothic", "NanumGothic"])

    def on_key(ev):
        if ev.key and ev.key in "12345678": state["sel"] = int(ev.key) - 1
        elif ev.key == "f": state["show_filt"] = not state["show_filt"]
        elif ev.key == "q": state["run"] = False
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: state.update(run=False))
    plt.ion(); plt.show()

    t_live0 = time.time()
    rate_hist = collections.deque()   # (시각, 누적프레임) — 최근 2 s 기준 수신률
    try:
        while state["run"]:
            codes, st = parser.take()
            if len(codes):
                x = codes * uv_per_code
                if first:      # 초기 DC로 필터 상태 초기화 → 시작 과도응답 최소화
                    zi = np.stack([signal.sosfilt_zi(SOS) * x[0, c] for c in range(NCH)], axis=-1)
                    first = False
                y = np.empty_like(x)
                for c in range(NCH):
                    y[:, c], zi[:, :, c] = signal.sosfilt(SOS, x[:, c], zi=zi[:, :, c])
                k = len(x)
                if k >= WIN:
                    raw_ring[:] = x[-WIN:]; flt_ring[:] = y[-WIN:]
                else:
                    raw_ring = np.roll(raw_ring, -k, 0); raw_ring[-k:] = x
                    flt_ring = np.roll(flt_ring, -k, 0); flt_ring[-k:] = y
                if csvf:
                    for r, sw in zip(x, st):
                        csvf.write(f"{nsamp}," + ",".join(f"{v:.3f}" for v in r) +
                                   f",{(sw >> 12) & 0xFF},{(sw >> 4) & 0xFF}\n")
                        nsamp += 1
                sat = np.any(np.abs(codes) > 0.98 * 2 ** 23, axis=0)
            else:
                sat = np.zeros(NCH, bool)

            show = flt_ring if state["show_filt"] else raw_ring - raw_ring.mean(0)
            for i in range(NCH):
                yv = show[::DEC, i]; lines[i].set_ydata(yv)
                lo, hi = np.percentile(yv, [0.5, 99.5]); pad = max(5.0, 0.15 * (hi - lo))
                axs[i].set_ylim(lo - pad, hi + pad)
            # PSD (마지막 2 s)
            seg = int(fs * 2); sel = state["sel"]
            f, pr = signal.welch(raw_ring[-seg:, sel] - raw_ring[-seg:, sel].mean(), fs, nperseg=int(fs / 2))
            _, pf = signal.welch(flt_ring[-seg:, sel], fs, nperseg=int(fs / 2))
            psd_raw.set_data(f, pr + 1e-12); psd_flt.set_data(f, pf + 1e-12)
            ax_psd.set_ylim(max(1e-6, min(pr.min(), pf.min()) * 0.5), max(pr.max(), pf.max()) * 2)
            ax_psd.set_title(f"CH{sel+1} 스펙트럼 (빨간 점선 = 60/120/180 Hz)", fontsize=9)

            # 지표
            now = time.time(); rate_hist.append((now, parser.frames))
            while len(rate_hist) > 2 and now - rate_hist[1][0] >= 2.0: rate_hist.popleft()   # 창 길이 ≥ 2 s 유지
            rate = ((rate_hist[-1][1] - rate_hist[0][1]) / (rate_hist[-1][0] - rate_hist[0][0])
                    if len(rate_hist) > 1 and rate_hist[-1][0] > rate_hist[0][0] else 0)
            rows = [f"모드: {hdr['mode']}   표시: {'필터 후' if state['show_filt'] else '원신호(DC 제거)'}   [f] 전환  [1-8] 채널",
                    f"수신 {rate:7.1f} Hz (기대 {fs:.0f})  누락 {parser.dropped}  깨진프레임 {parser.bad}", "",
                    " CH  RMSraw  RMSflt  60Hz_raw 60Hz_flt  att_dB",
                    "      (uV)    (uV)     (uV)     (uV)"]
            seg2 = raw_ring[-seg:]; segf = flt_ring[-seg:]
            n2 = len(seg2); win = np.hanning(n2); fr = np.fft.rfftfreq(n2, 1 / fs)
            i60 = np.argmin(np.abs(fr - 60))
            for c in range(NCH):
                r = seg2[:, c] - seg2[:, c].mean()
                A = np.abs(np.fft.rfft(r * win)) * 2 / win.sum()
                B = np.abs(np.fft.rfft(segf[:, c] * win)) * 2 / win.sum()
                a60 = A[i60 - 1:i60 + 2].max(); b60 = B[i60 - 1:i60 + 2].max()
                att = 20 * np.log10((b60 + 1e-9) / (a60 + 1e-9))
                flag = " SAT!" if sat[c] else (" FLAT?" if r.std() < 0.05 else "")
                if not (active >> c) & 1:
                    rows.append(f" {c+1:2d}   OFF (미사용 채널: 전원 차단)"); continue
                rows.append(f" {c+1:2d} {r.std():7.2f} {segf[:, c].std():7.2f}  {a60:8.2f} {b60:8.2f}  {att:6.1f}{flag}")
            rows += ["", "판정 기준",
                     " · short 모드: RMSraw < 1.5 uV 이면 잡음 정상",
                     " · emg 모드: 힘 줄 때 RMSflt 가 휴식 대비 수 배 이상 증가",
                     " · 60Hz_raw 가 클 때 att_dB < -20 이면 노치 동작",
                     " · SAT = ±187.5 mV 포화 → 전극 접촉/BIAS 확인",
                     " · 전극 접촉(리드오프)은 selftest T12 에서 확인"]
            txt.set_text("\n".join(rows))
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            plt.pause(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); time.sleep(0.2)
        if csvf:   # 남은 샘플까지 저장
            codes, st = parser.take()
            for r, sw in zip(codes * uv_per_code, st):
                csvf.write(f"{nsamp}," + ",".join(f"{v:.3f}" for v in r) +
                           f",{(sw >> 12) & 0xFF},{(sw >> 4) & 0xFF}\n")
                nsamp += 1
        stop_and_report(s)
        if csvf:
            csvf.close(); print(f"CSV 저장: {a.csv} ({nsamp} 샘플)")
        dur = time.time() - t_live0
        print(f"수신 프레임 {parser.frames}, 누락 {parser.dropped}, 깨진 프레임 {parser.bad}")
        if dur > 10:
            print(f"평균 샘플률 {parser.frames / dur:.1f} Hz (실행 {dur:.0f} s, 기대 2000 Hz — 차이가 1% 넘으면 알려주세요)")


# =========================== 필터 계수 검증 ===========================
def filter_response():
    from scipy import signal
    w, h = signal.sosfreqz(SOS, worN=np.array([5, 10, 15, 20, 30, 50, 60, 100, 120, 180, 250, 400, 450, 500, 700, 900.0]), fs=EMG_FS)
    return w, 20 * np.log10(np.abs(h) + 1e-15)


def cmd_filtercheck(a):
    from scipy import signal
    w, db = filter_response()
    print("emg_filters.h 계수 주파수 응답 (fs = 2000 Hz)")
    print("  주파수(Hz)  이득(dB)")
    for f, d in zip(w, db):
        print(f"  {f:8.0f}   {d:8.2f}")
    # 안정성
    poles = np.concatenate([np.roots([1, s_[4], s_[5]]) for s_ in SOS])
    print(f"\n최대 극점 크기 = {np.abs(poles).max():.5f} (<1 이면 안정)")
    # -3 dB 경계
    ff = np.linspace(1, 999, 9981); _, h = signal.sosfreqz(SOS, worN=ff, fs=EMG_FS); g = 20 * np.log10(np.abs(h))
    pb = ff[g > -3]
    print(f"-3 dB 통과대역 ≈ {pb.min():.1f} Hz ~ {pb.max():.1f} Hz (노치 주변 제외 전체 범위)")
    ok = (np.abs(poles).max() < 1 and db[list(w).index(60)] < -30 and db[list(w).index(250)] > -1)
    print("판정:", "PASS — 60 Hz 노치 30 dB 이상, 250 Hz 통과" if ok else "확인 필요")


# =========================== 하드웨어 없는 자가 점검 ===========================
def cmd_simulate(a):
    """가짜 프레임을 만들어 파서·스케일·필터·누락검출을 검증"""
    fs, gain, vref = 2000.0, 24.0, 4.5
    lsb = vref / gain / 2 ** 23 * 1e6
    t = np.arange(int(fs * 3)) / fs
    sig = np.zeros((len(t), NCH))
    sig[:, 0] = 100 * np.sin(2 * np.pi * 60 * t)                  # 60 Hz 100 µV
    sig[:, 1] = 50 * np.sin(2 * np.pi * 150 * t)                  # 대역 내 150 Hz 50 µV
    sig[:, 2] = 5000.0                                            # DC 5 mV
    sig[:, 3] = -1000 * np.sin(2 * np.pi * 10 * t)                # 대역 밖 10 Hz
    codes = np.round(sig / lsb).astype(np.int64)
    stream = bytearray(); drop_at = {200, 201, 400}   # 누락은 앞부분에 (필터 과도응답이 평가 구간에 안 섞이게)
    for k in range(len(t)):
        if k in drop_at: continue
        fr = bytearray([0xA0, k & 0xFF, 0xC0, 0x00, 0x00])
        for c in range(NCH):
            v = int(codes[k, c]) & 0xFFFFFF
            fr += bytes([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])
        fr.append(0xC0); stream += fr
        if k == 50: stream += b"\x00\xA0\x13"                     # 쓰레기 바이트 삽입
    p = FrameParser()
    for i in range(0, len(stream), 777): p.feed(bytes(stream[i:i + 777]))
    c, st = p.take()
    x = c * lsb
    from scipy import signal
    y = np.asarray(signal.sosfilt(SOS, x, axis=0))   # 타입 검사기(Pylance)용 명시 변환
    tail = slice(int(fs * 1.5), None)
    ok = True
    def chk(name, cond, val):
        nonlocal ok; ok &= cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}: {val}")
    print("파서/필터 자가 점검")
    chk("수신 프레임 수", len(x) == len(t) - 3, f"{len(x)} (기대 {len(t)-3})")
    chk("누락 검출", p.dropped == 3, p.dropped)
    chk("DC 5 mV 복원", abs(x[:, 2].mean() - 5000) < 0.1, f"{x[:, 2].mean():.3f} µV")
    chk("60 Hz 진폭 복원", abs(x[tail, 0].std() * np.sqrt(2) - 100) < 1, f"{x[tail,0].std()*np.sqrt(2):.2f} µV")
    chk("60 Hz 노치 감쇠 > 30 dB", 20 * np.log10(y[tail, 0].std() / x[tail, 0].std()) < -30,
        f"{20*np.log10(y[tail,0].std()/x[tail,0].std()):.1f} dB")
    chk("150 Hz 통과 (±1 dB)", abs(20 * np.log10(y[tail, 1].std() / x[tail, 1].std())) < 1,
        f"{20*np.log10(y[tail,1].std()/x[tail,1].std()):.2f} dB")
    chk("DC 제거", abs(y[tail, 2]).max() < 1, f"{abs(y[tail,2]).max():.3f} µV")
    chk("10 Hz 감쇠 > 6 dB", 20 * np.log10(y[tail, 3].std() / x[tail, 3].std()) < -6,
        f"{20*np.log10(y[tail,3].std()/x[tail,3].std()):.1f} dB")
    print("결과:", "전체 PASS" if ok else "FAIL 있음")
    return ok


def main():
    ap = argparse.ArgumentParser(description="ADS1299 보드 검진 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports")
    for name in ("selftest", "live"):
        p = sub.add_parser(name)
        p.add_argument("--port", default=None)
        p.add_argument("--baud", type=int, default=921600)
        if name == "live":
            p.add_argument("--mode", choices=["emg", "short", "test"], default="emg")
            p.add_argument("--csv", default=None)
            p.add_argument("--ch", default="1-8", help="전극을 붙인 채널. 예: 1  /  1,2  /  1-4  (emg 모드)")
    sub.add_parser("filtercheck")
    sub.add_parser("simulate")
    a = ap.parse_args()
    if a.cmd == "ports": list_ports(); return
    if a.cmd == "filtercheck": cmd_filtercheck(a); return
    if a.cmd == "simulate": sys.exit(0 if cmd_simulate(a) else 1)
    if not a.port:
        a.port = auto_port()
        if not a.port:
            print("COM 포트를 찾지 못함. python ads_monitor.py ports 로 확인 후 --port 지정"); return
        print(f"포트 자동 선택: {a.port}")
    {"selftest": cmd_selftest, "live": cmd_live}[a.cmd](a)


if __name__ == "__main__":
    main()
