#!/usr/bin/env python3
"""
통합 앱 (로컬 전용, 127.0.0.1:8765): 착용 → 보정 → 의수 + 근전도 실행·정지를 한 화면에서. 윈도우·리눅스(제슨) 둘 다.
기존 start_hand.py / finger_test.py / grip_model.py / grip_rt.py 를 그대로 실행하고, 정지는 Ctrl+C 와 같은 안전 종료를 보냄.
실행: 의수앱 바로가기 (VASH.vbs, 숨은 콘솔 창으로 실행해 정지 신호 유지). 기록을 보려면 의수앱_터미널.bat
"""
import collections, json, os, re, signal, subprocess, sys, threading, time, webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

PC = Path(__file__).resolve().parents[1]
ROOT = PC.parent
WIN = os.name == "nt"
# 경로·포트는 환경 변수로 바꿀 수 있음 (제슨: jetson/vash.env). 기본값 = 이 노트북 배치
DATA = Path(os.environ.get("VASH_DATA", PC / "data" / "S01"))
def first_existing(*paths):
    return next((p for p in paths if p.exists()), paths[-1])      # 아무것도 없으면 저장소 배치


# 노트북 배치 (hand/VGAT_ORCA/...) 가 없으면 저장소 배치 (emg_hand_control/orca_core, emg_hand_control/hand)
ORCA = Path(os.environ.get("VASH_ORCA") or first_existing(ROOT / "hand" / "VGAT_ORCA" / "orca_core", ROOT / "orca_core"))
HAND_DIR = Path(os.environ.get("VASH_HAND_DIR") or first_existing(ROOT / "hand" / "VGAT_ORCA" / "hand-config" / "right-no-wrist", ROOT / "hand"))


def venv_python(folder):
    p = folder / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
    return p if p.exists() else Path(sys.executable)


PC_PY = venv_python(PC)
HAND_PY = venv_python(ORCA)
PORT = int(os.environ.get("VASH_PORT", 8765))
GESTURES = "손 펴기,주먹,집기 (중지 펴기),세손가락 (중지 굽힘),가위,검지 펴기,엄지 펴기"
# 제슨은 udev 규칙(jetson/99-vash.rules)이 만드는 고정 이름을 씀
# 11-07: 대회까지 오른팔만 사용. right = 녹화 헤더 arm=right, 정밀 보정 '이 착용만으로 학습' 기본, 오른팔 위치 기준, 오른팔 배치도,
#   빠른 보정(왼팔 기본 모델에 휴식만 맞춤) 끔. left = 이전(왼팔) 방식
ARM = os.environ.get("VASH_ARM", "right")
REF_RIGHT = DATA / "grip7_ref_right.pkl"
EMG_COM = os.environ.get("VASH_EMG_PORT") or ("COM6" if WIN else "/dev/vash-emg")
HAND_COM = os.environ.get("VASH_HAND_PORT") or ("COM7" if WIN else "/dev/vash-hand")
NOISE = re.compile(r"findfont:|^\s*warnings\.warn|UserWarning")
LOCAL_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

# 녹화 종류: (제목, 대략 길이 s, finger_test 옵션)
RECORD = {
    "CAL": ("40초 보정", 50, ["--reps", "1", "--relax"]),
    "TRANS": ("전환 녹화", 125, ["--reps", "1", "--hold", "2", "--chain", "7", "--all-transitions", "--relax", "--rest-jitter", "1", "--next-cue", "1"]),
    "TEST": ("시험: 쉬는 동작", 130, ["--reps", "3", "--relax", "--rest-jitter", "1", "--next-cue", "1"]),
    "CHAIN": ("시험: 연속 동작", 72, ["--reps", "3", "--hold", "2", "--chain", "4", "--relax", "--rest-jitter", "1", "--next-cue", "1"]),
}


def command(kind, **kw):
    """실행할 명령과 폴더. 시험 때 가짜 명령으로 바꿈"""
    if kind == "record":
        return [str(PC_PY), "finger_test.py", "record", "--port", EMG_COM, "--ch", "1-8", "--side", "band", "--subject", "S01", "--session", kw["session"], "--task", kw["task"],
                "--arm", ARM, "--gestures", GESTURES, "--seed", str(int(time.time()) % 100000), *RECORD[kw["task"]][2], "--out", kw["out"]], PC
    if kind == "calib":
        ref = ["--ref", str(REF_RIGHT)] if ARM == "right" and REF_RIGHT.exists() else []
        return [str(PC_PY), "grip_model.py", "calib", *kw["files"], "--weight", str(kw["weight"]), "--out", kw["out"], *(["--fresh"] if kw.get("fresh") else []), *ref], PC
    if kind == "svm":
        return [str(PC_PY), "grip_model.py", "calib-svm", *kw["files"], "--weight", "30", "--out", kw["out"], *(["--fresh"] if kw.get("fresh") else [])], PC
    if kind == "monitor":
        quick = ["--quick-out", kw["quick_out"], "--quick-raw", kw["quick_raw"], "--base", str(DATA / "grip7_base.pkl")] if kw.get("quick_out") else []
        return [str(PC_PY), str(Path(__file__).with_name("signal_monitor.py")), "--port", EMG_COM, *quick], PC
    if kind == "hand":
        extra = (["--calibrate", "--calibrate-only"] if kw.get("calibrate") else []) + (["--update-baseline"] if kw.get("update_baseline") else [])
        return [str(HAND_PY), str(HAND_DIR / "start_hand.py"), "--log", kw["log"], *extra], ORCA
    if kind == "teach":                              # 자세 가르치기 (teach_live.py: 60 mA 로 낮춰 손으로 밀어 만듦)
        return [str(HAND_PY), str(HAND_DIR / "teach_live.py"), kw["pose"], "--soft", str(kw["soft"]), *(["--path"] if kw.get("path") else [])], ORCA
    if kind == "preview":                            # 동작 미리보기 (11-07): 대기 → 쥐기(엄지 동선) → 풀기(거꾸로) → 대기, 근전도 없이
        return [str(HAND_PY), str(HAND_DIR / "emg_bridge_guarded.py"), "--poses", "--only", kw["pose"], "--close-s", "1.5", "--hold", "1.0"], ORCA
    if kind == "emg":
        return [str(PC_PY), "grip_rt.py", "run", "--model", kw["model"], "--port", EMG_COM, "--udp", "127.0.0.1:9000", "--log", kw["log"], "--save-raw", kw["raw"],
                *(["--svm-temp", str(kw["temp"])] if kw.get("temp") is not None else [])], PC
    raise ValueError(kind)


class Job:
    """프로그램 하나: 출력 수집, 정지 신호. 줄바꿈 없이 \\r 로 덮어 쓰는 줄은 status 로 따로 둠"""

    def __init__(self, name, argv, cwd):
        self.name, self.argv, self.t0 = name, argv, time.time()
        self.lines = collections.deque(maxlen=400); self.status = ""; self.meta = None; self.quick = None; self.stopping = False; self.code = None; self.t_end = None
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        env.pop("VIRTUAL_ENV", None)
        # 정지 신호를 프로그램 묶음(자식 포함)에 보내려고 따로 묶음을 만듦: 윈도우 = 새 프로세스 그룹, 리눅스 = 새 세션
        group = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if WIN else dict(start_new_session=True)
        self.proc = subprocess.Popen(argv, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **group)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        while True:
            chunk = self.proc.stdout.read1(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                m = re.search(rb"\r\n|\n|\r(?=[^\n])", buf)      # Windows 줄끝 \r\n 은 한 줄, \r 만 있으면 덮어 쓰는 상태 줄
                if not m:
                    break
                text = buf[:m.start()].decode("utf-8", "replace").rstrip(); end = m.group(); buf = buf[m.end():]
                if not text or NOISE.search(text):
                    continue
                if text.startswith("#CH "):                     # 신호 확인 프로그램의 채널 값
                    self.meta = json.loads(text[4:])
                    continue
                if text.startswith("#QC "):                     # 대기 보정 진행 상태
                    self.quick = json.loads(text[4:])
                    continue
                if end == b"\r":
                    self.status = text
                else:
                    self.lines.append(text)
            if buf.strip(b"\r\n"):                                # 다음 \r 전까지 쓰인 상태 줄도 바로 표시
                self.status = buf.decode("utf-8", "replace").strip()
        if buf.strip():
            self.lines.append(buf.decode("utf-8", "replace").rstrip())
        self.code = self.proc.wait(); self.t_end = time.time()

    @property
    def running(self):
        return self.code is None

    def stop(self):
        """Ctrl+C 와 같은 안전 종료 요청 (의수: 손 펴고 토크 끔, 근전도: 기록 저장·보드 정지)"""
        if self.running:
            self.stopping = True
            if WIN:
                os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
            else:
                try:
                    os.killpg(self.proc.pid, signal.SIGINT)
                except ProcessLookupError:              # 이미 끝남
                    pass

    def force(self):
        if self.running:
            if WIN:
                subprocess.call(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def enter(self, text=""):
        if self.running:
            self.proc.stdin.write(text.encode() + b"\n"); self.proc.stdin.flush()

    def view(self):
        return dict(running=self.running, code=self.code, stopping=self.stopping, status=self.status, lines=list(self.lines)[-120:], elapsed=round((self.t_end or time.time()) - self.t0, 1))


class State:
    def __init__(self):
        self.jobs = {}; self.svm_out = None; self.session = next_session(); self.t_band = None; self.report = None
        self.run_no = None; self.record_kind = None; self.lock = threading.RLock(); self.ports = {}; self.t_ports = 0.0; self.teach_pose = None; self.teach_path = False; self.teach_preview = False; self.fresh = False; self.hand_mode = None; self.model = None; self.svm_temp = None; self.sum_cache = {}
        self.flow = dict(active=False, stage="", step=0, waiting=False, ask=None, message="", ok=None); self.answer = None; self.cancel = threading.Event(); self.proceed = threading.Event()


def next_session():
    """오늘 날짜 + 알파벳 (이미 쓴 것 다음)"""
    day = time.strftime("%Y%m%d")
    used = {m.group(1) for f in DATA.glob(f"*_{day}*.csv") if (m := re.search(day + r"([a-z])", f.name))}
    return day + next(c for c in "abcdefghijklmnopqrstuvwxyz" if c not in used)


def unused_run():
    nums = [int(m.group(1)) for f in DATA.glob("R*") if (m := re.match(r"(?:RUN|RAW)(\d+)", f.name))]
    return max(nums, default=0) + 1


def session_files(session, kind):
    return sorted(DATA.glob(f"{kind}_{session}*.csv"), key=os.path.getmtime)


def models():
    fs = [f for f in DATA.glob("grip7_*.pkl") if f.name != "grip7_base.pkl" and "check" not in f.name]
    return [f.name for f in sorted(fs, key=os.path.getmtime, reverse=True)]


_INFO = {}


def model_label(name):
    """목록용 이름: 파일 이름만 보고 만듦 (grip7_20261106b_svm.pkl → VASH SVM 1106b)"""
    m = re.match(r"grip7_\d{4}(\d{2})(\d{2})([a-z]?)_?(.*)\.pkl", name)
    if not m:
        return name
    rest = m.group(4)
    fresh = rest.startswith("fresh")                 # 이 착용만으로 학습 (11-07)
    rest = rest[len("fresh"):].lstrip("_") if fresh else rest
    engine = "SVM" if rest.startswith("svm") else ("대기 보정" if rest.startswith("rest") else "LDA")
    rev = re.search(r"(\d+)$", rest)
    return f"VASH {engine} {m.group(1)}{m.group(2)}{m.group(3)}" + (" 단독" if fresh else "") + (f" r{rev.group(1)}" if rev else "")


def model_info(name):
    """모델 카드: 이름·버전·판단 방식·보정 녹화·검증 기록. 파일이 바뀔 때만 다시 읽음"""
    import pickle
    if str(PC) not in sys.path:                  # 모델 파일을 풀려면 pc_tools 모듈(rt_bench 등)이 필요
        sys.path.insert(0, str(PC))
    p = DATA / name; key = (name, p.stat().st_mtime)
    if key not in _INFO:
        try:
            pk = pickle.load(open(p, "rb"))
        except Exception as e:
            return dict(file=name, title="VASH", version=name, engine="?", base="", made="", files=[], windows=None, weight=None, gestures=[], eval=f"모델 정보를 읽지 못했습니다: {e}", base_eval=None)
        cal = pk.get("calibrated") or {}
        engine = "SVM" if pk.get("svm") is not None else ("대기 보정" if cal.get("method") == "rest" else "LDA")
        m = re.match(r"grip7_(\d{4})(\d{2})(\d{2})([a-z]?)_?(.*)\.pkl", name)
        tag = f"{m.group(2)}{m.group(3)}{m.group(4)}" if m else name
        rev = re.search(r"(\d+)$", m.group(5)) if m else None
        evals = {}
        try:
            evals = json.load(open(DATA / "model_eval.json", encoding="utf-8"))
        except Exception:
            pass
        _INFO[key] = dict(file=name, title=f"VASH {engine}", version=f"{tag}" + (f" r{rev.group(1)}" if rev else ""),
                          engine=engine, base=pk.get("created", ""), made=cal.get("time", ""), files=cal.get("files", []),
                          windows=cal.get("windows"), weight=cal.get("weight"), gestures=pk["names"][1:], eval=evals.get(name),
                          base_eval=evals.get("grip7_base.pkl"))
    return _INFO[key]


def parse_report(lines):
    """grip_model calib 출력 → 화면용. severe = 그 보정을 쓰면 안 되는 경고"""
    rep = dict(match=None, items=[], severe=False, saved=None, snr=None)
    for ln in lines:
        s = ln.strip()
        if m := re.search(r"밴드 위치 일치도 ([\d.]+)", s):
            rep["match"] = float(m.group(1))
        elif s.startswith("채널별 신호 대 잡음"):
            rep["snr"] = s.split(":", 1)[1].strip()
        elif s.startswith("보정 저장"):
            rep["saved"] = s
        elif s.startswith(("위치 ", "휴식 잡음", "60 Hz", "신호 약함", "※")):
            severe = "쓰지 말 것" in s or "많이 어긋남" in s
            rep["items"].append(dict(text=s, level="bad" if severe else ("ok" if s.startswith("위치 좋음") else "warn")))
            rep["severe"] = rep["severe"] or severe
    return rep


S = None
app = FastAPI()


@app.middleware("http")
async def local_only(request: Request, call_next):
    # 다른 웹사이트가 이 PC 의 앱으로 손을 움직이는 요청을 보내지 못하게 함
    origin = request.headers.get("origin")
    if request.headers.get("host") not in LOCAL_HOSTS or (origin and origin.split("//")[-1] not in LOCAL_HOSTS):
        return JSONResponse({"detail": "로컬에서만 사용할 수 있습니다"}, status_code=403)
    return await call_next(request)


def running(name):
    j = S.jobs.get(name)
    return j is not None and j.running


def start(name, kind, **kw):
    with S.lock:
        if running(name):
            raise HTTPException(409, f"{name} 이미 실행 중")
        argv, cwd = command(kind, **kw)
        S.jobs[name] = Job(name, argv, cwd)


def get_ports():
    if time.time() - S.t_ports > 2:
        S.t_ports = time.time()
        if WIN:
            try:
                from serial.tools import list_ports
                names = {p.device for p in list_ports.comports()}
            except Exception:
                names = set()
            S.ports = {EMG_COM: EMG_COM in names, HAND_COM: HAND_COM in names}
        else:                                            # /dev/vash-emg 같은 링크는 목록에 안 나오므로 파일로 확인
            S.ports = {EMG_COM: os.path.exists(EMG_COM), HAND_COM: os.path.exists(HAND_COM)}
    return S.ports


def live():
    """근전도 기록 끝줄 = 지금 판단"""
    if not running("emg") or S.run_no is None:
        return None
    p = DATA / f"RUN{S.run_no}_emg.csv"
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 300))
            last = f.read().decode("utf-8", "replace").splitlines()[-1]
        wall, t, state, grip, com, z, level, *rest = last.split(",")
        post = [float(v) for v in rest[0].split(";")] if rest and rest[0].strip() else None
        return dict(age=round(time.time() - float(wall), 2), t=float(t), state=state, grip=grip, committed=com == "1", z=float(z), level=float(level), post=post)
    except Exception:
        return None


def run_summary(n):
    """RUN n 요약 (끝난 RUN 만, 파일이 바뀌지 않으면 다시 계산 안 함)"""
    import run_tools
    p = DATA / f"RUN{n}_emg.csv"
    if not p.exists():
        return None
    key = (p.stat().st_mtime, (DATA / f"RUN{n}_marks.csv").exists())
    c = S.sum_cache.get(n)
    if not c or c[0] != key:
        try:
            c = (key, run_tools.summarize(p, DATA / f"RUN{n}_marks.csv"))
        except Exception as e:
            c = (key, dict(error=str(e)))
        S.sum_cache[n] = c
    return c[1]


def last_run_view():
    """근전도를 끈 뒤: 이번 RUN 요약 + 바로 앞 RUN 요약 (비교) + 평가 저장 여부"""
    n = S.run_no
    if n is None or running("emg") or "emg" not in S.jobs:
        return None
    prev = max((int(m.group(1)) for f in DATA.glob("RUN*_emg.csv") if (m := re.fullmatch(r"RUN(\d+)_emg\.csv", f.name)) and int(m.group(1)) < n), default=None)
    note = DATA / f"RUN{n}_note.json"
    return dict(n=n, model=S.model, svm_temp=S.svm_temp, sum=run_summary(n), prev=prev, prev_sum=run_summary(prev) if prev is not None else None,
                note=json.loads(note.read_text(encoding="utf-8")) if note.exists() else None)


@app.post("/api/mark")
def mark():
    """실시간 중 '틀림' 표시: 그 순간의 판단과 함께 RUN{n}_marks.csv 에 남김"""
    if not running("emg") or S.run_no is None:
        raise HTTPException(409, "근전도가 켜져 있을 때만 표시합니다")
    p = DATA / f"RUN{S.run_no}_marks.csv"; lv = live() or {}
    new = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write("wall,t,grip\n")
        f.write(f"{time.time():.4f},{lv.get('t', '')},{lv.get('grip') or ''}\n")
    with open(p, encoding="utf-8") as f:
        return {"ok": True, "count": sum(1 for _ in f) - 1}


@app.post("/api/run_note")
async def run_note(req: Request):
    """근전도를 끈 뒤 평가 저장: RUN{n}_note.json (좋음/비슷/나쁨, 메모, 모델·온도, 요약)"""
    b = await req.json(); v = last_run_view()
    if not v:
        raise HTTPException(409, "끝난 RUN 이 없습니다")
    if b.get("rating") not in ("좋음", "비슷", "나쁨"):
        raise HTTPException(400, "평가를 고르세요")
    note = dict(run=v["n"], saved=time.strftime("%Y-%m-%d %H:%M:%S"), rating=b["rating"], memo=str(b.get("memo", ""))[:500],
                model=v["model"], svm_temp=v["svm_temp"], summary=v["sum"])
    (DATA / f"RUN{v['n']}_note.json").write_text(json.dumps(note, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True}


def hand_view():
    j = S.jobs.get("hand")
    if j is None:
        return dict(phase=None, ready=False, alert=None, waiting_enter=False, status="")
    text = list(j.lines)
    phases = [ln.strip("= ") for ln in text if ln.startswith("=====")]
    phase = phases[-1] if phases else None
    alert = None
    joined = "\n".join(text[-60:])
    if "Torque release unconfirmed" in joined or "힘 풀림을 확인하지 못했어요" in joined:
        alert = "토크 해제를 확인하지 못했습니다. 모터 전원을 끄세요."
    elif "이동량 이상" in joined:
        alert = "캘리브레이션 값이 이상합니다. 아래 기록을 확인하세요."
    elif "중립 도착 실패" in joined:
        alert = "중립 도착에 실패했습니다. 아래 기록을 확인하세요."
    elif j.code not in (None, 0, 130) and not j.stopping:
        alert = f"의수 프로그램이 오류로 멈췄습니다 (코드 {j.code}). 아래 기록을 확인하세요."
    return dict(phase=phase, ready=j.running and bool(phase) and phase.startswith("손 제어"), alert=alert,
                waiting_enter=j.running and bool(phase) and phase.startswith("장력"), status=j.status)


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"), headers={"Cache-Control": "no-store"})


GUIDES = {"electrode_guide.png", "electrode_guide_right.png", "section_gaps.png"}


@app.get("/favicon.png")
def favicon():
    return FileResponse(Path(__file__).with_name("assets") / "vash.png")


@app.get("/guide/{name}")
def guide(name: str):
    if name not in GUIDES:
        raise HTTPException(404, "그림 없음")
    return FileResponse(Path(__file__).with_name("assets") / name)


@app.get("/api/model/{name}")
def model_card(name: str):
    if name not in models():
        raise HTTPException(404, "모델 없음")
    return model_info(name)


@app.get("/api/state")
def state():
    jobs = {k: v.view() for k, v in S.jobs.items() if k != "monitor"}
    mon = S.jobs.get("monitor")
    return dict(ports=get_ports(), jobs=jobs, flow=S.flow, signal=mon.meta if mon and mon.running else None, signal_on=running("monitor"), quick=mon.quick if mon else None, svm_out=S.svm_out, gestures=GESTURES.split(","), model_labels={m: model_label(m) for m in models()},
                signal_error=(list(mon.lines)[-1] if mon and not mon.running and mon.code not in (0, None) and mon.lines else None), session=S.session, band_elapsed=round(time.time() - S.t_band) if S.t_band else None, report=S.report, record_kind=S.record_kind,
                record=RECORD_VIEW, files={k: [f.name for f in session_files(S.session, k)] for k in RECORD}, models=models(), hand=hand_view(), live=live(), run_no=S.run_no, svm_temp=S.svm_temp, last_run=last_run_view(), teach_pose=S.teach_pose, teach_path=S.teach_path, teach_preview=S.teach_preview, arm=ARM, hand_mode=S.hand_mode,
                next_run=unused_run(), emg_com=EMG_COM, hand_com=HAND_COM)


RECORD_VIEW = {k: dict(title=v[0], seconds=v[1]) for k, v in RECORD.items()}


@app.post("/api/session")
async def set_session(req: Request):
    name = (await req.json()).get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", name):
        raise HTTPException(400, "착용 이름은 영문·숫자만 쓸 수 있습니다")
    S.session, S.report = name, None
    return {"ok": True}


@app.post("/api/new_session")
def new_session():
    S.session, S.report, S.t_band = next_session(), None, time.time()
    return {"session": S.session}


@app.post("/api/record/{task}")
def record(task: str):
    if task not in RECORD:
        raise HTTPException(404, "녹화 종류 없음")
    if running("emg") or running("record") or S.flow["active"]:
        raise HTTPException(409, "근전도 포트를 쓰는 프로그램을 먼저 끄세요")
    stop_monitor()
    S.record_kind = task
    start("record", "record", session=S.session, task=task, out=str(DATA / f"{task}_{S.session}.csv"))
    return {"ok": True}


def stop_monitor():
    """근전도 포트를 다른 프로그램이 쓰기 전에 신호 확인을 끔"""
    if running("monitor"):
        S.jobs["monitor"].stop()
        t_end = time.time() + 5
        while running("monitor") and time.time() < t_end:
            time.sleep(0.05)


@app.post("/api/signal/start")
def signal_start():
    if running("emg") or running("record") or S.flow["active"]:
        raise HTTPException(409, "근전도 포트를 쓰는 중입니다. 먼저 끄세요")
    start("monitor", "monitor")
    return {"ok": True}


@app.post("/api/quick/start")
def quick_start():
    """대기 보정: 블록 실시간 표시 + 조용해지면 자동 측정 → 기본 모델에 적용해 새 모델 저장"""
    if ARM == "right":
        raise HTTPException(409, "오른팔은 아직 빠른 보정을 쓸 수 없습니다 (기본 모델이 왼팔 착용으로 만들어짐). 정밀 보정을 쓰세요")
    if running("emg") or running("record") or S.flow["active"]:
        raise HTTPException(409, "근전도 포트를 쓰는 중입니다. 먼저 끄세요")
    stop_monitor()
    out = DATA / f"grip7_{S.session}_rest.pkl"; raw = DATA / f"REST_{S.session}.csv"; k = 2
    while out.exists() or raw.exists():
        out = DATA / f"grip7_{S.session}_rest{k}.pkl"; raw = DATA / f"REST_{S.session}_{k}.csv"; k += 1
    start("monitor", "monitor", quick_out=str(out), quick_raw=str(raw))
    return {"ok": True}


def wait_job(name):
    while running(name):
        time.sleep(0.05)
    return S.jobs[name].code


def run_calib(files, weight, out, fresh=False):
    start("calib", "calib", files=files, weight=weight, out=str(out), fresh=fresh)
    code = wait_job("calib")
    return code, parse_report(list(S.jobs["calib"].lines))


def contact_check(seconds=2.0, timeout=15.0):
    """녹화 전 접촉 확인: 신호 확인을 켜서 안정된 뒤 2 s 동안 나쁨·주의 채널을 모음.
    문제가 있으면 신호 확인을 켜 둔 채 돌아옴 (팝업이 떠 있는 동안 블록을 보며 전극을 고치도록)"""
    if not running("monitor"):
        start("monitor", "monitor")
    mon = S.jobs["monitor"]; worst = {}; t_end = time.time() + timeout; t_ok = None
    t0 = mon.meta["t"] if mon.meta else 0.0
    rank = {"ok": 0, "warn": 1, "bad": 2}
    while time.time() < t_end and running("monitor") and not S.cancel.is_set():
        m = mon.meta
        if m and m["t"] >= max(4.0, t0 + 1.0) and all(c["state"] != "wait" for c in m["ch"]):
            t_ok = t_ok or time.time()
            for c in m["ch"]:
                if rank.get(c["state"], 0) >= rank.get(worst.get(c["n"], ("ok", ""))[0], 0):
                    worst[c["n"]] = (c["state"], c["why"])
            if time.time() - t_ok >= seconds:
                break
        time.sleep(0.1)
    err = None if t_ok else (list(mon.lines)[-1] if mon.lines else "신호를 받지 못했습니다")
    items = [dict(text=f"보드 신호 확인 실패: {err}", level="bad")] if err else \
        [dict(text=f"CH{n}: {why}", level=st) for n, (st, why) in sorted(worst.items()) if st != "ok"]
    if err or not items:
        stop_monitor()
    return items


def ask(kind, title, items, note=""):
    """사용자 확인 팝업: 'retry' (전극을 고친 뒤 다시) / 'go' (그대로 진행) / None (중단)"""
    S.answer = None
    S.flow.update(waiting=True, ask=dict(kind=kind, title=title, items=items, note=note))
    while S.answer is None and not S.cancel.is_set():
        time.sleep(0.05)
    S.flow.update(waiting=False, ask=None)
    return None if S.cancel.is_set() else S.answer


def calibration_worker():
    """캘리브레이션 = 접촉 확인 → 40 초 보정 녹화 → 품질 확인 → 전환 녹화 → 모델 (LDA·SVM).
    11-07 사용자 요청: 접촉·위치가 불안정하면 그대로 다음 단계로 넘어가지 않고, 고친 뒤 다시 할지 그대로 진행할지 팝업으로 물어봄"""
    f = S.flow
    def stage(step, text):
        f.update(step=step, stage=text)
    def ended(msg, ok):
        f.update(active=False, waiting=False, ask=None, message=msg, ok=ok, stage="")
    def record(task):
        S.record_kind = task
        start("record", "record", session=S.session, task=task, out=str(DATA / f"{task}_{S.session}.csv"))
        return wait_job("record")
    try:
        stop_monitor()
        while True:
            stage(1, "1/5 전극 접촉 확인 (힘을 빼고 기다려 주세요)")
            items = contact_check()
            if S.cancel.is_set(): return ended("캘리브레이션을 중단했습니다.", None)
            if not items: break
            ans = ask("contact", "전극 접촉이 불안정합니다", items, "표시된 채널의 전극을 눌러 보거나 다시 끼운 뒤 [다시 확인]을 누르세요. 이대로 녹화하면 동작 구분이 나빠질 수 있습니다.")
            if ans is None: return ended("캘리브레이션을 중단했습니다.", None)
            if ans == "go": break
        stop_monitor()
        while True:
            stage(2, "2/5 보정 녹화")
            code = record("CAL")
            if S.cancel.is_set(): return ended("캘리브레이션을 중단했습니다.", None)
            if code: return ended(f"녹화가 오류로 멈췄습니다 (코드 {code}). 아래 기록을 확인하세요.", False)
            stage(3, "3/5 신호 품질 확인")
            code, S.report = run_calib([str(session_files(S.session, "CAL")[-1])], 10, DATA / "grip7_check.pkl")
            if code: return ended("품질 확인이 오류로 멈췄습니다. 아래 기록을 확인하세요.", False)
            items = [it for it in S.report["items"] if it["level"] != "ok"]
            if not items: break
            pos_only = all(it["text"].startswith("위치 ") for it in items)
            note = ("위치 경고는 예전 착용과 비교한 값이라, 새 기준 위치로 찼다면 그대로 진행해도 됩니다." if pos_only else
                    "전극을 확인한 뒤 [다시 녹화]를 누르면 보정 녹화(약 50초)부터 다시 합니다.")
            ans = ask("quality", "보정 녹화 품질 경고", items, note)
            if ans is None: return ended("캘리브레이션을 중단했습니다.", None)
            if ans == "go": break
        stage(4, "4/5 전환 녹화")
        code = record("TRANS")
        if S.cancel.is_set(): return ended("캘리브레이션을 중단했습니다.", None)
        if code: return ended(f"녹화가 오류로 멈췄습니다 (코드 {code}). 아래 기록을 확인하세요.", False)
        stage(5, "5/5 모델 만들기")
        fr = "_fresh" if S.fresh else ""; out = DATA / f"grip7_{S.session}{fr}.pkl"; k = 2
        while out.exists():
            out = DATA / f"grip7_{S.session}{fr}_{k}.pkl"; k += 1
        code, S.report = run_calib([str(session_files(S.session, "CAL")[-1]), str(session_files(S.session, "TRANS")[-1])], 30, out, fresh=S.fresh)
        if code or not S.report["saved"]: return ended("모델을 만들지 못했습니다. 아래 기록을 확인하세요.", False)
        # 11-06 실사용 평가 "SVM 이 더 나았어" → 같은 녹화로 SVM 모델도 만들어 선택해 둠 (LDA 모델은 그대로 남음)
        stage(5, "5/5 SVM 모델 만들기 (약 20초)")
        start_svm()
        code = wait_job("svm")
        if S.cancel.is_set(): return ended(f"캘리브레이션을 중단했습니다. LDA 모델 {out.name} 은 저장됨", None)
        warn = " (경고 있음, 위 내용을 확인하세요)" if S.report["severe"] else ""
        if code:
            S.svm_out = None
            return ended(f"완료: {out.name}{warn}. SVM 모델은 만들지 못했습니다 (아래 기록 확인)", False)
        ended(f"완료: {S.svm_out} 선택됨 (LDA {out.name} 도 저장됨){warn}", not S.report["severe"])
    except Exception as e:
        ended(f"캘리브레이션 오류: {e}", False)


@app.post("/api/calibration/start")
async def calibration_start(req: Request):
    try:
        S.fresh = bool((await req.json()).get("fresh"))    # 11-07: 이 착용만으로 학습 (오른팔·처음 차는 위치)
    except Exception:
        S.fresh = False
    if S.flow["active"] or running("emg") or running("record"):
        raise HTTPException(409, "근전도 포트를 쓰는 프로그램을 먼저 끄세요")
    S.report = None; S.cancel.clear(); S.answer = None
    S.flow.update(active=True, step=0, stage="시작 중", waiting=False, ask=None, message="", ok=None)
    threading.Thread(target=calibration_worker, daemon=True).start()
    return {"ok": True}


@app.post("/api/calibration/continue")
def calibration_continue():
    S.answer = "go"
    return {"ok": True}


@app.post("/api/calibration/retry")
def calibration_retry():
    S.answer = "retry"
    return {"ok": True}


@app.post("/api/calibration/cancel")
def calibration_cancel():
    S.cancel.set()
    for name in ("record", "svm", "monitor"):
        if running(name):
            S.jobs[name].stop()
    return {"ok": True}


@app.post("/api/svm/build")
def svm_build():
    """정밀 보정 녹화로 SVM 모델을 같이 만듦 (선택 옵션). LDA 모델은 그대로 둠"""
    cal = session_files(S.session, "CAL")
    if not cal:
        raise HTTPException(409, "이 착용의 정밀 보정 녹화가 없습니다")
    if running("svm") or S.flow["active"]:
        raise HTTPException(409, "모델을 만드는 중입니다")
    start_svm()
    return {"ok": True}


def start_svm():
    cal, trans = session_files(S.session, "CAL"), session_files(S.session, "TRANS")
    fr = "_fresh" if S.fresh else ""; out = DATA / f"grip7_{S.session}{fr}_svm.pkl"; k = 2
    while out.exists():
        out = DATA / f"grip7_{S.session}{fr}_svm{k}.pkl"; k += 1
    S.svm_out = out.name
    start("svm", "svm", files=[str(cal[-1])] + ([str(trans[-1])] if trans else []), out=str(out), fresh=S.fresh)


# ---- 동작 자세 수정 (11-07 사용자 요청) ----
TEACH_EXTRA = [("감싸 쥐기 (물체 잡기)", "wrap"), ("대기 자세", "relax")]


def pose_list():
    """앱에서 고칠 수 있는 자세: 지금 쓰는 7 동작(emg_grips.yaml emg_map) + 감싸 쥐기·대기"""
    import yaml
    try:
        grips = yaml.safe_load((HAND_DIR / "emg_grips.yaml").read_text(encoding="utf-8")) or {}
        emap = grips.get("emg_map", {})
        tp = HAND_DIR / "taught_poses.yaml"
        taught = (yaml.safe_load(tp.read_text(encoding="utf-8")) or {}) if tp.exists() else {}
    except Exception as e:
        return dict(error=str(e), items=[])
    items = [(g, emap[g]) for g in GESTURES.split(",") if g in emap] + TEACH_EXTRA
    pp = HAND_DIR / "taught_paths.yaml"
    try:
        paths = (yaml.safe_load(pp.read_text(encoding="utf-8")) or {}) if pp.exists() else {}
    except Exception:
        paths = {}
    fp = paths.get("fist_thumb") or {}
    return dict(items=[dict(label=g, pose=p, taught=p in taught) for g, p in items],
                saved=time.strftime("%m-%d %H:%M", time.localtime(tp.stat().st_mtime)) if tp.exists() else None,
                paths=[dict(label="주먹 엄지 (쥐었다 펴기)", name="fist_thumb", saved=fp.get("saved"), points=len(fp.get("points") or []), designed=bool((grips.get("thumb_paths") or {}).get("fist")))])


@app.get("/api/teach/poses")
def teach_poses():
    return pose_list()


@app.post("/api/teach/start")
async def teach_start(req: Request):
    body = await req.json(); pose = body.get("pose", ""); is_path = bool(body.get("path"))
    soft = float(body.get("soft", 40))
    if not 15 <= soft <= 80:
        raise HTTPException(400, "모터 힘은 15 ~ 80 mA")
    pl = pose_list()
    if pose not in ({p["name"] for p in pl.get("paths", [])} if is_path else {it["pose"] for it in pl["items"]}):
        raise HTTPException(400, "고칠 동작을 고르세요")
    if running("hand") or running("teach"):
        raise HTTPException(409, "의수를 먼저 끄세요 (같은 포트를 씁니다)")
    start("teach", "teach", pose=pose, soft=soft, path=is_path)
    S.teach_pose, S.teach_path, S.teach_preview = pose, is_path, False
    return {"ok": True}


@app.post("/api/teach/preview")
async def teach_preview(req: Request):
    pose = (await req.json()).get("pose", "")
    if pose not in ("fist", "wrap"):
        raise HTTPException(400, "주먹 또는 감싸 쥐기")
    if running("hand") or running("teach"):
        raise HTTPException(409, "의수를 먼저 끄세요 (같은 포트를 씁니다)")
    with S.lock:
        argv, cwd = command("preview", pose=pose)
        S.jobs["teach"] = Job("teach", argv, cwd)
    S.teach_pose, S.teach_path, S.teach_preview = pose, False, True
    return {"ok": True}


@app.post("/api/teach/key")
async def teach_key(req: Request):
    """저장 = Enter, 다시 해보기 = r, 취소 = q (teach_live.py 키와 같음)"""
    k = {"save": "", "redo": "r", "quit": "q", "rec": "s"}.get((await req.json()).get("key"))
    if k is None or not running("teach"):
        raise HTTPException(409, "자세 수정이 실행 중이 아닙니다")
    S.jobs["teach"].enter(k)
    return {"ok": True}


@app.post("/api/teach/path/delete")
async def teach_path_delete(req: Request):
    """녹화한 경로 지우기 → 원래 자세 계산으로 돌아감 (이전 파일은 backup/ 에)"""
    import shutil, yaml
    name = (await req.json()).get("name", "")
    pp = HAND_DIR / "taught_paths.yaml"
    data = (yaml.safe_load(pp.read_text(encoding="utf-8")) or {}) if pp.exists() else {}
    if name not in data:
        raise HTTPException(404, "녹화한 경로가 없습니다")
    if running("hand") or running("teach"):
        raise HTTPException(409, "의수를 끈 뒤 지우세요")
    (HAND_DIR / "backup").mkdir(exist_ok=True)
    shutil.copy(pp, HAND_DIR / "backup" / f"taught_paths_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
    data.pop(name)
    pp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"ok": True}


@app.post("/api/hand/start")
async def hand_start(req: Request):
    body = await req.json()
    if running("hand"):
        raise HTTPException(409, "의수가 이미 실행 중입니다")
    if running("teach"):
        raise HTTPException(409, "자세 수정을 먼저 끝내세요")
    n = unused_run()
    cal = bool(body.get("calibrate"))                 # [장력·캘리브레이션]: 장력 → 캘리브레이션 → 이동량 검사만 하고 끝
    start("hand", "hand", log=str(DATA / (f"TUNE_{time.strftime('%Y%m%d_%H%M%S')}_hand.csv" if cal else f"RUN{n}_hand.csv")),
          calibrate=cal, update_baseline=cal and bool(body.get("update_baseline")))
    S.hand_mode = "tune" if cal else "run"
    return {"ok": True}


@app.post("/api/hand/enter")
def hand_enter():
    if not running("hand"):
        raise HTTPException(409, "의수가 실행 중이 아닙니다")
    S.jobs["hand"].enter()
    return {"ok": True}


@app.post("/api/emg/start")
async def emg_start(req: Request):
    body = await req.json(); model = body.get("model", ""); temp = body.get("temp")
    if model not in models():
        raise HTTPException(400, "모델을 고르세요")
    if temp is not None:                                  # 11-07: SVM 모델일 때만 온도 적용
        temp = float(temp)
        if not 0.5 <= temp <= 5:
            raise HTTPException(400, "SVM 온도는 0.5 ~ 5")
        if "_svm" not in model:
            temp = None
    if running("record") or running("emg") or S.flow["active"]:
        raise HTTPException(409, "근전도 포트를 쓰는 프로그램을 먼저 끄세요")
    stop_monitor()
    n = unused_run()
    start("emg", "emg", model=str(DATA / model), log=str(DATA / f"RUN{n}_emg.csv"), raw=str(DATA / f"RAW{n}.csv"), temp=temp)
    S.run_no, S.model, S.svm_temp = n, model, temp
    return {"ok": True}


@app.post("/api/stop/{name}")
def stop(name: str):
    if not running(name):
        raise HTTPException(409, "실행 중이 아닙니다")
    S.jobs[name].stop()
    return {"ok": True}


def stop_all_worker():
    """근전도·녹화를 먼저 끄고 의수를 끔 (손이 마지막 명령으로 움직이지 않게)"""
    S.cancel.set()
    for name in ("emg", "record", "monitor"):
        if running(name):
            S.jobs[name].stop()
    t_end = time.time() + 6
    while time.time() < t_end and (running("emg") or running("record") or running("monitor")):
        time.sleep(0.05)
    for name in ("hand", "teach"):
        if running(name):
            S.jobs[name].stop()


@app.post("/api/stop_all")
def stop_all():
    threading.Thread(target=stop_all_worker, daemon=True).start()
    return {"ok": True}


@app.post("/api/force/{name}")
def force(name: str):
    if not running(name):
        raise HTTPException(409, "실행 중이 아닙니다")
    S.jobs[name].force()
    return {"ok": True}


@app.post("/api/quit")
def quit_app():
    """앱 끄기 (11-07: 터미널 창 없이 켜므로 화면에서 끔). 전체 정지와 같은 순서로 끄고 서버 종료"""
    def worker():
        time.sleep(0.5)                                   # 응답을 먼저 보냄
        stop_all_worker()
        t_end = time.time() + 10
        while time.time() < t_end and any(running(n) for n in S.jobs):
            time.sleep(0.1)
        for n in list(S.jobs):
            if running(n):
                S.jobs[n].force()
        os._exit(0)
    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


def main():
    global S
    import socket
    with socket.socket() as so:
        if so.connect_ex(("127.0.0.1", PORT)) == 0:      # 이미 켜져 있으면 화면만 다시 엶
            webbrowser.open(f"http://127.0.0.1:{PORT}/")
            return
    S = State()
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    print(f"의수 앱: http://127.0.0.1:{PORT}/   이 창을 닫기 전에 화면의 [전체 정지] 를 누르세요", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
