"""오른팔 녹화 (앱 정밀 보정·시험 녹화와 같은 설정, 11-08)
  python rec_right.py CAL 20261108a      보정 2 회 (약 90 s)
  python rec_right.py TRANS 20261108a    전환 (약 2 분)
  python rec_right.py TEST 20261108a     시험: 쉬는 동작 (약 2 분)
  python rec_right.py CHAIN 20261108a    시험: 연속 동작 (약 1 분)
  python rec_right.py FOCUS 20261109a    집중: 손 펴기·집기·세손가락·가위만 짧게(1.5 s) 서로 바꿔 가며 (약 2 분, 11-09)
  (11-11 밤, 기본 모델용 팔 자세·집기 녹화 — 녹화 내내 그 자세 유지)
  python rec_right.py FWD 20261111c left    팔을 앞으로 뻗어 어깨 높이 (팔꿈치 폄), 7 동작 2 회 (약 90 s)
  python rec_right.py UP 20261111c left     팔꿈치 90° 접어 손이 위로 (위팔은 몸통 옆), 7 동작 2 회 (약 90 s)
  python rec_right.py PINCH 20261111c left  집기와 헷갈리는 동작(손 펴기·세손가락·가위·검지)만, 모든 순서, 길게 (약 4 분)
  python rec_right.py DOWN 20261111c left   팔꿈치 90°, 손바닥이 바닥 (엎침), 약 90 s (11-12)
  python rec_right.py SIDE 20261111c left   팔꿈치 90°, 엄지가 위, 약 90 s (11-12)
  python rec_right.py HELLO 20261112a left  위팔을 어깨 높이로 들고 팔꿈치 90°, 손이 위 (인사 자세), 약 90 s (11-12)
  세 번째 인자 = 착용 팔 (left/right, 없으면 right)
같은 이름 파일이 있으면 _2, _3 … 을 붙임 (덮어쓰지 않음)"""
import os, subprocess, sys, time
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

GESTURES = "손 펴기,주먹,집기 (중지 펴기),세손가락 (중지 굽힘),가위,검지 펴기,엄지 펴기"
OPTS = {
    "CAL": ["--reps", "2", "--relax"],
    "TRANS": ["--reps", "1", "--hold", "2", "--chain", "7", "--all-transitions", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "TEST": ["--reps", "3", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "CHAIN": ["--reps", "3", "--hold", "2", "--chain", "4", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    # 11-09: 실사용(RUN34)에서 집기↔세손가락·가위가 자주 바뀜 → 그 동작만 짧게, 모든 순서로. 라벨은 동작 이름으로 맞춰짐 (emg_classifier.load_csv)
    "FOCUS": ["--reps", "3", "--hold", "1.5", "--chain", "4", "--all-transitions", "--relax", "--rest-jitter", "1", "--next-cue", "0.7"],
    # 11-11 밤 왼팔 기본 모델: 팔 자세가 바뀌면 인식이 떨어짐 (앞으로 뻗기·팔꿈치 90° 위로) → 그 자세로 보정과 같은 녹화
    "FWD": ["--reps", "2", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "UP": ["--reps", "2", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    # 11-12: 아래팔 돌림 (평소 CAL 은 손바닥이 나를 향한 뒤침) → 엎침·중간 자세도 같은 녹화
    "DOWN": ["--reps", "2", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "SIDE": ["--reps", "2", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "HELLO": ["--reps", "2", "--relax", "--rest-jitter", "1", "--next-cue", "1"],   # 11-12: 인사 자세 (RUN65 손 펴기→휴식)
    # 집기 집중: 집기와 헷갈리는 쌍(손 펴기 0.89, 세손가락 0.87 상관)을 서로 바로 바꿔 가며, 2 s 유지, 4 번씩
    "PINCH": ["--reps", "4", "--hold", "2", "--chain", "4", "--all-transitions", "--relax", "--rest-jitter", "1", "--next-cue", "0.7"],
}
TASK_GESTURES = {"FOCUS": "손 펴기,집기 (중지 펴기),세손가락 (중지 굽힘),가위", "PINCH": "손 펴기,집기 (중지 펴기),세손가락 (중지 굽힘),가위,검지 펴기"}
POSTURE = {"FWD": "팔을 앞으로 뻗어 어깨 높이 (팔꿈치 폄)", "UP": "팔꿈치 90° 접어 손이 위로 (위팔은 몸통 옆)",
           "DOWN": "팔꿈치 90°, 손바닥이 바닥을 향함 (엎침)", "SIDE": "팔꿈치 90°, 엄지가 위 (손바닥이 옆을 향함)",
           "HELLO": "위팔을 어깨 높이로 들고 팔꿈치 90°, 손이 위 (인사 자세)"}
HERE = os.path.dirname(os.path.abspath(__file__))
task, session = sys.argv[1].upper(), sys.argv[2]
arm = sys.argv[3] if len(sys.argv) > 3 else "right"
if arm not in ("left", "right"):
    raise SystemExit("세 번째 인자 = left 또는 right")
if task not in OPTS:
    raise SystemExit(f"녹화 종류: {', '.join(OPTS)}")
out = os.path.join(HERE, "data", "S01", f"{task}_{session}.csv"); k = 2
while os.path.exists(out):
    out = os.path.join(HERE, "data", "S01", f"{task}_{session}_{k}.csv"); k += 1
print(f"녹화: {task} ({arm}) → {os.path.basename(out)}  (끝나면 이 창에 '저장' 이 나옵니다)", flush=True)
if task in POSTURE:
    print(f"*** 녹화 내내 자세: {POSTURE[task]} - 쉬는 동안에도 팔은 그 자세로 ***", flush=True)
PORT = os.environ.get("VASH_EMG_PORT") or ("COM6" if os.name == "nt" else "/dev/vash-emg")
cmd = [sys.executable, "finger_test.py", "record", "--port", PORT, "--ch", "1-8", "--side", "band", "--subject", "S01",
       "--session", session, "--task", task, "--arm", arm, "--gestures", TASK_GESTURES.get(task, GESTURES), "--seed", str(int(time.time()) % 100000),
       *OPTS[task], *(["--note", POSTURE[task]] if task in POSTURE else []), "--out", out]
raise SystemExit(subprocess.call(cmd, cwd=HERE))
