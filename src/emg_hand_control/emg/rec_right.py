"""오른팔 녹화 (앱 정밀 보정·시험 녹화와 같은 설정, 11-08)
  python rec_right.py CAL 20261108a      보정 2 회 (약 90 s)
  python rec_right.py TRANS 20261108a    전환 (약 2 분)
  python rec_right.py TEST 20261108a     시험: 쉬는 동작 (약 2 분)
  python rec_right.py CHAIN 20261108a    시험: 연속 동작 (약 1 분)
  python rec_right.py FOCUS 20261109a    집중: 손 펴기·집기·세손가락·가위만 짧게(1.5 s) 서로 바꿔 가며 (약 2 분, 11-09)
같은 이름 파일이 있으면 _2, _3 … 을 붙임 (덮어쓰지 않음)"""
import os, subprocess, sys, time

GESTURES = "손 펴기,주먹,집기 (중지 펴기),세손가락 (중지 굽힘),가위,검지 펴기,엄지 펴기"
OPTS = {
    "CAL": ["--reps", "2", "--relax"],
    "TRANS": ["--reps", "1", "--hold", "2", "--chain", "7", "--all-transitions", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "TEST": ["--reps", "3", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    "CHAIN": ["--reps", "3", "--hold", "2", "--chain", "4", "--relax", "--rest-jitter", "1", "--next-cue", "1"],
    # 11-09: 실사용(RUN34)에서 집기↔세손가락·가위가 자주 바뀜 → 그 동작만 짧게, 모든 순서로. 라벨은 동작 이름으로 맞춰짐 (emg_classifier.load_csv)
    "FOCUS": ["--reps", "3", "--hold", "1.5", "--chain", "4", "--all-transitions", "--relax", "--rest-jitter", "1", "--next-cue", "0.7"],
}
TASK_GESTURES = {"FOCUS": "손 펴기,집기 (중지 펴기),세손가락 (중지 굽힘),가위"}
HERE = os.path.dirname(os.path.abspath(__file__))
task, session = sys.argv[1].upper(), sys.argv[2]
if task not in OPTS:
    raise SystemExit(f"녹화 종류: {', '.join(OPTS)}")
out = os.path.join(HERE, "data", "S01", f"{task}_{session}.csv"); k = 2
while os.path.exists(out):
    out = os.path.join(HERE, "data", "S01", f"{task}_{session}_{k}.csv"); k += 1
print(f"녹화: {task} → {os.path.basename(out)}  (끝나면 이 창에 '저장' 이 나옵니다)", flush=True)
PORT = os.environ.get("VASH_EMG_PORT") or ("COM6" if os.name == "nt" else "/dev/vash-emg")
cmd = [sys.executable, "finger_test.py", "record", "--port", PORT, "--ch", "1-8", "--side", "band", "--subject", "S01",
       "--session", session, "--task", task, "--arm", "right", "--gestures", TASK_GESTURES.get(task, GESTURES), "--seed", str(int(time.time()) % 100000),
       *OPTS[task], "--out", out]
raise SystemExit(subprocess.call(cmd, cwd=HERE))
