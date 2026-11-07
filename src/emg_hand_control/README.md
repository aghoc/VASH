# EMG 의수 제어

팔에 감는 8채널 근전도 밴드(ADS1299 + ESP32-S3)로 손동작을 판단하고, ORCA Hand v2(오른손, 손목 없음, Dynamixel XC330 16개)를 움직이는 코드입니다. 착용·보정·실행·정지를 한 화면에서 하는 로컬 웹 앱이 들어 있고, Windows 노트북과 Jetson(JetPack 6)에서 같은 코드로 돌아갑니다.

```
ESP32-S3 + ADS1299 ──USB 921600──▶ emg/grip_rt.py ──UDP 9000──▶ hand/emg_bridge_guarded.py ──U2D2──▶ ORCA Hand
                                   (동작 판단)                   (자세·궤적·전류)
                                         ▲                               ▲
                                         └──── emg/app (127.0.0.1:8765) ─┘  실행·정지·보정·기록
```

판단하는 동작은 7가지입니다: 손 펴기 · 주먹 · 집기 · 세손가락 · 가위 · 검지 펴기 · 엄지 펴기. 감싸 쥐기(물체 잡기: 엄지를 벌렸다가 C 자로 감쌈)는 비전이 물체를 보고 POWER 를 제안할 때만 주먹 대신 쓰고, 비전이 없으면 주먹은 그냥 주먹입니다 (`emg_bridge_guarded.py --open-window 1.5` 를 주면 손 펴기 뒤 1.5 s 안의 주먹을 감싸 쥐기로).

## 폴더

```
emg/            근전도 측정·녹화·학습·실시간 판단
  app/          로컬 웹 앱 (FastAPI + 정적 HTML 한 장)
hand/           의수 제어 (orca_core 위에서 실행)
jetson/         Jetson 설치 스크립트, udev 규칙, 실행 스크립트
analysis/       연구 분석 (근육 피로, EMG·비전 융합, 그림)
```

### emg/

| 파일 | 역할 |
|---|---|
| `grip_rt.py` | 실시간 판단기. LDA 또는 SVM 확률 → 누적 증거 → 확정(0.9). `run` = 보드 실시간 / `--replay` = 녹화 재생, 결과를 UDP 로 의수에 보냄 |
| `grip_model.py` | 고정 기본 모델 + 착용마다 보정 (`calib`), SVM 보정 (`calib-svm`), 휴식만으로 빠른 보정, 밴드 위치·신호 검사, 합격 시험 (`gate`) |
| `finger_test.py` | 화면 지시를 따라 동작 녹화 (보정·전환·시험 녹화) |
| `emg_classifier.py`, `emg_math.py` | 필터(20–450 Hz, 60 Hz 노치), 200 ms 창 특징(TD + 리만), LDA |
| `rt_bench.py` | 세기(언제)·동작(무엇) 분리 판단, 녹화를 실시간처럼 재생해 채점 |
| `ads_monitor.py` | 보드 자가 검진, 실시간 파형, 시리얼 프레임 해석 |
| `latency_live.py` | 근전도 판단 → 손 움직임까지 실제 지연 측정 |
| `preprocess.py` | 녹화를 데이터 표준 v1 형식으로 정리 |

### emg/app/

| 파일 | 역할 |
|---|---|
| `server.py` | 앱 서버. 위 스크립트들을 그대로 실행하고, 정지는 Ctrl+C 와 같은 안전 종료로 보냄 |
| `index.html` | 화면 (기본 / 메카 테마) |
| `signal_monitor.py` | 8채널 블록 표시용 신호 상태, 휴식 15초 빠른 보정 |
| `run_tools.py` | 실사용 기록(RUN) 요약: 켜짐·확정·바뀜(끼어듦 추정)·짧은 확정·확정 시간·틀림 표시 |
| `start_windows.vbs` | Windows 에서 터미널 창 없이 실행 |

### hand/

| 파일 | 역할 |
|---|---|
| `start_hand.py` | 의수 시작 순서: 상태 확인 → (선택) 장력·캘리브레이션 → 중립 자세 → 손 제어 |
| `emg_bridge_guarded.py` | UDP 판단 → 손 자세. 확률 섞기, 부드러운 궤적, 닿으면 고정, 미끄러짐 반사, 비전 제안(UDP 9100). 주먹·감싸 쥐기의 엄지는 `emg_grips.yaml` 의 `thumb_paths` 동선을 따름 (손가락 먼저 → 엄지 끝을 펴서 띄움 → 마디 편 채 뿌리만 넘어감 → MCP → IP 마지막), 풀 때는 같은 동선을 거꾸로 (`--thumb-path-s` 0.6 s, `--fist-unwind` 0.6 s) |
| `teach_live.py` | 토크를 켠 채(60 mA) 손으로 밀어 자세 가르치기 → `taught_poses.yaml` |
| `emg_grips.yaml`, `taught_poses.yaml` | 자세 정의, 근전도 동작 ↔ 자세 연결 |
| `config.yaml`, `calibration.yaml`, `travel_baseline.yaml` | 모터 연결, 이 손의 캘리브레이션, 관절 이동 기준 |
| `neutral_guarded.py`, `calibrate_guarded.py`, `hold_tension.py` 등 | 0 전류 시작·한계 검사·종료 시 토크 해제를 지키는 실행기 (`README_team_bundle.md`) |

## 앱

```
1 밴드 착용     착용 이름, 전극 배치도
2 캘리브레이션  빠른 보정(휴식 15초) / 정밀 보정(접촉 확인 → 보정 녹화 → 품질 확인 → 전환 녹화 → LDA·SVM)
3 실행          모델·SVM 온도 선택, 의수 켜기, 동작 수정, 장력·캘리브레이션, 근전도 켜기, 틀림 표시
```

- 정밀 보정 중 접촉이나 품질에 문제가 있으면 팝업으로 묻습니다: 전극을 고친 뒤 다시 / 그대로 진행 / 중단.
- **이 착용만으로 학습** (정밀 보정 체크, 오른팔에서는 기본): 기본 모델(왼팔 착용들)을 섞지 않고 이번 녹화만으로 LDA·SVM 을 만듭니다. 오른팔 첫 착용 재생 (CAL 학습 → TRANS 49 회 채점): 기본+보정 LDA 29·SVM 22, 이 착용만 LDA 47·SVM 42 (`grip_model.py calib --fresh`, `calib-svm --fresh`).
- 근전도를 켜면 휴식이 5초 이어질 때까지 기다립니다. 첫 신호가 휴식 기준이 되기 때문입니다.
- 쓰는 중에 `Space` 를 누르면 그 순간이 `RUN{n}_marks.csv` 에 남습니다. 근전도를 끄면 요약과 앞 RUN 비교가 나오고, 평가(좋음/비슷/나쁨)와 메모를 `RUN{n}_note.json` 으로 저장합니다.
- **동작 미리보기** (동작 수정 팝업): 근전도 없이 주먹·감싸 쥐기를 실제 제어와 같은 동선으로 쥐었다 폅니다.
- **연속 경로 녹화** (동작 수정 팝업): 엄지를 손으로 편 자리에서 쥔 자리까지 천천히 움직이면 그 궤적을 `hand/taught_paths.yaml` 에 저장하고, `thumb_paths` 에 fist 가 없을 때 주먹 엄지가 그 궤적을 따라갑니다.
- **동작 수정하기**: 의수를 끈 상태에서 동작을 고르면 손이 그 자세로 간 뒤 모터 전류가 낮아집니다(25 / 40 / 60 mA, 기본 40). 손가락을 손으로 밀어 맞추고 저장합니다. XC330-T288 은 288:1 감속이라 전류를 낮춰도 어느 정도 버팁니다.
- **장력·캘리브레이션**: 줄이 느슨해졌을 때. 장력 조절(라쳇으로 조인 뒤 [조이기 끝]) → 캘리브레이션 → 이동량 검사까지만 하고 손 제어는 시작하지 않습니다. 줄을 새로 갈았으면 "줄 교체함" 을 켜서 이동량 기준을 새로 저장합니다.
- [전체 정지] 는 근전도·녹화를 먼저 끄고 의수를 마지막에 끕니다(손을 펴고 토크 해제). [앱 끄기] 는 같은 순서로 끈 뒤 서버를 닫습니다.

앱은 `127.0.0.1` 에서만 열립니다. 다른 컴퓨터에서 접속하는 기능은 없습니다.

## 설치

학습된 모델은 numpy·scikit-learn 버전에 묶여 있어서, 모델을 만든 PC 와 같은 버전(`emg/requirements.txt`)을 씁니다. Python 3.14 가 필요합니다.

### Jetson (JetPack 6)

```bash
git clone https://github.com/aghoc/VASH.git
cd VASH
bash src/emg_hand_control/jetson/setup_jetson.sh
```

스크립트가 하는 일:

1. `uv` 설치, `emg/.venv` 에 Python 3.14 와 패키지 설치 (JetPack 기본 Python 3.10 은 numpy 2.5 를 못 씀)
2. `orca_core` 를 `src/emg_hand_control/orca_core` 에 받아 `177f4e7`(v0.5.2)로 맞추고 `uv sync`
3. udev 규칙 설치: 근전도 보드 → `/dev/vash-emg`, U2D2 → `/dev/vash-hand` (FTDI 지연 1 ms 포함), `dialout` 그룹 추가
4. `hand/config.yaml` 의 포트를 `/dev/vash-hand` 로 바꿈
5. 바탕 화면에 VASH 실행 아이콘

설치 뒤 한 번 로그아웃했다가 다시 로그인하고, 장치를 꽂아 `ls -l /dev/vash-*` 로 두 이름이 보이는지 확인합니다.

### Windows

```powershell
cd src\emg_hand_control\emg
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

의수 쪽은 [orcahand/orca_core](https://github.com/orcahand/orca_core) 를 `src\emg_hand_control\orca_core` 에 받아 `uv sync --frozen --no-dev` 합니다. 포트 기본값은 근전도 `COM6`, 의수 `COM7` 이고 `hand\config.yaml` 의 `port` 도 맞춥니다.

### 모델과 데이터

S01 의 모델이 `emg/data/S01/` 에 들어 있습니다. 녹화 원본(CSV)은 넣지 않았습니다.

| 파일 | 용도 |
|---|---|
| `grip7_base.pkl` | 기본 모델 (7동작, 여러 착용 녹화로 학습) |
| `grip7_svm_train.npz` | SVM 보정에 쓰는 기본 학습 창 |
| `grip7_20261106b.pkl`, `grip7_20261106b_svm.pkl` | 11-06 착용 보정 모델 (LDA, SVM). 밴드를 새로 차면 앱에서 다시 보정 |
| `grip7_gate.json`, `model_eval.json` | 합격 시험 기준, 앱 모델 카드의 검증 기록 |

모델은 `requirements.txt` 의 numpy·scikit-learn 버전에서 불러야 합니다. 다른 사람에게는 맞지 않으므로 그 사람의 녹화로 새로 만듭니다(`docs/emg_hand_control/개편_측정녹화_가이드.md`). 기본 모델을 새로 만들 때는 `grip_model.py gate` 를 먼저 통과해야 합니다.

## 실행

**앱**: Jetson 은 바탕 화면 VASH 아이콘 또는 `jetson/start_vash.sh`, Windows 는 `emg/app/start_windows.vbs`. 이미 켜져 있으면 브라우저만 다시 엽니다. 서버 기록은 Jetson 에서 `~/.cache/vash/server.log`.

**명령으로 직접** (의수 먼저, 근전도 다음):

```bash
# 의수 — orca_core 폴더
cd src/emg_hand_control/orca_core
uv run --frozen --no-dev python ../hand/start_hand.py

# 근전도 — emg 폴더
cd src/emg_hand_control/emg
.venv/bin/python grip_rt.py run --model data/S01/grip7_<착용>_svm.pkl --port /dev/vash-emg --udp 127.0.0.1:9000
```

Windows 는 `.venv\Scripts\python`, `--port COM6`.

환경 변수로 기본값을 바꿀 수 있습니다: `VASH_EMG_PORT`, `VASH_HAND_PORT`, `VASH_DATA`, `VASH_ORCA`, `VASH_HAND_DIR`, `VASH_PORT`, `VASH_ARM`.

### 착용 팔 (`VASH_ARM`, 기본 `right`)

| 항목 | `right` | `left` |
|---|---|---|
| 정밀 보정 | 이 착용 녹화만으로 학습 (기본 체크) | 기본 모델 + 이번 녹화 |
| 밴드 위치 일치도 | `data/S01/grip7_ref_right.pkl` (오른팔 착용 기준) | 기본 모델의 기준 (왼팔) |
| 배치도 | `assets/electrode_guide_right.png` (왼팔 배치의 거울) | `assets/electrode_guide.png` |
| 빠른 보정 (휴식 15 초) | 끔 (기본 모델이 왼팔 착용으로 만들어짐) | 사용 |

오른팔은 같은 해부학 위치에 붙입니다: CH1 = 척골 능선 바로 옆 손등 쪽, 손등 → 엄지 쪽 → 손바닥 순서로 CH8 까지, BIAS = 팔꿈치 바깥 뼈(외측상과). 팔꿈치 쪽에서 보면 시계 반대 방향으로 감깁니다. 왼팔 기본 모델에 채널 순서를 거울·회전으로 맞춰 써 보면 최고 37/49 라, 오른팔은 오른팔 녹화로 만듭니다.

## 판단기 설정

| 항목 | 값 | 비고 |
|---|---|---|
| 창 / 간격 | 200 ms / 50 ms | 20–450 Hz, 60 Hz 노치 |
| 분류 | LDA 또는 SVM (RBF) | 착용 보정 뒤 SVM 모델이 기본 선택, LDA 모델도 같이 저장 |
| 확정 | 누적 확률 0.9, 최소 0.25 s | |
| SVM 온도 | 1.5 | 동작끼리만 적용 (휴식 확률은 그대로). 높을수록 끼어듦 감소, 확정 지연 |

실측(11-02, `latency_live.py`): 확정 0.27 s, 손 5° 0.31 s, 절반 0.41 s, 90 % 0.62 s.

착용 20261106b 시험 녹화(보정에 안 쓴 녹화)를 재생해 채점하면 쉬는 동작·연속 동작 모두 42/42 정답, 동작 사이 끼어듦은 LDA 11 회, SVM 8 회였습니다. 같은 사람·같은 날 재생 결과이고, 다른 사용자나 장기 사용 성능은 아직 재지 않았습니다.

## 문제 해결

| 증상 | 확인 |
|---|---|
| 앱 머리줄에 근전도/의수가 "없음" | `ls -l /dev/vash-*`. 없으면 `udevadm info -a -n /dev/ttyACM0` 로 idVendor 를 보고 `jetson/99-vash.rules` 에 추가 |
| CH34x 보드가 꽂자마자 끊김 | Ubuntu 22.04 의 `brltty` 가 장치를 가져감: `sudo apt remove brltty` |
| `Permission denied: /dev/vash-*` | `dialout` 그룹 추가 후 다시 로그인했는지 |
| 모델을 불러올 때 버전 경고·오류 | `emg/.venv` 가 `requirements.txt` 버전인지 (`.venv/bin/python -m pip list`) |
| 녹화 지시 창이 안 뜸 | 바탕 화면 세션에서 실행했는지 (`DISPLAY` 필요) |

## 출처

- 의수 구동: [orcahand/orca_core](https://github.com/orcahand/orca_core) (`177f4e7`, v0.5.2). `hand/` 의 실행기와 설정은 이 저장소에서 만든 것입니다.
