# EMG 기반 의수 제어 (ADS1299 8채널 → ORCA Hand v2)

팔에 감은 8채널 근전도 밴드(ADS1299 + ESP32-S3)로 손동작을 분류하고, 16자유도 ORCA Hand(오른손, 손목 없음)를 사람 손처럼 움직이는 코드입니다.

```
emg/        근전도 측정·녹화·학습·실시간 판단 (PC)
hand/       의수 제어 (ORCA Hand v2 오른손, 손목 없음, Dynamixel XC330 16개)
analysis/   연구 분석 — 근육 피로, EMG·비전 융합, 엄격 평가, 논문·포스터 그림
```

## 구성

```
ESP32-S3 + ADS1299 (hardware/ads1299_firmware)  ──USB 921600──▶  emg/grip_rt.py  ──UDP──▶  hand/emg_bridge_guarded.py  ──U2D2──▶  ORCA Hand
                                                                  (동작 판단)                (자세·궤적·힘 조절)
```

### emg/
| 파일 | 역할 |
|---|---|
| `ads_monitor.py` | 보드 검진(selftest)·실시간 파형 보기, 시리얼 프레임 해석 |
| `finger_test.py` | 동작 녹화 (화면 지시 + 휴식, `--gestures`, `--levels`, `--bend`) |
| `emg_classifier.py` | 필터(20–450 Hz + 60 Hz 노치), 200 ms 창 특징, 데이터셋 구성, 잡음 창 제외 |
| `emg_math.py` | TD + 리만 특징, LDA, 온도 보정 등 수학 파이프라인 |
| `rt_bench.py` | '언제(세기) / 무엇(동작)' 분리 디코더, 녹화를 실시간처럼 재생해 성능 측정 |
| `grip_rt.py` | 실시간 디코더: 학습(`train`) / 실행(`run`, 재생 `--replay`) → UDP 로 의수에 전달 |
| `gesture_analysis.py` | 동작 녹화 분석 |

### hand/
| 파일 | 역할 |
|---|---|
| `emg_bridge_guarded.py` | UDP 근전도 → 손 자세. 확률 섞기, 3차 임계감쇠 궤적, 손가락 시차, 스스로 쥐기(닿음 고정), 살살/꽉 2단계 전류, 미끄러짐 반사, 감싸 쥐기 거꾸로 풀기 |
| `emg_grips.yaml` | 자세(관절 가동범위 비율), 기준점 경로·중간 자세·호, 근전도 동작 ↔ 자세 |
| `taught_poses.yaml`, `teach_pose.py` | 손으로 직접 만든 자세 저장 (토크 끈 채 읽기만) |
| `contact_probe.py` | 물체를 쥘 때 손가락별 전류·위치 기록 (닿음 판단 기준 측정) |
| `config.yaml`, `calibration.yaml` | 관절↔모터 매핑, 실측 캘리브레이션 (이 손·현재 텐던 장력 기준) |
| `neutral_guarded.py`, `calibrate_guarded.py`, `hold_tension.py`, `stress_guarded.py`, `demo_guarded.py` | 팀 공유본 실행기 (0 전류 기동·한계 검사·종료 시 토크 해제). 설명: `README_team_bundle.md` |

현재 동작 (7동작 한 모델, 모드 전환 없음): 손 펴기 · 주먹 · 집기 · 세손가락 · 가위 · 검지 펴기 · 엄지 펴기.
손 펴기 직후 1.5 s 안의 주먹 = 감싸 쥐기(물체 잡기), 그냥 주먹 = 바위.
녹화 재생 검증 93/96 (96.9 %) — 자세한 수치: `docs/emg_hand_control/results/분석_모드_1031.md`

## 설치·실행

1. **근전도 (Python 3.10+)**: `pip install numpy scipy pyserial matplotlib` (SVM 비교 실험만 `scikit-learn` 선택)
2. **의수**: [orcahand/orca_core](https://github.com/orcahand/orca_core) 를 받아 (사용 커밋 `177f4e7`, v0.5.2) 그 옆 `hand-config/right-no-wrist/` 에 `hand/` 파일을 넣고, `orca_core` 폴더에서 `uv sync --frozen --no-dev`
3. 실행 순서와 조절 값: `docs/emg_hand_control/실행가이드_1031.md`

```powershell
# 터미널 B (의수, 먼저) — orca_core 폴더
uv run --frozen --no-dev python ../hand-config/right-no-wrist/emg_bridge_guarded.py
# 터미널 A (근전도) — emg 폴더
python grip_rt.py train <녹화 csv들> --classes "손 펴기,주먹,집기 (중지 펴기),세손가락 (중지 굽힘),가위,검지 펴기,엄지 펴기" --out grip_all7.pkl
python grip_rt.py run --model grip_all7.pkl --port COM6 --udp 127.0.0.1:9000
```

`analysis/` 스크립트는 `emg/` 모듈을 가져오므로 `PYTHONPATH` 에 `emg/` (그림 스크립트는 `analysis/` 도) 를 추가해 실행합니다.

## 포함하지 않은 것
- 근전도 녹화 원본(CSV, 약 1.2 GB)·학습된 모델(.pkl): 개인 생체신호이고 용량이 커서 제외. 녹화 방법은 `docs/emg_hand_control/개편_측정녹화_가이드.md`
- ORCA Core 소스: 원 저장소를 사용
- 접촉 측정 기록(probe/*.csv), 백업, 이전 버전 스크립트
