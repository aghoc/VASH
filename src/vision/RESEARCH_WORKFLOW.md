# 과거 연구·학습 워크플로

이 문서는 이전 YOLO26s/15종 연구 절차를 보존한 기록이다. 현재 원본 YOLO26m의 설치·배포는 [README](README.md)와 [현재 runbook](deployment/JETSON_RUNBOOK.md)을 따른다. Phase 6 자동 완료 판정은 현재 m 배포용이 아니다.

# VAGT-ORCA AI

AR-glass 시점의 물체 검출, 중앙 타깃 안정화와 그립 제안을 수행하는 GPU 전용 파이프라인이다.
현재 **추가학습하지 않은 공식 YOLO26m으로 손으로 다룰 물체 위주로 표시**한다.
책상·의자·사람·냉장고 등 배경 항목37종을 제외하고 나머지43종은 검출을 표시한다.
성능이 좋은 3~5종을 골라 시연한다는 사용자 결정은 화면의 검출 범위를 제한하는 뜻이 아니다.
병·휴대폰·포크는 현재 시연 후보이며, 초기 15종·4그립 설계는 [초기 설계](docs/history/VAGT_ORCA_AI_PLAN.md), 현재 상태는
`PROGRESS.md`를 참조한다.

기본 checkpoint는 `models/pretrained/yolo26m.pt`, Jetson 서비스는 장치에서 생성한
`yolo26m_coco_original_fp16.engine`이다. 원본 80종 가중치와 출력층을 유지하며
`configs/runtime.yaml`의 `detection_profile`에서 `cell phone`→`phone`, `sports ball`→`ball`과
프로젝트 지원 8종의 이름/ID만 연결한다. 배경 제외 목록은 `excluded_classes`다.
책·가위 등 다른 물체도 표시하며, 중앙 타깃은 `target.grip_supported_only: true`에 따라
그립을 지원하는 물체(원본 공통8종)에서 선택한다. confidence 0.35, CUDA FP16, square640, end2end 설정이다.
[모델 선정·배포·공개 평가 결과](reports/ORIGINAL_DEMO_DEPLOYMENT_20261016.md)에 근거와 한계를 기록했다.

2026-10-16 사용자 결정으로 캔·펜 추가 모델 사용을 중단하고 공식 원본 YOLO26m만 유지한다.
카메라는 종료했다(inactive/dead, MainPID=0, 장치 점유 없음). 로그인 자동 시작은 해제 상태이며
다음 사용자 시험 요청 때 수동 실행한다. 뷰어의 프레임·영상 자동 저장은 없다.
[캔·펜 시험 및 종료 기록](reports/CAN_PEN_SPECIALIST_TEST_20261016.md).
기존 추가학습 모델과 복구 백업은 보존하며, 두 추가학습 모델의 ELP 오탐 이력은
[오탐 점검 기록](reports/FALSE_POSITIVE_AUDIT_20261015.md)을 참조한다.

## 환경과 품질 검사

```bash
uv sync --extra dev --extra data
.venv/bin/python scripts/download_models.py --model yolo26s
.venv/bin/python scripts/check_gpu.py
.venv/bin/pytest -q
.venv/bin/ruff check .
```

CUDA를 찾지 못하면 inference/training은 의도적으로 실패하며 CPU로 전환하지 않는다.

YOLO26s pretrained baseline과 default end-to-end/NMS-free head는 다음 명령으로 확인한다.

```bash
.venv/bin/python scripts/phase1_yolo26s_baseline.py
```

## 데이터 준비

```bash
# Meta signed URL이 있으면 --urls urls.json, 공식 페이지 장애 시 pinned recovery mirror
.venv/bin/python scripts/download_egoobjects.py --huggingface-mirror

.venv/bin/python datasets/converters/egoobjects_to_yolo.py \
  --annotations data/raw/egoobjects/Meta_EgoObjects_Val.json \
                data/raw/egoobjects/Meta_EgoObjects_Test.json \
                data/raw/egoobjects/Meta_EgoObjects_Train.json \
  --images data/raw/egoobjects/images \
  --output data/processed/egoobjects_15

.venv/bin/python datasets/validate_dataset.py data/processed/egoobjects_15/data.yaml
```

EgoObjects에 없는 `card`, `small_block`은 공식 LVIS/COCO bbox로 보완한다. 상세 명령과
출처 revision은 `PROGRESS.md`에 기록한다.

## FP32 GPU 학습과 검증

```bash
.venv/bin/python training/train.py --data data/processed/egoobjects_15/data.yaml \
  --model-output artifacts/models/yolo26s_phase3.pt
.venv/bin/python training/validate.py \
  --model artifacts/models/yolo26s_phase3.pt \
  --data data/processed/egoobjects_15/data.yaml --split val \
  --project artifacts/validation_yolo26s
.venv/bin/python training/validate.py \
  --model artifacts/models/yolo26s_phase3.pt \
  --data data/processed/egoobjects_15/data.yaml --split test \
  --project artifacts/validation_yolo26s
.venv/bin/python training/analyze_failures.py \
  --model artifacts/models/yolo26s_phase3.pt \
  --data data/processed/egoobjects_15/data.yaml --batch 16 \
  --output artifacts/failures_yolo26s
.venv/bin/python scripts/summarize_phase3.py
```

학습은 train/val/test 각각에 15개 class가 모두 존재하지 않거나 CUDA가 없으면 즉시
중단된다. 중간 재개 명령과 현재 checkpoint 위치는 `PROGRESS.md`에 기록한다.

## Phase 4~5 인수 검사

```bash
.venv/bin/python scripts/phase45_acceptance.py
.venv/bin/python scripts/phase5_model_e2e.py
```

seed 42의 중앙 선택 100개, 5-frame/4-vote debounce 100개, 15-class grip mapping 및
No-target LKG 동작을 검사하고 `reports/phase45_acceptance.json`을 생성한다.
두 번째 명령은 학습 모델로 실제 validation frame 10개를 처리해
`reports/yolo26s_phase5_e2e.json`과 시각화 MP4를 생성한다.

## 영상 또는 webcam 추론

```bash
.venv/bin/vagt-orca infer --source sample.mp4 --output artifacts/result.mp4
.venv/bin/vagt-orca infer --source 0 --show
```

현재 카메라는 ELP IMX291이며, 측정한 UVC 모드는 `--source elp`로 사용한다. Jetson SSH
접속, 서비스, 브라우저 뷰어와 촬영 절차는 [ELP 카메라 문서](docs/ELP_IMX291_CAMERA.md)에 있다.

```bash
# Jetson /mnt/data/vagt-orca-ai에서 실행
.venv/bin/python -m vagt_orca.cli infer --source elp \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine
```

`--source`는 영상 경로, UVC 카메라 index, ELP 프로필 `elp`, Pixy2 USB 브리지 `pixy`를
받는다. Pixy2.1은 UVC 장치가 아니므로 별도 브리지 빌드가 필요하다. 과거 절차와 제약,
grab 모드 판정은 `docs/PIXY_CAMERA.md`에 있다.

```bash
PIXY2_ROOT=~/pixy2 scripts/build_pixy_bridge.sh
export VAGT_PIXY_BRIDGE=artifacts/deployment/pixy/libpixybridge.so
.venv/bin/python scripts/pixy_benchmark.py --frames 120
.venv/bin/vagt-orca infer --source pixy --show
```

## Phase 6 YOLO26 FP16 배포

```bash
uv sync --extra deployment --extra desktop-trt --extra dev
.venv/bin/python deployment/export_onnx.py
.venv/bin/python deployment/build_benchmark_dataset.py
# ONNX CUDA smoke test 후 x86 RTX용 FP16 engine 경로를 점검한다.
# 이 engine은 Jetson에 배포하지 않는다.
.venv/bin/python deployment/export_tensorrt.py --development-host
.venv/bin/python deployment/benchmark.py \
  --model artifacts/deployment/tensorrt/yolo26m_fp16_RTX_DEVELOPMENT_ONLY_NOT_FOR_JETSON.engine \
  --report reports/yolo26m_benchmark_tensorrt_desktop.json
.venv/bin/python deployment/package_jetson.py
```

TensorRT engine은 하드웨어/플랫폼 종속이므로 Jetson용 engine은 반드시 Jetson에서 생성한다.
대상 장치의 전체 실행 절차는 `deployment/JETSON_RUNBOOK.md`, 자동 완료 판정은
`deployment/summarize_phase6.py`에 있다.

Jetson에서는 `.venv/bin/python deployment/run_jetson_phase6.py` 한 명령으로 FP16 engine 생성,
검증, benchmark와 최종 판정을 순서대로 수행할 수 있다. INT8 calibration은 새 계획 범위가 아니다.

## Phase 7 UART

UART v1은 115200 8N1, 고정 16-byte frame, uint32 sequence와 CRC-16/CCITT-FALSE를 사용한다.
정확한 layout과 ESP32 처리 규칙은 `docs/UART_PROTOCOL.md`에 있다.

```bash
.venv/bin/python scripts/phase7_acceptance.py
.venv/bin/vagt-orca infer --source 0 \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine --uart /dev/ttyTHS1
```

## Phase 8~9 준비

실제 촬영 전 15 class × 최소 3 sequence의 수집표와 비하드웨어 준비 상태를 생성한다.

```bash
.venv/bin/python scripts/prepare_ar_glass_collection.py
```

결과는 `configs/ar_glass_collection_plan.csv`와 `reports/phase8_precollection.json`에 기록된다.
보고서가 `hardware_connection_pending`이면 촬영을 제외한 준비가 끝난 상태다. AR Glass가
연결되면 CSV의 command를 순서대로 실행해 조건을 명시한 domain frame을 수집한다.

```bash
.venv/bin/python tools/record_ar_glass.py --source 0 --class-name bottle \
  --session bottle_bright_near_001 --lighting bright --distance near \
  --background plain --rotation front --occlusion clear \
  --multi-object single --motion static
```

ORCA Hand 실측값은 `calibration/phase9/opening_observations.csv`와 `pose_checks.csv`에 기록한 뒤
`.venv/bin/python tools/fit_opening_calibration.py`로 적합한다. 현재 하드웨어 대기 상태는
`.venv/bin/python scripts/summarize_remaining_phases.py`로 갱신한다.
