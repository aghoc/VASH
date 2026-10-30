# VASH Jetson Vision

ELP IMX291 카메라 영상에서 물체를 검출하고 중앙 타깃을 안정화한 뒤 그립과 정규화된
개구량을 제안하는 GPU 전용 모듈이다. 프로젝트명은 **VASH**, 설치 패키지명은
`vash-vision`, 기본 CLI는 `vash`다. 기존 코드와 장치 설정의 호환성을 위해
Python import 경로 `vagt_orca`, 이전 CLI `vagt-orca`와 서비스 파일명은 유지한다.
이 모듈은 VASH의 비전 부분이며 전체 EMG·의수 통합 제어기는 아니다.

## 현재 선택 모델

- **추가학습하지 않은 공식 YOLO26m**, COCO 80종 출력층 유지.
- Jetson native TensorRT FP16, 640×640, batch 1, end-to-end/NMS-free, confidence 0.35.
- 배경 37종을 제외한 물체 43종을 표시한다. 그립 타깃은 병·컵·공·휴대폰·리모컨·
  숟가락·포크·칫솔 8종에서 선택한다.
- 기본 설정은 `configs/runtime.yaml`. 캔·펜 보조 모델과 중앙 crop 모델은 기본 실행에서 제외한다.
- 원본 checkpoint: `models/pretrained/yolo26m.pt`.
- Jetson 엔진: `artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine`.

최종 장치 기록(2026-10-16)은 카메라 수동 종료·자동 시작 해제 상태다.
2026-10-30 소스 공개 작업에서는 Jetson에 재접속하지 못했으므로 현재 장치 상태나
실제 카메라 정확도 합격을 주장하지 않는다. 개구량 계수도 실측 보정 전 초기값이다.

## 실행 구조

`ELP 입력 → YOLO26m → 중앙 후보 → 5프레임/4표 안정화 → 그립·개구량 제안`

| 역할 | 경로 |
| --- | --- |
| 브라우저 화면·상태 | `tools/camera_live_view.py` |
| 카메라 입력·검출·타깃 안정화 | `src/vagt_orca/vision/` |
| 그립 제안 | `src/vagt_orca/grip/` |
| 별도 CLI·UART 송신 | `src/vagt_orca/cli.py`, `communication/` |
| runtime·카메라·그립 설정 | `configs/` |
| 엔진 생성·서비스 | `deployment/` |
| 단위·회귀 검사 | `tests/` |

뷰어는 최신 JPEG만 RAM에 유지하고 자동 녹화하지 않는다. 카메라 서비스 자체는
UART를 송신하지 않으며 `infer --uart ...`가 별도 송신 경로다.

## 설치·모델 복원·실행

[현재 Jetson runbook](deployment/JETSON_RUNBOOK.md)에 CUDA PyTorch/TensorRT 환경,
엔진 생성, 수동 카메라 실행, 서비스 설치와 UART 예제가 있다.
모델 바이너리·데이터·가상환경·실험 산출물은 Git에 포함하지 않는다.

```bash
# 이 디렉터리(src/vision)에서 실행. 먼저 Jetson CUDA 환경을 준비한다.
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/check_gpu.py
```

다운로드 기본값은 원본 m이며 SHA-256을 검증한다. 과거 s 연구에는
`scripts/download_models.py --model yolo26s`를 명시한다.
TensorRT 엔진은 실행할 Jetson에서 직접 생성해야 한다. `desktop-trt` extra는 x86 전용이다.

## 개발 검사

CUDA가 필요 없는 단위 검사는 일반 개발 환경에서도 실행할 수 있다.

```bash
uv sync --extra dev
PYTHONPATH=src .venv/bin/python -m pytest -o addopts='' -ra
.venv/bin/ruff check .
```

GPU opt-in 검사는 실제 GPU·모델·데이터가 필요하다. 단위 검사 통과는 native 엔진·
실제 카메라·UART 실송신 검증을 대신하지 않는다. 추론과 학습은 CUDA가 없으면 실패한다.

## 결과와 연구 이력

- [원본 m 선정·공개 평가·배포 기록](reports/ORIGINAL_DEMO_DEPLOYMENT_20261016.md)
- [캔·펜 실험과 최종 사용 취소](reports/CAN_PEN_SPECIALIST_TEST_20261016.md)
- [UART 규격](docs/UART_PROTOCOL.md)
- [ELP 카메라 기록](docs/ELP_IMX291_CAMERA.md)
- [초기 15종 설계](docs/history/VAGT_ORCA_AI_PLAN.md)
- [과거 연구 워크플로](RESEARCH_WORKFLOW.md), [진행 기록](PROGRESS.md)

`training/`, `datasets/`, 추가 runtime 설정과 s/15종 Phase 6 자동화는 연구 이력으로
보존한다. 현재 m 배포에는 현재 runbook을 사용한다. 과거 보고서의 `runs/`, `data/`,
`artifacts/` 경로는 제외된 실험 자료를 가리키므로 새 clone에 해당 파일이 있는 것은 아니다.
공개 문서의 사용자명·장치 주소·개인 홈 경로는 예제 값으로 치환했다.
