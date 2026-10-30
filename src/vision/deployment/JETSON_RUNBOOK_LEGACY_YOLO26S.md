# Phase 6 Jetson Orin Nano 실행 절차

대상 모델은 `YOLO26s`, 배포 precision은 `TensorRT FP16`, 기본 head는 end-to-end/NMS-free로
고정한다. TensorRT engine은 생성한 GPU·TensorRT·JetPack 환경에 종속된다. RTX에서 만든
`*_NOT_FOR_JETSON.engine`은 기능 개발용이며 Jetson에 복사하지 않는다. 제공 bundle에는
checkpoint, ONNX, 코드만 포함되어 있고 engine은 Jetson에서 새로 만든다.

## 1. Bundle 배치와 환경 확인

저장소 루트에서 bundle을 풀고 JetPack이 제공한 CUDA/TensorRT가 보이는지 확인한다.
`desktop-trt` extra는 x86 개발 호스트 전용이므로 Jetson에서 설치하지 않는다.

```bash
tar -xzf vagt_orca_yolo26s_fp16_inputs.tar.gz
uv sync --extra deployment --extra dev
.venv/bin/python scripts/check_gpu.py
```

`nvidia-smi`가 없는 Jetson도 있으므로 GPU 확인은 위 Python 검사와 `tegrastats`를 사용한다.
어떤 단계에서도 CPU fallback을 허용하지 않는다.

NVMe를 데이터 디스크로 사용하는 구성에서는 프로젝트와 validation 데이터를 SSD 경로에
배치한다. 현재 검증 경로는 `/mnt/data/vagt-orca-ai`다. benchmark 전에는 25W mode를 확인하고
클럭 변동에 따른 결과 왜곡을 막기 위해 다음을 실행한다.

```bash
nvpmodel -q
sudo jetson_clocks
```

## 2. 자동 target 실행

검증 dataset까지 복사했다면 아래 한 명령이 engine 생성, 정확도 평가, 전체 benchmark와
완료 판정을 순서대로 수행한다. 시작 시 Jetson 여부와 CUDA를 강제 검사한다.

```bash
.venv/bin/python deployment/run_jetson_phase6.py
```

validation dataset을 아직 복사하지 않은 경우 `--skip-validation`으로 engine 생성과 benchmark만
진단할 수 있지만 Phase 6 완료 판정은 실패한다. 계획서 필수 조건인 전체 15-class validation을
수행해야만 최종 승인된다.

## 3. 개별 실행

```bash
.venv/bin/python deployment/export_tensorrt.py
```

성공하면 `artifacts/deployment/tensorrt/yolo26s_best_fp16.engine`과
`reports/yolo26s_phase6_tensorrt.json`이 생성되며 `accepted_target_build`가 `true`,
`precision`이 `fp16`이어야 한다.

## 4. 정확도와 전체 파이프라인 benchmark

```bash
.venv/bin/python training/validate.py \
  --model artifacts/deployment/tensorrt/yolo26s_best_fp16.engine \
  --data data/processed/egoobjects_15/data.yaml --split val --batch 4 \
  --project artifacts/deployment/validation/jetson

.venv/bin/python deployment/benchmark.py \
  --model artifacts/deployment/tensorrt/yolo26s_best_fp16.engine \
  --report reports/yolo26s_phase6_benchmark.json

.venv/bin/python deployment/summarize_phase6.py
```

정확도 검증에는 원본 프로젝트의 `data/processed/egoobjects_15`도 Jetson에 별도로 복사해야
한다. benchmark에는 bundle에 포함하지 않은 validation/benchmark 이미지를 별도로 복사하거나
`--images`로 Jetson에 존재하는 대표 이미지 디렉터리를 지정한다. FP16은 calibration dataset이나
calibrator를 사용하지 않는다.

최종 완료 조건은 Jetson report의 engine 승인, end-to-end latency/FPS/memory 기록,
`reports/yolo26s_phase6_status.json`의 `accepted: true`이다. 계획 문서에는 latency 합격 수치가
없으므로 수치는 변조 없이 기록하며, 운영 후보 기준은 p95 50 ms 이하로 별도 판단한다.

2026-08-23 기준 참조 결과는 val mAP50-95 `0.22228`, E2E p95 `17.46 ms`, `62.97 FPS`,
UART 포함 2,000회 p95 `19.87 ms`, GPU 최고 `55.625°C`다. `jetson_clocks` 미적용 결과는
약 `28.17 FPS`였으므로 두 상태의 수치를 섞지 않는다.
