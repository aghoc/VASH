# Jetson: 공식 원본 YOLO26m / TensorRT FP16

이 절차는 저장소 루트의 `src/vision/`에서 실행한다. 모델은 추가학습하지 않은
YOLO26m이며, 과거 YOLO26s/15종 Phase 6 완료 판정과 별개다.
카메라 자동 시작과 프레임 자동 저장은 기본적으로 사용하지 않는다.

## 1. 환경 준비

이전 배포 환경은 Orin Nano Super / JetPack L4T R36.4.7 / CUDA 12.6 /
NVIDIA PyTorch 2.8 / TensorRT 10.3이다. 새 장치에서는 해당 JetPack과 호환되는
CUDA PyTorch 및 TensorRT를 먼저 준비해야 한다. 일반 데스크톱용 PyTorch나
`desktop-trt` extra가 Jetson의 NVIDIA 패키지를 대체하게 하지 않는다.

JetPack용 CUDA PyTorch와 TensorRT를 사용할 수 있는 Python 가상환경을 `.venv`에
준비한 뒤 프로젝트와 추가 의존성을 설치한다. `pyproject.toml`의 Python 범위는
3.10 이상, 3.13 미만이다. 기존 검증 환경에 다음 명령으로 프로젝트를 설치할 수 있다.

```bash
cd /path/to/VASH/src/vision
uv pip install --python .venv/bin/python -e '.[dev,deployment]'
.venv/bin/python scripts/check_gpu.py
.venv/bin/python -c 'import tensorrt; print(tensorrt.__version__)'
```

`scripts/check_gpu.py`가 실패하면 CUDA PyTorch 설치를 먼저 수정한다.
CPU fallback은 지원하지 않는다. `uv.lock`은 개발 환경 기록이며 Jetson 전용
PyTorch/TensorRT 설치를 대체하지 않는다.

## 2. 원본 모델 복원과 장치 엔진 생성

```bash
.venv/bin/python scripts/download_models.py
.venv/bin/python deployment/export_tensorrt.py \
  --config configs/deployment.yaml \
  --model models/pretrained/yolo26m.pt \
  --output artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine \
  --report reports/yolo26m_original_tensorrt.json
```

복원 스크립트는 공식 release URL과 SHA-256을 확인하며 기본 모델은 m이다.
640 square, batch 1, 정적 입력, FP16, end-to-end/NMS-free 설정을 사용한다.
엔진은 실제 Jetson에서 생성한다. 데스크톱 RTX에서 만든 엔진을 복사하지 않는다.

## 3. ELP 카메라 수동 실행

`configs/camera_elp_imx291.yaml`의 `/dev/v4l/by-id/...`가 실제 카메라와 일치하는지
확인한다. 기본 협상 모드는 1280×720 YUYV / 요청 60 FPS / 60Hz 조명이다.
요청 카메라 FPS는 실제 추론·브라우저 FPS와 다르다. 사용자는 `video` 그룹 권한이 필요하다.

```bash
.venv/bin/python tools/camera_live_view.py \
  --source elp --runtime configs/runtime.yaml \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine
```

Jetson localhost `http://127.0.0.1:8765/`에서 화면을 제공한다. 다른 컴퓨터에서는
아래 접속 정보의 `jetson-user`와 `JETSON_HOST`를 실제 값으로 바꾼다.

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 jetson-user@JETSON_HOST
```

브라우저를 닫아도 카메라 프로세스는 계속 실행된다. 직접 실행은 터미널의 Ctrl+C로
종료한다. 서비스 실행은 아래 stop 명령으로 종료한다. 뷰어는 최신 JPEG만 RAM에서
갱신하며 디스크에 자동 저장하지 않는다. 이 뷰어 자체는 UART를 송신하지 않는다.

## 4. 사용자 systemd 서비스 설치 (선택)

서비스 파일의 기본 경로는 과거 배포 위치 `/mnt/data/vagt-orca-ai`이다.
현재 체크아웃 경로로 치환해서 설치한다. 아래 예시는 공백이 없는 프로젝트 경로를 전제로 한다.
로그인 자동 시작을 설정하지 않고 수동 start/stop을 사용한다.

```bash
mkdir -p "$HOME/.config/systemd/user"
sed "s|/mnt/data/vagt-orca-ai|$PWD|g" deployment/vagt-camera.service \
  > "$HOME/.config/systemd/user/vagt-camera.service"
systemctl --user daemon-reload
systemctl --user start vagt-camera.service
systemctl --user status vagt-camera.service
```

```bash
systemctl --user stop vagt-camera.service
systemctl --user show vagt-camera.service -p ActiveState -p SubState -p MainPID
```

## 5. UART 송신과 benchmark

카메라를 사용하는 뷰어/서비스를 먼저 종료한 뒤 한 프로세스에서만 카메라를 연다.

```bash
.venv/bin/python -m vagt_orca.cli infer --source elp \
  --runtime configs/runtime.yaml \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine \
  --uart /dev/ttyTHS1
```

UART는 115200 8N1, 16-byte packet이며 [규격](../docs/UART_PROTOCOL.md)에 정의되어 있다.
프로젝트에는 Jetson 제안 송신 코드가 포함되며 ESP32 모터 제어 펌웨어는 포함되지 않는다.
개구량 계수는 초기값으로, 실제 ELP/ORCA Hand 실측 보정 완료를 뜻하지 않는다.

benchmark를 수행하려면 별도 이미지 디렉터리를 준비한다. 이미지는 Git에 포함되지 않는다.

```bash
.venv/bin/python deployment/benchmark.py \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine \
  --images /path/to/benchmark/images \
  --runtime configs/runtime.yaml --serialize-uart \
  --report reports/yolo26m_original_benchmark.json
```

이 benchmark는 카메라 읽기·JPEG·브라우저 표시를 제외한다. `--serialize-uart`는 메모리
패킷 직렬화이며 물리 UART 송신이 아니다. 현재 절차에서 `run_jetson_phase6.py`,
`summarize_phase6.py`, `package_jetson.py`의 과거 s/15종 자동 완료 판정을 사용하지 않는다.
이전 절차는 [YOLO26s runbook](JETSON_RUNBOOK_LEGACY_YOLO26S.md)에 보존했다.
