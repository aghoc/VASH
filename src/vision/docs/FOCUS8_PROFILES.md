# 요청 8종 정확도/속도 프로필 (실험 이력)

**현재 기본은 캔·동전 개선 모델로 롤백했다.** 아래 8종 프로필은 ELP 펜 오탐으로 실사용 불합격이며 기본 실행에서 제외했다. 현재 서비스는 `configs/runtime.yaml`과 `yolo26s_can_coin_retention_fp16.engine`을 사용한다. [복구 결과](../reports/CAN_COIN_ROLLBACK_20261015.md).

2026-10-15 실험 대상은 병, 컵, 휴대폰, 리모컨, 숟가락, 포크, 칫솔, 펜이다.
실측 결과와 한계는 `reports/FOCUS8_MAX_RESULT_20261015.md`에 기록한다.

## 이전 정확도 우선 프로필

Jetson `/mnt/data/vagt-orca-ai`의 `configs/runtime_focus8_accuracy.yaml`을 사용한다.
추가 학습 YOLO26s의 원본+중앙 확대 예측으로 펜과 기존 비대상 클래스를 처리하고,
COCO 사전학습 YOLO26m 전문 모델이 나머지 요청 7종을 담당한다.
각 모델은 static640/batch1/native TensorRT FP16 엔진이다. 학습·추론에 CPU 대체를 사용하지 않는다.

이전에 `50-focus8-max-20261015.conf`가 이 프로필을 선택했다. 현재 40/50 drop-in은 롤백 백업 후 제거했다.
현재 서비스는 정지 및 자동 시작 해제 상태다. 카메라 연결 후 사용자가 시작을 요청했을 때 다음을 실행하면 **캔·동전 모델**이 실행된다.

```bash
cd /mnt/data/vagt-orca-ai
systemctl --user start vagt-camera.service
```

아래는 이전 8종 실험을 재현하는 명령이며 현재 기본 실행 명령이 아니다. 직접 실행 전 서비스를 정지한다.

```bash
systemctl --user stop vagt-camera.service
.venv/bin/python tools/camera_live_view.py --source elp \
  --runtime configs/runtime_focus8_accuracy.yaml \
  --model artifacts/deployment/tensorrt/yolo26s_focus8_refined_v8_fp16.engine
```

## 고속 단일 모델

같은 추가 학습 YOLO26s를 전체 화면 한 번만 추론한다. 전문 모델과 중앙 확대의 정확도 이득은 없다.
서비스를 정지한 상태에서 다음을 실행한다. 종료 후 기본 서비스를 시작하면 현재 캔·동전 모델로 돌아간다.

```bash
.venv/bin/python tools/camera_live_view.py --source elp \
  --runtime configs/runtime_focus8_fast.yaml \
  --model artifacts/deployment/tensorrt/yolo26s_focus8_refined_v8_fp16.engine
```

공통 개발용 `configs/runtime.yaml`은 현재 캔·동전 `.pt` 단일 모델로 복구했다.
Jetson의 기본 서비스는 해당 native engine을 명시한다. TensorRT 엔진은 대상 GPU 전용이며
데스크톱에서 빌드한 엔진을 Jetson에 복사하지 않는다.

## 이전 v5로의 복구 이력 (현재 캔·동전 롤백과 다름)

Jetson에서 카메라 서비스를 정지한 다음 최신 drop-in을 제거하고 원래 설정을 복구한다.
백업 파일의 원래 경로는 `runs/focus8_max_20261015/preapply_native/manifest.json`에 있다.
다음 코드는 해당 실행에서 변경한 세 파일만 byte 단위로 복구한다.

```bash
systemctl --user stop vagt-camera.service
.venv/bin/python - <<'PY'
import json
from pathlib import Path
backup = Path('runs/focus8_max_20261015/preapply_native')
for target, saved in json.loads((backup/'manifest.json').read_text()).items():
    Path(target).write_bytes((backup/saved).read_bytes())
dropin = Path.home()/'.config/systemd/user/vagt-camera.service.d/50-focus8-max-20261015.conf'
dropin.unlink(missing_ok=True)
PY
systemctl --user daemon-reload
```

기존 `yolo26s_focus8_v5_fp16.engine`은 보존했다. `40-focus8-20261015.conf`는 현재 `runs/can_coin_rollback_20261015/jetson_before_rollback/`에 백업되어 있다.
카메라 서비스 시작은 카메라를 연결한 뒤 별도로 한다. 프레임이나 영상을 자동 저장하는 기능은 추가하지 않았다.
