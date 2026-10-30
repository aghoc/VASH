# ELP IMX291 카메라 / Jetson 접속

2026-10-16 **추가학습하지 않은 공식 YOLO26m을 Jetson native FP16으로 적용했다**. 원본 COCO80종 가중치를 유지하고 책상·의자·사람·고정 가전 등 배경37종을 제외한 물체43종을 표시한다. 중앙 타깃은 그립을 지원하는 물체 중에서만 선택한다. 최초 시연 후보 3종을 표시 제한으로 잘못 적용했다가 사용자 지적에 따라 해제했다. 현재 시연 후보는 병·휴대폰·포크이며 표시 제한이 아니다. 프로젝트 지원 8종은 이름/ID를 연결하고, 그 외 물체도 표시하되 그립 매핑이 없으면 새 그립을 제안하지 않는다. confidence 0.35, square640, end2end 단일 전체 화면 추론이다. [선정·검증·배포 결과](../reports/ORIGINAL_DEMO_DEPLOYMENT_20261016.md)를 참조한다.

**현재 카메라는 사용자 지시로 종료했다.** 서비스 inactive/dead, MainPID=0, `/dev/video*` 점유 없음, 로그인 자동 시작 disabled. 캔·펜 추가 모델은 사용하지 않으며 시험 override를 백업 후 제거했다. 다음 수동 실행은 공식 원본 YOLO26m과 `configs/runtime.yaml`을 사용한다. 최신 상태는 `runs/can_pen_specialist_20261016/camera_stopped_original_selected.json`이다. 실제 정확도 합격 판정은 미확정이며 프레임/영상 자동 저장은 없다.

이전 v8 및 캔·동전 복구 모델은 ELP 오탐으로 실사용 불합격이었다. 해당 가중치와 설정 백업은 보존한다. [복구 이력](../reports/CAN_COIN_ROLLBACK_20261015.md), [오탐 점검](../reports/FALSE_POSITIVE_AUDIT_20261015.md).

2026-10-14에 Pixy2.1에서 ELP IMX291 USB UVC 카메라로 변경했다.
Pixy 기록은 `PIXY_CAMERA.md`에 보존한다. 새 입력은 `--source elp`로 선택한다.

## 접속과 프로젝트

- Jetson: `jetson-user@JETSON_HOST` (`jetson-host`)
- 프로젝트: `/mnt/data/vagt-orca-ai` (NVMe)
- JetPack/L4T: R36.4.7, CUDA 사용 가능, 전력 모드 25W
- 모델: `yolo26m_coco_original_fp16.engine` (공식 원본 m, Jetson에서 생성)
- 프로필: `configs/runtime.yaml`, 배경37종 제외/물체43종 표시, 그립 지원 물체만 타깃 선택. 40/50 focus8 및 60 can-pen 시험 drop-in은 백업 후 제거했다.
- USB 식별자: `15aa:1555`, 장치 이름 `3.0 USB Camera`
- 영상 노드: `/dev/video0`; `/dev/video1`은 두 번째 인터페이스이므로 사용하지 않는다.
- 고정 경로: `/dev/v4l/by-id/usb-3.0_USB_Camera_3.0_USB_Camera-video-index0`
- `permillion`은 `video` 그룹에 포함되어 있어 카메라 접근에 sudo가 필요 없다.

```bash
ssh -F /dev/null jetson-user@JETSON_HOST
cd /mnt/data/vagt-orca-ai
```

현재 데스크톱에서는 기존 SSH 인증으로 접속된다. `-F /dev/null`은 로컬 시스템 SSH
설정 파일의 권한 오류를 피한다. 호스트 키 검사를 끄지 않는다.

SSD 프로젝트의 `.venv`는 `/home/jetson-user/vagt-orca-ai/.venv`를 공유한다.
과거 editable import 경로가 이전 소스를 가리키던 문제를 수정해 현재
`vagt_orca.config.PROJECT_ROOT`는 `/mnt/data/vagt-orca-ai`다. 서비스도 이 경로를 명시한다.
변경 전 코드와 editable 경로는 Jetson `runs/elp_setup_20261014/`에 백업했다.

이전 이력: 2026-10-14 공개 캔·동전 GPU 추가 학습 모델로 전환했다. 당시 캔 confidence 0.75,
동전 0.25, 다른 클래스 0.35를 사용했다. 원래 모델·엔진과 배포 전 설정을 보존했다.
성능 및 기존 모델 복구 명령은 [학습 결과](../reports/CAN_COIN_OPEN_DATA_RESULT.md)를 참조한다.

2026-10-15 이전 v5 적용 단계에서는 요청 8종 COCO·Open Images 추가학습 모델을 Jetson native FP16으로 변환해 적용했다.
동일 Jetson validation·속도 비교 기준을 통과했으며, 캔·동전 confidence와 640 입력은 유지했다.
당시 사용자 unit의 `40-focus8-20261015.conf` drop-in이 v5 engine 경로를 지정했고 이후 50번 drop-in이 v8을 선택했다. 현재는 두 drop-in 모두 롤백 백업으로 보존하고 서비스에서 제거했다.
물체별 변화·성능 한계·복구 설정은 [Jetson 적용 기록](../reports/FOCUS8_JETSON_DEPLOYMENT_20261015.md)을 참조한다.

## 영상 모드

카메라가 열거한 모드는 MJPG/YUYV 각각 1920×1080, 1280×720, 640×480 @ 60 FPS다.
드라이버가 반환하는 FPS와 실제 프레임률은 구분한다.
설정 파일은 `configs/camera_elp_imx291.yaml`이다. 소스는 V4L2를 사용하고 버퍼는 1개다.
해상도·픽셀 형식·FPS 협상 결과가 설정과 다르면 실행을 중단해 조용한 모드 변경을 방지한다.

기본값은 **1280×720 YUYV / 요청 60 FPS / 조명 60Hz / 자동 노출**다.
카메라 전환 당시 기존 모델의 입력+FP16 추론은 120프레임 실패 0, 평균 24.40 FPS, p95 45.43ms였다.
동일 조건 MJPG는 20.59 FPS, p95 58.14ms여서 YUYV를 채택했다.
이 수치는 JPEG 인코딩, 브라우저 전송, 센서 노출 시간을 포함하지 않는다.
1080p의 경우 별도 프로필에서 해상도를 변경해 협상하고 다시 성능을 확인한다.
캡처 모드, 조명·노출 비교와 추론 결과 JSON은 `runs/elp_setup_20261014/`에 보관한다.

## 실시간 화면

사용자 서비스 정의는 `deployment/vagt-camera.service`다.

```bash
mkdir -p ~/.config/systemd/user
cp deployment/vagt-camera.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start vagt-camera.service
systemctl --user status vagt-camera.service
journalctl --user -u vagt-camera.service -n 30 --no-pager
```

현재 로그인 자동 시작은 해제되어 있다. 위 start는 사용자가 실시간 시험을 요청했을 때만 실행한다. 시험은 사용자가 완료/중지를 알릴 때 종료한다. 실행 중 오류 종료 시 재시작 설정은 유지된다.
영상은 Jetson localhost `127.0.0.1:8765`에서 제공한다. 데스크톱에서 SSH 터널을 열어 본다.

```bash
ssh -F /dev/null -N -L 127.0.0.1:8765:127.0.0.1:8765 jetson-user@JETSON_HOST
```

브라우저: `http://127.0.0.1:8765/`.
상태: `/status.json`, 현재 프레임: `/snapshot.jpg`, 영상: `/stream.mjpg`.
상태에는 입력 모드, 처리 FPS, 프레임 수, 캡처 실패 수, 최근 추론 p95가 포함된다.

직접 실행하려면 카메라를 사용하는 서비스를 먼저 정지한다.

```bash
systemctl --user stop vagt-camera.service
.venv/bin/python tools/camera_live_view.py --source elp \
  --runtime configs/runtime.yaml \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine
```

뷰어는 물체 검출·중앙 타겟·그립 제안을 표시한다. 실제 UART 송신은 별도 `infer --uart` 실행이다.

## 추론 / 수집 / 점검

카메라는 한 프로세스에서 사용한다. 아래 작업 전 서비스를 정지하고, 작업 후 다시 시작한다.

```bash
.venv/bin/python -m vagt_orca.cli infer --source elp \
  --runtime configs/runtime.yaml \
  --model artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine

.venv/bin/python tools/record_ar_glass.py --source elp --class-name bottle \
  --session bottle_bright_near_001 --lighting bright --distance near \
  --background plain --rotation front --occlusion clear \
  --multi-object single --motion static

.venv/bin/python tools/uvc_camera_check.py \
  --device /dev/v4l/by-id/usb-3.0_USB_Camera_3.0_USB_Camera-video-index0 \
  --width 1280 --height 720 --fps 60 --fourcc YUYV --frames 180
```

새 렌즈의 화각에 맞춘 개구량 보정과 ELP 실제 촬영 데이터의 정확도 검증은 별도 작업이다.
현재 `configs/opening.yaml`의 계수를 새 카메라 실측 보정 완료로 간주하지 않는다.

## 프레임 저장과 시작·종료

2026-10-16 실제 안경 시야 시험 후 사용자의 명시적 종료 요청으로 카메라를 정지했다.
원본 m만 유지하고 캔·펜 추가 모델은 사용하지 않는다. 서비스 inactive/dead·MainPID=0,
카메라 점유 없음과 자동 시작 disabled를 확인했다. 다음 시험은 사용자 요청 때 시작한다.
종료 시 `systemctl --user stop vagt-camera.service` 후 inactive/dead·MainPID=0 및
카메라 점유 해제를 확인하고 데스크톱의 해당 SSH 영상 터널도 종료한다.
브라우저 탭을 닫는 것만으로는 Jetson 카메라가 꺼지지 않는다. 로그인 자동 시작은 해제 상태다.

이전 이력: 2026-10-14 사용자 요청으로 카메라 서비스를 정지했다. 당시 로그인 자동 시작 설정은 유지했다.
2026-10-15 전원 인가 후 자동 시작된 것을 확인했고, 모델 비교 중 잠시 정지한 뒤 새 모델로 재시작했다.
적용 후 서비스 active, 캡처 실패 0회·자동 재시작 0회를 처리 통계로 확인했다. 카메라 화면은 조회하지 않았다.
Live viewer는 최신 JPEG 한 장을 메모리에서 교체하며 디스크에 프레임이나 영상을 자동 저장하지 않는다.
`/snapshot.jpg`도 메모리 응답이다. 처리 횟수는 저장 이미지 수가 아니다.
별도 `record_ar_glass.py` 실행 시에만 파일을 저장하며 기본 최대 300장 후 종료한다.
