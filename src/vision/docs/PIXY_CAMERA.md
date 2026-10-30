# Pixy2.1을 영상 소스로 사용하는 절차

Pixy2.1은 UVC 카메라가 아니라 온보드에서 컬러 추적을 수행하는 스마트 센서다. USB로
연결해도 `/dev/video*`가 생기지 않으므로 `cv2.VideoCapture`로 열 수 없다. 전체 프레임을
얻는 경로는 libpixyusb2의 `getRawFrame()` 하나뿐이며, 이 문서는 그 경로를 파이프라인에
연결하는 절차를 정리한다.

## 확정된 제약

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 프레임 해상도 | 316x208 고정 | `PIXY2_RAW_FRAME_WIDTH/HEIGHT` |
| 픽셀 포맷 | BGGR Bayer, 1 byte/px | `get_raw_frame.cpp` |
| 전체 프레임 인터페이스 | USB 전용 (SPI/I2C/UART 불가) | Pixy2 video API 문서 |
| 연속 grab API | libpixyusb2에 **미노출** | Charmed Labs: "We haven't exposed this type of streaming via libpixyusb2" |
| PixyMon 스트리밍 모드 | 약 20 fps (네이티브 60 fps 대비) | 동일 forum 답변 |
| Python SWIG 래퍼 | `getRawFrame` 미노출, `video_get_RGB` 픽셀 조회만 | `python_demos/pixy.i` |
| 최소 펌웨어 | 3.0.11 이상 | 3.0.10의 demosaic 버그로 segfault |

연속 스트리밍 API가 없다는 것이 핵심 제약이다. 따라서 `stop()` 후 `getRawFrame()`을
반복 호출하는 방식으로 구현했고, 이 방식이 실제로 새 프레임을 주는지는 문서화되어 있지
않으므로 **측정으로 확인해야 한다**(아래 3단계).

### 2026-08-12 Jetson 실기 확인

Jetson Orin Nano와 Pixy2.1 firmware 3.0.20에서 persistent 방식이 실제 새 프레임을
반환함을 확인했다. 30 frame 측정 결과 실패 0, 31.237 FPS, mean novelty 17.6228이었다.
cycled 방식은 실패 0이지만 12.247 FPS에 그쳤다. 따라서 현재 배포 기본값은
`cycle=False`(persistent)다.

## 1. 펌웨어와 색 보정 준비

Pixy2.1을 데스크톱에 연결하고 PixyMon에서 펌웨어를 3.0.11 이상으로 올린다. 3.0.10은
`demosaic()`에서 segfault가 발생한다. M12 렌즈는 초점 조절식이므로 실제 파지 거리에
맞춰 초점을 맞춰 둔다.

## 2. Jetson에서 브리지 빌드

```bash
sudo apt-get install -y git libusb-1.0-0-dev g++ build-essential
git clone https://github.com/charmedlabs/pixy2 ~/pixy2
cd ~/pixy2/src/host/libpixyusb2/src
make clean
make CFLAGS="-g -D__LINUX__ -fPIC"
mkdir -p ~/pixy2/build/libpixyusb2
cp ./lib/libpixy2.a ~/pixy2/build/libpixyusb2/libpixy2.a
```

공식 static library 기본 빌드에는 `-fPIC`가 없으므로 Python용 shared bridge에 링크할 때는
위와 같이 PIC object로 다시 빌드한다.

USB 인식 확인. `/dev/video*`가 안 생기는 것이 정상이다.

```bash
lsusb | grep -i -E "charmed|b1ac"
```

sudo 없이 접근하려면 공식 udev 규칙을 설치한 뒤 USB를 다시 연결한다.

```bash
sudo cp ~/pixy2/src/host/linux/pixy.rules /etc/udev/rules.d/99-pixy.rules
sudo chmod 644 /etc/udev/rules.d/99-pixy.rules
sudo udevadm control --reload-rules
```

프로젝트 브리지를 빌드하고 경로를 export 한다.

```bash
PIXY2_ROOT=~/pixy2 scripts/build_pixy_bridge.sh
export VAGT_PIXY_BRIDGE=artifacts/deployment/pixy/libpixybridge.so
```

## 3. grab 모드 결정 (필수)

`deployment/pixy_bridge.cpp`는 두 모드를 제공한다.

- `persistent`: 열 때 `stop()` 1회, 이후 `getRawFrame()`만 반복. 빠르지만 정지된
  버퍼를 다시 읽으면 같은 프레임이 계속 나온다.
- `cycled`: 매 grab마다 `resume()` → settle → `stop()` → `getRawFrame()`. 느리지만
  센서가 프레임을 진행시킨다.

어느 쪽이 유효한지는 문서에 없으므로 직접 측정한다. **렌즈 앞에서 손을 흔들면서**
실행해야 한다. 정지된 장면에서는 두 모드 모두 novelty가 낮게 나와 판별이 되지 않는다.

```bash
.venv/bin/python scripts/pixy_benchmark.py --frames 120
```

`reports/pixy_capture.json`의 `mean_novelty`가 0에 가까운 모드는 stale 버퍼를 반환하는
것이므로 fps와 무관하게 사용할 수 없다. `recommended_mode`가 판정 결과다.

동시에 `artifacts/deployment/pixy/bayer_reference.png`가 저장된다. 색이 뒤집혀 보이면
Bayer 패턴이 다른 것이므로 환경변수로 교정한다.

```bash
VAGT_PIXY_BAYER=BayerGB2BGR .venv/bin/python scripts/pixy_benchmark.py --frames 10
```

## 중앙 시선 기반 의도 선택

실기 데모에서 remote가 잠깐 미검출되자 중앙에서 벗어난 bottle이 유일한 detection이라는
이유로 candidate/target이 되는 문제가 확인됐다. 사용자의 시선을 대체하는 중앙 십자선 의도는
다음 정책으로 처리한다.

- 화면 중앙점이 detection bbox 내부에 있을 때만 candidate 자격을 갖는다.
- 여러 bbox가 중앙점을 포함하면 bbox 중심이 화면 중심에 가장 가까운 detection을 고른다.
- locked target이 사라지면 8 frame 동안 다른 target 출력을 보류하고 재검출을 기다린다.
- 유예 중에는 stale bbox를 현재 target으로 planner에 전달하지 않는다.
- 중앙 challenger가 지속되면 유예 종료 후 5 frame 중 4 vote debounce로 target을 전환한다.

설정은 `configs/runtime.yaml`의 `target.missing_grace_frames`에 있다. 이 변경은 순간 미검출이
다른 물체에 대한 사용자 의도로 해석되는 것을 막지만 detector 자체의 phone 약검출을 해결하지는
않는다. phone은 Pixy 도메인 영상을 추가 수집해 fine-tuning해야 한다.

브라우저 실시간 확인 도구는 중앙 십자선, candidate, target, center distance를 함께 표시한다.
영상 노출을 피하기 위해 Jetson localhost에 바인딩하고 SSH port forwarding으로 접속한다.

```bash
export VAGT_PIXY_BRIDGE=$PWD/artifacts/deployment/pixy/libpixybridge.so
.venv/bin/python tools/pixy_live_view.py \
  --host 127.0.0.1 --port 8765 \
  --model artifacts/deployment/tensorrt/yolo26s_best_fp16.engine
```

## 4. 파이프라인 실행

`--source pixy`가 Pixy 경로를 선택한다. 기존 `--source 0`(UVC)와 영상 파일 경로는
그대로 동작한다.

```bash
.venv/bin/vagt-orca infer --source pixy \
  --model artifacts/deployment/tensorrt/yolo26s_best_fp16.engine --uart /dev/ttyTHS1
```

도메인 프레임 수집도 동일하다.

```bash
.venv/bin/python tools/record_ar_glass.py --source pixy --class-name bottle \
  --session bottle_bright_near_001 --lighting bright --distance near \
  --background plain --rotation front --occlusion clear \
  --multi-object single --motion static
```

## 5. 남는 정확도 과제

브리지가 붙어도 아래 두 가지는 별도로 처리해야 한다.

**해상도**: 316x208은 학습 해상도 640 대비 작다. Ultralytics가 letterbox로 업스케일
하므로 실행은 되지만, 이미 `card`/`coin`/`small_block`/`can`이 사실상 검출되지 않는
상태(`PROGRESS.md`)에서 작은 물체 성능은 더 나빠진다. Phase 8 도메인 수집을 Pixy로
직접 수행해 이 해상도에서 재학습하는 것이 정공법이다.

**개구량 캘리브레이션**: `configs/opening.yaml`은
`opening = alpha * bbox_width/image_width + beta`로 화면 폭에 정규화되어 있으므로
해상도 변경 자체는 영향이 없다. 그러나 Pixy2.1의 80도 화각(2.0은 60도)은 같은 거리의
같은 물체가 차지하는 폭 비율을 바꾸므로 alpha/beta는 반드시 재적합해야 한다. 광각
구면왜곡 때문에 화면 가장자리 물체의 오차가 특히 커진다.

```bash
# calibration/phase9/opening_observations.csv를 Pixy로 다시 수집한 뒤
.venv/bin/python tools/fit_opening_calibration.py
```

이 재적합 전에는 ESP32로 나가는 개구량 값을 신뢰하면 안 된다.
