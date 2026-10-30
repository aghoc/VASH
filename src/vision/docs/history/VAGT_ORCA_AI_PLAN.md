# VAGT-ORCA AI Vision & Grip Planning Development Plan

작성일: 2026-08-07
최종 수정일: 2026-08-17
대상: AI/비전 파트 신규 개발자 및 통합 개발자
상태: Initial implementation plan / 주요 설계 결정 확정 / YOLO26s 기준 개정

## 1. 문서 목적

이 문서는 VAGT-ORCA 로봇 의수 프로젝트의 AI 파트를 처음 접하는 개발자가 전체 구조를 이해하고 바로 개발을 시작할 수 있도록 구현 범위, 확정된 설계 결정, 데이터 전략, 모델 학습 방식, 그립 계산 방식, Jetson 배포 방식, 테스트 기준을 정리한 개발 계획서다.

AI 파트의 핵심 역할은 사용자가 AR 글래스 카메라로 바라보는 물체를 인식하고, 화면 중심을 기준으로 목표 물체를 선택한 뒤, 해당 물체에 적합한 그립 종류와 목표 개구량을 Jetson에서 계산해 ESP32에 제안하는 것이다.

AI는 의수를 직접 닫거나 그립 힘을 결정하지 않는다. 최종 실행 권한은 EMG/ESP32 제어 루프에 있다.

## 2. 전체 시스템에서 AI 파트의 위치

전체 시스템은 다음과 같이 동작한다.

```text
AR Glass Camera
      ↓
Jetson Orin Nano
      ↓
YOLO26s Object Detection
      ↓
Center Bias Target Selection
      ↓
Debouncing + Hysteresis
      ↓
Grip Planner
  ├─ grip_type
  ├─ opening
  └─ confidence
      ↓
UART
      ↓
ESP32-S3
      ↓
EMG Intent 확인
      ↓
ORCA Hand Servo Control
```

역할 분리는 다음과 같다.

| 모듈 | 역할 |
| --- | --- |
| AR Glass | 사용자의 시점에 가까운 RGB 영상 제공 |
| Jetson Orin Nano | 물체 인식, 목표 선정, 그립 종류 및 개구량 계산 |
| ESP32-S3 | EMG 기반 실시간 제어, Jetson 제안 수신, 안전 제어 |
| EMG | 실제 동작 실행 여부 및 사용자의 힘 의도 결정 |
| Dynamixel/Servo | 실제 손가락 구동 및 전류 기반 접촉/힘 제한 |

## 3. AI 파트에서 하지 않는 것

초기 개발 범위를 명확히 하기 위해 다음 기능은 AI 파트의 책임이 아니다.

- YOLO가 실제 그립 힘을 직접 결정하지 않는다.
- YOLO가 서보 각도를 직접 전송하지 않는다.
- 비전 결과만으로 손가락을 즉시 움직이지 않는다.
- RGB 영상만으로 정밀한 물체 실제 크기(mm)를 초기 단계부터 추정하지 않는다.
- 초기 버전에서 비전 기반 slip detection을 핵심 기능으로 구현하지 않는다.
- EMG 처리 및 서보의 low-level control loop는 ESP32 파트에서 담당한다.

AI의 출력은 어디까지나 `그립 제안`이다.

## 4. 확정된 모델 전략

### 4.1 Detection Model

신규 개발 기준 detection model은 `YOLO26s`을 사용한다.

YOLO26s를 선택한 이유는 다음과 같다.

- YOLO26은 기본적으로 end-to-end, NMS-free inference를 지원한다.
- DFL-free detection head로 export/deployment 경로가 단순하다.
- YOLO26s는 YOLO26n보다 COCO mAP50-95가 7.7 point 높아 유사 생활물체와 작은 물체를 구분할 때 더 유리한 기본 용량을 제공한다.
- Jetson Orin Nano Super 공식 TensorRT benchmark에서 YOLO26s FP16은 7.17 ms/image로, 30~60 FPS eye-view 입력을 처리하기에 충분한 추론 여유가 있다.
- 우리 파이프라인에서 detector 이후 Center Bias, debounce, grip mapping, opening 계산은 상대적으로 가벼우므로, Nano 모델의 몇 ms 절약보다 detector 정확도를 우선할 가치가 크다.

따라서 기본 detector는 YOLO26s로 고정한다. 다만 Jetson 실제 통합 시 메모리 부족, 발열, 장시간 throttling 문제가 확인되면 YOLO26n은 명시적인 fallback 후보로만 유지한다.

기본 개발 흐름은 다음과 같다.

```text
Pretrained YOLO26s
      ↓
FP32 Fine-tuning
      ↓
Validation
      ↓
TensorRT FP16 Export
      ↓
Jetson Orin Nano Benchmark
      ↓
End-to-End Validation
```

배포 precision은 `TensorRT FP16`으로 확정한다. 모델 학습과 fine-tuning은 일반적인 FP32 학습으로 진행하고, 배포 단계에서 best checkpoint를 FP16 TensorRT engine으로 변환한다.

현재 프로젝트의 배포 precision은 FP16으로 고정하며, 다른 precision 비교는 MVP 범위에서 수행하지 않는다.

### 4.2 YOLO26s 기준 공식 성능 참고

Ultralytics 공식 YOLO26 detection 표에서 640 입력 기준 YOLO26s는 COCO mAP50-95 48.6, 9.5M parameters, 20.7B FLOPs다. YOLO26n은 mAP50-95 40.9, 2.4M parameters, 5.4B FLOPs다. 따라서 26s는 약 4배의 연산량을 사용하지만 일반 detection 정확도는 7.7 mAP point 높다.

Jetson Orin Nano Super Developer Kit 공식 benchmark의 YOLO26s TensorRT 결과는 다음과 같다. benchmark는 Ultralytics 8.4.33 기준이며, inference time에는 pre/post-processing이 포함되지 않는다.

| Precision | mAP50-95(B) | Inference time | Engine size |
| --- | ---: | ---: | ---: |
| TensorRT FP32 | 0.5664 | 13.60 ms/image | 38.2 MB |
| TensorRT FP16 | 0.5649 | 7.17 ms/image | 21.3 MB |

공식 수치상 FP16은 FP32 대비 정확도 손실이 매우 작으면서 추론시간을 거의 절반 수준으로 줄인다. 현재 프로젝트는 30~60 FPS 실시간 처리를 목표로 하므로 detection 안정성과 배포 단순성을 우선해 `YOLO26s TensorRT FP16`을 최종 배포 기준으로 확정한다.

### 4.3 NMS-Free 사용 원칙

YOLO26의 default one-to-one head는 최대 300개의 end-to-end detection을 NMS 없이 출력할 수 있다. 초기 구현은 default end-to-end 경로를 사용한다.

```text
YOLO26s Detection
      ↓
Center Bias
```

과거 YOLOv8처럼 별도 NMS 후처리를 시스템 설계의 필수 단계로 가정하지 않는다.

다만 Jetson/TensorRT export 조합에서 특정 버전 호환성 문제가 발생하거나 one-to-many head가 프로젝트 데이터에서 명확하게 더 높은 정확도를 보이는 경우 `end2end=False` fallback 경로를 허용한다. 이 선택은 benchmark 결과를 기록한 뒤 결정한다.

### 4.4 배포 타깃

- Device: Jetson Orin Nano 계열
- Runtime target: TensorRT
- Model: YOLO26s
- Training precision: FP32 fine-tuning
- Deployment precision: TensorRT FP16
- Input: AR 글래스 RGB frame
- Primary output: object class, confidence, bounding box
- Default inference head: end-to-end / NMS-free

### 4.5 라이선스 체크포인트

Ultralytics YOLO26 코드와 모델은 공식 문서 기준 AGPL-3.0 및 Enterprise 라이선스 체계로 제공된다.

연구/캡스톤 단계에서는 현재 개발을 진행하되, 상용 제품에 YOLO26 기반 모델과 Ultralytics runtime을 그대로 포함하려는 시점에는 반드시 라이선스 검토를 별도 milestone으로 수행한다.

이 항목은 데이터셋 라이선스와 별개다.

## 5. 초기 Object Class 정의

초기 MVP에서는 모든 생활 물체를 인식하려 하지 않고 15개 클래스로 범위를 제한한다.

| ID | Class | Default Grip |
| ---: | --- | --- |
| 0 | bottle | POWER |
| 1 | can | POWER |
| 2 | cup | POWER |
| 3 | box | POWER |
| 4 | ball | POWER |
| 5 | phone | POWER |
| 6 | remote | POWER |
| 7 | pen | TRIPOD |
| 8 | spoon | TRIPOD |
| 9 | fork | TRIPOD |
| 10 | toothbrush | TRIPOD |
| 11 | card | LATERAL_PINCH |
| 12 | key | LATERAL_PINCH |
| 13 | coin | PINCH |
| 14 | small_block | PINCH |

초기에는 `class → default grip` 룩업 테이블로 처리한다.

향후 동일 클래스가 크기나 orientation에 따라 다른 그립을 필요로 하는 문제가 확인되면 다음 단계에서 확장한다.

```text
class
+ bbox size
+ bbox aspect ratio
+ 추후 depth/거리 정보
      ↓
grip_type
```

초기 MVP에서는 이 복잡도를 넣지 않는다.

## 6. Grip Taxonomy

소프트웨어에서 지원하는 기본 그립은 총 4종이다.

### POWER

손바닥과 여러 손가락으로 물체를 감싸는 그립이다.

대표 대상:

- bottle
- can
- cup
- box
- ball
- phone
- remote

### TRIPOD

엄지, 검지, 중지를 중심으로 사용하는 정밀 그립이다.

대표 대상:

- pen
- spoon
- fork
- toothbrush

### PINCH

엄지와 검지 끝을 이용해 작은 물체를 집는 그립이다.

대표 대상:

- coin
- small_block

### LATERAL_PINCH

엄지와 검지 측면 사이에서 얇은 물체를 집는 그립이다.

대표 대상:

- card
- key

ORCA Hand의 엄지는 IP, MCP, ABD, CMC를 포함한 독립적인 자유도를 가지고 있어 lateral pinch에 필요한 기구학적 자세를 만들 수 있는 구조로 판단한다. 실제 하드웨어 통합 시 카드와 열쇠를 이용한 접촉 위치 calibration을 수행한다.

## 7. Dataset 전략

이 시스템은 AR 글래스 기반 eye/head-view만 사용한다. 따라서 일반적인 외부 시점 또는 로봇 손 근처에서 촬영한 데이터만을 핵심 학습 데이터로 사용하지 않는다.

### 7.1 Object Detection Primary Dataset: EgoObjects

EgoObjects를 object detector의 메인 공개 데이터셋으로 사용한다.

선정 이유:

- egocentric/head-mounted 시점
- Ray-Ban Stories, Snap Spectacles 등 wearable device 기반 영상 포함
- 실내 생활 물체 중심
- bounding box 및 category label 제공
- YOLO detection용 데이터로 변환 가능
- 공개 접근 가능
- 공식 repository 기준 MIT License

공식 v1.0은 약 114K annotated frames, 368 categories를 포함한다.

주의사항:

EgoObjects의 전체 클래스를 그대로 사용하는 것이 아니라 프로젝트의 15개 클래스와 대응되는 category를 추출하고 필요한 경우 class remapping을 수행한다.

### 7.2 Final Domain Adaptation Dataset: 자체 AR Glass 데이터

최종 모델 성능에 가장 중요한 데이터다.

실제 사용할 AR 글래스로 15개 대상 물체를 직접 촬영하고 bbox annotation을 추가한다.

공개 데이터만으로 끝내지 않는 이유는 다음과 같다.

- 실제 AR 글래스의 FOV 차이
- 렌즈 distortion
- exposure 및 white balance
- motion blur
- 카메라 장착 높이와 방향
- 사용자가 물체를 바라볼 때 나타나는 center bias

이 데이터는 하드웨어가 준비되는 즉시 수집하며 최종 fine-tuning에 사용한다.

### 7.3 Grip Mapping Reference: Human Grasping Database for ADL

35개 grasp taxonomy와 head-mounted GoPro RGB 영상, 실제 ADL grasp annotation을 제공한다.

이 데이터는 YOLO 메인 detection 학습이 아니라 다음 목적에 사용한다.

```text
기존 human grasp taxonomy
      ↓
우리 프로젝트의 4개 grip으로 축약
      ↓
POWER / TRIPOD / PINCH / LATERAL_PINCH
```

특히 grasp type, start/end frame, thumb configuration 등의 정보를 참고해 현재 grip mapping의 타당성을 검증한다.

### 7.4 EPIC-KITCHENS

필수 데이터셋이 아니다.

필요한 경우 주방/생활물체의 eye-view detection 보조 데이터로 사용한다.

라이선스가 CC BY-NC 4.0이므로 연구 단계에서만 보조적으로 사용하고, 상용화 가능성이 있는 최종 production model의 핵심 학습 데이터로 의존하지 않는다.

### 7.5 HANDS Dataset

prosthetic grasp intent 연구 목적의 eye-view dataset이라 내용상 적합하지만, 현재 실제 다운로드 경로의 지속적인 접근성을 확실하게 보장하기 어렵다.

따라서 다음 원칙을 적용한다.

- 다운로드가 가능하면 grip mapping 추가 검증에 사용
- 다운로드가 불가능해도 프로젝트 일정에 영향이 없어야 함
- 필수 dependency로 사용하지 않음

### 7.6 Ego4D

초기 MVP에서는 사용하지 않는다.

향후 hand-object interaction이나 contact timing 같은 기능을 추가할 때 FHO subset을 검토한다. 접근 절차와 데이터 규모가 더 크므로 현재 핵심 파이프라인에는 포함하지 않는다.

## 8. Dataset 구축 파이프라인

```text
EgoObjects
      ↓
15개 target class 추출
      ↓
Class normalization / remapping
      ↓
YOLO format 변환
      ↓
Train / Validation / Test split
      ↓
YOLO26s Fine-tuning
      ↓
AR Glass 자체 데이터 추가
      ↓
Final Fine-tuning
```

데이터 split 시 같은 실제 물체 또는 같은 촬영 sequence의 거의 동일한 frame이 train과 test에 동시에 들어가지 않도록 주의한다. 그렇지 않으면 실제 일반화 성능보다 높은 수치가 나올 수 있다.

## 9. Target Selection

아이트래킹은 사용하지 않는다.

각 YOLO detection의 bounding box center와 화면 center 사이의 거리를 계산하고 가장 가까운 객체를 candidate target으로 선택한다.

```text
screen_center = (W / 2, H / 2)
box_center    = ((x1 + x2) / 2, (y1 + y2) / 2)

distance = sqrt(
    (box_center_x - screen_center_x)^2
  + (box_center_y - screen_center_y)^2
)
```

가장 작은 distance를 가진 객체가 target candidate가 된다.

단, 한 프레임 결과만으로 target을 즉시 변경하지 않는다.

## 10. Target Stabilization

### 10.1 Debouncing

최근 5 frame의 candidate target을 저장한다.

동일 target이 5 frame 중 최소 4 frame에서 선택됐을 때 최종 target으로 lock한다.

```text
window_size = 5
required_same_target = 4
```

### 10.2 Hysteresis

이미 target이 lock되어 있을 경우 새로운 물체가 단순히 몇 pixel 더 중앙에 가까워졌다는 이유만으로 target을 변경하지 않는다.

새 candidate가 기존 target보다 center distance 기준 최소 20% 이상 유리할 때만 변경 대상으로 인정한다.

```text
switch_margin = 20%
```

이 값들은 초기 default이며 실제 AR Glass 데이터 테스트를 통해 조정할 수 있다. 구조 자체는 변경하지 않는다.

## 11. Grip Planner

Grip Planner는 stabilized target detection을 입력받아 ESP32에 전달할 그립 정보를 만든다.

입력:

```text
class_id
confidence
bbox
frame_size
```

출력:

```text
grip_type
opening
confidence
```

### 11.1 Grip Type 계산

초기 버전에서는 deterministic lookup table을 사용한다.

예시:

```python
GRIP_MAP = {
    "bottle": "POWER",
    "can": "POWER",
    "cup": "POWER",
    "box": "POWER",
    "ball": "POWER",
    "phone": "POWER",
    "remote": "POWER",
    "pen": "TRIPOD",
    "spoon": "TRIPOD",
    "fork": "TRIPOD",
    "toothbrush": "TRIPOD",
    "card": "LATERAL_PINCH",
    "key": "LATERAL_PINCH",
    "coin": "PINCH",
    "small_block": "PINCH",
}
```

그립 종류를 별도의 neural network로 학습하는 것은 초기 범위에서 제외한다.

## 12. Opening 계산

초기 RGB-only 단계에서는 실제 mm 단위의 물체 폭을 정확하게 추정할 수 없으므로 opening을 0.0~1.0의 normalized value로 정의한다.

```text
0.0 = 최소 개구
1.0 = 최대 개구
```

초기 계산은 bounding box width를 기준으로 한다.

```text
bbox_ratio = bbox_width / image_width

opening = clamp(
    alpha[class] * bbox_ratio + beta[class],
    0.0,
    1.0
)
```

`alpha`, `beta`는 object class별 calibration parameter다.

하드웨어가 나오기 전에는 heuristic/default 값을 사용하고, ORCA Hand 통합 후 실제 물체를 이용해 값을 fitting한다.

향후 카메라 거리 또는 depth 정보가 확보되면 실제 물체 폭을 추정하는 방식으로 교체할 수 있다.

```text
estimated_width ≈ bbox_width_px × Z / fx
```

하지만 현재 MVP에서는 depth estimation을 필수 기능으로 사용하지 않는다.

## 13. Confidence Gating

낮은 confidence의 detection은 ESP32의 실제 제어에 영향을 주면 안 된다.

Jetson은 detection confidence를 그대로 또는 정규화해서 ESP32에 전달한다.

confidence threshold는 validation 결과를 기준으로 tuning 가능한 configuration parameter로 유지한다.

confidence가 threshold 이하인 경우:

```text
새 grip 제안 금지
→ 현재 Last Known Good State 유지
```

## 14. Jetson → ESP32 Interface

AI 파트에서 논리적으로 전달해야 하는 값은 다음과 같이 고정한다.

| Field | 설명 |
| --- | --- |
| grip_id | POWER / TRIPOD / PINCH / LATERAL_PINCH |
| confidence | detection/target confidence |
| opening | normalized 0.0~1.0 |
| sequence | 최신 packet 여부 확인 |
| checksum | packet corruption 검출 |

개념적 packet 예시는 다음과 같다.

```text
{
  grip_id: POWER,
  confidence: 0.91,
  opening: 0.72,
  sequence: 1842,
  checksum: ...
}
```

정확한 byte layout, endian, checksum algorithm, timeout 값은 ESP32 통합 단계에서 양쪽 팀이 함께 확정한다.

Jetson inference loop와 ESP32 control loop는 동기식으로 묶지 않는다.

ESP32는 최신 Jetson packet을 참고할 뿐 Jetson inference를 기다리지 않는다.

## 15. Contact 이후 제어 권한

물체 접촉 이후에는 비전이 주 제어기가 아니다.

동작 순서는 다음과 같다.

```text
1. Vision: target + grip + opening 제안
2. EMG: 사용자의 접근/그립 의도 확인
3. Servo: 손가락 접근
4. Present Current + position 변화로 접촉 감지
5. 접촉 후 vision 기반 opening 변경 제한
6. EMG + Current Limit 기반 힘 제어
7. EMG release 신호는 언제나 최우선
```

따라서 YOLO 성능이 일시적으로 흔들려도 이미 물체를 잡은 이후 손가락이 갑자기 다른 형태로 움직이지 않아야 한다.

## 16. 개발 단계

### Phase 1. 개발 환경 및 Baseline

목표:

- 최신 stable Ultralytics 환경 구성
- pretrained `yolo26s.pt` baseline 실행
- 로컬 GPU 또는 개발 머신에서 inference pipeline 구성
- 영상 파일 및 webcam 입력 지원
- bbox, class, confidence 출력 확인
- default end-to-end/NMS-free output 구조 확인

완료 조건:

- 하나의 Python entry point에서 영상을 입력해 detection 결과를 실시간 출력 가능
- 결과 visualization 가능
- 사용한 Ultralytics/CUDA/PyTorch 버전을 기록

### Phase 2. Dataset Adapter

목표:

- EgoObjects 다운로드 및 구조 분석
- 프로젝트 15 classes 추출
- class remapping
- YOLO format converter 구현
- train/val/test dataset 생성

완료 조건:

- `data.yaml` 생성 가능
- dataset integrity check 통과
- 잘못된 bbox 및 missing image 검출 가능
- 각 class의 image/instance 수 통계 생성 가능

### Phase 3. YOLO26s Fine-tuning

목표:

- YOLO26s FP32 fine-tuning
- augmentation 설정
- validation metric 기록
- confusion matrix 분석

중점 확인 대상:

- bottle vs can
- spoon vs fork
- card/key/coin 같은 small object
- background clutter가 많은 eye-view 장면

완료 조건:

- 15 classes 전체 inference 가능
- validation 결과 및 model checkpoint 저장
- 실패 케이스 샘플 정리
- 재현 가능한 train config 저장

### Phase 4. Target Selector

목표:

- center bias 구현
- 5-frame / 4-frame debounce 구현
- 20% hysteresis 구현

완료 조건:

- 여러 물체가 한 화면에 있을 때 중앙 object가 안정적으로 lock됨
- 카메라가 약간 흔들려도 target이 frame마다 변경되지 않음

### Phase 5. Grip Planner

목표:

- 15 class → 4 grip mapping 구현
- bbox 기반 normalized opening 구현
- confidence gating 구현

완료 조건:

각 target에 대해 최소 다음 구조의 결과가 생성되어야 한다.

```json
{
  "class": "bottle",
  "grip": "POWER",
  "opening": 0.72,
  "confidence": 0.91
}
```

### Phase 6. Jetson FP16 Deployment & Benchmark

목표:

- YOLO26s TensorRT FP16 export
- Jetson Orin Nano에서 FP16 engine 실행
- 자체 validation set 기준 accuracy 확인
- latency / FPS / memory usage 측정
- end-to-end vision pipeline 성능 확인

배포 대상:

```text
best.pt
  ↓
TensorRT FP16 engine
```

완료 조건:

- Jetson에서 FP16 inference 동작
- end-to-end latency 측정 가능
- FPS 및 memory usage 기록
- 15-class validation 결과 기록
- FP16 engine 및 deployment environment 버전 고정

검증 기준:

- FP16은 현재 프로젝트의 고정 deployment precision이다.
- 실제 15-class eye-view validation에서 small-object recall, 유사 class confusion, end-to-end latency를 기록한다.
- JetPack/TensorRT/Ultralytics 버전에 따라 export 동작이 달라질 수 있으므로 deployment environment version을 반드시 고정한다.

### Phase 7. UART Integration

목표:

- Jetson output packet serialization
- sequence/checksum 구현
- ESP32 mock receiver와 통신 테스트

실제 손 하드웨어가 없어도 ESP32 또는 PC mock을 이용해 개발 가능하다.

완료 조건:

- target 변경에 따라 packet이 올바르게 변경됨
- packet loss 또는 stale sequence 상황을 재현 가능

### Phase 8. AR Glass Domain Adaptation

하드웨어가 준비되면 진행한다.

목표:

- 실제 AR Glass 영상 수집
- 15개 class 직접 촬영
- bbox annotation
- 공개 dataset 모델에 추가 fine-tuning
- TensorRT FP16 engine 재생성 및 validation 재수행

촬영 조건에는 다음 variation을 포함한다.

- 밝은 환경 / 어두운 환경
- 가까운 거리 / 먼 거리
- 여러 background
- 물체 회전
- partially occluded object
- 여러 object가 동시에 있는 장면
- walking/head motion blur

### Phase 9. ORCA Hand Calibration

목표:

- class별 `opening → 실제 손 개구` calibration
- POWER/TRIPOD/PINCH/LATERAL_PINCH pose 확인
- lateral pinch 접촉 위치 확인

이 단계에서 opening의 alpha/beta 값을 실제 하드웨어 기준으로 보정한다.

## 17. 권장 Repository 구조

```text
vagt-orca-ai/
├── README.md
├── configs/
│   ├── model.yaml
│   ├── classes.yaml
│   ├── grip_map.yaml
│   └── runtime.yaml
│
├── datasets/
│   ├── converters/
│   │   └── egoobjects_to_yolo.py
│   ├── remap.py
│   └── validate_dataset.py
│
├── training/
│   ├── train.py
│   ├── validate.py
│   └── export.py
│
├── vision/
│   ├── detector.py
│   ├── target_selector.py
│   ├── target_stabilizer.py
│   └── pipeline.py
│
├── grip/
│   ├── grip_types.py
│   ├── grip_mapper.py
│   └── opening_estimator.py
│
├── communication/
│   ├── packet.py
│   ├── uart_sender.py
│   └── mock_receiver.py
│
├── deployment/
│   ├── tensorrt/
│   │   ├── export_fp16.py
│   │   └── export_int8.py
│   └── benchmark.py
│
├── tests/
│   ├── test_target_selector.py
│   ├── test_stabilizer.py
│   ├── test_grip_mapper.py
│   ├── test_opening.py
│   └── test_packet.py
│
└── tools/
    ├── visualize_detection.py
    ├── record_camera.py
    └── annotate_samples.py
```

하드코딩을 줄이기 위해 grip mapping, threshold, debounce window, hysteresis margin, opening calibration parameter, detector head mode, deployment precision은 config 파일로 분리한다.

## 18. 테스트 지표

일반적인 YOLO mAP만으로 프로젝트 성공 여부를 판단하지 않는다.

최소 다음 지표를 기록한다.

| Metric | 의미 |
| --- | --- |
| Detection mAP / Precision / Recall | object detector 자체 성능 |
| Per-class AP / Confusion | 유사 물체 및 small object 취약점 확인 |
| Target Selection Accuracy | 사용자가 중앙에 둔 물체가 실제 target으로 선택되는 비율 |
| Target Switch Stability | 불필요한 target switching 빈도 |
| Grip Mapping Accuracy | target에 적절한 4종 grip 선택 여부 |
| Opening Error | 실제 요구 개구량과 제안값의 차이 |
| Inference Latency | YOLO inference 시간 |
| End-to-End Vision Latency | frame input부터 UART packet 생성까지 |
| FPS | Jetson 실시간 처리 속도 |
| Peak Memory | Jetson runtime 메모리 사용량 |

초기 engineering acceptance target은 다음과 같이 잡는다.

- Target Selection Accuracy: 95% 이상
- Grip Mapping Accuracy: 정의된 15 class에서 deterministic 100%
- End-to-End Vision Latency: Jetson에서 p95 50 ms 이하 목표
- hardware calibration 후 normalized Opening MAE: 0.10 이하 목표

YOLO detection metric 자체의 합격 threshold는 실제 데이터 분포를 확인한 뒤 baseline 결과를 기준으로 설정한다.

FP16 배포 검증에서는 단일 FPS 수치가 아니라 다음을 함께 본다.

```text
Detection accuracy
+ small-object recall
+ end-to-end latency
+ memory usage
+ export/runtime stability
```

## 19. 실패 상황 처리

### Detection 없음

```text
No target
→ 새로운 grip command 생성하지 않음
→ ESP32는 기존 상태 또는 EMG-only control 유지
```

### Confidence 낮음

```text
Do not update
→ Last Known Good State
```

### Target 흔들림

```text
Debounce + hysteresis
→ lock 유지
```

### Jetson packet 지연/손실

```text
ESP32가 stale sequence/timeout 감지
→ Vision assistance 비활성화
→ EMG-only fallback
```

### 이미 물체 접촉

```text
Vision correction 중단/제한
→ Current/EMG control 우선
```

### TensorRT export/runtime 실패

```text
JetPack / TensorRT / Ultralytics version 기록
→ FP16 export/runtime 경로를 우선 복구
→ 필요 시 end2end=False export 비교
→ 문제 원인과 재현 조건 문서화
```

## 20. 향후 확장 기능

MVP 안정화 이후 아래 기능을 검토한다.

### Object-aware Grip Selection

단순 class mapping 대신 bbox aspect ratio, orientation, depth 등을 이용해 같은 물체에서도 다른 grip을 선택한다.

### Monocular Depth 또는 Depth Sensor

RGB bbox의 상대 크기만 사용하는 한계를 해결하고 실제 opening을 mm 단위로 계산한다.

### Vision-based Slip Monitoring

물체 표면 특징점의 예상치 못한 drift를 감지해 servo current 기반 slip 판단을 보조한다.

비전 단독 slip control로 사용하지 않는다.

### Grip Intent Model

충분한 자체 eye-view + 사용자 interaction 데이터가 쌓인 이후 `object + geometry + context → grip` 모델을 학습할 수 있다.

초기에는 룰 기반 방식이 더 해석 가능하고 안전하다.

## 21. 현재 고정된 설계 결정

새 개발자가 임의로 변경하기 전에 팀과 논의해야 하는 항목이다.

| 항목 | 확정값 |
| --- | --- |
| Camera perspective | AR Glass eye/head-view |
| Detector | YOLO26s |
| Training precision | FP32 fine-tuning |
| Deployment runtime | TensorRT |
| Deployment precision | TensorRT FP16 |
| Default precision candidate | FP16 확정 |
| Default inference head | YOLO26 end-to-end / NMS-free |
| Compute | Jetson Orin Nano 계열 |
| Primary public detection data | EgoObjects |
| Final domain data | 자체 AR Glass 촬영 |
| Grip reference | Human Grasping Database |
| Grip count | 4 |
| Grip types | POWER, TRIPOD, PINCH, LATERAL_PINCH |
| Initial object classes | 15 classes |
| Target selection | Center Bias |
| Debounce | 5 frame 중 4 frame |
| Hysteresis | 20% switching margin |
| Opening representation | normalized 0.0~1.0 |
| AI authority | suggestion only |
| Communication | Jetson → ESP32 UART |
| Force control | EMG + servo current side |
| Ultralytics commercial use | 별도 라이선스 검토 필요 |

## 22. 하드웨어가 나오기 전에 완료할 수 있는 범위

AI 팀은 ORCA Hand와 AR 글래스 실물이 없어도 아래까지 완료할 수 있다.

```text
[가능]
YOLO26s baseline 실행
→ EgoObjects 확보
→ dataset converter
→ YOLO26s fine-tuning
→ validation
→ center bias
→ debounce/hysteresis
→ 4종 grip planner
→ bbox 기반 opening
→ packet serializer
→ PC/ESP32 mock 통신
→ TensorRT FP16 export
→ Jetson FP16 benchmark
→ end-to-end deployment validation
```

하드웨어가 필요한 부분은 다음뿐이다.

```text
[하드웨어 이후]
실제 AR Glass domain data 수집
AR Glass 데이터 추가 fine-tuning
FP16 최종 재검증
ORCA Hand grip pose 검증
class별 opening calibration
contact/current threshold calibration
전체 end-to-end 사용자 테스트
```

따라서 하드웨어 개발과 AI 개발을 병렬로 진행할 수 있다.

## 23. 신규 개발자의 첫 작업 순서

프로젝트에 처음 합류했다면 다음 순서로 시작한다.

1. 전체 아키텍처와 AI 권한 범위를 이해한다.
2. 최신 stable Ultralytics에서 `yolo26s.pt` pretrained baseline inference를 실행한다.
3. 실행 환경의 Python/PyTorch/CUDA/Ultralytics 버전을 고정한다.
4. EgoObjects 구조와 annotation schema를 확인한다.
5. 15개 클래스 remapping/conversion script를 구현한다.
6. YOLO26s baseline fine-tuning을 수행한다.
7. validation과 confusion matrix로 detector 실패 유형을 분석한다.
8. `target_selector.py`를 구현한다.
9. debounce/hysteresis unit test를 작성한다.
10. 4종 grip mapping을 구현한다.
11. normalized opening estimator를 구현한다.
12. 전체 파이프라인을 하나의 영상에서 실행한다.
13. Jetson에서 TensorRT FP16 export 및 benchmark를 수행한다.
14. 실제 15-class validation set에서 FP16 accuracy와 end-to-end latency를 검증한다.
15. UART mock integration까지 완료한다.

이 시점까지 완료하면 하드웨어가 준비되는 즉시 AR Glass domain adaptation과 ORCA Hand calibration 단계로 넘어갈 수 있다.

## 24. 첫 번째 개발 Milestone

하드웨어가 없는 현재 시점의 첫 번째 milestone은 다음으로 정의한다.

```text
EgoObjects 기반 15-class YOLO26s FP32 baseline 모델 완성
+ validation report
+ 실패 케이스 분석
```

그 다음 milestone은 다음이다.

```text
YOLO26s
+ Center Bias
+ Debounce/Hysteresis
+ 4종 Grip Planner
+ normalized opening
= PC 영상에서 완전한 Vision Proposal Pipeline 동작
```

Jetson 확보 후 세 번째 milestone은 다음이다.

```text
TensorRT FP16 deployment benchmark
→ accuracy / latency / memory / stability 검증
→ deployment environment 고정
```

## 25. 참고 자료

### 프로젝트 내부 기준 문서

- `실시간 반영형 로봇 의수 — 기술 아키텍처 정리` (2026-07-14)

### YOLO26 / Jetson

- YOLO26: https://docs.ultralytics.com/models/yolo26/
- YOLO26 vs YOLOv8: https://docs.ultralytics.com/compare/yolo26-vs-yolov8/
- NVIDIA Jetson + YOLO26: https://docs.ultralytics.com/guides/nvidia-jetson/
- TensorRT Export: https://docs.ultralytics.com/integrations/tensorrt/
- Ultralytics repository: https://github.com/ultralytics/ultralytics

### 데이터 및 하드웨어

- EgoObjects: https://github.com/facebookresearch/EgoObjects
- EgoObjects Dataset: https://ai.meta.com/datasets/egoobjects-dataset/
- Human Grasping Database for Activities of Daily Living: https://www.nature.com/articles/sdata2018101
- EPIC-KITCHENS: https://epic-kitchens.github.io/
- ORCA Hand: https://www.orcahand.com/

## 26. 변경 이력

### 2026-08-17

- Detector를 YOLOv8n에서 YOLO26 계열로 전환한 뒤, 최종 기본 detector를 YOLO26s로 확정
- YOLO26n 대비 YOLO26s의 정확도/연산량 trade-off 및 선택 근거 추가
- TensorRT 배포 precision을 FP16으로 최종 확정하고 INT8 비교 작업을 MVP 범위에서 제거
- YOLO26 end-to-end / NMS-free inference를 기본 경로로 지정
- Jetson Orin Nano Super 공식 YOLO26s TensorRT benchmark 참고값 추가
- YOLO26n은 메모리/발열 문제가 있을 때만 fallback 후보로 유지
- TensorRT export/runtime compatibility fallback 원칙 추가
- Ultralytics 라이선스 검토 checkpoint 추가
- Phase 1, 3, 6, 8 및 신규 개발자 작업 순서를 YOLO26s + FP16 기준으로 개정
