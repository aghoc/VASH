# 캔·동전 공개 데이터 GPU 추가 학습 결과

2026-10-14 KST. **RTX 4070 Ti SUPER에서 추가 학습을 완료하고 Jetson ELP 카메라 서비스에 반영했다.**
기존 15종 YOLO26s pretrained/fine-tuned checkpoint를 초기값으로 사용했다. 학습과 평가 모두 CUDA를 사용했으며,
Jetson에서도 직접 빌드한 TensorRT FP16 GPU 엔진으로 실행한다.

현재 카메라 서비스는 후속 사용자 요청으로 정지했다(`inactive/dead`). 위 배포 및 서비스 검증은 정지 전 기록이다.
물체별 비교와 프레임 저장 방식은 [별도 정리](PER_CLASS_PERFORMANCE_20261014.md)를 참조한다.

## 실제 배포 성능

동일한 공개 test 595장, 캔 146개·동전 2,841개 박스를 기존/신규 **Jetson FP16 엔진**으로 비교했다.
수치는 AP50–95이며 분류 정확도나 실제 카메라 성공률이 아니다.

| 클래스 | 기존 FP16 AP | 신규 FP16 AP | 배율 |
| --- | ---: | ---: | ---: |
| 캔 | 0.1622 | 0.6357 | 3.92× |
| 동전 | 0.2069 | 0.6611 | 3.20× |

실행 임계값은 공개 validation 367장과 기존 validation의 캔·동전 없는 1,024장을 사용해 선택했다.
캔 최소 precision 0.70, 동전 0.90 및 기존 음성 이미지 오검출 비율 ≤1% 조건에서 F1을 최대화했다.
threshold는 캔 **0.75**, 동전 **0.25**, 나머지 클래스 **0.35**다. 점수를 바꾸지 않고 클래스별 검출/그립 gate를 적용한다.

| 공개 test, IoU ≥0.5 | precision | recall | TP / FP / FN |
| --- | ---: | ---: | --- |
| 캔 | 89.0% | 50.0% | 73 / 9 / 73 |
| 동전 | 96.8% | 97.7% | 2775 / 92 / 66 |

작은 동전(640 입력에서 bbox 면적 제곱근 <32px)은 750개 중 714개 검출, recall 95.2%다.
보수적인 캔 임계값으로 test recall은 50.0%에 제한된다. 공개 test의 해당 클래스 없는 이미지에서는
캔 499장·동전 96장 모두 오검출 이미지가 0장이었다. 기존 음성 validation에서는 캔 3/1,024장(0.29%),
동전 10/1,024장(0.98%)이었다. 공개 주석의 누락 가능성과 표본 범위를 고려해야 한다.

## 속도와 서비스 확인

- Jetson Orin Nano Super, 25W, clocks 비고정, TensorRT 10.3.0, static 640×640, batch 1, FP16, NMS-free.
- 60회 warmup 후 300회 pipeline benchmark: **30.97 FPS**, 평균 **32.29ms**,
  p95 **35.48ms**. 16-byte UART 직렬화를 포함했으며 물리 UART 전송은 하지 않았다.
  기존 모델의 같은 작업 조건은 28.24 FPS, 평균 35.41ms, p95 38.63ms였다.
- 위 benchmark는 카메라 캡처, JPEG 인코딩, 네트워크와 센서 노출 지연을 포함하지 않는다.
- 배포 후 `vagt-camera.service` active/running, 재시작 0회. 10초 동안 프레임 3→207, 캡처 실패 0.
  초기화 직후 상태 표본의 뷰어 처리율은 19.42 FPS, pipeline 평균 28.13ms였다.
  카메라 자체의 이전 캡처 측정은 약 24.4 FPS이며 요청값 60 FPS와 구분한다.
- 추가 서비스 상태 확인: 334.7초 동안 6,587프레임, 캡처 실패 0, 재시작 0.
  해당 시점 뷰어 21.45 FPS, pipeline 평균 27.68ms, p95 28.23ms였다.
- 공개 validation 이미지로 4-frame 안정화를 실제 엔진에서 확인했다:
  캔 score 0.8213 → POWER,
  동전 score 0.6382 → PINCH.
  이는 기능 연결 검사이며 별도의 인식률 측정은 아니다. 현재 카메라 이미지는 관찰하지 않았다.

후속 기존 train 중복 감사에서 카드의 158개 파일 항목이 80개 고유 사진,
작은 블록의 72개 항목이 36개 고유 사진임을 확인했다. 아래 train 이미지 수는 파일/노출 항목 수이며
독립 장면 수를 뜻하지 않는다. 자세한 원인과 13종 보강안은 [추가 조사](ALL15_OPEN_DATA_RESEARCH.md)를 참조한다.

## 데이터와 학습

추가 공개 데이터는 [Open Images 공식 bbox](https://storage.googleapis.com/openimages/web/download_v7.html)와
[RF100 coins-1apki](https://universe.roboflow.com/roboflow-100/coins-1apki)를 사용했다.
확보한 Open Images 2,274장은 공식 메타데이터상 CC BY 2.0, 주석 CC BY 4.0이며 이미지별 저자·원본 URL을 보존했다.
RF100은 CC BY 4.0; COCO archive SHA-256
`0bbf2cc3299f1cb9b7595f0f41cdda867864800ad0bef987c655c98205b8245f`,
미러 revision `ec1591c4dd8fd3f93a2060a5a5a2a181c8d4e190`를 고정했다. TACO는 사용하지 않았다.

RF100의 원래 split 사이에 같은 영상 프레임이 섞여 있어 촬영 그룹으로 다시 분할했다.
동일 영상은 train에서 100ms당 1장만 사용하고, burst 분/인접 파일번호 그룹을 분리했다.
최종 RF100 1,208장을 사용했다. 확보 범위의 decoded pixel 중복 및 split 사이 pHash 거리 ≤4 검사는 중복 0건이었다.
누락된 촬영 메타데이터나 모든 장면 유사성을 보장하는 검사는 아니며 배경 편향이 남을 수 있다.
기존 frozen val/test 이미지와 라벨은 수정하지 않았다.

| 학습/평가 구성 | 이미지 수 |
| --- | ---: |
| 추가 공개 train / val / test | 2,520 / 367 / 595 |
| 원래 train 전체 | 25,804 |
| 1단계 집중 학습, 반복 포함 / 고유 | 17,088 / 11,772 |
| 2단계 전체 데이터 보완 학습 | 28,370 |
| 기존 frozen val / test | 6,560 / 6,103 |

LVIS beer_can/penny 주석 누락을 66개 이미지에서 복원했다(기존 train 18장, 추가 48장).
복원 이미지의 다른 호환 클래스 주석도 보존했다. 2단계 train 캔 박스 5,215개, 동전 5,162개.

- 1단계: FP32, batch 16, AdamW lr 0.0002, 7 epoch 실행(최대 10, early stop), best epoch 2.
- 2단계: 원래 train 전체를 복원하고 공개 데이터와 합쳐 FP32, batch 16, lr 0.00003, 1 epoch 보완.
- 검증셋으로 후보를 선택하고 test로 최종 보고 수치를 계산했다. 최종 모델은 native YOLO26s이며
  가중치 혼합 후보는 배포하지 않았다.
- 최종 FP32 공개 test AP: 캔 0.6501, 동전 0.7828. Open Images만 분리한 test도 캔 0.6559,
  동전 0.8078로 개선됐다. RF100의 특정 배경에만 의존하는 결과는 아니다.

## 남은 제한

기존 frozen **나머지 13종 평균 FP32 AP**는 val에서 0.2608→0.2552(-2.17%),
test에서 0.2011→0.1915(-4.77%)로 감소했다. 휴대폰 test AP는 0.4276→0.3310으로 감소 폭이 크다.
이 비교는 FP32이며 신규 엔진의 기존 13종 전체 FP16 평가를 대신하지 않는다.

FP16 변환 후 공개 coin AP는 FP32 0.7828→FP16 0.6611로 감소했다. IoU ≥0.5의 배포점 recall은
97.7%지만 엄격한 위치 정확도의 손실이 있으므로 FP32 수치를 배포 성능으로 사용하지 않는다.

기존 frozen 캔·동전 AP는 여전히 낮다. 기존 val 동전 18개는 640 입력에서 모두 32px 미만이며,
기존 test에는 금속 화폐와 외형이 다른 보드게임 토큰도 coin으로 포함되어 있다.
공개 데이터 개선을 ELP 실제 장면, 한국 동전, 손에 가린 물체 또는 원거리 성능으로 일반화하지 않는다.
실사용 영상에서의 정확도와 새 카메라 개구량 보정은 아직 측정하지 않았다.

## 파일과 복구

- 로컬 checkpoint: `artifacts/models/yolo26s_can_coin_retention.pt`
- Jetson 엔진: `/mnt/data/vagt-orca-ai/artifacts/deployment/tensorrt/yolo26s_can_coin_retention_fp16.engine`
- checkpoint SHA-256: `5ba5715728b996cfb11fca6b01072bd54256e38a313c27e691eb7ba34ea68726`
- 엔진 SHA-256: `b24d8fffcd1702a176b7743a7bade6de09b7b0202dfe63591ba16308e60263f0` (22,786,910 bytes)
- 설정: `configs/runtime.yaml`, `configs/deployment.yaml`, `deployment/vagt-camera.service`
- 원본 `yolo26s_best.pt`와 `yolo26s_best_fp16.engine`은 원래 checksum으로 보존했다.
- Jetson 배포 전 설정/서비스: `runs/can_coin_retention_20261014/predeploy/`
- 머신 판독 보고서: `reports/can_coin_open_data_result.json`
- 로컬 실험 및 Jetson 원시 결과: `runs/open_data_can_coin_20261014/`

Jetson에서 기존 모델로 즉시 복구:

```bash
cd /mnt/data/vagt-orca-ai
cp runs/can_coin_retention_20261014/predeploy/runtime.yaml configs/runtime.yaml
cp runs/can_coin_retention_20261014/predeploy/deployment.yaml configs/deployment.yaml
cp runs/can_coin_retention_20261014/predeploy/vagt-camera.service deployment/vagt-camera.service
cp deployment/vagt-camera.service ~/.config/systemd/user/vagt-camera.service
systemctl --user daemon-reload
systemctl --user restart vagt-camera.service
```

## 재현

데이터 준비의 revision/출처/라이선스와 이미지 attribution은 `DATA_SOURCES.md`,
`data/raw/can_coin_open_v1/oi_image_attribution.json`, supplement `manifest.jsonl`에 기록했다.
기존 raw 데이터 및 고정 checkpoint가 있는 CUDA 환경에서:

```bash
.venv/bin/python datasets/prepare_can_coin_open.py --raw data/raw/can_coin_open_v1 \
  --output data/supplements/can_coin_open_v1 --frozen data/processed/public_v3_adl_balanced_v2_15 \
  --class-csv runs/open_data_can_coin_20261014/openimages_boxable.txt
.venv/bin/python datasets/build_can_coin_v4.py --base data/processed/public_v3_adl_balanced_v2_15 \
  --supplement data/supplements/can_coin_open_v1 --lvis-train data/raw/lvis/lvis_v1_train.json \
  --output data/processed/public_v4_can_coin_15
.venv/bin/python training/train.py --data data/processed/public_v4_can_coin_15/data.yaml \
  --config configs/model_yolo26s_can_coin_v4.yaml --project artifacts/training \
  --name yolo26s_can_coin_v4_20261014 --model-output artifacts/models/yolo26s_can_coin_v4.pt
.venv/bin/python datasets/build_can_coin_v4.py --base data/processed/public_v3_adl_balanced_v2_15 \
  --supplement data/supplements/can_coin_open_v1 --lvis-train data/raw/lvis/lvis_v1_train.json \
  --per-class 100000 --coin-repeat 1 --can-repeat 1 --output data/processed/public_v4_can_coin_retention_15
.venv/bin/python training/train.py --data data/processed/public_v4_can_coin_retention_15/data.yaml \
  --config configs/model_yolo26s_can_coin_retention.yaml --project artifacts/training \
  --name yolo26s_can_coin_retention_20261014 --model-output artifacts/models/yolo26s_can_coin_retention.pt
# 아래 export는 Jetson에서 실행한다.
.venv/bin/python deployment/export_tensorrt.py --model artifacts/models/yolo26s_can_coin_retention.pt \
  --output artifacts/deployment/tensorrt/yolo26s_can_coin_retention_fp16.engine \
  --report runs/can_coin_retention_20261014/export.json
.venv/bin/python training/validate.py \
  --model artifacts/deployment/tensorrt/yolo26s_can_coin_retention_fp16.engine \
  --data data/benchmark/can_coin_v4_public/data.yaml --split test --batch 1 \
  --project runs/can_coin_retention_20261014/public
```

전체 pytest **77개**, 변경 Python 파일 Ruff 검사 통과. 데이터/임계값/그립/고정배치 입력 회귀검사를 포함한다.
