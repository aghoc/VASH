# 캔·동전 공개 데이터 보강 계획

작성: 2026-10-14 KST. 사용자 조건: 자체 촬영·수동 데이터 구축 없이 공개 주석 데이터를 사용한다.
아래 내용은 추가 학습 전 조사 기록이다. GPU 추가 학습 및 Jetson FP16 배포를 완료했으며,
현재 성능과 복구 절차는 [최종 결과](CAN_COIN_OPEN_DATA_RESULT.md)를 참조한다.

## 확인된 현재 데이터

최종 학습 입력 `data/processed/public_v3_adl_balanced_v2_15`의 실제 라벨을 집계했다.

| 클래스 | 학습 이미지 | 학습 박스 | 기존 val 박스 | 기존 test 박스 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 6,313 | 12,942 | 2,698 | 2,251 |
| can | 1,552 | 2,860 | 61 | 23 |
| coin | 141 | 428 | 18 | 62 |

동전은 학습 이미지가 적고, 작은 박스를 포함한 원거리 장면이 많다. 병·컵·휴대폰·리모컨·칫솔만
포함한 ADL 마지막 학습 단계는 캔·동전 예시를 추가하지 않았다. 이를 약한 성능의 원인 후보로
보고 데이터/표본 추출 실험을 진행한다. 기존 val의 캔·동전 표본 수가 적어 별도 공개 holdout이 필요하다.

실제 집계와 시각적 확인 자료:

- `runs/open_data_can_coin_20261014/current_dataset_audit.json`
- `runs/open_data_can_coin_20261014/audit_can_val.jpg`
- `runs/open_data_can_coin_20261014/audit_coin_train.jpg`

크기 통계는 정규화된 박스를 640×640 canvas로 환산한 값이다. 실제 letterbox 입력의 픽셀
크기는 원본 종횡비에 따라 달라지므로 같은 수치로 간주하지 않는다.

기존 YOLO26s FP16 frozen val의 AP50–95는 can 0.001654, coin 0.004917이다.
이는 분류 정확도 퍼센트가 아니다. FP32 단계에서도 두 클래스가 약했으므로 FP16 변환만의 문제로 보지 않는다.

## 사용할 공개 데이터 후보

| 출처 | 사용할 클래스 | 확인된 정보 | 적용 원칙 |
| --- | --- | --- | --- |
| Open Images | Tin can `/m/02jnhm`, Coin `/m/0242l` | 공식 600 boxable 클래스 CSV에 두 클래스 존재 | bbox 주석이 있는 이미지 사용; 이미지 출처·주석 메타데이터 보존 |
| RF100 coins-1apki | coin | 공식 프로젝트 약 8.4k 이미지, coin/screw/nut/nail, CC BY 4.0 | 전체 이미지가 동전 양성이라고 계산하지 않음; coin만 프로젝트 13번으로 매핑; 나사·너트 등은 음성 예시 |
| 기존 LVIS v1 | can, beer_can, coin, penny_(coin) | 기존 변환에서 beer_can/penny_(coin) 누락 | 음료/식품 캔과 동전만 의미에 맞게 매핑; watering_can/aerosol 등은 자동 편입하지 않음 |
| TACO (보조 후보) | Drink can, Food Can | 공식 JSON: 229 / 34 박스 | 찌그러진 폐기물 장면 편향과 개별 이미지 출처 조건을 확인한 뒤 보조 자료로 판단 |

공식 출처:

- Open Images 다운로드: https://storage.googleapis.com/openimages/web/download_v7.html
- 공식 boxable 클래스: https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
- RF100 프로젝트: https://universe.roboflow.com/roboflow-100/coins-1apki
- RF100 원 출처: https://universe.roboflow.com/labelimg/label_coin
- RF100 공식 목록: https://github.com/roboflow/roboflow-100-benchmark
- TACO 공식 데이터: https://github.com/pedropro/TACO

RF100 다운로드 후보 미러 `Francesco/coins-1apki`의 조사 revision은
`ec1591c4dd8fd3f93a2060a5a5a2a181c8d4e190`이다. 공식 RF100 프로젝트와의 출처 연결,
파일 checksum, 주석 형식/좌표/클래스 수를 확인한 뒤 실제 학습에 넣는다.
미러 card의 모호한 `cc` 표기 대신 원 프로젝트의 CC BY 4.0 출처를 보존한다.

## 수정 및 다음 실험

1. LVIS 변환에 `beer_can → can`, `penny_(coin) → coin`을 추가했다.
   원본 train 주석에 beer_can 203박스, penny_(coin) 15박스가 존재함을 확인했다.
   전체 218박스가 새 학습 예시가 된다는 의미는 아니다. 이미 확보된 이미지에서 빠진 라벨을
   복구할 수도 있으며, split/중복 검사와 이미지 확보 후 실제 추가 수를 기록해야 한다.
   watering_can을 잘못 매핑하지 않는 회귀검사 포함, 관련 테스트 2개와 Ruff 검사 통과.
2. Open Images 캔·동전과 RF100 coin의 bbox를 기존 15개 class 번호로 통일한다.
   동일 이미지의 나머지 목표 클래스 주석도 가능한 한 유지해 배경 오라벨을 줄인다.
3. 원본 이미지·영상 그룹·RF 증강 원본을 기준으로 중복을 제거하고 split을 고정한다.
   기존 frozen val/test는 유지하고, 새 공개 데이터 holdout을 추가한다.
4. 기존 15종 데이터를 섞으면서 캔·동전의 학습 노출을 늘리는 표본 추출을 비교한다.
   근거리 crop/다양한 크기/조명/회전/블러를 사용하고, 작은 물체를 과도하게 줄이는 증강은 조절한다.
5. 같은 YOLO26s/640 입력에서 추가 학습한다. 기존 checkpoint, 공개 stage checkpoint에서 시작하는
   짧은 실험을 비교하고 캔·동전 AP/recall·음성 예시 오검출·나머지 13종 성능을 함께 기록한다.
6. 개선 후보만 Jetson에서 FP16 engine을 다시 만들고 기존 모델과 비교한다.
   더 높은 해상도나 타일 추론은 작은 물체 성능과 Jetson 지연을 실측한 별도 실험으로 다룬다.

공개 데이터만으로 보강 학습과 공개 holdout 평가는 가능하다. ELP 시점과 한국 동전에 대한
실사용 성능은 공개 데이터 결과만으로 보장하지 않으며, 향후 실제 사용 확인과 구분해 기록한다.
