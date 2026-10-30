# 요청 8종 GPU 추가 학습 결과

2026-10-15 KST. 대상: 병·컵·휴대폰·리모컨·숟가락·포크·칫솔·펜.

GPU 통합 추가 학습과 분리된 테스트 평가를 완료했다. 데이터셋별 결과는 아래와 같다. 클래스별 AP50–95의 산술 평균이며 단순 정답률이 아니다.

| 평가셋 | 사진 | 대상 클래스 | 이전 평균 AP | 추가 학습 평균 AP |
| --- | ---: | ---: | ---: | ---: |
| 기존 frozen | 6,103 | 8 | 24.22 | 24.75 |
| 신규 COCO | 125 | 7 (펜 제외) | 21.13 | 23.45 |
| 신규 Open Images | 324 | 8 | 26.68 | 35.66 |

새 공개 사진에서는 개선 폭이 더 컸고 기존 frozen에서는 소폭 개선됐다. 포크와 칫솔은 여전히 보완이 필요하다. 실제 ELP 카메라 성능으로 이 숫자를 그대로 해석하지 않는다.

## 고정 테스트 비교

요청 8종 평균 mAP@50–95: **24.22 → 24.75**, +0.53 percentage points, 상대 변화 +2.18%.
수치는 검출 AP이며 단순 정답률이 아니다. 같은 frozen test 6,103장과 CUDA FP32 설정을 사용했다.

| 물체 | 이전 AP | 추가 학습 AP | 변화(pp) | test 박스 |
| --- | ---: | ---: | ---: | ---: |
| 병 | 21.36 | 22.20 | +0.84 | 2,251 |
| 컵 | 41.47 | 42.74 | +1.27 | 1,761 |
| 휴대폰 | 33.10 | 34.71 | +1.60 | 516 |
| 리모컨 | 27.70 | 27.38 | -0.32 | 695 |
| 펜 | 17.04 | 17.41 | +0.37 | 1,263 |
| 숟가락 | 11.65 | 12.15 | +0.50 | 467 |
| 포크 | 41.45 | 41.42 | -0.03 | 289 |
| 칫솔 | 0.00 | 0.00 | +0.00 | 6 |

칫솔 frozen test는 정답 박스가 6개뿐이고 모두 640 입력에서 작은 물체다. 일반적인 칫솔 인식 성능을 이 표 하나로 확정할 수 없다.
새 COCO/Open Images 평가도 함께 보되 실제 ELP 카메라 장면의 정확도는 별도 실측이 필요하다.

## 독립 COCO 테스트

학습과 분리한 신규 COCO test 125장. 기존 사용 COCO/LVIS ID를 제외했고 이미지 내용 기준으로도 train과 분리했다.
사진에 존재하는 클래스만 표시한다. 펜은 COCO 직접 범주가 아니므로 이 표 대신 frozen 평가를 사용한다.

| 물체 | 이전 AP | 추가 학습 AP | test 박스 |
| --- | ---: | ---: | ---: |
| 병 | 28.29 | 31.36 | 101 |
| 컵 | 25.00 | 27.94 | 102 |
| 휴대폰 | 25.13 | 26.89 | 33 |
| 리모컨 | 28.91 | 32.05 | 22 |
| 숟가락 | 6.78 | 8.96 | 23 |
| 포크 | 17.43 | 17.67 | 22 |
| 칫솔 | 16.38 | 19.28 | 18 |

## 독립 Open Images 테스트

공식 test split에서 선별한 새 사진만 사용했다. 기존 사용 Open Images ID, 원본 MD5, COCO와 같은 Flickr 사진 및 decoded-pixel 중복을 제외했다.
Open Images는 모든 물체의 라벨이 완전한 데이터셋이 아니므로 이 AP도 실제 카메라 정확도와 다를 수 있다.

| 물체 | 이전 AP | 추가 학습 AP | test 박스 |
| --- | ---: | ---: | ---: |
| 병 | 36.79 | 44.79 | 142 |
| 컵 | 23.61 | 36.52 | 71 |
| 휴대폰 | 26.34 | 43.02 | 66 |
| 리모컨 | 48.38 | 57.39 | 30 |
| 숟가락 | 19.88 | 25.87 | 83 |
| 포크 | 29.16 | 29.03 | 66 |
| 칫솔 | 13.10 | 18.80 | 19 |
| 펜 | 16.19 | 29.83 | 83 |

## 캔·동전 및 다른 클래스 유지

별도 공개 can/coin test 595장, CUDA FP32:

| 물체 | 이전 AP | 추가 학습 AP |
| --- | ---: | ---: |
| 캔 | 65.01 | 62.80 |
| 동전 | 78.28 | 77.67 |

별도 공개 validation에서 캔·동전 각각 이전 AP의 90% 이상 유지 조건: **통과**. 공개 test에서는 두 클래스 모두 소폭 하락했다.

나머지 클래스의 기존 frozen test:

| 물체 | 이전 AP | 추가 학습 AP | 변화(pp) |
| --- | ---: | ---: | ---: |
| 캔 | 0.02 | 0.00 | -0.01 |
| 상자 | 3.92 | 3.81 | -0.11 |
| 공 | 44.28 | 43.92 | -0.36 |
| 카드 | 0.80 | 0.71 | -0.09 |
| 열쇠 | 6.21 | 5.51 | -0.70 |
| 동전 | 0.01 | 0.01 | +0.00 |
| 작은 블록 | 0.01 | 0.00 | -0.01 |

## 데이터와 학습

- 이전 checkpoint: `artifacts/models/yolo26s_can_coin_retention.pt`.
- 고유 train 사진 43,419장: 기존 15종 retention + EgoObjects 잔여 replay + 신규 COCO + 신규 Open Images.
- 기존 train의 내용 중복을 제거하고 같은 이미지의 호환되는 라벨을 병합했다.
- 칫솔 양성에 명시적 2배 표본 가중치: 반복 2,659개를 포함한 train 항목 46,078개. 반복을 새 사진으로 세지 않는다.
- 원래 val 6,927 / test 6,698개 파일·라벨 보존. 신규 COCO val 136 / test 125장은 별도로 유지했다.
- 신규 Open Images val 248 / test 324장도 기존 frozen 및 COCO 평가와 분리해 유지했다.
- train과 모든 holdout의 decoded-pixel 중복 0건. EgoObjects는 기존 group split을 유지했다.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, CUDA 13.0, PyTorch 2.13.0+cu130.
- FP32, 640px, batch 16, AdamW. 선택된 실험 3 epoch, 16.6분. 전체 학습 실험 합계 58.7분.
- 선택: stable 실험 epoch 3. 요청 8종 validation 평균 및 클래스별 하락·can/coin 유지 조건으로 선택했다. test는 선택에 쓰지 않았다.
- 세부 설정: `configs/model_yolo26s_focus8_v5.yaml`, 보존 위주의 추가 실험 설정 `configs/model_yolo26s_focus8_stable_v5.yaml`.

| 출처 / split | 고유 사진 |
| --- | ---: |
| Existing 15-class retention/val | 6,927 |
| Existing 15-class retention/test | 6,698 |
| COCO 2017/test | 125 |
| COCO 2017/val | 136 |
| EgoObjects replay/train | 8,667 |
| COCO 2017/train | 2,818 |
| Existing 15-class retention/train | 28,239 |
| Open Images focus8/val | 248 |
| Open Images focus8/test | 324 |
| Open Images focus8/train | 3,695 |

COCO 공식 annotation, Open Images 공식 box/이미지 정보 CSV, 각 이미지의 원본 URL·저작자·라이선스와 매핑은 manifest 및 [DATA_SOURCES.md](DATA_SOURCES.md)에 보존했다.
최종 데이터: `data/processed/public_v6_focus8_coco_oi_15/data.yaml`.
COCO에서는 동일 물체에 COCO/LVIS 라벨이 겹치지 않도록 COCO 직접 범주에 원본 COCO 라벨만 사용했다. 그 외 호환 범주의 LVIS 라벨은 유지했다.

## 모델과 재현 자료

- 추가 학습 후보: `artifacts/models/yolo26s_focus8_v5.pt`, SHA-256 `c60b573217426a0f600cbc7d2dea9fc67a9708ce53e4ce46fbc7248d9c794439`.
- run: `artifacts/training/yolo26s_focus8_v5_stable_20261015`.
- 선택 근거 및 평가: `runs/focus8_20261015/validation_selection.json`, 각 evaluation 폴더의 `metrics_summary.json`.
- 전체 수치: `reports/focus8_finetune_result_20261015.json`.
- Jetson의 기존 배포 설정/engine은 이번 학습 실험에서 변경하지 않았다. 카메라 재시작 명령도 실행하지 않았다.

후속 2026-10-15 작업에서 이 후보를 Jetson native FP16으로 변환하고 기존 engine과 같은 validation·속도 기준으로 비교한 뒤 카메라 서비스에 적용했다. 최신 상태와 물체별 결과는 [Jetson 비교·적용 기록](FOCUS8_JETSON_DEPLOYMENT_20261015.md)을 참조한다.
