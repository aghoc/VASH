# 요청 8종 집중 보강 실험

2026-10-15 KST. 병·컵·휴대폰·리모컨·숟가락·포크·칫솔·펜을 우선 보강하되 15종 출력과 모든 호환 라벨을 유지했다.

## 확인한 문제

- 이전 추가 학습은 backbone 앞 10블록을 고정하고 640px·낮은 학습률로 3 epoch 실행한 보존 위주 실험이었다. 기존 고정 test의 8종 평균 AP 개선은 24.22→24.75였다.
- SSH로 확인한 Jetson은 기존 `yolo26s_can_coin_retention_fp16.engine`을 사용 중이었다. 직전 추가학습 `yolo26s_focus8_v5.pt`는 Jetson에 없었고 카메라 서비스는 active였다. 이 확인 시점의 실시간 화면에는 새 학습 결과가 반영되지 않았다.
- 기존 test에서 칫솔은 6개 박스 모두 640 입력 환산 면적 기준 32px보다 작다. 펜·리모컨도 약 절반이 작은 물체다. 사진 수 외에도 크기와 촬영 환경 차이를 고려해야 한다.

## 입력 크기만 변경한 GPU 검증

이전 추가학습 모델의 validation만 비교했다. FP32, batch 8 고정. 최종 test는 이 선택에 사용하지 않았다.

| validation | 640px 평균 AP | 960px 평균 AP |
| --- | ---: | ---: |
| openimages | 32.83 | 15.05 |
| frozen | 29.53 | 31.40 |

크기 변경의 효과는 데이터셋에 따라 달랐다. 고해상도만 켜서 일반적인 개선을 보장할 수 없으므로 다음 학습에는 여러 입력 크기를 함께 사용했다.

## 집중 학습 설정

- 원본 train 사진 12,150장, 원본에서 만든 확대 예제 780개. 확대 예제를 새 원본 사진으로 세지 않는다.
- 8종 클래스·촬영 그룹별 균형 선택. 칫솔·펜·숟가락·포크·리모컨의 표본을 우선 확보했다.
- 작은 물체 확대 crop은 train 원본에서만 만들었다. 잘린 다른 물체가 라벨 없이 배경이 되는 crop은 제외하고, 남은 모든 호환 물체 box를 변환했다.
- 별도 can/coin replay 및 기존 다른 클래스 사진 일부를 유지했다. train/holdout decoded-pixel 교집합 0건.
- RTX 4070 Ti SUPER, FP32, AdamW 2e-5, batch 8, backbone 앞 5블록 고정(이전 10), 입력 800에 multi_scale=0.2(640–960 범위).
- 학습 2 epoch, 6.6분. 후보 비교는 FP32 640px batch 16으로 통일했다.
- 기존 combined·COCO·Open Images validation의 영역별 평균 개선과 클래스별 큰 하락, can/coin 유지 여부를 평가했다. 선택 뒤에만 test를 측정한다.

공식 근거: [Ultralytics train의 multi_scale/freeze 설정](https://docs.ultralytics.com/modes/train/), [SAHI의 작은 물체 확대/분할 추론 설명](https://docs.ultralytics.com/guides/sahi-tiled-inference/). 이 실험은 SAHI 라이브러리를 사용하지 않았으며 학습용 crop을 별도로 생성했다.

## 후보 채택 결과

이번 집중 실험의 후보는 미리 정한 validation 개선·유지 조건을 통과하지 못했다. 새 모델을 최종 후보로 채택하지 않았으며 이전 추가학습 모델을 유지한다. test를 보고 후보를 다시 고르지 않았다.

아래는 직전 추가학습 모델 대비 8종 평균 AP 변화다. COCO는 펜 정답이 없어 7종이다.

| 후보 | 기존 combined 변화(pp) | COCO 변화(pp) | Open Images 변화(pp) | 채택 |
| --- | ---: | ---: | ---: | --- |
| epoch1 | -5.19 | -4.66 | +1.23 | 미채택 |
| epoch2 | -4.83 | -4.29 | +1.85 | 미채택 |
| blend25 | -1.28 | +0.77 | +1.52 | 미채택 |
| blend50 | -2.67 | +0.52 | +2.82 | 미채택 |
| bn_reference | -28.38 | -23.82 | -32.47 | 미채택 |

raw epoch 1/2와 사전에 정한 25%/50% 가중치 보간을 비교했다. 이후 BN 통계 이동 가능성을 확인하려고 직전 모델의 BN running statistics만 복원한 대조 후보를 추가했지만 더 악화됐다. 이 대조 후보는 validation 실패를 본 후 추가했으며, 사전 계획하거나 test로 선택한 후보라고 주장하지 않는다.

## 적용 범위와 해석

수치는 공개 데이터 검출 AP이며 ELP 카메라에서의 정답률이 아니다. 실제 카메라로 평가한 새 정답 데이터는 만들지 않았다. 이 보고서는 학습 후보의 평가이며 Jetson engine·서비스 변경 여부는 [Jetson 비교·적용 기록](FOCUS8_JETSON_DEPLOYMENT_20261015.md)에서 확인한다.
