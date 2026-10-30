# 15종 공개 데이터 보강 조사

2026-10-14 KST. 이번 작업은 데이터 조사와 기존 데이터 감사다. 새 학습/모델 교체는 실행하지 않았다.
카메라 서비스는 사용자 요청에 따른 정지 상태를 유지했다.

2026-10-15 추가 감사: [EgoObjects 잔여분](EGOOBJECTS_REMAINDER_AUDIT_20261015.md)을 확인했다.
전체 train 49,830장 중 현재 포함 11,256장, 빠진 프레임 38,574장이 맞다. 초기 YOLO26s 전체 학습에서는 이미 사용한 데이터다.
상자·펜·리모컨 등은 이 잔여분을 먼저 선별해 재활용하는 것을 우선한다. 칫솔·카드·작은 블록의 추가 양성은 이 잔여분에 없어 아래 외부 데이터 보강도 필요하다.

## 확인한 원인과 데이터 규모

현재 추가 학습 데이터 `public_v4_can_coin_retention_15`의 이미지/라벨을 직접 읽었다.
이미지 수는 파일 항목 수이며 촬영 장면 수와 다르다. 카드·블록은 decoded RGB hash로 중복을 추가 확인했다.

| 클래스 | train 이미지 항목 | train 박스 | 기존 test 박스 | train 작은 박스 비율 |
| --- | ---: | ---: | ---: | ---: |
| 병 | 6,369 | 13,091 | 2,251 | 28.9% |
| 컵 | 6,592 | 10,323 | 1,761 | 24.0% |
| 상자 | 3,899 | 9,114 | 3,877 | 25.8% |
| 공 | 1,382 | 1,990 | 373 | 37.8% |
| 휴대폰 | 3,194 | 3,820 | 516 | 32.4% |
| 리모컨 | 4,078 | 5,864 | 695 | 36.3% |
| 펜 | 1,857 | 2,758 | 1,263 | 51.9% |
| 숟가락 | 2,706 | 3,827 | 467 | 21.6% |
| 포크 | 2,376 | 3,262 | 289 | 21.0% |
| 칫솔 | 2,492 | 3,929 | 6 | 33.4% |
| 카드 | 158 | 442 | 39 | 33.9% |
| 열쇠 | 1,476 | 1,902 | 249 | 41.2% |
| 작은 블록 | 72 | 1,040 | 161 | 72.3% |

작은 박스는 원본 종횡비를 유지한 640 letterbox에서 bbox 면적의 제곱근 <32px 기준이다.

- 카드: 158개 파일 항목, **실제 서로 다른 사진 80장**. 작은 블록: 72개 항목, **서로 다른 사진 36장**.
  기존 LVIS 초기 supplement와 후속 supplement가 같은 사진을 중복 포함했다. 반복 노출은 장면 다양성을 늘리지 않는다.
- 카드/블록의 표본 부족은 확인됐다. 상자 9,114개·칫솔 3,929개 박스 등은 양만으로 낮은 AP를 설명할 수 없다.
  촬영 그룹·물체 크기·가림·클래스 범위·위치 주석 및 학습 편향을 함께 확인해야 한다.
- 칫솔의 기존 test 표본은 6개뿐이다. AP 0을 모든 실사용 칫솔 장면의 실패율로 해석하지 않는다.
- 직전 보강 후 휴대폰 AP 0.4276→0.3310. 다른 13종 평균 AP도 4.77% 감소했으므로 기존 능력 유지가 새 학습의 목표에 포함되어야 한다.

## 즉시 활용 가능한 미사용 LVIS train

원래 train annotation에 존재하되 현재 train/val/test 파일의 LVIS 원본 image ID에 없는 후보다.
이미지 checksum, 유효 박스 및 출처 병합을 검사한 뒤 사용해야 하며, 이미 신규 데이터로 구축된 수가 아니다.

| 원본 클래스 | 미사용 후보 이미지 |
| --- | ---: |
| box | 693 |
| cellular_telephone | 776 |
| bottle | 727 |
| cup | 388 |
| fork | 676 |
| remote_control | 86 |
| spoon | 94 |

pen/toothbrush/key/card/Lego/cube는 현재 사용 중인 정확한 LVIS 클래스 이름 범위에서 추가 미사용 image ID가 0이다.
LVIS를 다시 추천하는 것만으로 이 클래스의 새 장면이 늘어나지는 않는다.
water_bottle/wine_bottle 등 일부 세부 유형은 기존 매핑 밖에 있지만, 클래스 정의와 중복 검토 후 확장할 후보다.
carton은 소형 상자에 해당하는 주석만 사용할 후보이며 컵·음료 용기와 혼동되면 제외한다.
현재 small_block의 Lego/cube 병합도 점검한다. 단순 큐브 형태라는 이유로 상자·큰 조립품을 블록으로 섞지 않는다.

## Open Images 실제 후보 수

[공식 Open Images](https://storage.googleapis.com/openimages/web/download_v7.html)의 기존 로컬 bbox CSV를 전부 집계했다.
개별 IsGroupOf=0 / IsDepiction=0 행을 대상으로 한 후보 수다. 이미지 전체 조건, 중복 및 실제 다운로드 검수 전이다.
클래스별 이미지 수를 합하면 같은 이미지가 중복 계산될 수 있다.

| 프로젝트 클래스 | train 후보 이미지 | 이전 캔·동전 supplement에 없던 후보 |
| --- | ---: | ---: |
| 상자 | 2,150 | 2,147 |
| 휴대폰 | 3,965 | 3,963 |
| 펜 | 951 | 950 |
| 숟가락 | 1,009 | 1,007 |
| 포크 | 1,008 | 1,007 |
| 컵 | 1,509 | 1,506 |
| 병 | 10,981 | 10,927 |
| 공 | 8,752 | 8,749 |
| 리모컨 | 181 | 180 |
| 칫솔 | 62 | 62 |

600개 bbox 클래스 목록에는 key/card/small_block에 대응하는 직접 클래스가 없다.
칫솔은 Open Images train 후보가 62장뿐이므로 COCO 보완과 별도 평가 표본 확보가 더 중요하다.
Open Images의 Cup 대신 Mug 매핑을 사용했다. 구기류는 실제 ball 하위 클래스만 병합하며 Balloon은 제외한다.

## 클래스별 추천 데이터와 적용 방식

| 대상 | 우선 출처 | 적용 |
| --- | --- | --- |
| 병·컵·공·휴대폰·리모컨·숟가락·포크·칫솔 | [COCO 2017 클래스/형식](https://docs.ultralytics.com/datasets/detect/coco/) + Open Images | 해당 8종 직접 매핑, COCO/LVIS 동일 이미지의 기존 주석 병합 |
| 상자·펜 | Open Images + 미사용 LVIS | 상자 유형별/작은 펜 표본 선별, pen에 pencil을 자동 병합하지 않음 |
| 열쇠 | [Keys Data v1](https://universe.roboflow.com/keys-detection-data/keys-data/dataset/1) | 페이지 총 6,769장(train 5,306), CC BY 4.0. 전체를 key 양성으로 계산하지 않고 실제 class/bbox와 물리 열쇠 여부 확인 |
| 작은 블록 | [Lego v2](https://universe.roboflow.com/legos/lego-7kwjy/dataset/2) | 287장(train 200), CC BY 4.0. 블록 11종을 small_block으로 통합; 큰 조립품/판은 제외 |
| 카드 | [MIDV-Holo 저자 자료](https://github.com/SmartEngines/midv-holo) + [MIDV-2020 논문](https://arxiv.org/abs/2107.00396) | ID 카드만 선택, 문서 외곽 quadrangle의 min/max를 bbox로 변환. 스캔/여권 제외, 문서/촬영 그룹별 분할 |
| 카드 보조 | [ball_marker_card](https://universe.roboflow.com/object-detection-kmblq/ball_marker_card) | 프로젝트 총 700장, CC BY 4.0. 전체 카드 경계·카드 양성 수·원본 중복 검수 후 사용 |

COCO는 초기 pretrained 가중치가 이미 학습한 출처지만, 15종 fine-tuning 중 잃은 일반 물체 표현을 유지하는 재학습 자료로 사용할 수 있다.
현재 데이터도 LVIS/COCO를 이미 포함하므로 무조건 모두 추가하지 않고 기존 이미지/평가셋과 중복을 제거한다.

MIDV-Holo는 저자 README에 총 700개 문서 영상, 프레임별 문서 경계 주석을 명시한다.
여권도 포함되므로 700개가 모두 카드 영상이라는 뜻은 아니다. CC BY-SA 2.5 및 저자/Generated Photos attribution을 보존한다.
MIDV-2020은 논문상 전체 문서 유형 합계 72,409개 annotated image이며 이는 카드 사진 수가 아니다.
아카이브 license.txt와 템플릿/이미지 attribution 확인은 데이터 편입 전에 필요하다.

Roboflow 프로젝트 숫자는 확인 당시 버전의 페이지 표기이며 실제 다운로드 파일의 고유 이미지 수와 같다고 보장하지 않는다.
공식 다운로드는 무료 계정/API 인증이 필요할 수 있다. 인증 없는 미러를 쓰면 revision·archive checksum과 원본 라이선스를 고정한다.
후보 페이지의 mAP@50 수치를 우리 모델의 향상 예측으로 사용하지 않는다.

## 추가 후보와 제외 조건

- [RF100 poker cards](https://universe.roboflow.com/roboflow-100/poker-cards-cxcvz/dataset/1): 1,285장, CC BY 4.0.
  rank/suit 구석 박스인지 카드 전체 박스인지 먼저 확인한다. 구석 주석을 card 전체 bbox로 변환하지 않는다.
- [BOP YCB-V](https://bop.felk.cvut.cz/datasets/): 포장 상자·wood block의 가림/자세 보강 후보, BOP 페이지상 MIT.
  amodal bbox와 visible mask의 범위를 기존 라벨과 맞추고 특정 물체 브랜드/인스턴스에 과적합하지 않도록 제한한다.
  공식 BOP 안내에 따라 test와 자세를 공유하는 원래 YCB-V 80K synthetic train은 사용하지 않는다.
- [EPIC-KITCHENS VISOR](https://epic-kitchens.github.io/VISOR/site): 1인칭 손/가림 장면의 mask-to-bbox 보강 후보.
  원본 영상 전체 대신 sparse RGB frame과 수동 mask부터 선별할 수 있다. CC BY-NC 4.0이므로 주 경로와 분리해 연구용 후보로 기록한다.
- [Objects365](https://www.objects365.org/download.html): 공식 페이지가 academic purpose 및 image 재배포 제한을 명시해 주 경로에서 제외했다.
  annotation CC BY 4.0을 전체 이미지 사용 조건으로 해석하지 않는다.
- 기존 ADL knife/spoon/fork 통합 label은 목표 spoon/fork에 임의 분배하지 않는다.
- ASL 데이터의 credit card/spoon/fork는 물체가 아니라 수어 의미인 경우가 있어 같은 문자열만 보고 병합하지 않는다.

## 다음 GPU 학습 실행안

1. 기존 train 중복 사진을 제거하고 촬영 그룹·물체 크기·양성/음성·source별 실제 고유 수를 고정한다.
2. 미사용 LVIS 이미지부터 복원하고, Open Images 10종과 COCO 8종을 class별 균형으로 선별한다.
   카드·블록·열쇠 전용 데이터는 단순 클래스 이름 매칭보다 전체 물체 경계와 원본 촬영 그룹을 우선 검수한다.
3. COCO와 LVIS는 같은 이미지 ID의 모든 호환 주석을 병합한다. 누락된 target annotation을 음성으로 취급하지 않도록 검사한다.
4. 원래 15종 `yolo26s_best.pt`에서 기존 데이터+캔/동전 공개 데이터+새 보강분을 함께 CUDA FP32 학습하는 후보를 우선한다.
   현재 can/coin checkpoint에서도 같은 입력으로 비교 후보를 만들 수 있다. 새 실험은 train/validation에서 선택한다.
   source/class별 균형, 소형 물체 crop/크기 증강을 사용하고 특정 두 클래스 반복 노출이 학습을 지배하지 않게 한다.
5. 기존 frozen val/test는 보존한다. 부족한 칫솔·카드·블록 평가 표본을 독립 공개 촬영 그룹으로 추가하되 기존 평가와 따로 보고한다.
   새 holdout과 frozen test를 train/후보선택에 사용하지 않는다. 배포 판단은 모든 클래스의 AP/P/R, 기존 능력 유지 및 실패 유형을 포함한다.
6. 최종 후보만 Jetson FP16에서 15종 전체 성능과 처리 속도를 다시 확인한다. 큰 입력 크기는 속도 검증 후 결정한다.

수치 예측이나 개선 보장은 하지 않는다. 데이터 편입과 재학습이 끝나야 실제 13종 개선 여부를 판단할 수 있다.

감사 원본: `runs/open_data_all15_20261014/active_dataset_audit.json`, `rare_class_duplicate_audit.json`,
`openimages_candidates.json`, `lvis_candidate_inventory.json`, `source_candidates.json`.
