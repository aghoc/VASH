# 데이터 및 모델 출처

## 2026-10-15 COCO + 신규 Open Images 요청 8종 보강

- 공식 Open Images box CSV는 기존 `data/raw/can_coin_open_v1/oid_{train,val,test}.csv`를 재사용한다. 클래스 표는 `runs/open_data_can_coin_20261014/openimages_boxable.txt`.
- 신규 사진/manifest/공식 이미지 저작자: `data/raw/focus8_openimages/`. 원본 URL, landing URL, Author, License, OriginalMD5, split을 보존한다. 이미지 CC BY 2.0/4.0만 허용하고 annotation은 CC BY 4.0이다.
- 매핑: Bottle→병, Mug→컵, Mobile phone→휴대폰, Remote control→리모컨, Pen→펜, Spoon→숟가락, Fork→포크, Toothbrush→칫솔. 함께 등장하는 다른 호환 15종 box도 보존한다. GroupOf/Depiction=1인 이미지는 제외하며, 속성이 -1(미확인)인 공식 human-verified bbox는 포함한다. 과거 후보 수(칫솔 62장)는 두 속성이 모두 0인 box만 센 값이므로 현재 편입 수와 직접 비교하지 않는다.
- 모든 기존 사용 OI ID, 기존 OI 원본 MD5, COCO에 등장하는 동일 Flickr 사진, train/holdout decoded-pixel 중복을 제외한다. OI 공식 train/validation/test 분할을 유지한다.
- 공식 COCO 범주에는 원본 COCO box만 사용한다. LVIS는 COCO 직접 범주 이외의 호환 물체 라벨을 보완한다. COCO/LVIS의 겹친 box가 동일 물체의 정답으로 중복되지 않도록 했다.
- 생성 코드: `datasets/prepare_focus8_openimages.py`, `datasets/build_focus8_coco_oi.py`, `datasets/clean_focus8_labels.py`. 최종 manifest: `data/processed/public_v6_focus8_coco_oi_15/manifest.jsonl`.
- [Open Images 공식 다운로드](https://storage.googleapis.com/openimages/web/download_v7.html), [COCO 공식 다운로드](https://cocodataset.org/#download).


## 2026-10-15 요청 8종 추가 학습 데이터

- 구성: `data/processed/public_v5_focus8_15`; 재현 코드 `datasets/build_focus8_v5.py`, `datasets/weight_focus8_v5.py`.
- 기존 15종 retention 자료 + 로컬 EgoObjects replay 8,667장 + COCO 2017 신규 train 2,818장.
- COCO 공식 annotation archive: `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`.
  원본 `instances_train2017.json` / `instances_val2017.json` 및 selected image 파일을 보존했다.
- 공식 COCO image `license` ID 4/5/7/8만 추가 후보로 허용했다. 각 이미지의 license 항목, coco_url, flickr_url, image_id와 원본 split은 manifest에 보존했다.
- COCO의 bottle/cup/cell phone/remote/spoon/fork/toothbrush/sports ball을 대응 클래스에 매핑하고, 동일 이미지의 호환되는 LVIS 주석도 병합했다.
- 기존 데이터에 등장했던 모든 COCO/LVIS image ID와 target crowd 이미지를 신규 후보에서 제외했다. 신규 COCO 평가 사진 261장은 train과 분리했다.
- decoded-pixel 기준 중복 제거 후 train 고유 사진 39,724장. 칫솔 양성 가중치 2배 적용으로 학습 파일 항목은 42,287개이며 반복 2,563개를 고유 사진으로 세지 않는다.
- provenance: `data/processed/public_v5_focus8_15/manifest.jsonl`, `build_report.json`, `sampling_weights.json`.

## 2026-10-15 EgoObjects 잔여 학습 이미지 확인

- 변환 train의 EgoObjects 49,830장 중 현재 균형셋/추가 학습셋에 포함된 실제 원본은 11,256장이다. 빠진 38,574장이라는 수치는 파일명·원본 경로 모두 일치한다.
- 초기 YOLO26s는 전체 변환 train으로 학습했고, 이후 추가 학습에서 클래스별 최대 1,000개 후보의 합집합으로 축소했다. 처음 보는 데이터 38,574장이라는 뜻은 아니다.
- 상자 14,441 / 펜 3,873 / 리모컨 3,304 / 휴대폰 1,962 / 열쇠 1,972장 등의 잔여 후보가 있다. 칫솔·카드·동전·작은 블록은 추가 양성 0장이다.
- 현재 train에서 완전히 빠진 원본 촬영 그룹은 20개(317장), 영상 조합은 1,008개다. 기존 그룹의 반복 프레임을 줄여 선별·순환 학습하는 경로를 우선한다.
- 상세 근거와 수정한 우선순위: [EGOOBJECTS_REMAINDER_AUDIT_20261015.md](EGOOBJECTS_REMAINDER_AUDIT_20261015.md).

## 2026-10-14 나머지 13종 공개 데이터 보강 조사

[ALL15_OPEN_DATA_RESEARCH.md](ALL15_OPEN_DATA_RESEARCH.md)에 출처·라이선스·실제 후보 수와 다음 학습안을 기록했다.
현재 train의 카드 158개 파일 항목은 decoded pixel 기준 80장, 작은 블록 72개 항목은 36장이다.
초기 LVIS supplement와 후속 supplement의 중복이 남아 있어 파일 수를 독립 장면 수로 계산하지 않는다.
미사용 LVIS 원본의 box 693 / cellular_telephone 776개 이미지 후보와 Open Images 10종 후보를 집계했다.
COCO 8종, Keys Data v1, Lego v2, MIDV 카드 경계 자료를 추가 후보로 확인했다.
이번 조사에서 새 학습·데이터 편입·모델 교체는 하지 않았다.

## 2026-10-14 캔·동전 공개 데이터 보강 조사

자체 데이터 수집 대신 공개 bbox 데이터를 사용하는 보강 계획을
`reports/CAN_COIN_OPEN_DATA_PLAN.md`에 기록했다. Open Images의 Tin can/Coin,
RF100 coins-1apki를 확인했고, LVIS 변환에서 빠진 beer_can/penny_(coin)을 추가했다.
RTX 4070 Ti SUPER에서 FP32 7+1 epoch 추가 학습과 Jetson FP16 검증·배포를 완료했다.
실제 사용 데이터, 고정 검증셋 및 결과는 [학습 결과](CAN_COIN_OPEN_DATA_RESULT.md)에 기록했다.

### Open Images / RF100 실제 확보

- Open Images 공식 bbox CSV: v6 train, v5 validation/test. Tin can → 1, Coin → 13.
  같은 이미지의 호환되는 병·컵·상자·공·휴대폰·리모컨·펜·숟가락·포크·칫솔 주석도 보존했다.
- 사용 이미지 2,274장(원본 split 유지), CVDF 공개 S3에서 다운로드했다.
  공식 메타데이터로 2,274장 모두 CC BY 2.0 라이선스, 원본 URL, 저자를 확인했다.
  `data/raw/can_coin_open_v1/oi_image_attribution.json` 참조. 주석 라이선스 CC BY 4.0.
- RF100 coins-1apki 원본 COCO archive: 8,419장, CC BY 4.0. 아카이브 README에 출처와 라이선스가 있다.
  미러 `Francesco/coins-1apki`, revision `ec1591c4dd8fd3f93a2060a5a5a2a181c8d4e190`,
  SHA-256 `0bbf2cc3299f1cb9b7595f0f41cdda867864800ad0bef987c655c98205b8245f`.
- RF100은 원래 split 간 같은 영상 프레임이 섞여 있어 재분할했다. 숫자 timestamp 영상은
  train에서 100ms당 1장만 선택하고, 원래 val/test의 같은 영상 프레임은 제외했다.
  버스트는 촬영 분 단위 그룹, 일반 사진은 인접 파일번호 30개 단위 그룹으로 분할했다.
  일반 사진은 전부 holdout으로 사용했다. 최종 RF100 1,208장.
- RF100의 coin만 13번으로 변환했다. nail/nut/screw는 동전 음성 예시이며,
  `coins` 상위 분류를 모든 동전 박스로 잘못 변환하지 않았다.
- 통합 supplement는 train 2,520장, val 367장, test 595장이다.
  `data/supplements/can_coin_open_v1/manifest.jsonl`에 출처·그룹·원본 split·좌표·파일 checksum을 보존했다.
- TACO는 이번 학습에 사용하지 않았다.

## EgoObjects v1

- 공식 프로젝트: https://github.com/facebookresearch/EgoObjects
- 공식 다운로드 페이지: https://ai.meta.com/datasets/egoobjects-downloads/
- 공식 저장소 명시 규모: annotated frame 약 114K, image archive 약 40 GB, 368 categories
- 라이선스: 공식 저장소의 MIT License
- 2026-08-07 상태: 공식 다운로드 view가 빈 component를 반환함
- 복구 미러: `Myszz/EgoObjects`, revision
  `b9b6b37cddcddf28b467f657edc81b7824eb906d`
- 미러는 제3자 경로이므로 파일 크기와 SHA-256을 `download_manifest.json`에 고정하고,
  공식 파일을 다시 확보할 수 있으면 교차 검증해야 한다.

## LVIS v1 / COCO supplement

- LVIS 공식 사이트: https://www.lvisdataset.org/
- annotation host: `dl.fbaipublicfiles.com/LVIS`
- image host: `s3.amazonaws.com/images.cocodataset.org`
- 사용 목적: EgoObjects taxonomy에 없는 `card`, `small_block` bootstrap
- mapping: card/business_card/identity_card/keycard → card; Lego/cube → small_block
- 생성 보고서: `reports/lvis_supplement.json`

### 2026-08-12 전체 15-class 보강

- 공식 LVIS v1 train annotation과 COCO train 이미지에서 목표 15개 class 전체를 다시 추출했다.
- 추가 train 이미지 8,320장, bbox 26,591개를 사용했다.
- mapping과 class별 bbox 수: `data/supplements/lvis_all15_v2/lvis_supplement_report.json`
- EgoObjects의 기존 val/test는 변경하지 않았고, 공개 데이터 보강분은 train에만 넣었다.
- 학습용 집중 구성은 기존 train에서 class별 최대 1,000장을 뽑은 11,372장과 LVIS/COCO
  8,320장을 합친 총 19,692장이다. `data/processed/public_v2_focus_15/build_report.json`에 기록했다.

## 계획서 기재 데이터 사용 상태 (2026-08-18)

| 데이터 | 계획상 역할 | 이번 detector 추가 학습 사용 여부 | 사유 |
| --- | --- | --- | --- |
| EgoObjects v1 | 주 공개 detection 데이터 | 사용 | 기존 15-class checkpoint 및 집중 재학습의 기반 |
| 자체 AR Glass | 최종 domain adaptation | 미사용 | 아직 수집 데이터 없음 |
| Human Grasping Database for ADL | grip mapping 참고 | 검출 보강에도 제한 사용 | 모호한 통합 label은 제외하고 5개 신뢰 class의 bbox만 train에 사용 |
| EPIC-KITCHENS | 선택적 eye-view detection 보조 | 미사용 | 필수가 아니며 공식 다운로드가 학습 시간 내 확보되지 않음; CC BY-NC 4.0 |
| HANDS | 선택적 grip mapping 검증 | 미사용 | detector 학습용이 아니며 안정적 다운로드를 필수로 두지 않음 |
| Ego4D | 향후 FHO 후보 | 미사용 | 계획서가 초기 MVP에서 명시적으로 제외 |

따라서 계획서의 데이터를 모두 detector 학습에 쓰는 구조가 아니며, 최종 공개 데이터 모델에는
`EgoObjects + LVIS/COCO + ADL 1인칭 5-class 균형 subset`을 사용한다. EPIC-KITCHENS annotation 일부는 조사 목적으로 받았지만
이미지를 확보하지 못했으므로 어떤 checkpoint에도 포함되지 않았다.

## YOLO26s

- Ultralytics 공식 asset release의 `models/pretrained/yolo26s.pt`
- SHA-256: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`
- COCO pretrained 가중치는 Phase 1 baseline 및 Phase 3 초기화에 사용한다.
- Ultralytics 8.4.115에서 default end-to-end/NMS-free head가 활성화된 것을 실제 확인했다.
- 학습은 FP32, 최종 Jetson 배포는 TensorRT FP16으로 수행한다. YOLOv8n/INT8 결과는 과거
  실험 이력으로만 보존하고 새 모델의 승인 근거로 사용하지 않는다.

## ADL Dataset (2026-08-13 추가)

- 공식 페이지: https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/
- 공식 1인칭 ADL 영상 20개, 43,087,304,390 bytes를 모두 확보했다.
- `ffprobe`로 20개 모두 1280x960 영상이며 annotation의 마지막 frame까지 존재함을 확인했다.
- 공식 bbox는 640x480 좌표계이므로 정규화 시 annotation canvas를 사용했다.
- 신뢰 가능한 목표 클래스만 변환: 14,916장, bbox 27,192개.
- bbox: bottle 10,310, cup 11,050, phone 1,224, remote 2,813, toothbrush 1,795.
- 원본의 `knife/spoon/fork`는 하나의 통합 label이라 프로젝트의 fork/spoon으로 오염 없이
  분리할 수 없어 제외했다.
- 전체 투입 실험은 반복 bottle/cup frame 때문에 val 성능이 하락했다. 최종 학습에는 매 두 번째
  annotation을 유지하고 class별 최대 2,000개 frame 후보를 체계 표본 추출한 6,112장만 사용했다.
- 모든 ADL 데이터는 train에만 사용했으며 기존 EgoObjects val/test는 변경하지 않았다.
- 변환 보고서: `data/supplements/adl_v1_5class/adl_supplement_report.json`
- 균형 표본 보고서: `data/supplements/adl_v1_balanced_v2/adl_balanced_report.json`
