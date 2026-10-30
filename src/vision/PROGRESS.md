# VAGT-ORCA AI 작업 기록

최종 갱신: 2026-10-16 KST

현재 기본 설정: **추가학습하지 않은 공식 YOLO26m**, Jetson native `yolo26m_coco_original_fp16.engine`. 원본 COCO80종 가중치를 유지하고 책상·가구·사람 등 배경37종을 제외한 물체43종을 표시한다. 중앙 타깃은 그립 지원 물체 중에서 선택한다. 병·휴대폰·포크는 현재 시연 후보이며 표시 제한이 아니다. 원본 80종 가중치/출력층을 유지하고 프로젝트 지원 8종의 이름/ID만 연결한다. [선정·검증·배포](reports/ORIGINAL_DEMO_DEPLOYMENT_20261016.md).
현재 카메라 상태: **사용자의 종료 지시로 정지했다.** 서비스 inactive/dead, MainPID=0, 카메라 장치 점유 없음, 로그인 자동 시작 disabled. 캔·펜 시험 override를 백업 후 제거했고 다음 수동 실행은 공식 원본 YOLO26m만 사용한다. 프레임·영상 자동 저장 없음. 최신 확인: `runs/can_pen_specialist_20261016/camera_stopped_original_selected.json`.
실사용 판정: 원본 m은 사용자 관찰 시험 진행 중이며 정확도 합격은 미판정이다. 이전 v8과 캔·동전 복구 모델은 ELP 오탐으로 불합격이었다. 해당 가중치·실험 기록·복구 백업을 보존한다. 공개 AP 개선을 실사용 합격으로 간주하지 않는다. [학습 이력과 원인 분리 기록](reports/FALSE_POSITIVE_AUDIT_20261015.md).
8종 공개 평가 이력(실사용 불합격): [FOCUS8_MAX_RESULT_20261015.md](reports/FOCUS8_MAX_RESULT_20261015.md), [프로필 실행·복구](docs/FOCUS8_PROFILES.md).
이전 v5 배포 이력: [FOCUS8_JETSON_DEPLOYMENT_20261015.md](reports/FOCUS8_JETSON_DEPLOYMENT_20261015.md).
이전 배포 성능(이력): [PER_CLASS_PERFORMANCE_20261014.md](reports/PER_CLASS_PERFORMANCE_20261014.md).
GPU 추가 학습: [기존 8종 보강 결과](reports/FOCUS8_FINETUNE_RESULT_20261015.md), [후속 집중 실험](reports/FOCUS8_TARGETED_RESULT_20261015.md).

## 2026-10-16 캔·펜 시험 종료, 공식 원본만 사용

- 사용자 결정: 캔·펜은 사용하지 않고 추가학습하지 않은 공식 원본 YOLO26m을 유지한다. 카메라를 종료했다.
- 캔·펜 별도 모델의 공개 native test: 캔 정밀도85.7%/재현율42.6%, 펜64.3%/4.7%. 실사용 합격으로 판정하지 않았다. 원본 가중치·기본 runtime·엔진 checksum은 보존했다.
- 실제 A/B 표본에서 원본 약15.2~18.1 FPS/p95 48.46~49.33ms, 추가 모드 약10.9~12.1 FPS/p95 66.51~67.99ms. 보수적 threshold에서 화면의 캔·펜이 검출되지 않은 장면을 확인했다. 일반 정확도 평가로 해석하지 않는다.
- 사용자가 overcurrent 경고를 알렸다. OC3 누적26회, OC1/OC2 각각0회. 측정 전 기준값이 없어 발생 시점이나 이번 작업과의 인과는 미확정이다. 기본 제공 DC 어댑터 사용, 전력 모드25W. 보호 설정·전력 모드·클럭은 변경하지 않았다.
- 서비스 stop/disable, 시험 drop-in 백업·제거, daemon-reload를 완료했다. inactive/dead, MainPID=0, 장치 점유0개, 다음 실행은 기본 원본 설정이다. 종료 후 OC3 누적26회였다. 60초 관찰 스크립트는 실행하지 않았다.
- 시험 가중치/평가 기록은 이력으로 보존하며 사용하지 않는다. [시험 결과와 종료 기록](reports/CAN_PEN_SPECIALIST_TEST_20261016.md).

## 2026-10-16 배경 과다 표시·책상 타깃 선택 수정

- 사용자가 책상까지 표시·선택되는 문제를 지적했다. `excluded_classes`로 책상(dining table)·의자·사람·고정 가전·동물·차량 등37종을 제외하고, 휴대/조작 물체43종을 표시한다. 시연 후보3종 제한은 적용하지 않는다.
- `target.grip_supported_only: true`를 추가해 표시와 타깃 선택을 분리했다. 책/가위 등은 표시할 수 있고, 중앙 타깃은 그립 매핑이 있는 물체에서만 고른다. 원본 공통8종인 병·컵·공·휴대폰·리모컨·숟가락·포크·칫솔이 해당한다.
- 선택기와 안정화기 양쪽에 동일한 적격 검출 목록을 전달한다. 배경/미지원 물체가 더 중앙에 있어도 타깃을 빼앗거나 기존 타깃으로 재등장하지 않도록 검사했다.
- detector는 backend의 classes 필터와 결과 재검사 모두에 배경 제외를 적용한다. 원본 가중치/엔진/confidence0.35는 유지한다. CLI/라이브/benchmark 경로에 같은 타깃 정책을 반영했다.
- 관련50검사와 Ruff 통과. Jetson 이전 상태는 `runs/original_demo_20261015/before_object_focus_20261016/`에 백업 후 적용하고 서비스를 재시작했다. 최신 상태는 `object_focus_state.json`이며 시험은 사용자 종료 지시까지 유지한다.

## 2026-10-16 사용자 지적 반영: 시연 후보와 표시 범위 분리, 원본 80종 전체 표시

- 사용자 의도는 좋은 3~5종을 골라 시연하는 것이며 검출 화면을 해당 종류로 제한하는 것이 아니었다. 잘못 적용한 3종 필터를 제거했다.
- 원본 80종을 모두 추론/표시한다. `cell phone`→`phone`, `sports ball`→`ball` 및 프로젝트 공통 8종의 ID만 연결한다. 프로젝트 외 물체 ID는 100+COCO ID로 구분해 person=0과 bottle=0 등의 타깃 정체성 충돌을 방지한다.
- 그립 매핑 없는 물체는 검출/타깃 표시가 가능하며 기존 planner의 unknown_class 처리에 따라 새 그립을 만들지 않는다. 모델/엔진/confidence는 유지했다.
- 제한 해제 전 설정·코드 백업은 Jetson `runs/original_demo_20261015/before_full_display_20261016/`에 보존했다. 관련 adapter/deployment/pipeline 13검사, grip/stabilizer/packet 35검사와 Ruff 통과.
- 서비스를 한 번 재시작해 적용하고 실제 브라우저 스트림을 재연결해 시계·냉장고 등 기존 3종 외 물체 표시를 확인했다. 카메라는 사용자 종료 지시까지 계속 실행하며 자동 저장 없음. 최신 상태는 `all_class_display_state.json`에 기록한다.

## 2026-10-16 원본 YOLO26m 선정·배포와 최초 3종 필터 시험 이력

- 사용자 결정: 15종 시연은 하지 않는다. 성능이 좋은 3~5종만 골라 추가학습하지 않은 공식 YOLO26을 사용한다. 이전 15종/4그립 전체 지원 조건을 이번 시연의 합격 조건으로 적용하지 않는다.
- 주력은 포크·리모컨·휴대폰 3종. 공은 추가 공개 기능 검사에서 미검출이 있어 예비 후보로 둔다. 공식 원본 m의 COCO 80종 가중치·출력층과 checksum을 보존한다. 새 학습 없음.
- Jetson native FP16의 같은 COCO 125장/OI 324장 평가에서 m의 3종 AP가 모두 s보다 높았다. 고정 confidence 0.35의 정밀도가 모든 항목에서 향상된 것은 아니다. 모델·물체 선택에 이 공개 평가 결과를 사용했으므로 새로운 blind test로 부르지 않는다.
- 후보 4종 프로필 동일 공개 benchmark: s 40.10 FPS/p95 29.26ms, m 36.40 FPS/p95 34.80ms. 최종 3종 프로필·중앙 공개 사진 23장/200회는 m 26.26 FPS/p95 39.88ms였다. 이미지 구성 차이로 속도 차이를 클래스 수 효과로 해석하지 않는다. 카메라/화면 전송은 제외, 메모리 UART 직렬화 포함, 클럭 비고정.
- 최종 native 엔진으로 공개 23장×5회 검출·5/4 안정화·그립 제안·메모리 UART 패킷 해독을 통과했다. 원본80/CUDA/end2end 확인. 선별 정지 사진 반복은 정확도나 실제 시간 안정성 평가로 사용하지 않는다.
- 기존 코드·설정·unit을 로컬/Jetson에 백업한 뒤 runtime/deployment/서비스와 원본 s/m 선택 프로필을 동기화했다. 활성 모델은 m, confidence 0.35/square640/FP16/end2end, focus/crop/specialist 및 캔·동전 전용 threshold 없음.
- 사용자 요청으로 15:49 KST 카메라를 시작하고 실제 브라우저 영상 표시를 확인했다. 1280×720 YUYV 요청60FPS, 실제 뷰어 약17~18FPS. 15:51 상태의 pipeline p95는 48.94ms이며 센서부터 표시까지 지연은 측정하지 않았다. 최초 화면이 매우 흐려 렌즈 가림/초점 확인을 안내했다.
- 로컬 전체 검사 81통과/5 GPU opt-in 생략, 전환 후 adapter/deployment/pipeline 관련 11검사 및 변경 Python Ruff 통과. 실제 CUDA/native smoke는 별도 통과했다. 결과·백업은 `runs/original_demo_20261015/`에 있다.

## 2026-10-15 원본 YOLO26의 안경 시점 적합성 판단

- 판단: 지원 물체에 한정한 안경 시점 검증용 기준 모델로 원본 s/m은 타당한 후보. 목표 15종·4종 그립의 최종 단독 인식기로는 부족하다. 원본의 실제 ELP 중앙 타깃/오탐/지연 합격은 아직 미확인이다.
- EgoObjects는 실제 스마트안경을 포함한 1인칭 데이터여서 간접 근거가 된다. 공개 전체 화면 AP/P/R은 프로젝트의 중앙 타깃 선택 정확도 95% 목표나 Jetson p95 50ms 목표를 검증하지 않는다.
- 포크 m은 정밀도/재현율 87.2/79.9%로 유망. 병/컵 정밀도 약 50%, 휴대폰·리모컨·숟가락 미검출, 칫솔 정답 6개의 한계가 남는다. 미주석 물체가 FP로 집계될 가능성을 고려한다.
- 원본 미지원 7종 때문에 PINCH와 LATERAL_PINCH 대상이 빠진다. 지원 8종 출력 필터·COCO 이름 연결은 추가학습 없이 가능하나 현재 설정의 가중치 경로만 바꾸는 것으로는 충분하지 않다.
- 추가학습보다 원본 고정 상태에서 실제 장착 시점 검증을 먼저 권고했다. 이번 작업은 기록/판단만 수행했고 카메라·가중치·배포·임계값은 변경하지 않았다. [상세 판단](reports/YOLO26_ORIGINAL_GLASS_VIEW_ASSESSMENT_20261015.md).

## 2026-10-15 원본과 캔·동전 / 마지막 추가학습 모델 동일 조건 비교 완료

- 사용자 지적에 따라 원본 수치만 제시한 누락을 보완했다. 캔·동전 retention 및 마지막 refined_v8 s 단독 모델을 동일 6,552장·CUDA FP32/640 square/b16/end2end/단일 전체 화면으로 새로 평가했다. 고정 confidence 0.35의 TP/FP/FN도 다시 계산했다.
- 공통 요청 7종(원본에서 미지원인 펜 제외) 평균 AP50–95×100, 순서 원본 s / 캔·동전 / 마지막 추가학습 s / 원본 m: frozen **34.23 / 25.01 / 25.85 / 41.91**, COCO **42.76 / 19.44 / 22.29 / 49.47**, OI **64.68 / 29.23 / 36.74 / 69.42**.
- 마지막 실제 배포는 s 단독이 아닌 원본 m 전문 모델+추가학습 s+중앙 확대 조합이다. 공통 7종×3영역의 최종 조합과 원본 m AP 차이는 최대 0.18포인트였다. 큰 공개 AP 향상을 추가학습 자체의 효과로 해석하면 안 된다.
- 정밀도 변화는 클래스별로 다르다. frozen 리모컨은 80.9 / 63.8 / 68.8%, 펜은 캔·동전 54.3 → 추가학습 s 46.7%. 컵은 정밀도가 높아졌지만 재현율이 감소했다. AP 저하를 모든 클래스의 오탐 증가와 동일시하지 않는다.
- 원본 대비 공통 클래스 성능 저하를 확인했다. 어떤 학습 단계/라벨/도메인 차이가 주원인인지는 미확정이다. 새 학습·배포·카메라 사용 없음. 6회 GPU 평가/정답 합계/checksum 및 두 파일 Ruff 통과. [물체별 비교와 정밀도·재현율](reports/YOLO26_ORIGINAL_VS_FINETUNED_20261015.md).

## 2026-10-15 추가학습하지 않은 공식 YOLO26 원본 GPU 평가 완료

- 공식 YOLO26s·YOLO26m의 80종 출력층과 가중치를 그대로 유지하고 RTX 4070 Ti SUPER CUDA FP32/640/단일 화면/end2end로 평가했다. checkpoint 전후 SHA-256 일치를 확인했다.
- 공개 test 6,552장씩: 기존 frozen 6,103(EgoObjects 6,085 + LVIS 18), COCO subset 125, Open Images 324. 공통 8종 정답 ID만 COCO ID로 변환했으며 원본 사진·라벨은 수정하지 않았다.
- 요청 8종 중 펜을 제외한 7종 평균 AP50–95×100: s는 frozen 34.23 / COCO 42.76 / OI 64.68, m은 41.91 / 49.47 / 69.42. 서로 다른 영역의 점수를 합산하지 않았다.
- confidence 0.35·IoU 0.5에서 TP/FP/FN을 다시 매칭해 정밀도·재현율도 계산했다. frozen 병/컵 정밀도는 s와 m 모두 약 50%이며 m이 모든 클래스의 오탐을 줄이지는 않았다. 미주석 물체가 FP로 집계될 가능성을 고려해야 한다.
- 펜·캔·동전·상자·카드·열쇠·작은 블록은 원본 클래스 미지원이다. frozen 칫솔은 정답 6개에서 AP 0이며 다른 두 공개 영역에서는 검출되므로 칫솔 전체 인식률 0으로 해석하지 않는다.
- 학습·모델 배포·임계값 변경·카메라 사용 없음. 카메라는 정지 상태를 유지한다. 새 blind test 또는 ELP 실측 결과는 아니다.
- 6회 평가의 이미지 수·정답/TP/FN 합계·CUDA/end2end·원본 checksum 검증 및 두 Python 파일 Ruff 검사 통과. [물체별 전체 수치와 평가 조건](reports/YOLO26_ORIGINAL_BASELINE_20261015.md).

## 2026-10-15 복구 모델 실시간 시험도 오탐으로 중지

- 사용자가 실제 ELP 화면에서 캔·동전 복구 모델도 오탐이 많다고 확인했다. 서비스·장치·영상 터널을 종료했다. 원인 확인을 위한 추가 촬영이나 학습은 시작하지 않았다.
- 학습 metadata/code 확인: can_coin_v4와 can_coin_retention 모두 전체 모델 학습이며 freeze 설정이 없다. “캔·동전 성능 개선”이 두 클래스 가중치만 변경했다는 의미는 아니다.
- 복구 모델 역시 여러 차례 추가학습된 15종 모델이다. 공식 COCO 사전학습 원본과 구별해야 한다.
- 과적합, 기존 인식 능력 손상, 라벨/도메인 차이, 변환/후처리 문제 중 무엇이 주원인인지는 미확정이다. 공통 클래스에서 공식 원본·프로젝트 PyTorch·Jetson engine을 같은 입력으로 비교해야 한다.

## 2026-10-15 캔·동전 개선 모델로 롤백 완료

- 사용자가 8종 개선을 실패로 판단하고 캔·동전 개선 모델로 복구를 요청했다. `yolo26s_can_coin_retention.pt` 및 해당 Jetson FP16 engine의 SHA-256이 10월 14일 기록과 일치한다.
- `configs/runtime.yaml`은 당시 runtime_candidate와 같은 내용으로 복구했다. 캔 0.75·동전 0.25·나머지 0.35, 640 입력 및 기존 안정화 설정을 사용한다. focus_profile 없음.
- deployment 모델 경로와 서비스 ExecStart도 can_coin_retention으로 복구했다. 서비스의 40/50 focus8 drop-in은 byte 백업 후 제거했다. 가중치와 실험 기록은 보존한다.
- Jetson CUDA:0에서 기존 공개 benchmark 이미지 5장으로 실제 TensorRT pipeline 동작 확인. focus_profile 비활성 확인. 설정/클래스 기준 회귀검사 6개 통과.
- 카메라는 이번 복구 중 열지 않았다. 서비스 inactive/dead, MainPID=0, disabled. 이 모델의 실카메라 오탐·과적합 여부는 아직 판정하지 않았다.
- 로컬·Jetson 백업 및 검증 JSON은 `runs/can_coin_rollback_20261015/`에 있다.

## 2026-10-15 ELP 실시간 시험 중단 — 펜 오탐

- 사용자의 명시적 요청으로 실제 화면을 열고 v8 정확도 우선 프로필을 실행했다. 임의 시간 제한 없이 사용자의 종료 지시까지 실행했다.
- 1280×720 YUYV/요청 60 FPS, 상태 확인 시 870프레임·읽기 실패 0·화면 포함 약 10.25 FPS. 이 수치는 인식 정확도가 아니다.
- 사용자가 펜이 아닌 물체가 펜으로 검출되는 문제를 지적했다. 즉시 서비스와 로컬 SSH 영상 터널을 종료했다. 서비스 inactive/dead, MainPID=0, disabled; 프로젝트 카메라 프로세스 없음.
- 기존 모델 선택은 공개 AP와 보존 기준 중심이었다. 배포 confidence=0.35에서의 펜 오탐 허용량과 실사용 비대상 물체 검증이 합격 조건에 없었다.
- 펜은 COCO 전문 모델의 담당 클래스가 아니며 추가학습 모델의 전체/중앙 확대 검출을 사용한다. 과적합·도메인 차이·확대 효과의 기여는 아직 분리 확인하지 않았다.
- 실사용 합격 판단을 철회하고 문제를 기록했다. 이번 중단 뒤 추가 촬영이나 학습을 시작하지 않았다.

## 2026-10-15 요청 8종 정확도 우선 v8 공개 평가 완료 (실사용 불합격)

- RTX 4070 Ti SUPER CUDA에서 BN 통계/box/backbone을 고정하고 분류 행 6 epoch 및 분류 특징 4 epoch를 비교했다.
- 공개 원본 12,150장 + 기존 train crop 780개 재사용. 새 카메라 프레임 및 새 raw 공개 사진 0장.
- 세 validation 영역에서 classifier 마지막 epoch를 선택하고, COCO 공식 s/m 원본 전문 모델과 단일/중앙/4방향 확대·route/union을 비교했다.
- 최종은 7종 COCO m 전문 모델 + 펜/기타 클래스 추가 학습 s 모델 + 중앙 확대. 작은 칫솔 32/64px 보존 규칙은 오탐으로 탈락했다.
- 동일 FP32 square640/b16 공개 test의 물체별 영역 균등 8종 평균 AP50–95×100: 27.09→49.96. 후보 선택에 test 미사용.
- pen 개선 폭이 작고 frozen 극소형 칫솔은 test AP 0 유지. 모든 영역/물체가 동시에 개선된 것은 아니다.
- Jetson native FP16/640/b1 두 engine 및 전체 공개 validation 완료. 공개 pipeline FPS: 기존 29.14, 고속 27.76, 정확도 우선 13.56. 카메라/화면 전송/실제 UART 없음.
- 두 프로필과 50번 service drop-in 적용, 이전 B engine·설정·40번 drop-in 보존. 카메라 서비스는 정지 상태로 유지했다.
- TensorRT 출력 버퍼 재사용으로 전체 화면 결과가 덮어써지지 않게 보존했다. 실제 CUDA 병합 검사 5개 및 기존 회귀검사 20개, Ruff, native systemd 문법 검사 통과.

## 2026-10-15 요청 8종 집중 실험 및 Jetson 모델 적용 완료

- 전원 인가 후 SSH 확인: Jetson 서비스는 active였으나 여전히 캔·동전 모델을 사용했고 직전 8종 보강 checkpoint는 없었다.
- RTX 4070 Ti SUPER에서 12,150장 원본 + train crop 780개로 multi-scale 640–960, freeze5, FP32 2 epoch(6.6분) 추가 실험했다.
- raw 2개·가중치 보간 2개·BN 통계 대조 1개가 모두 validation 퇴행으로 탈락했다. 이 v7 후보는 배포하지 않고 test 선택도 하지 않았다.
- 직전 검증 모델 `yolo26s_focus8_v5.pt`를 Jetson에서 native TensorRT FP16/static640/batch1/workspace1GiB로 변환했다.
- 동일 Jetson validation 대상 평균 AP50–95×100: frozen 8종 28.04→28.57, COCO 7종 24.36→24.82, Open Images 8종 24.24→32.87.
- 모든 물체가 모든 영역에서 개선된 것은 아니다. frozen 리모컨·숟가락·포크 소폭 하락, COCO 칫솔 55.01→45.65. Open Images 칫솔 3.78→17.89이나 아직 약하다.
- 공개 can/coin validation 캔 65.26→65.58, 동전 66.54→67.58. 전체 15종 및 기존 클래스 유지 기준 통과.
- 공개 사진 24장/200회 pipeline+UART 직렬화: 31.06→27.45 FPS, p95 36.31→40.70ms. 실제 UART 송신 없음, 클럭 비고정.
- 새 engine SHA-256 `748ec009527b76679afab7a9e875c2099aad9feb916c6337cfb981c984d9f26a`.
- 새 모델로 서비스 재시작 후 326프레임, 읽기 실패/자동 재시작 0회. 상태 JSON만 확인했으며 카메라 이미지는 조회·수집하지 않았다.
- 이전 engine과 원래 설정 byte 백업을 보존했다. 로컬 runtime·deployment·service 기본 경로도 동기화했다.
- 관련 기존 회귀검사 12개 및 변경 Python Ruff 검사 통과. 상세 수치/복구는 위 최신 Jetson 기록 참조.

## 2026-10-15 COCO·Open Images 요청 8종 GPU 추가 학습 완료

- 대상: 병·컵·휴대폰·리모컨·숟가락·포크·칫솔·펜. RTX 4070 Ti SUPER / CUDA FP32에서 학습·평가 완료.
- 최종 데이터 `public_v6_focus8_coco_oi_15`: 기존 고유 train 28,239 + EgoObjects replay 8,667 + COCO 2,818 + 신규 Open Images 3,695 = 고유 43,419장. 칫솔 2배 표본 가중치 포함 train 항목 46,078개.
- 기존 val 6,927 / test 6,698개 파일·라벨 byte/path 보존, train/holdout decoded-pixel 중복 0건. 신규 COCO val/test 136/125, OI val/test 248/324는 별도 평가.
- 첫 COCO/Ego 6-epoch 실험은 채택하지 않았다. COCO 범주의 중복 LVIS box를 정리하고 기존 모델 backbone 고정·AdamW 3e-5·3 epoch로 다시 학습했다.
- 최종 후보 `artifacts/models/yolo26s_focus8_v5.pt`, SHA-256 `c60b573217426a0f600cbc7d2dea9fc67a9708ce53e4ce46fbc7248d9c794439`.
- 동일 batch 16 GPU validation의 요청 8종 평균 AP 28.84→29.52로 epoch 3 선택. test는 후보 선택에 쓰지 않았다.
- 최종 test 평균 AP50–95: 기존 frozen 8종 24.22→24.75, 신규 COCO 7종 21.13→23.45, 신규 OI 8종 26.68→35.66. 각 테스트의 범주/사진 수가 다르므로 서로 섞어 평균하지 않는다.
- 기존 frozen에서 병·컵·휴대폰·숟가락·펜 개선. 리모컨/포크는 소폭 하락, 칫솔은 6개 정답에서 AP 0 유지. 신규 OI 칫솔 AP 13.10→18.80으로 개선됐으나 여전히 약하다.
- 공개 can/coin test AP: 캔 65.01→62.80, 동전 78.28→77.67. validation 유지 기준은 통과했지만 test 하락도 보고했다.
- 최종 학습 16.6분, 첫 실험 포함 순수 학습 합계 58.7분. GPU 평가, 데이터 무결성 검사 및 변경 Python Ruff 검사 통과.
- 해당 학습 단계에서는 기존 Jetson engine/설정을 변경하지 않았다. 후속 배포는 상단의 최신 적용 항목 참조.
- 결과/모든 물체 수치/재현: [FOCUS8_FINETUNE_RESULT_20261015.md](reports/FOCUS8_FINETUNE_RESULT_20261015.md).

## 2026-10-15 EgoObjects 잔여분 감사 완료

- 사용자 제시 49,830 / 11,256 / 38,574장 수치를 파일명·symlink 원본 경로 기준으로 확인했다. 현재 can/coin retention 학습셋의 EgoObjects 원본도 같은 11,256장이다.
- 초기 YOLO26s 전체 EgoObjects 학습 기록은 17 epoch이며, 후속 fine-tuning에서 고정 subset으로 축소했다. 잔여분 전체가 모델이 처음 보는 데이터라는 의미는 아니다.
- 상자 14,441 / 펜 3,873 / 숟가락 3,771 / 리모컨 3,304 / 휴대폰 1,962 / 열쇠 1,972장 잔여 후보. 칫솔·카드·동전·작은 블록은 추가 양성 0장이다.
- 현재 포함 630/650개 촬영 그룹. 완전히 빠진 20개 그룹의 사진은 317장이고, 빠진 영상 조합은 1,008개다. 남은 프레임 대부분은 기존 그룹의 추가 장면이다.
- 원본 경로/group_id 기준 현재 val/test 중복은 0건. 콘텐츠 hash·프레임 유사도 검사는 미실시다.
- 외부 다운로드 전에 잔여 EgoObjects를 클래스·촬영 그룹별로 선별/순환하는 실험을 우선한다. 새 학습·모델 교체·카메라 재시작은 실행하지 않았다.
- 근거 및 프레임 목록: [EGOOBJECTS_REMAINDER_AUDIT_20261015.md](reports/EGOOBJECTS_REMAINDER_AUDIT_20261015.md).

## 2026-10-14 나머지 13종 보강 조사 완료

- 현재 데이터/원본 Open Images·LVIS annotation을 직접 집계했다. 신규 학습·모델 교체는 하지 않았다.
- 카드 train 실제 고유 사진 80장, 작은 블록 36장. 초기/후속 LVIS supplement 중복이 있어 파일 항목 수보다 적다.
- 기존 train·holdout의 LVIS 원본 ID를 제외하면 box 693 / phone 776개 미사용 이미지 후보가 남는다.
- Open Images train 후보: box 2,150 / phone 3,965 / pen 951 / toothbrush 62장(다운로드·전체 검수 전).
- COCO 8종 및 카드·열쇠·Lego 전용 공개 데이터를 추천했다. source/class 균형과 15종 유지 검증을 포함한다.
- 출처·라이선스·수치·실행안: [ALL15_OPEN_DATA_RESEARCH.md](reports/ALL15_OPEN_DATA_RESEARCH.md).

## 2026-10-14 캔·동전 공개 데이터 GPU 보강 완료

- Open Images Tin can/Coin + RF100 coins-1apki 공개 supplement train 2,520 / val 367 / test 595장.
  촬영 그룹 분할, decoded pixel / pHash 중복 검사, 출처·라이선스·checksum manifest를 보존했다.
- LVIS beer_can/penny 누락을 66개 이미지에서 복원했다(기존 18, 추가 48).
- RTX 4070 Ti SUPER CUDA FP32에서 7 epoch 집중 학습 후 원래 train 전체를 복원한
  28,370장으로 1 epoch 보완했다. 최종 `artifacts/models/yolo26s_can_coin_retention.pt`.
- Jetson에서 TensorRT FP16 engine을 직접 빌드하고 ELP 카메라 서비스에 배포했다.
  당시 `artifacts/deployment/tensorrt/yolo26s_can_coin_retention_fp16.engine` 사용(최신 모델은 상단 참조).
- 같은 공개 test의 Jetson FP16 AP50–95: 캔 0.1622→0.6357, 동전 0.2069→0.6611.
  배포 임계값 can 0.75 / coin 0.25 / 나머지 0.35. 공개 test 동전 recall 97.7%, 캔 50.0%.
- 300회 pipeline+UART 직렬화: 30.97 FPS, 평균 32.29ms, p95 35.48ms(clocks 비고정).
  서비스 active, 재시작 0, 캡처 실패 0. 공개 이미지 4-frame 안정화 POWER/PINCH E2E 통과.
- 기존 13종 평균 FP32 test AP는 4.77% 감소, 휴대폰 0.4276→0.3310. 기존 frozen 캔·동전은
  여전히 약하다. 한국 동전 및 ELP 실사용 정확도는 아직 측정하지 않았다.
- 원래 checkpoint/engine checksum을 보존하고 Jetson `runs/can_coin_retention_20261014/predeploy`
  에 복구 프로필을 저장했다. 전체 pytest 77개 및 변경 파일 Ruff 검사 통과.
- 상세 결과·재현·복구 명령: [CAN_COIN_OPEN_DATA_RESULT.md](reports/CAN_COIN_OPEN_DATA_RESULT.md).

## 2026-10-14 ELP IMX291 카메라 전환

- Jetson `jetson-user@JETSON_HOST` SSH 접속 및 USB 3.0 UVC `15aa:1555` 인식 확인.
- 최신 프로젝트는 `/mnt/data/vagt-orca-ai`; 공유 `.venv`의 과거 editable import 경로를
  수정해 SSD 프로젝트 소스를 사용하도록 했다. 원래 경로와 변경 전 코드는 백업했다.
- `--source elp` 프로필 추가: 고정 by-id 경로, 1280×720, YUYV, 드라이버 60 FPS,
  V4L2 버퍼 1, 국내 조명 60Hz 설정. 카메라 모드 협상 결과를 검사한다.
- MJPG/YUYV × 720p/1080p 각각 180프레임, 총 720프레임 캡처 실패 0.
  실제 캡처 속도는 720p MJPG 24.22 / YUYV 24.40 FPS, 1080p MJPG 16.18 / YUYV 23.87 FPS.
  50/60Hz 설정과 수동 10ms 노출도 시험했으나 캡처 속도는 약 24 FPS로 유지됐다.
- 기존 YOLO26s TensorRT FP16 GPU 모델로 각 모드 120프레임 실시간 추론 실패 0.
  720p YUYV 카메라 read+pipeline 평균 24.40 FPS, p95 45.43ms;
  pipeline만 평균 28.66ms, p95 29.93ms. MJPG는 20.59 FPS / p95 58.14ms로 느려 YUYV를 채택했다.
  이 측정은 브라우저 JPEG 인코딩·전송과 센서 노출 지연을 포함하지 않는다.
- 현재 실기 장면은 사무실 배경이어서 목표 15종 검출은 없었다. 새 카메라 정확도 승인이나
  개구량 실측 보정 완료로 간주하지 않는다.
- ELP 브라우저 뷰어, 상태 JSON/스냅샷, 오류 재시작 사용자 서비스 정의 추가.
  접속·시작·정지·수집 명령은 `docs/ELP_IMX291_CAMERA.md`에 기록한다.
- 로컬 관련 회귀검사 43개 통과, 변경 Python 파일 Ruff 검사 통과.
- 원시 실기 측정은 Jetson/로컬 `runs/elp_setup_20261014/`에 보관한다.

## 2026-08 YOLO26s + TensorRT FP16 전환 이력

2026-08-17 개정 계획에 따라 active detector를 YOLOv8n에서 YOLO26s로, Jetson precision을
INT8에서 FP16으로 변경했다. 아래의 2026-08-14 이전 YOLOv8n/INT8 결과는 이력으로 보존하지만
새 Phase 완료나 승인 근거로 사용하지 않는다.

- pretrained `yolo26s.pt` SHA-256 고정 및 Ultralytics 8.4.115 CUDA baseline 완료
- default end-to-end/NMS-free head 확인 완료
- 15-class FP32 fine-tuning과 공개 데이터·ADL staged fine-tuning 완료
- FP32 batch 32 OOM 확인 후 RTX 4070 Ti SUPER 15.56 GiB 기준 batch 16으로 고정
- 당시 runtime/config는 `artifacts/models/yolo26s_best.pt`, TensorRT FP16, static 640, batch 1
  (현재 모델은 위 2026-10-14 공개 데이터 보강 항목 참조)
- INT8 calibration은 새 MVP 경로에서 제거하고 class-balanced benchmark 128장을 별도 생성
- center/debounce/grip/LKG 및 UART PC mock 재검증 완료
- ONNX/RTX FP16/13초 demo/engine 미포함 bundle까지 완료
- Jetson target FP16 engine·전체 val·benchmark·지속 부하 검증 완료

상세 체크리스트와 Jetson 경계는 `reports/YOLO26S_FP16_MIGRATION.md`, 실제 target 명령은
`deployment/JETSON_RUNBOOK.md`에 기록한다.

## 목표와 원칙

`01_Docs/VAGT_ORCA_AI_PLAN.md`의 Phase 1~9를 가능한 하드웨어 범위까지 실제 데이터, GPU 학습,
checkpoint, 검증 및 배포 산출물로 진행한다. 학습·추론은 CUDA 전용이며 CPU fallback은 금지한다.

애플리케이션은 문서 폴더에서 분리해 저장소 루트의 `vagt-orca-ai/`에 배치했다.
`01_Docs/`에는 계획과 사양 문서만 둔다.

## 확인된 환경

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 15.56 GiB
- Driver/CUDA: NVIDIA 595.71.05 / CUDA 13.2
- Python: uv 관리 CPython 3.12.13
- PyTorch: 2.13.0+cu130, `torch.cuda.is_available() == True`
- Ultralytics: 8.4.115
- GPU 접근은 실행 샌드박스 밖 권한이 필요하다. CPU로 우회하지 말 것.

## Phase 상태

| Phase | 상태 | 근거 / 다음 작업 |
| --- | --- | --- |
| 1 | 재완료 | `reports/yolo26s_phase1_baseline.json`: YOLO26s CUDA/NMS-free/10-frame 검증 |
| 2 | 완료 | 62,609장, 15 classes, sequence-safe 80/10/10 split, integrity 오류/경고 0 |
| 3 | 재완료 | YOLO26s FP32 학습·val/test·failure 분석 및 공개/ADL staged fine-tuning 완료 |
| 4 | 재완료 | center bias 100/100, 5/4 debounce 100/100, 20% hysteresis/identity 테스트 통과 |
| 5 | 재완료 | 15→4 grip/LKG 및 최종 YOLO26s checkpoint 영상 E2E 승인 완료 |
| 6 | 재완료 | Jetson FP16 engine·전체 val·E2E/UART·2,000회 지속 부하 승인 완료 |
| 7 | PC mock 재완료 | 16-byte UART v1, CRC16, sequence/loss/stale PC PTY mock 재검증 |
| 8 | 진행 중 | ELP IMX291 UVC 입력·TensorRT FP16 실시간 확인. 새 카메라 도메인 수집/재학습 필요 |
| 9 | 하드웨어 대기 | ORCA Hand 없음. opening 회귀·pose/lateral contact 템플릿·상태 검사 준비 완료 |

과거 YOLOv8n ADL 모델의 Phase 6 실기 판정은 `reports/phase6_status.json`에 `accepted: true`로
남아 있으나 YOLO26s FP16의 승인으로 간주하지 않는다.

현재 전체 검증은 이 문서 하단의 최신 검사 결과를 기준으로 한다.

## 2026-08-23 Jetson YOLO26s FP16 최종 검증

- Jetson Orin Nano Super는 `JETSON_HOST`, JetPack/L4T `R36.4.7`, CUDA `12.6`,
  TensorRT `10.3.0`, PyTorch `2.8.0`, Ultralytics `8.4.117`, 25W mode에서 검증했다.
- 루트는 `/dev/mmcblk0p1`이지만 238.5GB NVMe SSD가 `/mnt/data`에 정상 마운트되어 있다.
  새 프로젝트, validation 데이터, engine은 `/mnt/data/vagt-orca-ai`에 배치했다.
- bundle·checkpoint·ONNX와 전체 val 6,560장/라벨 6,560장을 전송하고 전 파일 checksum
  dry-run으로 차이가 없음을 확인했다.
- target FP16 engine은 Jetson에서 새로 생성했다. 22,879,912 bytes, SHA-256
  `9d05b0229d368e8ae020485eef3d9a5c5cfae70bd7177101b4e254a0f3f46464`, static
  640×640, batch 1, end-to-end/NMS-free, `accepted_target_build: true`다.
- 전체 frozen val mAP50-95 `0.22228`, mAP50 `0.28852`, precision `0.50116`, recall
  `0.28092`다. FP32 mAP50-95 `0.22681` 대비 절대 `0.00453` 감소했다.
- `jetson_clocks` 고정 전에는 평균 `35.50 ms`, p95 `38.43 ms`, `28.17 FPS`였다.
  고정 후 공식 200회 E2E는 평균 `15.88 ms`, p95 `17.46 ms`, `62.97 FPS`, peak device
  memory `3,537.29 MiB`로 운영 후보 p95 50ms 및 30–60 FPS 목표를 충족했다.
- UART 16-byte 직렬화 포함 200회도 평균 `15.88 ms`, p95 `17.46 ms`, `62.97 FPS`였다.
- UART 포함 2,000회 지속 부하는 평균 `15.76 ms`, p95 `19.87 ms`, 최대 `20.77 ms`,
  `63.44 FPS`, GPU/Tj 최고 `55.625°C`, thermal throttling 없음으로 통과했다.
- `reports/yolo26s_phase6_status.json`은 모든 gate가 true이며 `complete/accepted`다.
- 최초 전달 bundle에서 `training/` 누락을 실행 전 발견했다. 패키저에 해당 디렉터리를
  추가하고 회귀검사 및 2회 동일 SHA 검사를 다시 수행했다.

## 2026-08-18 YOLO26s + FP16 재수행 결과

- EgoObjects Phase 3 최고 epoch 12: val/test mAP50-95 `0.20981 / 0.14274`.
- LVIS/COCO 공개 보강 최고 epoch 3: val/test `0.23071 / 0.17297`.
- ADL 균형 stage 최고 epoch 1: val/test `0.22681 / 0.17432`; test recall `0.23469`.
  공개 stage보다 val은 `0.00390` 낮지만 미사용 test mAP50-95는 `0.00135`, recall은
  `0.00338` 높아 최종 egocentric checkpoint로 채택했다.
- 최종 `artifacts/models/yolo26s_best.pt`: 20,370,165 bytes, SHA-256
  `36446774b1da4a61dcfe3da4d386c30686e09bb0abf13deb3b4d5fac09b705f5`.
- Phase 5 영상 E2E, ONNX checker/CUDA smoke, 13초/130-frame FP16 demo가 모두 통과했다.
- RTX 전용 TensorRT FP16 참고 결과: 평균 `1.992 ms`, p95 `2.295 ms`, `502.11 FPS`,
  val mAP50-95 `0.22235`. 플랫폼 종속이므로 Jetson 승인 근거가 아니다.
- engine 미포함 Jetson bundle SHA-256은
  `7a91cefe1d9fc46c826da3424089d729938b2a593871ae377a03df4372bd7c4d`이며 2회 재현됐다.
- `reports/yolo26s_phase6_status.json`은 `target_hardware_pending`: target FP16 engine,
  frozen 15-class val, E2E benchmark만 남았다.
- 상세 정량 비교: `reports/yolo26s_training_summary.json`; 전환 체크리스트:
  `reports/YOLO26S_FP16_MIGRATION.md`.
- 최신 품질 검사: `pytest` 64개 통과, `ruff check .` 통과, Python compileall 통과,
  `uv lock --check` 통과.

## 2026-08-12 공개 데이터 추가 학습

- 자체 AR Glass 데이터가 없는 상태에서 공식 LVIS v1 annotation/COCO 이미지로 목표 15개 class를
  보강했다. 추가분은 train 8,320장, bbox 26,591개이며 기존 EgoObjects val 6,560장과 test
  6,103장은 그대로 동결했다.
- 집중 학습 구성은 기존 train의 class별 최대 1,000장(11,372장)과 추가 공개 데이터
  8,320장, 총 19,692장이다. FP32, 640px, batch 32, AdamW, lr 0.0002로 기존 Phase 3
  `best.pt`에서 fine-tuning했다.
- 최고는 epoch 3이며, epoch 4~6의 val mAP50-95가 연속 하락해 epoch 7 진행 중 조기 종료했다.
- 동일 val: mAP50-95 `0.18333 → 0.20071`(+9.48%), recall `0.24859 → 0.27631`(+11.15%).
- 미사용 test: mAP50-95 `0.13609 → 0.15638`(+14.91%), recall
  `0.20649 → 0.21859`(+5.86%).
- 새 모델을 `artifacts/models/best.pt`로 승격했고 기존 Phase 3 모델은
  `artifacts/models/best_phase3.pt`로 보존했다. 새 SHA-256은
  `f1acf9aa7ee59240aba2ad18a13d4bf59b9a761e19b0dab177a618d273cef749`이다
  (optimizer 제거 후 6,271,020 bytes).
- 상세 결과: `reports/public_v2_training_summary.json`, 검증 산출물:
  `artifacts/validation_public_v2/{val,test}/`.
- 기존 ONNX/TensorRT/Jetson bundle은 아직 Phase 3 모델 기반이다. 새 모델 배포 전 재export,
  INT8 calibration, Jetson engine 재생성이 필요하다.

## 2026-08-13 ADL 1인칭 데이터 추가 학습

- 공식 ADL 1인칭 영상 20개(43,087,304,390 bytes)를 모두 확보하고 20개 영상의 metadata와
  annotation 마지막 frame 존재를 검사했다.
- bottle, cup, phone, remote, toothbrush의 신뢰 가능한 bbox만 14,916 frame/27,192개로 변환했다.
  `knife/spoon/fork` 통합 label은 클래스 오염을 막기 위해 제외했다.
- 전체 14,916장을 넣은 첫 실험은 val mAP50-95가 0.1900 이하로 하락해 3 epoch 후 중단했다.
- 시간 중복과 클래스 편향을 줄인 20명 전체의 균형 subset 6,112장을 구성해 기존 공개 데이터
  train 19,692장과 합쳤다. 최종 train은 25,804장이며 val/test는 그대로 동결했다.
- FP32, 640px, batch 32, AdamW, lr 0.00005로 추가 학습했고 epoch 1이 최고였다. epoch 2~4
  연속 하락 후 patience 3으로 자동 조기 종료했다.
- 동일 val mAP50-95 `0.20071 → 0.20305`(+1.16%).
- 미사용 test mAP50-95 `0.15638 → 0.16011`(+2.39%), mAP50 `0.20711 → 0.21002`,
  recall `0.21859 → 0.22099`.
- Phase 3 최초 모델 대비 test mAP50-95 누적 개선은 `0.13609 → 0.16011`(+17.65%)다.
- 새 모델을 `artifacts/models/best.pt`로 승격했고 직전 모델은
  `artifacts/models/best_pre_adl.pt`로 보존했다. 새 SHA-256은
  `3d095807e8e74dfb347005fa5bd2367a88839b1ae8ffe1ee3cbbfebc8c114a90`이다.
- 상세 보고서: `reports/adl_v3_training_summary.json`; 검증 산출물:
  `artifacts/validation_adl_v3/{val,test}/`.
- 당시 ONNX/TensorRT/Jetson 배포 산출물은 이전 checkpoint 기반이어서 재생성이 필요했다.

## 2026-08-14 새 모델 배포 및 장착 시점 수집 직전 준비

- 현재 checkpoint SHA-256 `3d095807...c114a90`에 맞춰 ONNX opset 17을 다시 생성했다.
  ONNX SHA-256은 `9d3c33cc...cb4c32`, frozen val mAP50-95는 `0.20177`로 FP32
  `0.20305` 대비 절대 `0.00128` 감소했다.
- INT8 calibration을 frozen val 384장 + ADL 1인칭 128장, 총 512장으로 재구성했다.
  15 class 포함, 전 이미지 decode, portable YAML 검사를 통과했다.
- RTX 개발용 INT8 engine의 frozen val mAP50-95는 `0.19224`다. ONNX 대비 절대
  `0.00952` 낮으며, RTX에서 UART 직렬화 포함 E2E는 평균 `1.540 ms`, p95 `1.681 ms`,
  `649.28 FPS`였다. 이 engine과 수치는 Jetson 승인 근거로 사용하지 않는다.
- 새 Jetson 전달 bundle을 생성했고 platform-specific engine은 포함하지 않았다. 재연결된
  Jetson Orin Nano Super(`JETSON_HOST`)에 bundle을 SHA-256 검증 후 적용했다. 이전 Phase 3
  모델·engine·보고서는 target과 local 양쪽에 보존 이름으로 격리했다.
- target의 기존 calibration 디렉터리에 남은 구형 128장을 발견해 첫 빌드를 중단했다. 혼합
  640장과 구형 TensorRT cache를 격리한 뒤 local/target 파일 트리 SHA가 동일한 512장으로
  GPU calibration을 처음부터 다시 수행했다.
- 새 Jetson INT8 engine은 5,037,884 bytes, SHA-256 `81ed28ad...b8343`이다. TensorRT 10.3
  CUDA calibrator, 고정 640×640, batch 1이며 `accepted_target_build: true`다.
- Jetson E2E 200회는 평균 `22.956 ms`, p95 `25.559 ms`, `43.56 FPS`였다. UART 16-byte
  packet 직렬화 포함 측정은 평균 `26.419 ms`, p95 `29.053 ms`, `37.85 FPS`로 운영 후보
  p95 50 ms 기준을 통과했다. target에는 frozen validation 전체가 없어 정확도 평가는 생략했고,
  동일 엔진 계열의 정확도는 desktop frozen val 검증 결과로 관리한다.
- 장착 시점 수집 계획은 15 class × 3 sequence = 45 session, 최소 4,500장이다.
  각 class에서 밝기/거리/배경/회전/가림/다중 물체/동작의 모든 조건을 포함한다.
  `configs/ar_glass_collection_plan.csv`에 세션별 실행 명령을 생성했다.
- `reports/phase8_precollection.json`은 `non_hardware_preparation_complete: true`,
  `actual_collection_started: false`, `status: hardware_connection_pending`이다. Jetson 새 engine과
  Pixy bridge는 준비됐으며, 현재 USB에서 보이지 않는 Pixy를 실제 장착하고 화각·초점을 확인한
  뒤 첫 촬영만 시작하면 된다.

## 2026-08-12 Pixy2.1 실기 통합 및 중앙 의도 선택 개선

- Jetson Orin Nano에서 USB `b1ac:f000 Charmed Labs Pixy`, firmware `3.0.20 general`을
  확인했고 공식 raw-frame 예제로 316x208 PPM 캡처에 성공했다.
- `stop()` 1회 후 `getRawFrame()`을 반복하는 persistent 모드는 30/30 frame, 실패 0,
  31.237 FPS, mean novelty 17.6228이었다. 매 frame resume/stop을 반복하는 cycled 모드는
  12.247 FPS였으므로 persistent를 기본값으로 확정했다.
- Pixy→Python/OpenCV→Jetson TensorRT YOLO 실시간 시험은 25초 동안 275 frame을 처리했고,
  260 frame에서 detection이 있었다. 시험 장면에서는 bottle과 remote가 검출됐지만 phone은
  약했고, remote가 잠깐 미검출될 때 화면 밖 bottle로 target이 넘어가는 현상을 관찰했다.
- 원인은 기존 `CenterBiasSelector`가 중앙에서 멀어도 유일하게 남은 detection을 candidate로
  선택하고, locked target이 detection에서 사라진 동안 challenger debounce를 즉시 시작하는
  정책이었다.
- 변경 정책: 화면 중앙 십자선이 bbox 내부에 있는 detection만 candidate로 허용한다. 기존
  target이 사라지면 `missing_grace_frames: 8` 동안 새 target 출력을 보류하면서 lock을 내부에
  유지하고 재검출을 기다린다. challenger가 계속 중앙에 있으면 유예 종료 후 기존 5/4
  debounce 결과로 전환한다. 유예 중 stale detection을 현재 target처럼 planner에 전달하지 않는다.
- phone 약검출은 의도 선택 정책과 별개인 detector/domain 문제다. threshold 완화는 데모용
  임시 조치일 뿐이며 Pixy 실촬영 phone 데이터의 추가 annotation/fine-tuning이 필요하다.

## Phase 7 UART Integration

- protocol: 115200 baud, 8N1, little-endian 고정 16 bytes
- fields: magic/version/flags/class/grip/confidence/opening/uint32 sequence/CRC16-CCITT-FALSE
- flags로 새 plan과 Last Known Good를 구분하고 no-target은 neutral packet으로 전송한다.
- ESP32 mock은 CRC 손상, duplicate/out-of-order, uint32 wrap-around, packet gap을 처리한다.
- 마지막 유효 target packet 이후 250 ms stale이면 vision assistance를 끄고 EMG-only로 fallback한다.
- Jetson PTY mock: target 변경, LKG, 2 packet loss, CRC 손상, stale fallback 모두 재현 및 통과
- Jetson PTY UART 왕복 p95 0.296 ms
- Jetson INT8 detection→grip→packet 200회: p50 30.247 ms, p95 32.613 ms,
  34.602 FPS, packet 16 bytes. p95 50 ms 목표를 유지했다.
- protocol 문서: `docs/UART_PROTOCOL.md`; 보고서: `reports/phase7_uart.json`,
  `reports/phase7_jetson_e2e.json`

Jetson에는 `/dev/ttyTHS1`, `/dev/ttyTHS2`가 있으나 `permillion` 사용자는 현재 `dialout` group에
없고 ESP32 serial device도 연결되지 않았다. 실제 UART pin/ESP32 연결 시 OS 권한과 전압 레벨을
확인한 뒤 physical loopback을 추가한다.

## Phase 8~9 하드웨어 상태

- Jetson에 `/dev/video*` 장치는 없지만 비-UVC Pixy2.1을 libpixyusb2 C++ bridge로 연결했다.
  정식 AR Glass 도메인 수집 세션은 아직 시작하지 않았다.
- `tools/record_ar_glass.py`가 15 class와 밝기/거리/background/회전/가림/multi-object/motion
  variation metadata를 강제해 frame과 JSONL manifest를 수집한다.
- Phase 9는 `calibration/phase9/opening_observations.csv`, `pose_checks.csv` 템플릿을 준비했다.
- `tools/fit_opening_calibration.py`는 class별 bbox ratio→normalized opening의 alpha/beta와 MAE를
  계산하며, 전체 MAE 0.10 이하만 승인한다. physical pose 검증 전 기본 config를 덮어쓰지 않는다.
- 현재 정확한 잔여 상태: `reports/phase8_status.json`, `reports/phase9_status.json`

## Phase 6 배포 준비 및 검증

- ONNX opset 17: `artifacts/deployment/onnx/best.onnx`, 12,276,337 bytes,
  SHA-256 `7a42392fe57586c302d36a0d030c53a2e9d4c10fdaf48347ebd7d45d0a05837e`
- ONNX checker 및 ONNX Runtime `CUDAExecutionProvider` smoke test 통과
- ONNX val mAP50-95 0.18104, FP32 대비 절대 감소 0.00230
- INT8 calibration: 512장, 15개 class 모두 포함, 전 이미지 decode 및 portable YAML 검사 통과
- 실제 AR Glass sample은 현재 0장이므로 공개 egocentric/LVIS image만 사용했다.
- RTX 개발용 INT8 engine: 7,936,134 bytes, val mAP50-95 0.17449
- RTX 전체 파이프라인 INT8: p50 1.723 ms, p95 2.106 ms, 583.50 FPS,
  peak device memory 940.125 MiB (warmup 20, 측정 200)
- 최종 ONNX 전체 파이프라인: p50 2.563 ms, p95 2.937 ms, 393.26 FPS
- 8개 실제 sequence, 고유 130 frame의 13초 입력/결과 MP4 검증 완료. 결과는 130-frame
  JSONL과 정확히 일치하며 detection 92개, target lock frame 14개, plan update 14개다.
- Jetson 전달 bundle:
  `artifacts/deployment/jetson_bundle/vagt_orca_phase6_inputs.tar.gz` (166,885,147 bytes),
  SHA-256 `0ab900bbcbf2f5d04effa0d2abde58cf44374dd0a3fae94780d742864d9a91f8`
- bundle은 uid/gid/mtime 및 gzip timestamp를 정규화해 연속 2회 생성 시 동일 SHA임을 확인했다.
- RTX engine은 `*_NOT_FOR_JETSON.engine`으로 명시했고 bundle 포함 여부가 false임을 검사했다.
- `run_jetson_phase6.py`가 Jetson/CUDA와 TensorRT 7~10 CUDA calibrator를 강제하며,
  TensorRT 11의 ModelOpt CPU calibration 경로를 engine 생성 전에 거부한다.
- Jetson Orin Nano Super target engine: 5,039,597 bytes, SHA-256
  `bcad7281b23eac7c00d767c9057680fea7053c5e60a6b938b6b2f5ac9be54a27`
- Jetson 환경: L4T 36.4.7, CUDA 12.6, TensorRT 10.3.0, PyTorch 2.8.0, 25W mode
- TensorRT 10.3 dynamic-shape calibration graph 오류를 피하기 위해 target engine은 고정
  640x640, batch 1, ONNX opset 17로 생성했다. desktop development 경로의 dynamic 설정은 유지한다.
- Jetson 전체 파이프라인 200회: p50 28.979 ms, p95 32.677 ms, p99 33.234 ms,
  35.219 FPS, peak device memory 5,356.398 MiB. p95 50 ms 운영 후보 목표를 충족했다.
- target bundle에는 전체 validation dataset이 없어 정확도 재평가는 건너뛰었으며, 계획서의
  Phase 6 필수 조건인 engine 동작, latency, FPS, memory 기록은 모두 충족했다.

Jetson target 보고서는 `reports/phase6_tensorrt.json`과 `reports/phase6_benchmark.json`,
최종 승인은 `reports/phase6_status.json`에 보존했다. target engine은 Jetson의
`/home/jetson-user/vagt-orca-ai/artifacts/deployment/tensorrt/best_int8.engine`에 있으며
하드웨어/플랫폼 종속이므로 다른 장치에 재사용하지 않는다.

### Phase 3 학습 및 평가 결과

- 실행: 2026-08-07 약 20:25~23:25 KST, 10,815.089초
- 설정: 50 epoch, image 640, batch 32, workers 8, AdamW, `amp=False`, CUDA:0
- GPU: RTX 4070 Ti SUPER, 학습 중 약 7.57 GiB 사용
- run: `artifacts/training/yolov8n_egoobjects_15/`
- best epoch: 17 (val mAP50 0.24580, mAP50-95 0.18332)
- final epoch 50: val mAP50 0.22034, mAP50-95 0.16716; 후반 과적합으로 best 사용
- 당시 배포에 사용한 모델: `artifacts/models/best_phase3.pt` (6,255,402 bytes; 2026-08-12 공개 데이터 재학습 후 이름 변경)
- SHA-256: `04709825b5b570c2a2548cf9400a187400e9f9f00bb3a81cbd764853de793062`
- 별도 val: precision 0.47828, recall 0.24859, mAP50 0.24563, mAP50-95 0.18333
- 별도 test: precision 0.40878, recall 0.20649, mAP50 0.18245, mAP50-95 0.13609
- GPU inference: val 0.497 ms/image, test 0.519 ms/image (모델 연산만, desktop RTX 기준)
- confusion matrix/plot: `artifacts/validation/{val,test}/`
- 통합 보고서: `reports/training_summary.json`

완료 전 중단 상황의 재개 명령은 아래와 같다. 현재 학습은 완료됐으므로 재실행할 필요가 없다.

```bash
cd /home/developer/Documents/VAGT/vagt-orca-ai
.venv/bin/python training/train.py \
  --data data/processed/egoobjects_15/data.yaml \
  --resume artifacts/training/yolov8n_egoobjects_15/weights/last.pt
```

## 데이터 현황

### EgoObjects v1

Meta 공식 다운로드 페이지는 2026-08-07 현재 다운로드 component가 빈 상태다. 복구 미러를
commit `b9b6b37cddcddf28b467f657edc81b7824eb906d`로 고정해 사용한다.

완료 파일:

- `data/raw/egoobjects/Meta_EgoObjects_Train.json` (72,487,045 bytes; 실제 eval val/test 주석)
- `data/raw/egoobjects/Meta_EgoObjects_Test.json` (161,152,266 bytes; 실제 train 주석)
- `data/raw/egoobjects/Meta_EgoObjects_Val.json` (85,217 bytes; category metadata)

완료 파일:

- `data/raw/egoobjects/Meta_EgoObjects.zip` (39,826,913,851 bytes)
- SHA-256: `0d7e62e839a427b5e9d24e57fd9c77d044f29731f979bd5f32bf27943fdf3f89`
- ZIP 전체 CRC 검사 통과, 이미지 114,317장 압축 해제 완료

다운로드 manifest는 `data/raw/egoobjects/download_manifest.json`에 저장했다.

### 변환 완료 데이터셋

- 총 이미지/라벨: 62,609 / 62,609
- train: 49,946장, EgoObjects group 650개
- val: 6,560장, EgoObjects group 83개
- test: 6,103장, EgoObjects group 81개
- 15개 class 모두 train/val/test 각각에 annotation 존재
- integrity validator: errors 0, warnings 0, missing classes 0
- 상세 결과: `reports/dataset_summary.json`
- 원본의 val/test 표기는 대부분 동일 `group_id`에 함께 나타나므로 원본 subset을 그대로 쓰지 않고,
  완전한 `group_id` 단위로 seed 42의 80/10/10 결정적 재분할을 수행했다. 최초 분할에서
  val에 빠진 희소 `coin`은 전체 그룹 1개를 train에서 val로 옮겨 보강했다.

### LVIS/COCO 보완 데이터

EgoObjects v1 taxonomy에는 `card`, `small_block`이 없어서 LVIS v1의 `card`,
`business_card`, `identity_card`, `keycard`, `Lego`, `cube`를 보완 데이터로 사용한다.

- 공식 LVIS train/val annotation 다운로드 및 압축 해제 완료
- COCO 이미지 148장 다운로드 완료
- bbox: card 280개, small_block 695개
- 보고서: `reports/lvis_supplement.json` (dataset 내부에도 동일 report 저장)

## 실패 분석과 Phase 5 E2E

- val 1,000장을 15-class 결정적 round-robin 방식으로 샘플링해 GPU FP32 분석 완료
- 최초 분석은 1,000장을 단일 GPU batch로 처리해 OOM 발생. CPU로 전환하지 않고 batch 16
  micro-batch 구현으로 수정 후 완료했다.
- 실패 시각화 100개: `artifacts/failures/`; 보고서: `failure_report.json`
- 약점: card/coin/small_block/can은 공개 데이터의 validation sample과 학습 다양성이 부족해 사실상
  검출되지 않는다. Phase 8 실제 AR Glass 수집 전에도 oversampling/추가 annotation이 우선 개선점이다.
- 학습 모델 영상 E2E: validation 후보 300장 중 ball(confidence 0.92694) 자동 선택
- 10-frame 영상에서 frame 0~2는 명령 없음, frame 3에서 debounce lock 및
  `POWER`, opening `0.420851`, confidence `0.926062` 제안 생성
- 결과: `reports/phase5_e2e.json`, 영상: `artifacts/phase5_e2e/visualized.mp4`

## 폴더 정리

- 실수로 `01_Docs/` 아래 있던 앱 전체를 저장소 루트 `vagt-orca-ai/`로 이동했다.
- `01_Docs/`에는 `VAGT_ORCA_AI_PLAN.md`, `spec.md`만 남겼다.
- 코드 `src/`, 설정 `configs/`, 데이터 `data/`, pretrained `models/pretrained/`, 실행 산출물
  `artifacts/`, 추적 보고서 `reports/`로 구분했다.
- 실패한 다운로드의 `Meta_EgoObjects.zip.curl-part` 557,166,592 bytes와 빈 mirror cache는
  정상 ZIP/annotation/manifest를 확인한 뒤 삭제했다. 삭제 파일은 재다운로드 가능하다.

## 검증된 명령

```bash
uv sync --extra dev --extra data
.venv/bin/python scripts/check_gpu.py
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python training/validate.py --model artifacts/models/best.pt \
  --data data/processed/egoobjects_15/data.yaml --split test
.venv/bin/python training/analyze_failures.py --model artifacts/models/best.pt \
  --data data/processed/egoobjects_15/data.yaml --split val --batch 16
.venv/bin/python scripts/phase5_model_e2e.py
.venv/bin/vagt-orca infer --source artifacts/phase1_baseline_input.mp4 \
  --model models/pretrained/yolov8n.pt --output artifacts/phase1_baseline_visualized.mp4
```

## 주의점

- 미러의 Train/Test 파일명과 실제 split 의미가 반대처럼 보인다. converter는 파일명이 아니라
  모든 JSON의 이미지를 합친 뒤 `group_id`를 기준으로 결정적 재분할한다.
- category metadata는 별도 JSON이고 이미지명 필드는 `file_name`이 아니라 `url`이다.
- split은 `group_id`/video sequence 단위이며 동일 sequence leakage를 허용하지 않는다.
- YOLO metric 합격 threshold는 계획서상 baseline 이후 설정 대상이다. Phase 3 완료 조건은 충족했지만
  현재 test 성능은 production-ready가 아니며 희소 class 보강 없이는 Phase 6 배포를 권장하지 않는다.
- 실제 실행하지 않은 metric/checkpoint를 완료로 기록하지 않는다.
