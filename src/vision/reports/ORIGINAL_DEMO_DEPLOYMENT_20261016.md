# 공식 원본 YOLO26m 시연 배포 — 2026-10-16

**선정 모델은 추가학습하지 않은 공식 YOLO26m이며 현재 책상·가구·사람 등 배경37종을 제외하고 물체43종을 표시한다.**
Jetson에 native TensorRT FP16 엔진을 적용하고 실제 안경 ELP 화면을 시험했다.
**최신 사용자 결정은 공식 원본만 사용하고 캔·펜을 제외하는 것이다. 카메라는 종료했다.**
서비스 inactive/dead, MainPID=0, 장치 점유 없음, 자동 시작 disabled이며 다음 실행은 기본 원본 설정이다.
실제 인식 정확도 합격 판정은 미확정이다. [캔·펜 비교 시험과 종료 기록](CAN_PEN_SPECIALIST_TEST_20261016.md).

사용자 지적 반영: 시연할 좋은 3~5종을 고르라는 뜻을 3종만 표시하라는 뜻으로 잘못 해석했다.
최초 배포의 3종 필터를 제거하고 서비스를 재시작했다. 현재 병·휴대폰·포크는 시연 후보이며
표시 제한이 아니다. 원본 80종 가중치/출력층/엔진과 confidence 0.35는 유지했다.
아래 3종 필터 benchmark/상태는 최초 배포 이력이며 현재 배경 제외 설정의 속도 측정값으로 사용하지 않는다.
최신 상태는 `runs/can_pen_specialist_20261016/camera_stopped_original_selected.json`이다.
`runs/original_demo_20261015/object_focus_state.json`은 배경 제외 적용 당시의 실행 이력이다.

## 최신 수정: 배경 제외와 물체 타깃 선택

사용자가 80종 전체 표시에서 책상이 타깃이 되는 문제를 확인했다. 현재 `excluded_classes`로
책상·의자·사람·고정 가전·동물·차량 등37종을 제외하고 물체43종을 표시한다.
`target.grip_supported_only: true`를 활성화해 책·가위 등 미지원 물체는 표시할 수 있으나
중앙 타깃은 프로젝트 그립을 지원하는 물체에서만 선택한다. 원본 공통8종(병·컵·공·휴대폰·
리모컨·숟가락·포크·칫솔)이 대상이며, 좋은3~5종을 골라 시연한다는 의도와 표시 범위는 별개다.
선택기와 안정화기 모두 같은 적격 목록을 사용한다. 배경/미지원 물체가 병보다 중앙에 있어도
타깃을 빼앗지 않고, 적격 물체가 사라지면 새 그립을 만들지 않는 회귀검사를 포함해50검사와 Ruff 통과.
원본 엔진과 confidence0.35는 그대로다. `object_focus_state.json`과 기존 3종/80종 화면 기록은
수정 당시 이력이다. 현재 카메라는 사용자의 중지 지시에 따라 정지했다.

## 선정과 적용 범위

- 사용자가 기존 15종 시연을 취소하고 성능이 좋은 3~5종으로 범위를 줄였다.
- 같은 공개 이미지의 Jetson native 평가에서 m은 주력 3종의 AP가 모두 s보다 높고 공개 이미지 pipeline p95가 50ms 이내였다. 정확도를 우선해 m을 선택했다.
- 공은 예비 후보로 남겼다. 추가 중앙 공개 사진 기능 검사에서 미검출이 있어 최초 3종 필터에서 제외했으나 현재는 다른 물체와 함께 표시한다. 이 검사는 공 전체 성능의 대표 평가가 아니다.
- 원본 COCO 80종 가중치/출력층/이름 metadata는 유지한다. 현재 배경37종을 제외하며 `cell phone`→`phone`, `sports ball`→`ball` 및 프로젝트 공통 8종의 이름/ID를 연결한다. 그 외 물체도 표시하며 그립 매핑이 없으면 새 그립을 제안하지 않는다. 미매핑 물체 ID는 100+COCO ID로 구분해 프로젝트 ID와 충돌하지 않는다.
- 포크의 기존 그립 매핑은 TRIPOD, 리모컨/휴대폰은 POWER다. 이번 3종 시연이 4종 그립 전체 지원을 뜻하지 않는다.
- confidence 0.35, square640, batch1, CUDA FP16, end2end/NMS-free, 단일 전체 화면. 추가학습·중앙 crop·전문 모델 혼합·클래스별 threshold 조정 없음.

## Jetson native FP16 공개 평가

같은 COCO 프로젝트 subset 125장과 Open Images 324장, 총449장을 두 모델에 사용했다.
AP는 AP50–95×100이다. 카메라 프레임을 사용하지 않았으며 기존 공개 test 평가를 재사용했다.
**이번 모델/물체 선정에 아래 결과를 사용했으므로 새로운 blind test로 해석하지 않는다.**

| 공개 영역 | 물체 | 원본 s AP | 원본 m AP | m−s |
|---|---|---:|---:|---:|
| COCO 125장 | 포크 | 32.06 | 42.28 | +10.22 |
| COCO 125장 | 리모컨 | 56.35 | 58.69 | +2.34 |
| COCO 125장 | 휴대폰 | 42.48 | 47.84 | +5.36 |
| Open Images 324장 | 포크 | 63.67 | 70.28 | +6.61 |
| Open Images 324장 | 리모컨 | 82.00 | 87.30 | +5.30 |
| Open Images 324장 | 휴대폰 | 74.49 | 81.31 | +6.82 |

배포 confidence 0.35·정답 매칭 IoU 0.5의 m 정밀도/재현율은 다음과 같다.

| 공개 영역 | 물체 | 정밀도 % | 재현율 % | TP / FP / FN |
|---|---|---:|---:|---:|
| COCO 125장 | 포크 | 80.0 | 36.4 | 8 / 2 / 14 |
| COCO 125장 | 리모컨 | 86.7 | 59.1 | 13 / 2 / 9 |
| COCO 125장 | 휴대폰 | 78.3 | 54.5 | 18 / 5 / 15 |
| Open Images 324장 | 포크 | 85.2 | 78.8 | 52 / 9 / 14 |
| Open Images 324장 | 리모컨 | 71.4 | 83.3 | 25 / 10 / 5 |
| Open Images 324장 | 휴대폰 | 82.5 | 78.8 | 52 / 11 / 14 |

AP 향상이 모든 confidence 고정 정밀도의 향상을 뜻하지 않는다. 예를 들어 OI 리모컨 정밀도는
s 73.5%→m 71.4%다. 공개 자료의 미주석 물체도 FP로 집계될 수 있다.
이 수치는 ELP 실제 안경 시야의 오탐률이나 중앙 타깃 성공률이 아니다.
더 큰 기존 EgoObjects 중심 평가는 [원본 상세 평가](YOLO26_ORIGINAL_BASELINE_20261015.md),
추가학습 모델 대비는 [같은 조건 비교](YOLO26_ORIGINAL_VS_FINETUNED_20261015.md)를 참조한다.

## 처리 속도와 기능 확인

Jetson Orin Nano Super / L4T R36.4.7 / Torch2.8 CUDA12.6 / TensorRT10.3, 전력25W,
클럭 비고정. 공개 이미지 benchmark는 카메라 읽기·JPEG·전송·화면 표시를 제외하고
검출→안정화→그립 제안→메모리 UART 패킷 직렬화를 포함한다. warmup20/측정200회다.

| 조건 | FPS | 평균 ms | p95 ms |
|---|---:|---:|---:|
| 후보4종 s, 같은 공개 benchmark | 40.10 | 24.93 | 29.26 |
| 후보4종 m, 같은 공개 benchmark | 36.40 | 27.48 | 34.80 |
| 최종3종 m, 중앙 공개 사진23장 | 26.26 | 38.07 | 39.88 |

마지막 행은 이미지 구성이 달라 앞 행과 클래스 수의 효과를 비교하는 실험이 아니다.
선별 공개 사진23장×5회=115프레임에서 최종 native80/CUDA/end2end,
3종 검출·5/4 안정화·그립 제안·115개 메모리 UART 패킷 해독을 확인했다.
선별 정지 사진 반복은 정확도 평가나 실제 영상의 시간 안정성 평가가 아니다.
물리 UART는 송신하지 않았다.

로컬 전체 pytest81통과/5개 opt-in GPU 검사 생략(실제 CUDA/native 기능 검사는 별도 통과).
활성 설정 전환 후 adapter/deployment/pipeline 11검사와 변경 Python Ruff를 통과했다.
로컬과 Jetson의 활성 runtime/deployment/service 및 실행 코드7개 SHA-256이 모두 일치한다.

## 실제 안경 ELP 화면

- 2026-10-16 15:49 KST 사용자 요청으로 시작, 브라우저 실제 영상 표시 확인.
- 2026-10-16T15:51:11.399534+09:00 상태 확인: active/running, MainPID=5248, 재시작0회.
- 1280×720 YUYV, 드라이버 요청60FPS. 실제 뷰어는 약17~18FPS, 최근 확인값17.15FPS.
- 2초 간격3회 확인에서 프레임 1700→1769, 읽기 실패0회. 장치 점유는 /dev/video0.
- 최근 pipeline 평균37.70ms/p9548.94ms. 센서 노출부터 브라우저 표시까지의 지연은 측정하지 않았다.
- 최초 확인 화면은 매우 흐리게 보여 렌즈 앞 가림/초점 확인을 안내했다. 광학 상태가 불명확한 화면으로 모델의 정확도를 판정하지 않는다.
- 주소: [실시간 화면](http://127.0.0.1:8765/). SSH localhost 터널을 사용한다.
- 이후 사용자 중지 지시로 카메라를 종료했다. 서비스 inactive/dead·MainPID=0, 장치 점유 없음, 로그인 자동 시작 disabled를 확인했다.
- 최신 JPEG만 메모리에 유지하며 프레임/영상 자동 저장은 없다. 이번 작업은 상태 JSON과 기존 공개 이미지 검증 결과만 기록했다.
- 실제 관찰에서 오탐/미검출·중앙 타깃 선택을 확인해야 한다. 초점·장착 위치·개구량 실측 보정 완료를 주장하지 않는다.

## 가중치·설정과 복구 기록

- checkpoint: `models/pretrained/yolo26m.pt`
- checkpoint SHA-256: `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`
- Jetson engine: `artifacts/deployment/tensorrt/yolo26m_coco_original_fp16.engine`
- engine SHA-256: `22eedf1aca4bc67b55b08407d6dbb401fc7f9fc103558c8d0c9ed8e7acd02c72`
- 활성 설정: `configs/runtime.yaml`, `configs/deployment.yaml`, `deployment/vagt-camera.service`
- 선택 프로필: `configs/runtime_original_demo_s.yaml`, `configs/runtime_original_demo_m.yaml` (현재 각각 배경 제외/물체43종 표시)
- 로컬 전환 전 백업: `runs/original_demo_20261015/local_before_original/`, `local_backup_manifest.json`
- Jetson 전환 전 백업: `/mnt/data/vagt-orca-ai/runs/original_demo_20261015/jetson_before_original/manifest.json`
- 기존 can_coin/v8 가중치와 엔진은 보존했다. 복구를 원할 때 카메라를 먼저 정지하고 백업 설정·코드·unit을 함께 복원한다.

근거 JSON은 `runs/original_demo_20261015/`의 `native_results.json`, `s_export.json`, `m_export.json`,
`native_pipeline_m.json`, `selected_benchmark.json`, `deployment_result.json`, `live_test_state.json`이다.
`deployment_result.json`의 카메라 정지 상태는 설정 적용 직후/사용자 시험 시작 직전 기록이며,
당시 실행 상태는 `live_test_state.json`, 현재는 `object_focus_state.json`을 참조한다. 작업 디렉터리는 10월15일 시작 이름을 유지했다.

80종 표시 수정 뒤 adapter/deployment/pipeline 13검사와 grip/stabilizer/packet 35검사 및 Ruff를 통과했다.
80종 전체 표시 이력은 `all_class_display_state.json`이며 이후 배경 제외를 적용한 최신 `object_focus_state.json`이 우선한다.

80종 설정에서 실제 브라우저 영상의 시계·냉장고 표시를 확인했다. 15:57 상태는 약15.35FPS/읽기 실패0/재시작0/프레임761→791이며 이후 UI에서17.78FPS를 확인했다.

배경 제외 적용 후 16:03 상태: active/running, 약15~17FPS, 프레임1037→1102, 읽기 실패/재시작0회. 실제 화면에서 책상은 표시되지 않고 병이 빨간 타깃으로 선택되는 것을 확인했다. pipeline p95는 해당 표본에서50.22ms이며 50ms 목표를 엄격히 통과했다고 주장하지 않는다. 최신 활성 파일10개는 로컬/Jetson SHA-256이 모두 일치한다.
