# 원본 유지 + 캔·펜 전용 모델 시험 — 2026-10-16

**최종 결정: 사용자가 캔·펜 사용을 취소했다. 공식 원본 YOLO26m만 유지하고 카메라를 종료했다.**
시험 override는 백업 후 제거했으며 서비스 inactive/dead·MainPID=0, 장치 점유 없음, 자동 시작 disabled다.
아래는 채택하지 않은 실험 이력이다. 원본 가중치·엔진은 변경하지 않았다.

공식 원본 YOLO26m의 가중치·80종 출력층·기본 runtime·기본 서비스 파일을 보존했다.
새 모델은 공식 원본 YOLO26s에서 별도로 학습한 **캔/펜 2종 출력 모델**이다.
원본 검출 결과에 캔/펜 검출만 덧붙이며 원본 상자·이름·점수를 교체하거나 억제하지 않는다.
추가 검출이 중앙 타깃 선택에 영향을 줄 수 있고, 두 모델 실행으로 속도는 떨어질 수 있다.
원본의 기존 물체43종 표시/배경37종 제외 정책을 유지한다. 추가 모드에서는 최대45종을 표시한다.
사용자 시연 후보는 병·휴대폰·포크이며 이 3종으로 표시를 제한하지 않는다.

## 학습

- RTX 4070 Ti SUPER CUDA0, 공식 `yolo26s.pt`에서 2종 헤드 생성, 첫10블록 고정.
- FP32, 640 square, batch16, AdamW lr0.0003, seed20261016. 최대8회 중6회 후 조기 종료, best=3회, 418초.
- 학습9,506장(캔 양성2,392장 / 펜 양성3,677장 / 양쪽 주석이 없는 이미지3,500장).
- 기존 공개 데이터만 사용: EgoObjects/기존 공개 retention, Open Images, COCO. 신규 카메라 프레임0장.
- 같은 EgoObjects 그룹의 양성은 최대30장으로 제한하고 pixel SHA256 중복을 병합했다. 기존 train/val/test 분할은 그대로다.
- pixel train/holdout 중복0, EgoObjects 그룹 분할 중복0. `build_report.json`과 integrity 보고서 참조.
- 다른 물체의 주석은 이 전용 모델에 한해 배경으로 다룬다. 공개 주석에 캔·펜이 없다는 뜻이며 실제 부재를 수동 검증한 음성 데이터는 아니다.
- 원본 m은 학습하지 않았다. 2종 전용 모델의 neck/head 등은 학습되므로 '캔·펜에 해당하는 가중치 몇 개만 변경했다'는 뜻은 아니다.

## 공개 검증과 테스트 — PC FP32

검증2,708장으로 임계값을 고르고 기존 test7,111장에서는 그대로 고정했다.
IoU0.5로 정답을 매칭하며 사각형 겹침/중복 검출도 오탐에 포함한다.
두 클래스 모두 검증에서 선택된 threshold는0.80이다(정밀도≥85%, 음성 이미지 오탐률≤1%, TP≥10 중 F1 최대).
아래 test는 과거에도 사용한 공개 분할이며 완전히 새로운 blind set은 아니다. 이번 test 결과로 threshold를 바꾸지 않았다.

| 물체 | 검증 정밀도 | 검증 재현율 | 테스트 정밀도 | 테스트 재현율 | 테스트 TP/FP/FN | 주석 음성 이미지 오탐 |
|---|---:|---:|---:|---:|---:|---:|
| 캔 | 87.5% | 18.8% | 81.3% | 51.5% | 87/20/82 | 13/6,995 (0.19%) |
| 펜 | 87.3% | 4.2% | 64.6% | 4.8% | 64/35/1,282 | 28/6,245 (0.45%) |

일반 threshold0.35에서는 test 캔 정밀도32.6%/재현율73.4%, 펜36.0%/35.3%로 오탐이 많았다.
**펜은 높은 threshold로 오탐을 줄이면 대부분을 놓친다. 성능 개선 성공이나 시연용 합격으로 판정하지 않는다.**
캔도 정밀도85% 검증 기준이 test에서는 유지되지 않았다. 이번 모델은 사용자가 요청한 비교 시험용이다.
원본 COCO80에는 can/pen 출력 클래스가 없어 직접 인식의 TP/P/R 비교 대상이 아니다.
공개 데이터의 미주석 물체는 FP로 계산될 수 있으며 이 수치는 실제 안경 카메라 정확도가 아니다.

학습 중 최종 best 검증 AP50–95: 캔21.05 / 펜24.80. 서로 다른 지표인 P/R과 혼동하지 않는다.

## 실행과 복구

- 기본 `configs/runtime.yaml`, `configs/deployment.yaml`, `deployment/vagt-camera.service`는 보존한다.
- 비교용 `configs/runtime_can_pen_test_jetson.yaml`에서만 `supplement_profile`을 사용한다.
- 화면의 **원본** 버튼은 보조 추론을 생략한다. **원본 + 캔·펜**은 추가 추론을 실행한다.
- 모드 변경 시 타깃 투표와 이전 그립 상태를 초기화해 이전 모드의 결과가 남지 않게 한다.
- 검출 표시43종은 유지하며 캔ID1/펜ID7만 더한다. 타깃은 그립 지원 물체에서만 선택한다.
- 원본 가중치/기본 설정은 `runs/can_pen_specialist_20261016/original_manifest.json`으로 대조한다.
- 서비스 시험 override `~/.config/systemd/user/vagt-camera.service.d/60-can-pen-test.conf`는 `runs/can_pen_specialist_20261016/overcurrent_backup/`에 byte 백업 후 제거했다. daemon-reload 후 기본 원본 설정으로 복구하고 서비스를 정지·자동 시작 해제했다.
- 사용자 종료 지시를 이행해 카메라 장치를 해제했다. 시험 중에는 최신 JPEG 한 장만 RAM에서 갱신했으며 프레임/영상 자동 저장은 없었다.
- Jetson용 고정 공개 검증 이미지 약3.1GiB를 별도로 복사했다. 녹화 누적이 아니며 이미지/주석19,638개 SHA256이 PC 원본과 일치했다. Jetson 전송 당시 가용 공간218GiB.

## 검증 파일

- `runs/can_pen_specialist_20261016/calibration_fp32.json`, `test_fp32.json`
- `artifacts/models/yolo26s_can_pen_specialist_20261016.training.json`
- `data/processed/can_pen_specialist_20261016/build_report.json`
- `runs/can_pen_specialist_20261016/holdout_files.sha256`
- 전체 Python 검사92통과/5 GPU opt-in 생략, 변경 Python Ruff 통과. GPU 장치 인덱스 전달과 평가 매칭 회귀검사를 포함했다. PC GPU 공개32장에서는 보조 활성/비활성 두 경우 모두 원본 검출의 이름·좌표·점수가 정확히 같았다. 실제 native GPU 검사는 별도로 기록한다.

## Jetson native FP16 평가

Orin Nano Super / JetPack R36.4.7 / Torch2.8 CUDA12.6 / TensorRT10.3에서 직접 생성한
정적640·batch1·FP16·end2end 엔진을 사용했다. PC에서 만든 엔진을 배포하지 않았다.
공개 검증2,708장으로 다시 정한 threshold는 캔0.85/펜0.80이며 test7,111장에는 고정했다.
검증 캔 정밀도95.0%/재현율17.0%, 펜87.3%/4.2%였다.

| 물체 | threshold | test 정밀도 | test 재현율 | TP/FP/FN | 주석 음성 이미지 오탐 |
|---|---:|---:|---:|---:|---:|
| 캔 | 0.85 | 85.7% | 42.6% | 72/12/97 | 7/6,995 (0.10%) |
| 펜 | 0.80 | 64.3% | 4.7% | 63/35/1,283 | 28/6,245 (0.45%) |

캔은 약57%, 펜은 약95%의 주석 정답을 놓쳤다. 펜은 test 정밀도 기준도 충족하지 못해 실사용 합격으로 판정하지 않는다.
PC/Jetson의 캔 수치는 threshold가 다르므로 차이를 FP16 변환 효과만으로 해석하지 않는다.
공개 평가/엔진 생성 중에는 원본 카메라를 계속 실행했다. 이때 얻은 속도를 단독 benchmark로 사용하지 않는다.

공개32장 native 동작 검사에서 원본 검출56개의 좌표·점수·이름이 보조 활성/비활성 두 경우 모두 정확히 같았다.
같은 입력에서 보조 검출3개만 추가됐으며 원본 엔진 SHA256도 그대로다. 이는 기능 검사이며 정확도 검사가 아니다.

- 전용 PT SHA256: `9f688b21ef93eb01e898cb412f23efa8002a5c512131841481fdbb421133b7bf`
- 전용 Jetson 엔진 SHA256: `497fee2132acf0181094db5c32b9fbd2f3ee3835d4cfe12a5659ff52fc9c005b`
- 원본 m PT SHA256: `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`
- 원본 m 엔진 SHA256: `22eedf1aca4bc67b55b08407d6dbb401fc7f9fc103558c8d0c9ed8e7acd02c72`
- `calibration_fp16.json`, `test_fp16.json`, `isolation_jetson.json`, `native_verification_done.json` 참조.

## 실제 A/B 관찰과 사용자 종료

화면에서 원본/추가 모드 전환과 상태 초기화를 확인했다. 각 모드 상태3회 표본은 `mode_samples.json`에 있다.

| 모드 | 화면 처리 FPS | pipeline 평균 ms | pipeline p95 ms |
|---|---:|---:|---:|
| 원본 | 15.24~18.09 | 37.33~38.60 | 48.46~49.33 |
| 원본 + 캔·펜 | 10.87~12.14 | 61.93~63.74 | 66.51~67.99 |

순차 표본이며 통제된 단독 benchmark가 아니다. 추가 모드는 50ms 지연 목표를 초과했다.
보수적 threshold에서 실제 화면에 놓인 캔·펜이 검출되지 않은 장면을 확인했다. 이 관찰을 일반 인식률로 환산하지 않는다.

사용자가 Jetson의 overcurrent throttling 경고를 알렸다. 즉시 원본 모드로 바꾼 뒤 확인한 누적값은
OC3(순간 과전류)26회, OC1(저전압)0회, OC2(평균 과전류)0회였다. 이후 몇 차례와 카메라 종료 후에도 같은 값이었다.
경고 전 기준값이 없어 이 26회의 발생 시점이나 이번 작업의 인과를 확정할 수 없다.
사용자는 기본 제공 DC 어댑터 사용을 확인했다. 전력 모드는25W이며 보호 임계값·클럭·전력 모드를 변경하지 않았다.
확인 당시 온도는 CPU52.9/GPU53.5/TJ53.8°C, 입력4968mV·1744mA였고 현재 전류 alarm은0이었다.
이는 과거 온도·전력 피크 측정이 아니다. kernel journal 조회는 항목이 없었고, sudo 무암호 권한은 없어 전력 모드를 바꾸지 않았다.
OC 카운터와 하드웨어 제한 동작은 [NVIDIA 공식 문서](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)에 설명되어 있다.
60초 관찰 스크립트는 사용자 종료 요청으로 실행하지 않았다. 전류 경고 해결이나 원인 규명 완료를 주장하지 않는다.

사용자의 최종 지시로 캔·펜 모델을 사용하지 않고 공식 원본만 유지했다. 시험 override를 백업 후 제거하고
서비스 stop/disable 및 daemon-reload를 완료했다. **inactive/dead, MainPID=0, 카메라 장치 점유0개**를 확인했다.
현재 실행 중인 카메라가 없으며 다음 수동 실행은 기본 `configs/runtime.yaml`과 원본 m 엔진을 사용한다.
최종 상태는 `runs/can_pen_specialist_20261016/camera_stopped_original_selected.json`에 기록했다.
시험 가중치·평가 자료는 이력으로 보존한다. 추가 학습·변환·평가를 계속 실행하지 않는다.
