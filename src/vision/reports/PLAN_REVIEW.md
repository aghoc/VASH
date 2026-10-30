# YOLO26s 계획 검토 및 구현 추적표

기준 문서: `01_Docs/VAGT_ORCA_AI_PLAN.md` (2026-08-17 개정)

## 검토 결과

현재 계획의 고정 경로는 `YOLO26s FP32 fine-tuning → validation → TensorRT FP16 → Jetson
benchmark`다. 과거 YOLOv8n/INT8 모델·engine·수치는 새 계획의 완료 근거가 아니며 이력으로만
보존한다. YOLO26 default end-to-end/NMS-free head를 우선 사용하고, Jetson 호환성 문제가 실제로
재현될 때만 `end2end=False`를 별도 비교한다.

데이터, target selection, 4종 grip, normalized opening, confidence/LKG, UART 권한 경계는 모델
변경과 무관하므로 기존 구현을 보존하되 새 설정에서 다시 검증한다. AI는 여전히 제안만 만들며
서보 각도·힘·실행 여부를 결정하지 않는다.

## 요구사항 추적

| 계획 요구사항 | 구현 | 현재 검증 |
| --- | --- | --- |
| YOLO26s pretrained GPU baseline | `scripts/phase1_yolo26s_baseline.py` | CUDA 실행, checkpoint SHA 고정 |
| end-to-end/NMS-free 확인 | baseline 및 export gate | pretrained model에서 확인 완료 |
| video/webcam/Pixy 입력 | `frame_source.py`, CLI, Pixy bridge | 기존 기능 테스트 통과 |
| EgoObjects 15-class adapter | dataset converter/validator | 62,609장, split integrity 유지 |
| FP32 fine-tuning | `training/train.py`, `amp=False` | EgoObjects→공개 데이터→ADL 단계 완료 |
| validation/confusion/failure | validate/analyze/summarize scripts | val/test/failure 및 단계별 비교 완료 |
| center bias + 5/4 + 20% | selector/stabilizer | 100/100 재검증 |
| 15→4 grip/opening/LKG | GripPlanner | 15/15, no-target LKG 재검증 |
| 모델 영상 E2E | `scripts/phase5_model_e2e.py` | 새 checkpoint로 승인 완료 |
| ONNX CUDA smoke | `deployment/export_onnx.py` | opset 17 checker/CUDA smoke 통과 |
| TensorRT FP16 export | `deployment/export_tensorrt.py` | RTX 개발 및 Jetson target build 완료 |
| portable benchmark images | `deployment/build_benchmark_dataset.py` | class-balanced 128장 생성 완료 |
| engine 없는 Jetson bundle | `deployment/package_jetson.py` | 생성 및 2회 SHA 재현성 확인 |
| UART mock | `scripts/phase7_acceptance.py` | loss/CRC/stale fallback 통과 |
| AR Glass 수집 준비 | capture plan/recorder | Jetson·카메라 장착 대기 |
| ORCA Hand calibration 준비 | opening/pose templates | 손 하드웨어 대기 |

## Jetson 전/후 경계

현재 머신의 학습·검증·전체 proposal pipeline·ONNX·RTX 전용 FP16·UART PC mock·engine 미포함
bundle을 완료한 뒤, Jetson에서도 FP16 engine 생성과 전체 val, latency/FPS/memory/thermal
안정성을 완료했다. Target engine은 RTX 산출물과 분리해 Jetson/로컬에 보존한다.

실시간 승인 기준은 target selection 95% 이상, deterministic grip 100%, Jetson end-to-end p95
50 ms 이하 목표이며 detector metric은 새 YOLO26s baseline 결과를 그대로 기록하고 희소 class
취약점을 별도로 판단한다. 실제 AR Glass와 ORCA Hand가 없으므로 domain adaptation과 opening
실측 calibration은 아직 완료로 표시하지 않는다. Phase 6은 `jetson_clocks` 고정 후 p95
`17.46 ms`, `62.97 FPS`, val mAP50-95 `0.22228`로 승인했다.
