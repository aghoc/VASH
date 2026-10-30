# YOLO26s FP16 전환 실행 기록

기준 문서: `../01_Docs/VAGT_ORCA_AI_PLAN.md` (2026-08-17 개정)

## 고정 결정

- Detector: YOLO26s
- Fine-tuning: FP32 (`amp=False`)
- Default head: end-to-end / NMS-free
- Jetson runtime: TensorRT FP16, static 640×640, batch 1
- YOLO26n: Jetson 메모리·발열·throttling 문제가 실측될 때만 fallback
- INT8 calibration/precision 비교: 현재 MVP 범위에서 제외

## 기존 결과의 취급

YOLOv8n/INT8 checkpoint, engine, benchmark와 보고서는 과거 실험 이력으로 보존한다. 새 모델의
Phase 1·3·5·6 완료 또는 정확도/지연 승인 근거로 재사용하지 않는다. 새 산출물은 `yolo26s_*`
경로에 생성해 구형 파일과 섞이지 않게 한다.

## Jetson 연결 전 실행 범위

- [x] 공식 YOLO26s checkpoint SHA-256 고정
- [x] Ultralytics/PyTorch/CUDA/GPU 환경 기록
- [x] pretrained CUDA baseline과 end-to-end/NMS-free head 확인
- [x] 기존 15-class 데이터 integrity 및 고정 split 유지
- [x] EgoObjects YOLO26s FP32 fine-tuning 및 val/test/failure 분석
- [x] LVIS/COCO 공개 보강 fine-tuning
- [x] ADL 1인칭 균형 subset fine-tuning
- [x] center bias, debounce/hysteresis, grip/opening/LKG 재검증
- [x] 새 checkpoint 기반 영상 E2E 재검증
- [x] ONNX export/checker/CUDA smoke test
- [x] RTX 개발용 TensorRT FP16 export 및 benchmark
- [x] portable 128-image benchmark set 생성
- [x] engine 미포함 Jetson 전달 bundle 생성 및 재현성 검사
- [x] UART PC mock의 loss/CRC/stale fallback 재검증

Jetson이 준비되면 `deployment/JETSON_RUNBOOK.md`에 따라 target에서 FP16 engine을 새로 만들고
15-class accuracy, p95 latency, FPS, peak memory, 장시간 안정성을 기록한다. Jetson target engine은
RTX engine을 복사하지 않는다.

## 현재 확인된 baseline

- Checkpoint: `models/pretrained/yolo26s.pt`
- SHA-256: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`
- Ultralytics: 8.4.115
- PyTorch/CUDA: 2.13.0+cu130 / CUDA 13.0
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 15.56 GiB
- Model: 9,959,474 parameters, 22.8 GFLOPs after 15-class head remapping
- Default end-to-end/NMS-free head: 확인 완료
- 최초 요청 batch 32는 FP32에서 OOM이 발생해 batch 16으로 자동 축소됨. 재현 config도 16으로 고정함.

## Jetson 연결 전 최종 결과

- Phase 3 FP32: val/test mAP50-95 `0.20981 / 0.14274`
- 공개 데이터 stage: val/test mAP50-95 `0.23071 / 0.17297`
- ADL 최종 stage: val/test mAP50-95 `0.22681 / 0.17432`; test recall `0.23469`
- 최종 checkpoint: 20,370,165 bytes, SHA-256
  `36446774b1da4a61dcfe3da4d386c30686e09bb0abf13deb3b4d5fac09b705f5`
- Phase 5 모델 E2E: 300개 후보 중 confidence `0.9352` spoon 선택, frame 3 lock,
  `TRIPOD` proposal 생성, 승인 완료
- ONNX: opset 17, `(1,300,6)` NMS-free output, CUDA smoke 통과, SHA-256
  `d94db44fe919a07fec6e578ace40a0cbda92a4cddd665a10ca7ffce76a6153c7`
- RTX 개발용 TensorRT FP16: 83,683,312 bytes, 평균 `1.992 ms`, p95 `2.295 ms`,
  `502.11 FPS`; 전체 val mAP50-95 `0.22235`. Jetson 승인 근거로 사용하지 않음
- 13초/130-frame 시퀀스 demo 무결성 검사 통과
- Jetson bundle: 89,161,183 bytes, engine 미포함, `training/` 포함, 연속 2회 동일 SHA-256
  `7a91cefe1d9fc46c826da3424089d729938b2a593871ae377a03df4372bd7c4d`
- `reports/yolo26s_phase6_status.json`: ONNX/demo/bundle 통과, target engine/전체 val/benchmark만 대기

세부 단계 비교는 `reports/yolo26s_training_summary.json`에 기록했다. 이후
`deployment/JETSON_RUNBOOK.md`에 따라 target에서 FP16 engine을 새로 생성하고 전체 15-class
val, E2E latency/FPS/memory 및 장시간 안정성을 기록했다. RTX engine은 복사하지 않았다.

## Jetson 최종 승인 (2026-08-23)

- [x] SSD `/mnt/data` 배포 및 bundle/model/ONNX checksum 확인
- [x] 전체 frozen val 이미지/라벨 각 6,560개 checksum 확인
- [x] Jetson TensorRT 10.3 FP16 engine target-side 생성
- [x] 전체 15-class FP16 accuracy 기록
- [x] 200회 E2E 및 UART 포함 benchmark 기록
- [x] 2,000회 sustained benchmark와 tegrastats 온도 기록
- [x] Phase 6 자동 gate `complete`, `accepted: true`

Target engine SHA-256은 `9d05b0229d368e8ae020485eef3d9a5c5cfae70bd7177101b4e254a0f3f46464`,
val mAP50-95는 `0.22228`이다. `jetson_clocks` 고정 후 E2E p95는 `17.46 ms`,
`62.97 FPS`; UART 포함 2,000회 p95는 `19.87 ms`, GPU 최고 온도는 `55.625°C`다.
결과는 `reports/yolo26s_phase6_{tensorrt,benchmark,benchmark_uart,benchmark_sustained,status}.json`과
`reports/yolo26s_phase6_thermal.json`에 보존한다.
