# 캔·동전 개선 모델 롤백 — 2026-10-15

사용자 요청으로 8종 개선 설정을 기본 실행에서 제외하고 2026-10-14의 **캔·동전 개선 모델**로 복구했다.

- checkpoint: `artifacts/models/yolo26s_can_coin_retention.pt`
- Jetson TensorRT FP16 engine: `artifacts/deployment/tensorrt/yolo26s_can_coin_retention_fp16.engine`
- checkpoint SHA-256: `5ba5715728b996cfb11fca6b01072bd54256e38a313c27e691eb7ba34ea68726`
- engine SHA-256: `b24d8fffcd1702a176b7743a7bade6de09b7b0202dfe63591ba16308e60263f0`
- `configs/runtime.yaml`: 당시 runtime_candidate와 내용 동일. 캔 0.75·동전 0.25·나머지 0.35, image_size=640, 기존 안정화 설정 유지, focus_profile 없음.
- deployment 설정 및 서비스 ExecStart도 위 모델로 복구했다. 40/50 focus8 drop-in은 byte 백업 후 서비스에서 제거했다. 중앙 확대와 COCO 전문 모델은 현재 실행에 참여하지 않는다.
- Jetson CUDA:0에서 기존 공개 benchmark 이미지 5장으로 실제 복구 engine의 pipeline 동작을 확인했다. 별도 카메라 캡처나 실제 UART 송신은 없었다.
- 로컬 기존 설정/검출 기준 회귀검사 6개 통과. 서비스 unit 문법 확인 통과.
- 현재 카메라 상태: inactive/dead, MainPID=0, UnitFileState=disabled. 로그인 자동 시작은 해제되어 있다.

**후속 실제 ELP 시험에서도 사용자가 오탐 과다를 확인해 실사용 불합격으로 기록했다.** 사용자의 중지 지시에 따라 서비스와 영상 터널을 종료했고 MainPID=0·장치 점유 없음·자동 시작 해제를 확인했다. 과적합 여부와 주원인은 아직 확정하지 않았다. 위 GPU 확인은 복구한 모델이 정상 로드되고 pipeline이 동작하는지만 검사한 것이다. 기존 공개 성능 기록은 [캔·동전 학습 결과](CAN_COIN_OPEN_DATA_RESULT.md)에 있으며 실사용 정확도 보장으로 해석하지 않는다. [후속 원인 분리 기록](FALSE_POSITIVE_AUDIT_20261015.md).

로컬 이전 설정은 `runs/can_coin_rollback_20261015/local_before_rollback/`, Jetson 이전 설정·서비스·drop-in은 같은 run의 `jetson_before_rollback/manifest.json`에 기록했다. `rollback_result.json`에 실제 모델 체크섬, 설정, 서비스 실행 경로 및 GPU 확인 결과가 있다. 8종 실험 가중치와 평가는 비교를 위해 보존했다.
