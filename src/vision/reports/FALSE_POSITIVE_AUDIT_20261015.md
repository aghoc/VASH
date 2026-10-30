# 실제 ELP 오탐과 모델 이력 확인 — 2026-10-15

v8 및 롤백한 can_coin_retention 모두 사용자가 실제 ELP 화면에서 오탐 과다를 확인했다. 공식 YOLO26 사전학습 원본의 동일 장면 결과는 아직 확인하지 않았다. 원인을 모델 계열 자체나 과적합 하나로 단정할 수 없다.

카메라는 사용자의 “그만” 지시 후 정지했다. 서비스 inactive/dead, MainPID=0, 같은 사용자 소유의 /dev/video 장치 점유 없음, 자동 시작 disabled를 확인했다. SSH 영상 터널도 종료했다. 새 촬영 파일은 수집하지 않았다. 재시험 상태는 `runs/can_coin_rollback_20261015/live_test_state.json`에 기록했다.

## 확인한 학습 이력

- `yolo26s_best.training.json`: public_v2를 초기값으로 사용해 15종 데이터로 추가학습했다. 공식 COCO 원본 자체가 아니다.
- `yolo26s_can_coin_v4.training.json`: best를 초기값으로 전체 모델 학습, freeze 설정 없음.
- `yolo26s_can_coin_retention.training.json`: can_coin_v4를 초기값으로 전체 모델 보완 학습, freeze 설정 없음.
- `training/train.py`는 설정의 freeze를 명시한 경우에만 전달한다. 위 두 학습은 캔·동전 출력 가중치만 따로 수정하는 방식이 아니다. 다른 클래스 성능도 바뀔 수 있다.
- 기존 캔·동전 보고서에 나머지 13종 평균 공개 FP32 test AP가 0.2011→0.1915(-4.77%), phone AP가 0.4276→0.3310으로 낮아진 사실이 이미 기록되어 있다. 이 값들은 실사용 오탐률이나 공식 COCO benchmark와 직접 비교할 수 없다.
- 기존 고정 confidence 오탐 보정은 캔·동전만 대상으로 했다. 15종 전체에 대해 실제 ELP 비대상 물체의 오탐을 배포 합격 기준으로 검사하지 않았다.

## 필요한 원인 분리

병·컵·휴대폰·리모컨·숟가락·포크·칫솔의 같은 입력에서 공식 원본과 현재 프로젝트 모델을 비교해야 한다. 공식 원본에도 오탐이 있으면 해당 촬영 조건의 기여를 확인하고, 프로젝트 모델에서만 심하면 추가학습/라벨을 조사한다. 같은 프로젝트 가중치의 PyTorch와 TensorRT 결과도 비교해 변환·클래스 번호·후처리 문제를 분리해야 한다. 임계값만 올린 결과를 인식 성능 개선으로 보고하지 않는다.

[Ultralytics 공식 YOLO26 문서](https://docs.ultralytics.com/models/yolo26/)의 COCO 사전학습 모델은 80개 클래스다. [공식 COCO 클래스 목록](https://docs.ultralytics.com/datasets/detect/coco/)에는 펜·캔·동전이 별도 클래스로 없으므로 공식 원본과 프로젝트 15종을 그대로 일대일 비교할 수 없다.

현재 배포 가중치/임계값은 사용자가 요청한 can_coin_retention 롤백 상태를 유지한다. 이번 확인에서는 카메라를 다시 켜거나 추가 학습·모델 교체를 실행하지 않았다.
