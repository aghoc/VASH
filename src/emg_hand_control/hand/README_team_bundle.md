# VGAT 오른손 — 손목 없는 설정

팀 공유본입니다. 아래 `cd orca_core` 예시는 압축을 푼 최상위 폴더에서 시작합니다. 최초 환경 설치는 최상위 `README.md`를 참고하세요.

손가락 모터 ID 2–17만 사용하는 16개 관절 설정입니다. 기본 v2 오른손 설정에서 손목 모터, 관절, 가동 범위, 중립 위치와 캘리브레이션 단계를 제외했습니다. 모터 종류·포트·통신 속도는 연결 시 자동으로 찾습니다.

텐던트를 교체한 뒤에는 먼저 수동으로 장력을 조절합니다. 연결된 XC330-T288-T 모터가 전류 기반 위치 제어 모드이고 토크가 꺼져 있을 때 아래 실행기를 사용할 수 있습니다.

```sh
cd orca_core
uv run --frozen --no-dev python ../hand-config/right-no-wrist/hold_tension.py --hold
```

자동 왕복 없이 현재 모터 위치를 목표로 기록하고, 전류를 0에서 설정된 유지 전류까지 서서히 올려 고정합니다. `HOLDING` 출력 후 스풀을 수동으로 조절합니다. Enter 또는 Ctrl+C로 종료하며, 10분이 지나면 자동 해제됩니다. 이 실행기는 제어 모드를 바꾸지 않으며, 통신 실패·하드웨어 오류·시작 위치에서 10도 넘는 이동이 확인되면 유지 작업을 끝내고 모든 모터의 토크 해제를 시도합니다. 해제 확인에 실패하면 모터 전원을 끕니다. `--hold` 없이 실행하면 읽기 전용으로 상태만 확인합니다.

교체 전의 미완료 캘리브레이션은 `backup/calibration-before-tendon-replacement-2026-10-29.yaml`에 보관했습니다. 새 텐던트에는 다시 측정한 캘리브레이션을 사용합니다.

캘리브레이션은 아래 전용 실행기를 사용합니다. 손목을 제외한 관절 목록을 명시적으로 넘기므로, 기본 루틴이 손목 단계를 자동 추가하더라도 손목을 구동하지 않습니다.

```sh
cd orca_core
uv run --frozen --no-dev python ../hand-config/right-no-wrist/calibrate_guarded.py
```

이 실행기는 기존 전류 기반 위치 모드를 유지하고, 토크를 켜기 전 전류 0을 확인합니다. 토크를 켠 뒤에도 전류를 0으로 유지한 채 현재 위치를 목표로 기록·검증하고, 그 다음 캘리브레이션 전류로 올립니다. 연결된 ID 17은 토크가 꺼져 있을 때 목표 위치를 실제 위치로 갱신하는 동작이 관찰됐습니다. 전체 실행 중 각 모터의 실제 위치와 명령 목표가 차지하는 범위가 360도를 넘으면 중단합니다. 이는 과회전 감시값이며 텐던트의 안전 장력을 보증하는 값은 아닙니다. Ctrl+C는 통신 중 예외를 발생시키지 않고 중단을 요청합니다. 종료할 때 모든 모터의 토크 해제를 읽어 확인합니다.

특정 관절만 다시 측정하려면 `--joints thumb_cmc`처럼 관절 이름을 지정합니다. 측정이 끝난 다른 관절 결과는 유지합니다. 현재 설정에는 기대 모터 회전량(`joint_motor_travel`) 기준이 없으므로, 측정 범위가 조립체의 실제 관절 가동 범위에 맞는지는 별도로 확인해야 합니다.

결과는 이 폴더의 `calibration.yaml`에 저장됩니다. 이후 제어 스크립트에도 이 폴더의 `config.yaml` 경로를 전달합니다. 설정의 `default` 전류는 다이나믹셀에서 현재 300mA로 해석됩니다. 이 값은 줄과 조립 상태에 대한 안전 보증이 아닙니다.

ORCA README의 다음 단계인 중립 자세 이동은 아래 실행기로 진행합니다.

```sh
cd orca_core
uv run --frozen --no-dev python ../hand-config/right-no-wrist/neutral_guarded.py
```

중립 목표는 이 설정의 `neutral_position`이며, 원래 `scripts/neutral.py`와 같은 `set_neutral_position()` 함수를 사용합니다. 제어 모드는 유지하고, 전류 0 상태에서 현재 위치 목표를 확인한 뒤 유지 전류를 서서히 올려 기동합니다. 목표 도달 검증과 하드웨어 오류를 확인하고 종료할 때 토크를 해제합니다. `neutral_result.json`에는 토크를 해제하기 전의 목표·실제 모터 위치, 정지 상태, 오류 상태를 기록합니다. 도달 판정 실패는 성공으로 처리하지 않습니다.

원래 `scripts/stress_test.py`의 OPEN/CLOSE 자세를 횟수를 정해 반복하려면 아래 실행기를 사용합니다. 원본 파일에서 자세를 읽으며, 손목은 현재 설정에 없으므로 제외됩니다.

```sh
cd orca_core
uv run --frozen --no-dev python ../hand-config/right-no-wrist/stress_guarded.py --cycles 3 --num-steps 50 --step-size 0.02 --hold 2
```

전류 0에서 현재 위치를 확인하는 기동 절차와 캘리브레이션 한계 내 목표 검증을 사용합니다. 온도·하드웨어 오류·통신 오류를 감시하고, 정해진 반복이 끝나거나 중단되면 토크를 해제합니다. `stress_result.json`에 완료 횟수, 최고 온도, 자세별 모터 위치 오차를 기록합니다. 반복 완료는 정밀 위치 도달 검증 통과를 뜻하지 않습니다.

원본 `examples/main_demo.py`와 `examples/main_demo_abduction.py`가 사용하는 동작은 `orca_core/data/demo_poses.yaml`에 정의되어 있습니다. 두 데모의 원래 자세·순서를 각각 1회 재생하려면 아래 실행기를 사용합니다.

```sh
cd orca_core
uv run --frozen --no-dev python ../hand-config/right-no-wrist/demo_guarded.py --demo both
```

기본 데모는 `open_hand → power_grasp → pinch → neutral`, 벌림 데모는 `fan_out → fan_in → spread_grasp → neutral` 순서입니다. `--demo main` 또는 `--demo abduction`으로 하나만 선택할 수 있습니다. 원본 데모 데이터를 읽고 현재 손에 없는 손목을 제외하며, 각 자세를 50단계로 보간하고 2초 유지합니다. 스트레스 실행기의 기동·온도·오류 감시와 종료 시 토크 해제를 공유합니다. 결과는 `demo_result.json`에 기록합니다.

집기 자세만 천천히 확인하려면 아래 명령을 사용합니다. 원래 `open_hand → pinch → open_hand → neutral` 자세를 사용하고 각 자세를 5초 유지합니다. 목표와 실제 모터 위치는 `pinch_result.json`에 기록합니다.

```sh
uv run --frozen --no-dev python ../hand-config/right-no-wrist/demo_guarded.py --demo pinch --cycles 1 --num-steps 100 --step-size 0.03 --hold 5
```
