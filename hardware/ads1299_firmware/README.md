# ADS1299 Firmware

ESP32-S3-DevKitC-1용 펌웨어입니다. 보드를 검진하고 8채널 데이터를 PC로 보냅니다.

## Build

- Arduino IDE, Arduino-ESP32 3.x
- Open `ads1299_selftest/ads1299_selftest.ino`
- USB CDC On Boot: Disabled
- DevKit의 UART USB 포트를 사용합니다 (921600 baud)

배선은 [보드 README](../ads1299_board/README.md#esp32-s3-connections)를 참고하세요.

## Serial commands

| Key | Function |
|---|---|
| `T` | 전체 검진 |
| `L` | 실시간 EMG 스트리밍 |
| `S` | 실시간 스트리밍, 입력 단락 |
| `G` | 실시간 스트리밍, 내부 테스트 신호 |
| `W` | 배선·칩 구동 상태 |
| `R` | 선 단락 검사 |
| `D` | 전극 DC 검사 |
| `I` | 레지스터 덤프 |
| `X` | 스트리밍 중지 |
| `H` | 도움말 |

## Stream format

30바이트 프레임, 2000 Hz.

| Byte | Content |
|---|---|
| 0 | `0xA0` |
| 1 | Sequence |
| 2–4 | ADS1299 status |
| 5–28 | 8 channels, 24-bit each |
| 29 | `0xC0` |

스트리밍 시작 시 `#LIVE,fs=2000,gain=24,vref=4.5,...` 줄을 먼저 보냅니다. PC 쪽 프로그램은 `src/emg_hand_control/emg/ads_monitor.py`, `src/emg_hand_control/emg/grip_rt.py`입니다.
