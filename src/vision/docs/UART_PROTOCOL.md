# VASH Jetson → ESP32 UART Protocol v1

UART는 115200 baud, 8 data bits, no parity, 1 stop bit(8N1)을 사용한다. 각 packet은
little-endian 고정 16 bytes다. Jetson은 inference frame마다 packet을 보내며 ESP32는 유효한
최신 sequence만 채택한다.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 2 | bytes | magic `A5 5A` |
| 2 | 1 | uint8 | version = 1 |
| 3 | 1 | uint8 | flags: bit0 target valid, bit1 new plan, bit2 LKG |
| 4 | 1 | uint8 | class ID 0–14, no target = 255 |
| 5 | 1 | uint8 | grip: 0 none, 1 POWER, 2 TRIPOD, 3 PINCH, 4 LATERAL_PINCH |
| 6 | 1 | uint8 | confidence percent 0–100 |
| 7 | 1 | uint8 | reserved = 0 |
| 8 | 2 | uint16 | normalized opening × 1000 |
| 10 | 4 | uint32 | sequence, modulo 2^32 |
| 14 | 2 | uint16 | CRC-16/CCITT-FALSE over bytes 0–13 |

`target valid` packet은 `new plan` 또는 `LKG` 중 정확히 하나만 설정한다. no-target packet은
class 255, grip/confidence/opening 0으로 전송한다. ESP32는 CRC 오류, duplicate 및 역행 sequence를
폐기한다. sequence gap은 packet loss로 기록한다. 마지막 유효 target packet 이후 250ms가 지나거나
no-target packet을 받으면 vision assistance를 비활성화하고 EMG-only control로 fallback한다.

Jetson은 제안만 전송한다. 이 protocol은 servo angle, force, contact override 권한을 갖지 않는다.
