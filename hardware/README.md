# 하드웨어

```
ads1299_board/        ADS1299 8채널 EMG 보드 (KiCad, 대회용 수정본)
ads1299_firmware/     ESP32-S3 펌웨어 (Arduino-ESP32 3.x): 보드 검진 + 실시간 스트리밍
emg_electrode_band/   전극 블록·뚜껑 CAD (STEP/STL), 렌더링·전극 배치 그림 스크립트
```

## ads1299_board
- 27 × 96 mm, 2층, ENIG. ESP32-S3-DevKitC-1 위에 J4/J5 로 적층
- `emg_ads1299.kicad_pro / .kicad_sch / .kicad_pcb` + 프로젝트 심볼·풋프린트·3D (`*.kicad_sym`, `*.pretty`, `MY_3D/`)
- `fabrication/`: Gerber·드릴, JLCPCB BOM/CPL, 수납땜 BOM
- `docs/`: 대회용 수정 내역(CHANGELOG), 설계 검토, 파일 구조, 보드·전원/신호 흐름 그림

## ads1299_firmware
- `ads1299_selftest.ino`: 배선·전원·SPI·아날로그 경로 검진(`T`), 실시간 EMG 스트리밍(`L`, 921600 baud) — PC 쪽은 `src/emg_hand_control/emg/ads_monitor.py`, `grip_rt.py`
- 배선·명령 설명은 파일 머리말

## emg_electrode_band
- `emg_electrode_block.*`, `emg_electrode_lid.*`: 밴드 전극 블록과 뚜껑 (CadQuery 스크립트 `emg_electrode_block.py`)
- `render_*.py`, `export_worn_assembly.py`: 미리보기·착용 렌더링, `draw_electrode_*.py`: 전극 배치 그림
