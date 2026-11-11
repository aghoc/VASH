# 근전도 녹화 원본 (S01)

정현 결정(11-12)으로 올린 녹화 원본입니다. 이 폴더 밖의 녹화(CSV·RAW·RUN)는 올리지 않습니다.

| 파일 | 내용 |
|---|---|
| `HELLO_20261112a.csv` | 왼팔, 인사 자세(위팔을 어깨 높이로 들고 팔꿈치 90°, 손이 위), 7 동작 × 2 회, 2000 Hz, 8 채널 µV. 밴드는 착용 20261111c 그대로 (위치 일치도 0.89) |

- 형식: 첫 줄 `#` 헤더(녹화 설정), 그다음 `label,rep,ch1_uV … ch8_uV`. label 0 = 휴식, 1–7 = 헤더 `gestures` 순서
- 읽기: `emg_classifier.load_csv`, 채점 `python grip_model.py --arm left classcheck <파일> --model data/S01/grip7_main_left_svm.pkl`
