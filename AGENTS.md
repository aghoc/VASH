# AGENTS.md — VASH 저장소 작업 규칙 (Claude Code · Codex 공통)

이 저장소는 여러 컴퓨터(Windows 노트북 2 대, Jetson)에서 AI 도구가 함께 고칩니다.

> **비전 + 근전도 통합 앱(제슨용)을 고치는 AI 는 먼저 `docs/emg_hand_control/통합앱_적용가이드_1111.md` 를 읽고, 거기 '꼭 넣을 것' 을 통합 앱에 같은 동작으로 넣어야 합니다.** `src/emg_hand_control/emg/app/` 근전도 앱은 정현 전용이라 제슨에서는 쓰지 않으므로, 그 앱만 바뀌어도 제슨에는 반영되지 않습니다.
> **11-11 밤: 근전도 밴드를 왼팔로 옮김 (거치대는 오른팔).** 앱에 '착용 팔' 선택이 생겼습니다. 통합 앱에는 적용 가이드 **7번(착용 팔 선택)** 을 넣으세요. 왼팔은 `--arm left`, 메인 모델 `grip7_main_left_*.pkl`, 착용 목록·기본 모델 강화 없음.
**작업을 시작하기 전에 이 파일과 `docs/emg_hand_control/변경기록_및_오류대처.md` 를 먼저 읽으세요.**
답은 한국어로, 짧고 직접적으로 합니다. 모의 계산·재생 결과를 실측이라고 말하지 않습니다.

## 1. 받기 전에 할 일
1. `git status` 로 로컬에 고친 파일이 있는지 봅니다. 있으면 덮어쓰기 전에 사람에게 묻습니다.
2. `git pull` 뒤에 `docs/emg_hand_control/변경기록_및_오류대처.md` 의 맨 위 항목을 읽고, 무엇이 바뀌었는지 사람에게 한두 줄로 알립니다.
3. **`src/emg_hand_control/hand/calibration.yaml` 은 손의 줄 장력 상태 값입니다.** 손을 다른 컴퓨터에서 쓰고 왔다면 이 파일이 지금 손과 안 맞을 수 있습니다 (아래 오류 표 1번). 맞지 않으면 그 컴퓨터에서 다시 캘리브레이션합니다. 다른 컴퓨터의 파일로 함부로 덮어쓰지 않습니다.

## 2. 고칠 때 규칙
- **새 동작·새 판단 규칙은 옵션으로 넣고 기본값은 꺼 둡니다.** 끈 상태가 이전과 똑같은지 확인합니다 (가짜 손·모의 계산·녹화 재생으로 판단이 같은지 비교).
- 손을 움직이는 명령(데모, 캘리브레이션, 장력)은 사람이 직접 실행하거나, 사람이 요청했을 때만 보이는 터미널에서 실행합니다.
- 개인 근전도 녹화(CSV·RAW·RUN 기록)는 올리지 않습니다. 모델 파일(`emg/data/S01/*.pkl`, gate·eval json, svm_train npz)만 올립니다. `**/data/` 는 `.gitignore` 에 있으니 `git add -f` 로 파일을 하나씩 지정합니다.
- 고친 내용은 `docs/emg_hand_control/변경기록_및_오류대처.md` 맨 위에 **날짜 · 파일 · 새 옵션과 기본값 · 확인한 것 / 확인 안 한 것** 형식으로 적습니다.
- 커밋은 기능 단위로 나누고, main 에는 PR 로 합칩니다.

## 3. 실행 요약
- **대회·시연 절차는 `docs/emg_hand_control/대회_보정절차_1111.md`**: 착용 → [위치 확인] → 정밀 보정 (거치대면 `거치대 착용` 체크) → SVM 자동 선택 → [분류 확인] → 사용
- 앱: Jetson 은 바탕 화면 VASH 아이콘(`jetson/start_vash.sh`), Windows 는 `emg/app/start_windows.vbs`
- 의수만: `cd src/emg_hand_control/orca_core && uv run --frozen --no-dev python ../hand/start_hand.py`
- 모델: 오른팔은 `emg/data/S01/` 의 **`grip7_main_lda.pkl`, `grip7_main_svm.pkl`**, 왼팔은 **`grip7_main_left_lda.pkl`, `grip7_main_left_svm.pkl`** 2 개만 씁니다 (앱 목록도 이 2 개). 앱 정밀 보정을 하면 새 파일을 만들지 않고 이 2 개를 덮어씁니다 (이전 것은 `models_old/` 에 시간 붙여 보관). 밴드를 다시 차서 인식이 나빠지면 정밀 보정을 다시 합니다.
- 기본 모델 강화(`grip7_base_right.pkl` 다시 만들기)는 녹화 원본이 있는 컴퓨터(정현 Windows 노트북)에서만 됩니다. 원본이 없는 컴퓨터에서는 정밀 보정해도 기본 모델은 그대로이고 메인 2 개만 그 착용으로 보정됩니다. 이 경우 `wears_right.json` 에 착용이 추가되지만 원본이 없으므로 그 변경은 올리지 말고 사람에게 알립니다.
- 자세한 설치·앱 설명은 `src/emg_hand_control/README.md`
