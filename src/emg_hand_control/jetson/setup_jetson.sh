#!/usr/bin/env bash
# Jetson (JetPack 6 / Ubuntu 22.04, aarch64) 설치. 여러 번 실행해도 됩니다.
#   bash src/emg_hand_control/jetson/setup_jetson.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"                      # src/emg_hand_control
ORCA_REPO="https://github.com/orcahand/orca_core.git"
ORCA_COMMIT="177f4e7"                          # orca_core v0.5.2 (이 코드를 시험한 버전)

step() { printf '\n== %s\n' "$*"; }

step "시스템 패키지"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "근전도 환경 (Python 3.14, emg/.venv)"
# JetPack 6 기본 Python 은 3.10 이라 numpy 2.5 를 못 씀 → uv 가 따로 받은 3.14 를 씀
[ -x "$BASE/emg/.venv/bin/python" ] || uv venv --python 3.14 "$BASE/emg/.venv"
uv pip install --python "$BASE/emg/.venv/bin/python" -r "$BASE/emg/requirements.txt"

step "의수 제어 (orca_core $ORCA_COMMIT)"
if [ ! -d "$BASE/orca_core/.git" ]; then
  git clone -q "$ORCA_REPO" "$BASE/orca_core"
fi
git -C "$BASE/orca_core" fetch -q origin
git -C "$BASE/orca_core" checkout -q "$ORCA_COMMIT"
(cd "$BASE/orca_core" && uv sync --frozen --no-dev)

step "USB 고정 이름 (/dev/vash-emg, /dev/vash-hand)"
sudo cp "$HERE/99-vash.rules" /etc/udev/rules.d/99-vash.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG dialout "$USER"

step "의수 포트 설정 (hand/config.yaml)"
sed -i 's#^port: .*#port: /dev/vash-hand   # Jetson (jetson/99-vash.rules)#' "$BASE/hand/config.yaml"
grep '^port:' "$BASE/hand/config.yaml"

step "데이터 폴더"
mkdir -p "$BASE/emg/data/S01"
if [ ! -f "$BASE/emg/data/S01/grip7_base.pkl" ]; then
  echo "emg/data/S01/ 에 기본 모델이 없습니다. PC 의 data/S01 폴더(grip7_base.pkl, grip7_svm_train.npz, 착용 모델)를 복사하세요."
fi

step "실행 아이콘"
chmod +x "$HERE/start_vash.sh"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/vash.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=VASH
Comment=근전도 의수 제어
Exec=$HERE/start_vash.sh
Icon=$BASE/emg/app/assets/vash.png
Terminal=false
Categories=Utility;
EOF
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
  cp "$APPS/vash.desktop" "$DESKTOP_DIR/vash.desktop"
  chmod +x "$DESKTOP_DIR/vash.desktop"
  gio set "$DESKTOP_DIR/vash.desktop" metadata::trusted true 2>/dev/null || true
fi

step "끝"
echo "dialout 그룹이 새로 추가됐다면 한 번 로그아웃 후 다시 로그인하세요."
echo "실행: 바탕 화면의 VASH 아이콘, 또는 $HERE/start_vash.sh"
echo "장치 확인: ls -l /dev/vash-*"
