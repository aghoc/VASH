#!/usr/bin/env bash
# VASH 앱 실행 (터미널 없이). 이미 켜져 있으면 브라우저만 다시 엽니다.
# 끄기: 화면의 [앱 끄기]. 기록: ~/.cache/vash/server.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../emg"
mkdir -p "$HOME/.cache/vash"
exec .venv/bin/python app/server.py >>"$HOME/.cache/vash/server.log" 2>&1
