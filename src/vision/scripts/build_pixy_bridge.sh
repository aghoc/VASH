#!/usr/bin/env bash
# Builds libpixybridge.so from deployment/pixy_bridge.cpp against libpixyusb2.
#
# Prerequisites on the Jetson:
#   sudo apt-get install -y git libusb-1.0-0-dev g++ build-essential
#   git clone https://github.com/charmedlabs/pixy2
#   cd pixy2/scripts && ./build_libpixyusb2.sh
#
# Usage: PIXY2_ROOT=/path/to/pixy2 scripts/build_pixy_bridge.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXY2_ROOT="${PIXY2_ROOT:-$HOME/pixy2}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/artifacts/deployment/pixy/libpixybridge.so}"

if [[ ! -d "$PIXY2_ROOT" ]]; then
  echo "PIXY2_ROOT not found: $PIXY2_ROOT" >&2
  echo "Clone https://github.com/charmedlabs/pixy2 and set PIXY2_ROOT." >&2
  exit 1
fi

# Locate the headers and the static library the charmedlabs build script emits.
# Layout has moved between releases, so search instead of hardcoding paths.
header="$(find "$PIXY2_ROOT" -name libpixyusb2.h -print -quit)"
if [[ -z "$header" ]]; then
  echo "libpixyusb2.h not found under $PIXY2_ROOT; run scripts/build_libpixyusb2.sh first." >&2
  exit 1
fi

archive="$(find "$PIXY2_ROOT" -type f \( -name 'libpixy2.a' -o -name 'libpixyusb2.a' \) -print -quit)"
if [[ -z "$archive" ]]; then
  echo "libpixy2.a/libpixyusb2.a not found under $PIXY2_ROOT; build libpixyusb2 first." >&2
  exit 1
fi

# Resolve the three public include roots used by the official Makefile. Avoid
# device firmware include directories because they contain colliding headers.
tpixy_header="$(find "$PIXY2_ROOT" -path '*/host/arduino/libraries/Pixy2/TPixy2.h' -print -quit)"
chirp_header="$(find "$PIXY2_ROOT" -path '*/common/inc/chirp.hpp' -print -quit)"
if [[ -z "$tpixy_header" || -z "$chirp_header" ]]; then
  echo "TPixy2.h or chirp.hpp not found under $PIXY2_ROOT." >&2
  exit 1
fi
includes=(
  "-I$(dirname "$header")"
  "-I$(dirname "$tpixy_header")"
  "-I$(dirname "$chirp_header")"
)

usb_cflags="$(pkg-config --cflags libusb-1.0 2>/dev/null || echo -I/usr/include/libusb-1.0)"
usb_libs="$(pkg-config --libs libusb-1.0 2>/dev/null || echo -lusb-1.0)"

mkdir -p "$(dirname "$OUTPUT")"
echo "libpixyusb2.h : $header"
echo "Pixy archive    : $archive"
echo "output        : $OUTPUT"

g++ -O2 -fPIC -shared -std=c++11 \
  "$PROJECT_ROOT/deployment/pixy_bridge.cpp" \
  -o "$OUTPUT" \
  "${includes[@]}" $usb_cflags \
  -Wl,--whole-archive "$archive" -Wl,--no-whole-archive \
  $usb_libs -lpthread

echo "Built $OUTPUT"
echo "Export it for the Python frame source:"
echo "  export VAGT_PIXY_BRIDGE=$OUTPUT"
