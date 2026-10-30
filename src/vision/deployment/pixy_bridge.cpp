// C ABI shim exposing Pixy2 raw-frame capture to Python via ctypes.
//
// libpixyusb2 does not expose PixyMon's streaming mode, so the only full-frame
// path is Link2USB::getRawFrame(), which requires the running program to be
// stopped first. Two grab modes are provided because charmedlabs does not
// document whether a single stop() lets getRawFrame() observe new sensor
// frames; scripts/pixy_benchmark.py measures both and reports frame novelty.
//
//   persistent (cycle=0): stop() once at open, then getRawFrame() repeatedly.
//   cycled     (cycle=1): resume(), settle, stop(), getRawFrame() per call.
//
// Frames are returned as raw BGGR Bayer, 1 byte per pixel. Demosaic happens in
// Python with cv2.cvtColor so we do not duplicate OpenCV's optimized path.

#include <cstdint>
#include <cstring>
#include <unistd.h>

#include "libpixyusb2.h"

namespace {

Pixy2 *g_pixy = nullptr;
bool g_stopped = false;

constexpr int kFrameBytes = PIXY2_RAW_FRAME_WIDTH * PIXY2_RAW_FRAME_HEIGHT;

}  // namespace

extern "C" {

int pixy_bridge_frame_width(void) { return PIXY2_RAW_FRAME_WIDTH; }

int pixy_bridge_frame_height(void) { return PIXY2_RAW_FRAME_HEIGHT; }

int pixy_bridge_frame_bytes(void) { return kFrameBytes; }

// Returns 0 on success, or the negative libpixyusb2 error code.
int pixy_bridge_open(void) {
  if (g_pixy != nullptr) {
    return 0;
  }
  Pixy2 *pixy = new Pixy2();
  const int rc = pixy->init();
  if (rc < 0) {
    delete pixy;
    return rc;
  }
  g_pixy = pixy;
  g_stopped = false;
  return 0;
}

// Stops the on-device program so getRawFrame() has a stable frame buffer.
int pixy_bridge_stop(void) {
  if (g_pixy == nullptr) {
    return -1;
  }
  const int rc = g_pixy->m_link.stop();
  if (rc >= 0) {
    g_stopped = true;
  }
  return rc;
}

int pixy_bridge_resume(void) {
  if (g_pixy == nullptr) {
    return -1;
  }
  const int rc = g_pixy->m_link.resume();
  if (rc >= 0) {
    g_stopped = false;
  }
  return rc;
}

// Copies one BGGR Bayer frame into `out`. `size` must be >= frame_bytes().
// cycle != 0 runs resume/settle/stop first so the sensor advances a frame.
int pixy_bridge_grab(uint8_t *out, int size, int cycle, int settle_us) {
  if (g_pixy == nullptr || out == nullptr) {
    return -1;
  }
  if (size < kFrameBytes) {
    return -2;
  }
  if (cycle != 0) {
    const int resumed = g_pixy->m_link.resume();
    if (resumed < 0) {
      return resumed;
    }
    g_stopped = false;
    if (settle_us > 0) {
      usleep(static_cast<useconds_t>(settle_us));
    }
  }
  if (!g_stopped) {
    const int stopped = g_pixy->m_link.stop();
    if (stopped < 0) {
      return stopped;
    }
    g_stopped = true;
  }
  uint8_t *bayer = nullptr;
  const int rc = g_pixy->m_link.getRawFrame(&bayer);
  if (rc < 0) {
    return rc;
  }
  if (bayer == nullptr) {
    return -3;
  }
  memcpy(out, bayer, kFrameBytes);
  return 0;
}

int pixy_bridge_set_lamp(int upper, int lower) {
  if (g_pixy == nullptr) {
    return -1;
  }
  return g_pixy->setLamp(static_cast<uint8_t>(upper), static_cast<uint8_t>(lower));
}

// Leaves Pixy running so the device is not stranded in the stopped state.
void pixy_bridge_close(void) {
  if (g_pixy == nullptr) {
    return;
  }
  g_pixy->m_link.resume();
  delete g_pixy;
  g_pixy = nullptr;
  g_stopped = false;
}

}  // extern "C"
