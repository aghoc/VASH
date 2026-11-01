/*
 * ads1299_selftest.ino  —  ADS1299 EMG 보드 초기 검진 펌웨어 (ESP32-S3, Arduino-ESP32 3.x)
 *
 * 목적: 점퍼선 14개 연결 상태, 전원(AVDD/DVDD), SPI, 아날로그 신호 경로를
 *       데이터시트(TI SBAS499C) 기준으로 하나씩 검증하고, 이후 실시간 스트리밍으로
 *       PC(ads_monitor.py)에서 원신호/필터 결과를 확인한다.
 *
 * 배선 (보드 J4/J5 ↔ DevKitC-1 J1/J3, 점퍼 14개)
 *   J4.1  DVDD  ← 3V3          J4.15 DRDY  → GPIO9
 *   J4.16 CS    ← GPIO10       J4.17 DIN   ← GPIO11 (MOSI)
 *   J4.18 SCLK  ← GPIO12       J4.19 DOUT  → GPIO13 (MISO)
 *   J4.20 START ← GPIO14       J4.21 5V    ← 5V        J4.22 GND
 *   J5.17 RESET ← GPIO47       J5.18 PWDN  ← GPIO21    J5.1/21/22 GND   (WIRING_MODE 0)
 *   WIRING_MODE 1: 왼쪽 열 J4 직결, RESET←GPIO4(J4.4), PWDN←GPIO5(J4.5) 보드 뒷면 전선
 *
 * 시리얼: 921600 baud (DevKit의 "UART/COM" USB 포트 사용, USB CDC On Boot = Disabled)
 * 명령:  T = 전체 검진    L = 실시간 EMG    S = 실시간(입력 단락)    G = 실시간(내부 테스트신호)
 *        W = 배선/칩 구동 상태   R = 선 단락 검사(멀티미터 없이)   M/N = 멀티미터 측정 모드(HIGH/LOW 고정)
 *        X = 스트리밍 중지  I = 레지스터 덤프   H = 도움말
 */
#include <Arduino.h>
#include <SPI.h>
#include <algorithm>
#include "esp_efuse.h"
#include "esp_efuse_table.h"
#include "driver/gpio.h"
#include "driver/rtc_io.h"
#include "selftest_types.h"

// ---------------- 핀 ----------------
static const int PIN_DRDY  = 9;
static const int PIN_CS    = 10;
static const int PIN_MOSI  = 11;
static const int PIN_SCLK  = 12;
static const int PIN_MISO  = 13;
static const int PIN_START = 14;
// ---- 배선 방식 선택 ----
// 0 = 점퍼선 14개 (RESET←GPIO47 J5.17, PWDN←GPIO21 J5.18)
// 1 = ESP32 왼쪽 열을 J4에 직결 + ADS 보드 뒷면 전선 2개
//     (J4.4 [GPIO4] → J5.17 RESET,  J4.5 [GPIO5] → J5.18 PWDN)
#define WIRING_MODE 0
#if WIRING_MODE == 1
static const int PIN_RESET = 4;
static const int PIN_PWDN  = 5;
#define RST_TXT  "RESET(GPIO4→J4.4→뒷면전선→J5.17)"
#define PWDN_TXT "PWDN(GPIO5→J4.5→뒷면전선→J5.18)"
#else
static const int PIN_RESET = 47;
static const int PIN_PWDN  = 21;
#define RST_TXT  "RESET(J5.17↔GPIO47)"
#define PWDN_TXT "PWDN(J5.18↔GPIO21)"
#endif

// ---------------- ADS1299 명령/레지스터 (SBAS499C §9.5) ----------------
enum : uint8_t { C_WAKEUP=0x02, C_STANDBY=0x04, C_RESET=0x06, C_START=0x08, C_STOP=0x0A,
                 C_RDATAC=0x10, C_SDATAC=0x11, C_RDATA=0x12 };
enum : uint8_t { R_ID=0x00, R_CONFIG1=0x01, R_CONFIG2=0x02, R_CONFIG3=0x03, R_LOFF=0x04,
                 R_CH1SET=0x05, R_BIAS_SENSP=0x0D, R_BIAS_SENSN=0x0E, R_LOFF_SENSP=0x0F,
                 R_LOFF_SENSN=0x10, R_LOFF_FLIP=0x11, R_LOFF_STATP=0x12, R_LOFF_STATN=0x13,
                 R_GPIO=0x14, R_MISC1=0x15, R_MISC2=0x16, R_CONFIG4=0x17 };
// CONFIG1: 0b1 0 0 10 DR  (bit7=1, DAISY_EN=0, CLK_EN=0, bit4:3=10)
static const uint8_t CFG1_250  = 0x96, CFG1_500 = 0x95, CFG1_1K = 0x94, CFG1_2K = 0x93;
static const uint8_t CFG2_NORMAL = 0xC0;   // test signal off
static const uint8_t CFG2_TEST   = 0xD1;   // INT_CAL=1, 1x amp, fCLK/2^20 (~1.95 Hz)
static const uint8_t CFG3_REF    = 0xE0;   // PD_REFBUF=1 (내부 4.5V 기준), bias off
static const uint8_t CFG3_BIAS   = 0xEC;   // + BIASREF_INT=1, PD_BIAS=1 (bias 구동 on)
// CHnSET: PD | GAIN[6:4] | SRB2 | MUX[2:0]
static const uint8_t MUX_NORMAL=0, MUX_SHORT=1, MUX_MVDD=3, MUX_TEMP=4, MUX_TEST=5;
static const uint8_t GAIN_1=0x00, GAIN_24=0x60;

static const float VREF = 4.5f;            // VREFP-VREFN (내부 기준)

// ---------------- 상태 ----------------
SPIClass spi(FSPI);
static uint32_t spiHz = 1000000;
static volatile uint32_t drdyCount = 0;
static volatile bool drdyFlag = false;
static bool useStartCmd = false;           // START 핀이 불량이면 명령으로 대체
static int nPass=0, nWarn=0, nFail=0;

void IRAM_ATTR onDrdy() { drdyCount = drdyCount + 1; drdyFlag = true; }

// ---------------- 출력 헬퍼 ----------------
static void result(const char* id, const char* lvl, const String& msg) {
  if (!strcmp(lvl,"PASS")) nPass++; else if (!strcmp(lvl,"WARN")) nWarn++; else nFail++;
  Serial.printf("RESULT,%s,%s,%s\n", id, lvl, msg.c_str());
}
static void info(const String& s) { Serial.print("  "); Serial.println(s); }

// ---------------- SPI 저수준 ----------------
static inline void csLow()  { digitalWrite(PIN_CS, LOW);  delayMicroseconds(1); }
static inline void csHigh() { delayMicroseconds(3); digitalWrite(PIN_CS, HIGH); delayMicroseconds(2); } // tSCCS ≥ 4 tCLK
static void beginTx() { spi.beginTransaction(SPISettings(spiHz, MSBFIRST, SPI_MODE1)); }
static void endTx()   { spi.endTransaction(); }

static void cmd(uint8_t c) {
  beginTx(); csLow(); spi.transfer(c); csHigh(); endTx();
  delayMicroseconds(4);                    // tSDECODE (4 tCLK ≈ 2 µs)
}
static uint8_t rreg(uint8_t a) {
  beginTx(); csLow();
  spi.transfer(0x20 | a); delayMicroseconds(3);
  spi.transfer(0x00);     delayMicroseconds(3);
  uint8_t v = spi.transfer(0x00);
  csHigh(); endTx(); delayMicroseconds(4);
  return v;
}
static void wreg(uint8_t a, uint8_t v) {
  beginTx(); csLow();
  spi.transfer(0x40 | a); delayMicroseconds(3);
  spi.transfer(0x00);     delayMicroseconds(3);
  spi.transfer(v);
  csHigh(); endTx(); delayMicroseconds(4);
}
static void allCh(uint8_t v) { for (int i=0;i<8;i++) wreg(R_CH1SET+i, v); }

static void startConv() {
  if (useStartCmd) { digitalWrite(PIN_START, LOW); cmd(C_START); }
  else digitalWrite(PIN_START, HIGH);
}
static void stopConv() {
  digitalWrite(PIN_START, LOW);
  cmd(C_STOP);
  delay(2);
}
static void hwReset() {
  digitalWrite(PIN_RESET, LOW);  delayMicroseconds(20);   // ≥ 2 tCLK
  digitalWrite(PIN_RESET, HIGH); delay(2);                 // ≥ 18 tCLK
}
static uint32_t countDrdy(uint32_t ms) {
  uint32_t c0 = drdyCount; delay(ms); return drdyCount - c0;
}

// 1 프레임(27 byte) 읽기 — RDATAC 모드, DRDY 하강 후 호출
static void readFrame(uint8_t* b) {
  beginTx(); csLow();
  for (int i=0;i<27;i++) b[i] = spi.transfer(0x00);   // DIN은 계속 Low
  csHigh(); endTx();
}
static inline int32_t s24(const uint8_t* p) {
  int32_t v = ((int32_t)p[0]<<16) | ((int32_t)p[1]<<8) | p[2];
  if (v & 0x800000) v |= 0xFF000000;
  return v;
}
static inline float codeToV(int32_t c, float gain) { return (float)c * (VREF/gain) / 8388608.0f; }

// n개 샘플 수집 (skip개 버림). 콜백으로 채널 코드 전달. 실패 시 false
static uint32_t badHeader = 0;
static bool acquire(int n, int skip, SampleFn fn, void* ctx) {
  stopConv();
  cmd(C_RDATAC);
  drdyFlag = false; badHeader = 0;
  startConv();
  uint8_t b[27]; int32_t ch[8];
  for (int k=0; k < n+skip; k++) {
    uint32_t t0 = millis();
    while (!drdyFlag) { if (millis()-t0 > 200) { stopConv(); cmd(C_SDATAC); return false; } }
    drdyFlag = false;
    readFrame(b);
    if ((b[0] & 0xF0) != 0xC0) badHeader++;
    if (k < skip) continue;
    for (int c=0;c<8;c++) ch[c] = s24(&b[3+3*c]);
    uint32_t st = ((uint32_t)b[0]<<16)|((uint32_t)b[1]<<8)|b[2];
    fn(ch, st, ctx);
  }
  stopConv();
  cmd(C_SDATAC);
  return true;
}

// ---------------- 통계 ----------------
static void statInit(Stat& s){ memset(&s,0,sizeof(s)); for(int i=0;i<8;i++){s.mn[i]=INT32_MAX;s.mx[i]=INT32_MIN;} }
static void statAdd(const int32_t ch[8], uint32_t, void* ctx) {
  Stat& s = *(Stat*)ctx; s.n++;
  for (int i=0;i<8;i++) {
    double x = ch[i], d = x - s.mean[i];
    s.mean[i] += d / s.n; s.m2[i] += d * (x - s.mean[i]);
    if (ch[i]<s.mn[i]) s.mn[i]=ch[i]; if (ch[i]>s.mx[i]) s.mx[i]=ch[i];
  }
}

// ======================= 검진 항목 =======================

static void t0_system() {
  Serial.println("\n[T0] ESP32-S3 모듈 정보 / GPIO47 전압 도메인");
  info(String("칩: ") + ESP.getChipModel() + " rev" + ESP.getChipRevision() +
       "  Flash " + (ESP.getFlashChipSize()>>20) + "MB  PSRAM " + (ESP.getPsramSize()>>20) + "MB");
  bool force = esp_efuse_read_field_bit(ESP_EFUSE_VDD_SPI_FORCE);
  bool tieh  = esp_efuse_read_field_bit(ESP_EFUSE_VDD_SPI_TIEH);
  if (force && !tieh) {
    result(WIRING_MODE ? "T0" : "T0", WIRING_MODE ? "WARN" : "FAIL", "VDD_SPI=1.8V (WROOM-2 'V' 모델) → GPIO47/48 은 1.8V. (WIRING_MODE 1 은 GPIO4 사용이라 영향 없음)");
  } else if (force && tieh) {
    result("T0","PASS","VDD_SPI=3.3V (eFuse 고정) → GPIO47 3.3V 정상");
  } else {
    result("T0","PASS","VDD_SPI eFuse 미설정 → GPIO45 스트랩 기본값 3.3V (WROOM-1) → GPIO47 3.3V 정상");
  }
}

// T1: SPI 없이 START 핀 → DRDY 펄스 확인 (리셋 직후 기본 250 SPS, RDATAC)
static bool t1_ok = false;
static void t1_drdy_start() {
  Serial.println("\n[T1] 전원·RESET·PWDN·START·DRDY 연결 (SPI 사용 안 함)");
  digitalWrite(PIN_START, LOW);
  hwReset();                                   // 기본값: CONFIG1=0x96 (250 SPS)
  info(String("START Low 상태 DRDY 레벨 = ") + digitalRead(PIN_DRDY));
  digitalWrite(PIN_START, HIGH); delay(20);
  uint32_t e = countDrdy(400);                 // 기대 100개 (250 Hz × 0.4 s)
  digitalWrite(PIN_START, LOW); delay(20);
  uint32_t e2 = countDrdy(200);                // 정지 확인
  info(String("START High 0.4 s 동안 DRDY 하강 = ") + e + " (기대 100)  /  START Low 후 = " + e2 + " (기대 0)");
  if (e >= 90 && e <= 110 && e2 == 0) {
    t1_ok = true;
    result("T1","PASS","DRDY 250 Hz 정상 → 3V3·GND·" RST_TXT "·" PWDN_TXT "·START(GPIO14)·DRDY(GPIO9) 연결 OK, 내부 클럭 OK");
  } else if (e >= 90 && e <= 110 && e2 > 0) {
    result("T1","WARN","DRDY는 나오나 START Low에도 멈추지 않음 → START 점퍼(J4.20↔GPIO14) 확인");
  } else if (e > 0) {
    result("T1","WARN", String("DRDY 주파수 이상 (") + e*2.5f + " Hz) → DRDY 점퍼 접촉 불량 또는 잡음 의심");
  } else {
    result("T1","FAIL","DRDY 펄스 없음 → T2/T5 결과로 원인 분리 (SPI 통하면 START/DRDY 점퍼, 안 통하면 전원/RESET/PWDN)");
  }
}

// T2: ID 읽기 (SPI 4선 + DVDD)
static bool t2_ok = false;
static void t2_id() {
  Serial.println("\n[T2] SPI 통신 — ID 레지스터 (기대 0x3E: ADS1299 8ch)");
  spiHz = 1000000;
  hwReset(); cmd(C_SDATAC);
  int good=0; uint8_t last=0, seen[4]={0}; int ns=0;
  for (int i=0;i<20;i++) {
    uint8_t id = rreg(R_ID); last=id;
    if ((id & 0x1F) == 0x1E) good++;
    bool dup=false; for(int k=0;k<ns;k++) if(seen[k]==id) dup=true;
    if(!dup && ns<4) seen[ns++]=id;
  }
  String vals; for(int k=0;k<ns;k++){ char t[8]; sprintf(t,"0x%02X ",seen[k]); vals+=t; }
  info("읽은 값: " + vals + " (20회 중 정상 " + good + ")");
  if (good == 20) {
    t2_ok = true;
    char t[64]; sprintf(t,"ID=0x%02X (REV_ID=%d, 8ch) → CS·SCLK·MOSI·MISO·DVDD 정상", last, last>>5);
    result("T2","PASS",t);
  } else if (good > 0) {
    result("T2","WARN","ID가 간헐적으로만 맞음 → SPI 점퍼 접촉 불량 (J4.16~19) 또는 선 길이/잡음");
  } else if (last == 0xFF) {
    result("T2","FAIL","항상 0xFF → MISO(J4.19↔GPIO13) 단선, 또는 칩 무응답(CS/SCLK/MOSI 단선, PWDN/RESET Low, DVDD 없음)");
  } else if (last == 0x00) {
    result("T2","FAIL","항상 0x00 → MISO가 GND에 붙음, 또는 SCLK(J4.18) 단선");
  } else {
    result("T2","FAIL","ID 불일치 → MOSI/SCLK 순서 바뀜 또는 SPI 모드 문제. 점퍼 J4.16~19 순서 재확인");
  }
}

// T3: 레지스터 쓰기/읽기 패턴 (MOSI 비트 단위 검증)
static void t3_pattern() {
  Serial.println("\n[T3] 레지스터 쓰기/읽기 패턴 (LOFF_SENSP, 비트별 MOSI/MISO 검증)");
  if (!t2_ok) { result("T3","FAIL","T2 실패로 생략"); return; }
  uint8_t pats[] = {0x00,0xFF,0x55,0xAA,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80};
  uint8_t stuck1=0, stuck0=0; int err=0;
  for (uint8_t p : pats) {
    wreg(R_LOFF_SENSP, p); uint8_t r = rreg(R_LOFF_SENSP);
    if (r != p) { err++; stuck1 |= (r & ~p); stuck0 |= (~r & p); }
  }
  wreg(R_LOFF_SENSP, 0x00);
  if (!err) result("T3","PASS","12개 패턴 모두 일치 → 8비트 전부 정상");
  else { char t[96]; sprintf(t,"%d개 불일치, 1로 고정 비트=0x%02X, 0으로 고정 비트=0x%02X → MOSI/MISO 점퍼 확인", err, stuck1, stuck0);
         result("T3","FAIL",t); }
}

// T4: SPI 속도 여유 (점퍼선 품질)
static uint32_t bestHz = 1000000;
static void t4_speed() {
  Serial.println("\n[T4] SPI 속도 여유 (점퍼선 신호 품질)");
  if (!t2_ok) { result("T4","FAIL","T2 실패로 생략"); return; }
  uint32_t hz[] = {1000000, 2000000, 4000000, 8000000};
  bestHz = 1000000;
  for (uint32_t h : hz) {
    spiHz = h; int e=0;
    for (int i=0;i<200;i++) { if ((rreg(R_ID)&0x1F)!=0x1E) e++; uint8_t p=(uint8_t)(i*37+11); wreg(R_LOFF_SENSN,p); if(rreg(R_LOFF_SENSN)!=p) e++; }
    info(String(h/1000000) + " MHz: 오류 " + e + " / 400");
    if (e==0) bestHz = h; else break;
  }
  spiHz = 1000000; wreg(R_LOFF_SENSN,0x00);
  bestHz = min(bestHz, (uint32_t)4000000);     // 스트리밍은 최대 4 MHz로 제한
  if (bestHz >= 2000000) result("T4","PASS", String("오류 없음 → 스트리밍 SPI ") + bestHz/1000000 + " MHz 사용");
  else result("T4","WARN","2 MHz 이하에서 오류 → 점퍼선 길이 줄이기/GND선 추가 권장. 1 MHz로 진행");
}

// T5: START 핀 vs START 명령 (T1 실패 시 원인 분리)
static void t5_start_isolate() {
  Serial.println("\n[T5] START/DRDY 원인 분리");
  if (t1_ok) { result("T5","PASS","T1 통과 → START 핀 정상 (분리 시험 불필요)"); return; }
  if (!t2_ok) { result("T5","FAIL","SPI도 실패 → 칩 전원/RESET/PWDN 쪽 문제. 멀티미터로 DVDD(3.3V)·5V·RESET·PWDN 핀 전압 확인"); return; }
  hwReset(); cmd(C_SDATAC);
  digitalWrite(PIN_START, LOW);
  cmd(C_RDATAC); cmd(C_START); delay(20);
  uint32_t e = countDrdy(400);
  cmd(C_STOP); cmd(C_SDATAC);
  info(String("START 명령으로 DRDY = ") + e + " (기대 100)");
  if (e >= 90 && e <= 110) {
    useStartCmd = true;
    result("T5","FAIL","START 명령은 동작, START 핀은 무반응 → START 점퍼(J4.20↔GPIO14) 불량. 이후 명령 방식으로 대체 진행");
  } else {
    // DRDY 선만 끊긴 경우: RDATA로 데이터 헤더 확인
    cmd(C_START); delay(50);
    beginTx(); csLow(); spi.transfer(C_RDATA); delayMicroseconds(3);
    uint8_t h = spi.transfer(0); for(int i=0;i<26;i++) spi.transfer(0); csHigh(); endTx();
    cmd(C_STOP);
    if ((h & 0xF0) == 0xC0) result("T5","FAIL","변환은 진행 중(헤더 0xC_), DRDY 신호만 안 들어옴 → DRDY 점퍼(J4.15↔GPIO9) 불량");
    else result("T5","FAIL","변환 자체가 안 됨 → AVDD(아날로그 5V) 또는 클럭 문제. T8 전원 측정/멀티미터 확인");
  }
}

// T6: RESET 핀
static void t6_reset() {
  Serial.println("\n[T6] " RST_TXT);
  if (!t2_ok) { result("T6","FAIL","T2 실패로 생략"); return; }
  cmd(C_SDATAC);
  wreg(R_CONFIG1, CFG1_500);
  uint8_t before = rreg(R_CONFIG1);
  hwReset(); cmd(C_SDATAC);
  uint8_t after = rreg(R_CONFIG1);
  char t[96]; sprintf(t,"CONFIG1 변경 0x%02X → RESET 펄스 후 0x%02X (기대 0x96)", before, after); info(t);
  if (before == CFG1_500 && after == CFG1_250) result("T6","PASS","RESET 핀 정상 (레지스터 기본값 복귀)");
  else if (after == CFG1_500) result("T6","FAIL","RESET 펄스가 칩에 안 들어감 → " RST_TXT " 배선 확인");
  else result("T6","WARN","판정 불가 (쓰기 실패) → T3 결과 확인");
}

// T7: PWDN 핀
static void t7_pwdn() {
  Serial.println("\n[T7] " PWDN_TXT);
  if (!t2_ok) { result("T7","FAIL","T2 실패로 생략"); return; }
  cmd(C_SDATAC);
  digitalWrite(PIN_PWDN, LOW); delay(20);
  uint8_t idOff = rreg(R_ID);
  digitalWrite(PIN_PWDN, HIGH); delay(200);    // 발진기·기준전압 기상
  hwReset(); cmd(C_SDATAC);
  uint8_t idOn = rreg(R_ID);
  char t[96]; sprintf(t,"PWDN Low 중 ID=0x%02X (기대 ≠0x3E) / 복귀 후 ID=0x%02X (기대 0x3E)", idOff, idOn); info(t);
  if ((idOff & 0x1F) != 0x1E && (idOn & 0x1F) == 0x1E) result("T7","PASS","PWDN 핀 정상 (전원 차단/복귀 확인)");
  else if ((idOff & 0x1F) == 0x1E) result("T7","FAIL","PWDN Low에도 칩이 응답 → " PWDN_TXT " 배선 확인");
  else result("T7","FAIL","PWDN 복귀 후 칩 무응답 → 전원 재인가 후 재시험");
}

// 공통 설정: 내부 기준 켜고 대기
static void refOn(uint8_t cfg3) {
  cmd(C_SDATAC);
  wreg(R_CONFIG3, cfg3);
  delay(200);                                  // 내부 기준 + VREFP 10µF 충전
}

// T8: 전원 전압 측정 (MVDD, gain 1)
static void t8_supply() {
  Serial.println("\n[T8] 전원 전압 (ADS1299 내부 측정, MUX=MVDD, Gain 1)");
  if (!t2_ok) { result("T8","FAIL","T2 실패로 생략"); return; }
  spiHz = bestHz;
  // 전극이 몸에 붙어 있어도 측정이 흔들리지 않도록 BIAS 를 켜서 몸을 중간전압(≈2.5 V)에 고정.
  // BIAS_SENS=0 이면 BIAS 출력 = 내부 기준(중간전압). (v1.8 까지는 BIAS off → 전극 부착 시 T8~T11 오차)
  refOn(CFG3_BIAS);
  wreg(R_BIAS_SENSP, 0x00); wreg(R_BIAS_SENSN, 0x00);
  wreg(R_CONFIG1, CFG1_250); wreg(R_CONFIG2, CFG2_NORMAL);
  allCh(GAIN_1 | MUX_MVDD);
  Stat s; statInit(s);
  if (!acquire(100, 20, statAdd, &s)) { result("T8","FAIL","데이터 수집 타임아웃 (DRDY 없음)"); return; }
  // ch1,2,5,6,7,8: 0.5*(AVDD+AVSS) / ch3,4: DVDD/4  (SBAS499C §9.3.1.1.4)
  double a=0; int na=0; int idxA[]={0,1,4,5,6,7};
  for (int i: idxA) { a += codeToV((int32_t)s.mean[i],1); na++; }
  double avdd = 2.0*a/na;
  double dvdd = 4.0*(codeToV((int32_t)s.mean[2],1)+codeToV((int32_t)s.mean[3],1))/2.0;
  char t[128]; sprintf(t,"AVDD = %.3f V (기대 5.00)   DVDD = %.3f V (기대 3.30)   헤더오류 %lu", avdd, dvdd, (unsigned long)badHeader); info(t);
  // 채널간 일치도
  double amin=9,amax=0; for(int i: idxA){double v=2*codeToV((int32_t)s.mean[i],1); amin=min(amin,v); amax=max(amax,v);}
  sprintf(t,"AVDD 채널간 편차 %.1f mV", (amax-amin)*1000); info(t);
  if (avdd>4.85 && avdd<5.15) result("T8a","PASS", String("AVDD ") + String(avdd,3) + " V → 차지펌프(MAX1683)+LDO(LT1763) 정상");
  else if (avdd>4.6 && avdd<5.3) result("T8a","WARN", String("AVDD ") + String(avdd,3) + " V → 허용범위 경계. 5V(J4.21) 공급·C36 전압 멀티미터 확인");
  else result("T8a","FAIL", String("AVDD ") + String(avdd,3) + " V → 아날로그 전원 이상. 5V 점퍼(J4.21), U2 출력(≈9V), U3 출력(5V) 순서로 멀티미터 확인");
  if (dvdd>3.0 && dvdd<3.6) result("T8b","PASS", String("DVDD ") + String(dvdd,3) + " V 정상");
  else result("T8b","WARN", String("DVDD ") + String(dvdd,3) + " V → 3V3 점퍼(J4.1) 확인");
}

// T9: 칩 온도
static void t9_temp() {
  Serial.println("\n[T9] 칩 내부 온도 (MUX=TEMP)");
  if (!t2_ok) { result("T9","FAIL","T2 실패로 생략"); return; }
  allCh(GAIN_1 | MUX_TEMP);
  Stat s; statInit(s);
  if (!acquire(50, 20, statAdd, &s)) { result("T9","FAIL","타임아웃"); return; }
  double uv = codeToV((int32_t)s.mean[0],1)*1e6;
  double tc = (uv - 145300.0)/490.0 + 25.0;          // SBAS499C 식(3)
  char t[80]; sprintf(t,"%.0f µV → %.1f °C", uv, tc); info(t);
  if (tc>10 && tc<70) result("T9","PASS", String(tc,1) + " °C 정상 범위");
  else result("T9","WARN", String(tc,1) + " °C → 비정상 (기준전압/AVDD 확인)");
}

// T10: 내부 테스트 신호 (gain 24) — 8채널 신호경로/PGA/ADC
#define TS_N 500
static int32_t tsBuf[8][TS_N]; static int tsIdx=0;
static void tsAdd(const int32_t ch[8], uint32_t, void*) { if (tsIdx<TS_N){ for(int i=0;i<8;i++) tsBuf[i][tsIdx]=ch[i]; tsIdx++; } }
static void t10_testsig() {
  Serial.println("\n[T10] 내부 테스트 신호 (CONFIG2=0xD1, ~1.95 Hz 구형파, Gain 24, 250 SPS, 2 s)");
  if (!t2_ok) { result("T10","FAIL","T2 실패로 생략"); return; }
  wreg(R_CONFIG2, CFG2_TEST); wreg(R_CONFIG1, CFG1_250);
  allCh(GAIN_24 | MUX_TEST);
  tsIdx=0;
  if (!acquire(TS_N, 30, tsAdd, nullptr)) { result("T10","FAIL","타임아웃"); return; }
  wreg(R_CONFIG2, CFG2_NORMAL);
  float pp[8], fr[8];
  for (int c=0;c<8;c++) {
    int32_t mn=INT32_MAX,mx=INT32_MIN; for(int i=0;i<TS_N;i++){mn=min(mn,tsBuf[c][i]);mx=max(mx,tsBuf[c][i]);}
    double mid=(mn+(double)mx)/2, hi=0, lo=0; int nh=0, nl=0, cross=0; bool above = tsBuf[c][0]>mid;
    for(int i=0;i<TS_N;i++){ double x=tsBuf[c][i]; if(x>mid){hi+=x;nh++;} else {lo+=x;nl++;}
      bool a = x>mid; if(a!=above){cross++; above=a;} }
    pp[c] = (nh&&nl) ? codeToV((int32_t)(hi/nh - lo/nl),24)*1000.0f : 0;   // mV
    fr[c] = cross / 2.0f / (TS_N/250.0f);
    char t[80]; sprintf(t,"CH%d: %.3f mVpp, %.2f Hz", c+1, pp[c], fr[c]); info(t);
  }
  float s[8]; memcpy(s,pp,sizeof(s)); std::sort(s,s+8); float med=(s[3]+s[4])/2;
  int bad=0; String which;
  for(int c=0;c<8;c++) if (fabs(pp[c]-med) > 0.05f*med || fr[c]<1.6f || fr[c]>2.3f) { bad++; which += String(" CH")+(c+1); }
  char t[120]; sprintf(t,"중앙값 %.3f mVpp (기대 3.75 mVpp = ±(VREFP-VREFN)/2.4 = ±1.875 mV)", med); info(t);
  if (!bad && med>3.4f && med<4.1f) result("T10","PASS","8채널 모두 동일 진폭·주파수 → PGA/ADC/기준전압 정상");
  else result("T10","FAIL", String("이상 채널:") + (bad?which:String(" 전체(진폭 범위 밖)")));
}

// T11: 입력 단락 잡음 (gain 24) — 250 SPS(데이터시트 비교) + 2000 SPS(실사용)
static void noiseRun(uint8_t cfg1, float fs, float limPass, float limWarn, const char* id) {
  wreg(R_CONFIG1, cfg1); allCh(GAIN_24 | MUX_SHORT);
  Stat s; statInit(s);
  int n = (int)(fs*2);
  if (!acquire(n, (int)(fs*0.2f), statAdd, &s)) { result(id,"FAIL","타임아웃"); return; }
  float worst=0; int wc=0; String line;
  for (int c=0;c<8;c++) {
    float rms = codeToV(1,24)*1e6f * sqrt(s.m2[c]/(s.n-1));
    float pp  = codeToV(s.mx[c]-s.mn[c],24)*1e6f;
    float off = codeToV((int32_t)s.mean[c],24)*1e6f;
    char t[96]; sprintf(t,"CH%d: %.3f µVrms  %.2f µVpp  오프셋 %.1f µV", c+1, rms, pp, off); info(t);
    if (rms>worst){worst=rms; wc=c+1;}
  }
  char t[120];
  if (worst <= limPass) { sprintf(t,"최대 %.3f µVrms (CH%d) → 잡음 정상", worst, wc); result(id,"PASS",t); }
  else if (worst <= limWarn) { sprintf(t,"최대 %.3f µVrms (CH%d) → 약간 높음 (AVDD 리플/디커플링 확인)", worst, wc); result(id,"WARN",t); }
  else { sprintf(t,"최대 %.3f µVrms (CH%d) → 비정상 잡음. AVDD/GND/기준전압 확인", worst, wc); result(id,"FAIL",t); }
}
static void t11_noise() {
  Serial.println("\n[T11] 입력 단락 잡음 (MUX=SHORT, Gain 24) — 칩+전원 고유 잡음");
  if (!t2_ok) { result("T11","FAIL","T2 실패로 생략"); return; }
  info("(a) 250 SPS — 데이터시트 표 비교용");
  noiseRun(CFG1_250, 250, 0.5f, 1.5f, "T11a");
  info("(b) 2000 SPS — 실제 EMG 운용 속도");
  noiseRun(CFG1_2K, 2000, 1.5f, 3.0f, "T11b");
}

// T12: 전극 접촉 (DC lead-off, 6 nA, 95%)
static void t12_leadoff() {
  Serial.println("\n[T12] 전극 접촉 (DC 리드오프 6 nA) — 전극 미부착이면 전부 OFF가 정상");
  if (!t2_ok) { result("T12","FAIL","T2 실패로 생략"); return; }
  refOn(CFG3_BIAS);
  wreg(R_CONFIG1, CFG1_250);
  wreg(R_BIAS_SENSP, 0xFF); wreg(R_BIAS_SENSN, 0xFF);
  wreg(R_LOFF, 0x00);                 // COMP_TH 95%, 6 nA, DC
  wreg(R_LOFF_SENSP, 0xFF); wreg(R_LOFF_SENSN, 0xFF); wreg(R_LOFF_FLIP, 0x00);
  wreg(R_CONFIG4, 0x02);              // PD_LOFF_COMP=1
  allCh(GAIN_24 | MUX_NORMAL);
  struct L { uint16_t p[8], n[8]; uint16_t cnt; } l; memset(&l,0,sizeof(l));
  if (!acquire(250, 50, [](const int32_t*, uint32_t st, void* c){
        L* q=(L*)c; q->cnt++;
        for(int i=0;i<8;i++){ if((st>>(12+i))&1) q->p[i]++; if((st>>(4+i))&1) q->n[i]++; } }, &l)) {
    result("T12","FAIL","타임아웃"); return; }
  wreg(R_LOFF_SENSP,0); wreg(R_LOFF_SENSN,0); wreg(R_CONFIG4,0x00);
  uint8_t offP=0, offN=0;                       // 1초 중 절반 이상 OFF면 떨어짐으로 판정
  for(int i=0;i<8;i++){ if(l.p[i]*2>l.cnt) offP|=1<<i; if(l.n[i]*2>l.cnt) offN|=1<<i; }
  String s="  채널:  1 2 3 4 5 6 7 8\n  P(+):  "; for(int c=0;c<8;c++) s += ((offP>>c)&1)?"X ":"O ";
  s += "\n  N(-):  "; for(int c=0;c<8;c++) s += ((offN>>c)&1)?"X ":"O ";
  Serial.println(s); info("(O=접촉됨, X=떨어짐)");
  int off = __builtin_popcount(offP)+__builtin_popcount(offN);
  if (off==0) result("T12","PASS","16개 전극 모두 접촉");
  else if (off==16) result("T12","WARN","전극 미부착 상태 (부착 후 D 명령으로 확인)");
  else result("T12","WARN", String(off) + "개 X — 이 보드에서는 입력필터(4.7 nF) 때문에 DC 리드오프 판정이 부정확할 수 있음. D 명령(전극 DC 검사)으로 확인");
}

static void dumpRegs() {
  cmd(C_SDATAC);
  Serial.println("레지스터:");
  for (int a=0;a<=0x17;a++) Serial.printf("  0x%02X = 0x%02X\n", a, rreg(a));
}


// ---------------- 배선/칩 구동 상태 판별 (풀업 vs 풀다운 비교) ----------------
// 칩이 핀을 '구동'하면 풀업/풀다운을 바꿔도 값이 같고, 선이 끊겼거나 칩이 죽어 있으면 풀 방향을 따라간다.
static int readWithPull(int pin, gpio_pull_mode_t m) {
  gpio_set_pull_mode((gpio_num_t)pin, m); delayMicroseconds(300);
  int v = gpio_get_level((gpio_num_t)pin);
  return v;
}
static const char* driveState(int up, int dn) {
  if (up == dn) return up ? "칩이 High로 구동" : "칩이 Low로 구동 (또는 GND 단락)";
  if (up == 1 && dn == 0) return "떠 있음 (구동 없음)";
  return "비정상";
}
static void wiringTest() {
  Serial.println("\n[T0W] 배선/칩 구동 상태 (풀업·풀다운 비교, SPI 사용 안 함)");
  digitalWrite(PIN_PWDN, HIGH); digitalWrite(PIN_RESET, HIGH); digitalWrite(PIN_START, LOW);
  hwReset(); delay(5);
  // DRDY: 리셋 직후 칩이 High로 구동해야 정상
  int du = readWithPull(PIN_DRDY, GPIO_PULLUP_ONLY), dd = readWithPull(PIN_DRDY, GPIO_PULLDOWN_ONLY);
  gpio_set_pull_mode((gpio_num_t)PIN_DRDY, GPIO_PULLUP_ONLY);
  info(String("DRDY(GPIO9): 풀업 ") + du + " / 풀다운 " + dd + " → " + driveState(du, dd));
  // MISO, CS High: DOUT은 하이임피던스여야 함 → '떠 있음'이 정상
  digitalWrite(PIN_CS, HIGH);
  int mu = readWithPull(PIN_MISO, GPIO_PULLUP_ONLY), md = readWithPull(PIN_MISO, GPIO_PULLDOWN_ONLY);
  info(String("MISO(GPIO13), CS High: 풀업 ") + mu + " / 풀다운 " + md + " → " + driveState(mu, md) + "  (정상: 떠 있음)");
  // MISO, CS Low: 칩이 DOUT을 구동해야 함
  digitalWrite(PIN_CS, LOW); delayMicroseconds(50);
  int cu = readWithPull(PIN_MISO, GPIO_PULLUP_ONLY), cd = readWithPull(PIN_MISO, GPIO_PULLDOWN_ONLY);
  digitalWrite(PIN_CS, HIGH);
  gpio_set_pull_mode((gpio_num_t)PIN_MISO, GPIO_PULLUP_ONLY);
  info(String("MISO(GPIO13), CS Low : 풀업 ") + cu + " / 풀다운 " + cd + " → " + driveState(cu, cd) + "  (정상: 칩이 구동)");

  bool drdyDriven = (du == dd && du == 1);
  bool misoHiZ    = (mu == 1 && md == 0);
  bool misoDriven = (cu == cd);
  if (drdyDriven && misoHiZ && misoDriven) {
    result("T0W","PASS","DRDY·MISO 모두 칩이 정상 구동 → 칩 디지털 전원(DVDD)·GND·RESET·PWDN·CS 정상");
  } else if (!misoHiZ && mu == 0) {
    result("T0W","FAIL","CS High인데 MISO가 Low에 고정 → MISO 선이 GND에 단락, 또는 ADS1299 DVDD 미공급(무전원 칩이 핀을 끌어내림). J4.1 3.3V·GND 점퍼, C33 양단 3.3V 측정");
  } else if (!drdyDriven && !misoDriven) {
    result("T0W","FAIL","DRDY·MISO 둘 다 칩이 구동하지 않음 → 칩이 꺼져 있거나 리셋/파워다운 상태. DVDD 3.3V, GND, RESET·PWDN 핀이 칩 쪽에서 3.3V인지 측정. 이상 없으면 ADS1299 핀 납땜(특히 35·36·48·49·50·51·52번) 확인");
  } else if (!drdyDriven) {
    result("T0W","FAIL","MISO는 정상, DRDY만 구동 없음 → DRDY 점퍼(J4.15↔GPIO9) 또는 ADS1299 47번 핀 납땜");
  } else if (!misoDriven) {
    result("T0W","FAIL","DRDY는 정상, CS Low에도 MISO 구동 없음 → CS 점퍼(J4.16↔GPIO10)·MISO 점퍼(J4.19↔GPIO13) 또는 39·43번 핀 납땜");
  } else {
    result("T0W","WARN","판정 애매 → 'M' 명령으로 핀 전압 직접 측정");
  }
}

// ---------------- 멀티미터 측정용 레벨 고정 ----------------
static void holdLevels(bool high) {
  int v = high ? HIGH : LOW;
  digitalWrite(PIN_RESET, high ? HIGH : LOW);
  digitalWrite(PIN_PWDN,  high ? HIGH : LOW);
  digitalWrite(PIN_START, v);
  digitalWrite(PIN_CS,    v);
  Serial.printf("\n[측정 모드] RESET·PWDN·START·CS 를 모두 %s 로 고정했습니다.\n", high ? "HIGH(3.3V)" : "LOW(0V)");
  Serial.println("  멀티미터 COM(검정)을 보드 GND(J4.22)에 대고 아래를 측정하세요.");
  Serial.println("  ① 헤더 쪽 (배선 확인):  J5.17 RESET / J5.18 PWDN / J4.20 START / J4.16 CS");
  Serial.println("  ② 칩 다리 쪽 (납땜 확인): ADS1299 36번 RESET / 35번 PWDN / 38번 START / 39번 CS");
  Serial.println("  ③ 전원: J4.1(3.3V) · C33 양단(DVDD 3.3V) · J4.21(≈5V) · U3 1번핀/C36(AVDD 5.0V) · 52번 CLKSEL(3.3V)");
  Serial.printf("  기대값: ①②는 %s, ③은 괄호 안 값\n", high ? "약 3.3 V" : "0 V");
  Serial.println("  끝나면 아무 키나 입력 → RESET·PWDN 은 HIGH 로 복귀");
  while (!Serial.available()) delay(10);
  while (Serial.available()) Serial.read();
  digitalWrite(PIN_RESET, HIGH); digitalWrite(PIN_PWDN, HIGH);
  digitalWrite(PIN_START, LOW);  digitalWrite(PIN_CS, HIGH);
  delay(200); hwReset();
}

// ---------------- 멀티미터 없이 선 단락 검사 (ESP32 ADC + 내부 풀업 ≈ 45 kΩ) ----------------
// 원리: 핀을 입력으로 두고 내부 풀업을 켠 채 전압을 재면, GND 쪽 저항 R 에 따라
//       V = Vopen × R / (R + Rpu) 로 떨어진다. 아무 것도 안 연결된 GPIO4 로 Vopen 을 보정한다.
static const int PIN_REF_OPEN = 6;          // 미연결 핀 (보정용, J4.6 은 보드에서 미연결)
static void padPull(int pin, bool up) {
  gpio_num_t g = (gpio_num_t)pin;
  if (up) { gpio_pulldown_dis(g); gpio_pullup_en(g); rtc_gpio_pulldown_dis(g); rtc_gpio_pullup_en(g); }
  else    { gpio_pullup_dis(g); gpio_pulldown_en(g); rtc_gpio_pullup_dis(g); rtc_gpio_pulldown_en(g); }
}
static int mvWithPull(int pin, bool up) {
  analogReadMilliVolts(pin);                // ADC 채널 초기화 (이때 풀 설정이 풀림)
  padPull(pin, up); delay(5);
  uint32_t acc = 0; for (int i=0;i<16;i++) acc += analogReadMilliVolts(pin);
  return acc / 16;
}
static String classify(int mv, int vopen, float* rOut) {
  const float RPU = 45000.0f;
  float r = (mv >= vopen - 30) ? 1e9f : RPU * mv / (float)max(1, vopen - mv);
  if (rOut) *rOut = r;
  if (mv > 0.85f * vopen) return "단락 없음";
  if (r > 3000 && r < 40000) return String("약 ") + (int)(r/1000) + " kΩ 로 GND → 옆 핀(10 kΩ 풀다운 핀)과 브리지 의심";
  if (r <= 3000) return "GND 에 강하게 붙음 (단락 또는 칩이 Low 구동)";
  return String("약 ") + (int)(r/1000) + " kΩ 누설";
}
static void lineShortTest() {
  Serial.println("\n[R] 선 단락 검사 (멀티미터 대신 ESP32 ADC 사용, 대략값)");
  // SPI 해제 → 모든 선을 입력으로
  spi.end();
  digitalWrite(PIN_START, LOW);
  int pins[]  = { PIN_MISO, PIN_START, PIN_CS, PIN_MOSI, PIN_SCLK };
  const char* names[] = { "MISO/DOUT (J4.19, 칩43)", "START (J4.20, 칩38)", "CS (J4.16, 칩39)",
                          "MOSI/DIN (J4.17, 칩34)", "SCLK (J4.18, 칩40)" };
  int vopen = mvWithPull(PIN_REF_OPEN, true);
  int vref0 = mvWithPull(PIN_REF_OPEN, false);
  info(String("보정: 미연결 GPIO4 풀업 ") + vopen + " mV / 풀다운 " + vref0 + " mV");
  if (vopen < 2500 || vref0 > 400) {
    result("R","WARN","내부 풀업으로 ADC 측정이 안 됨 (보정 실패) → 이 검사 결과는 무시");
  }
  for (int pass=0; pass<2; pass++) {
    bool chipOn = (pass==0);
    digitalWrite(PIN_PWDN, chipOn ? HIGH : LOW);
    if (chipOn) { delay(50); hwReset(); }
    delay(30);
    Serial.printf("  -- 칩 %s --\n", chipOn ? "동작 (PWDN High)" : "꺼짐 (PWDN Low)");
    for (int k=0;k<5;k++) {
      // 측정 대상이 아닌 제어선은 정상 상태로 구동 (CS High = DOUT 하이임피던스, START Low)
      pinMode(PIN_CS, OUTPUT);    digitalWrite(PIN_CS, HIGH);
      pinMode(PIN_START, OUTPUT); digitalWrite(PIN_START, LOW);
      delayMicroseconds(100);
      pinMode(pins[k], INPUT);
      int mv = mvWithPull(pins[k], true);
      float r; String c = classify(mv, vopen, &r);
      Serial.printf("  %-26s %4d mV  → %s\n", names[k], mv, c.c_str());
    }
  }
  // 복구
  digitalWrite(PIN_PWDN, HIGH);
  pinMode(PIN_CS, OUTPUT); digitalWrite(PIN_CS, HIGH);
  pinMode(PIN_START, OUTPUT); digitalWrite(PIN_START, LOW);
  spi.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, -1);
  padPull(PIN_MISO, true);
  delay(200); hwReset();
  info("해석: 칩 동작 중에만 MISO 가 Low → CS 가 칩에 안 들어감(CS 점퍼/39번 납땜).");
  info("      칩 꺼져도 MISO 가 Low → 보드 쪽 단락/브리지 (43번 주변) 또는 점퍼가 GND 핀에 꽂힘.");
  info("      START/CS/MOSI/SCLK 가 Low → 해당 선 단락. '단락 없음'이면 선은 정상(단선 여부는 별도).");
}

// ---------------- 멀티미터 대용 전압계 (GPIO7 = J4.7, ADC1_CH6) ----------------
// ADS 보드 뒷면 J4.7 핀 끝에 전선을 연결해 측정봉으로 사용. 입력 한계 3.1 V!
// 5 V 이상 지점은 10 kΩ+10 kΩ 분압(측정봉→10k→J4.7, J4.7→10k→GND)으로만 측정.
static const int PIN_PROBE = 7;
static void probeMeter() {
  Serial.println(WIRING_MODE ? "\n[V] 전압계 모드 — 측정봉(J4.7 뒷면 전선) 끝을 측정점에 대세요. 아무 키 = 종료"
                             : "\n[V] 전압계 모드 — ESP32 '7'번 핀에 꽂은 점퍼선 끝을 측정점에 대세요. 아무 키 = 종료");
  Serial.println("  ⚠ 3.3V 초과 지점(5V, AVDD, 9V)은 분압 없이 직접 대지 마세요 (ESP32 손상)");
  pinMode(PIN_PROBE, INPUT);
  analogReadMilliVolts(PIN_PROBE);
  gpio_num_t g=(gpio_num_t)PIN_PROBE; gpio_pullup_dis(g); gpio_pulldown_dis(g); rtc_gpio_pullup_dis(g); rtc_gpio_pulldown_dis(g);
  while (Serial.available()) Serial.read();
  while (!Serial.available()) {
    uint32_t acc=0; for(int i=0;i<32;i++) acc += analogReadMilliVolts(PIN_PROBE);
    int mv = acc/32;
    Serial.printf("  측정봉 %4d mV   (10k+10k 분압 사용 시 실제 ≈ %4d mV)%s\n", mv, mv*2, mv>3000 ? "  ← 범위 초과 가능" : "");
    delay(400);
  }
  while (Serial.available()) Serial.read();
}

static void runSelftest() {
  nPass=nWarn=nFail=0; useStartCmd=false; t1_ok=t2_ok=false; spiHz=1000000;
  Serial.println("\n#BEGIN_SELFTEST");
  Serial.println("==== ADS1299 보드 초기 검진 ====");
  t0_system();
  wiringTest();
  t1_drdy_start();
  t2_id();
  t3_pattern();
  t4_speed();
  t5_start_isolate();
  t6_reset();
  t7_pwdn();
  t8_supply();
  t9_temp();
  t10_testsig();
  t11_noise();
  t12_leadoff();
  Serial.printf("\n==== 요약: PASS %d / WARN %d / FAIL %d ====\n", nPass, nWarn, nFail);
  Serial.println("#END_SELFTEST");
  hwReset(); cmd(C_SDATAC);
}

// ======================= 실시간 스트리밍 =======================
// 프레임: 0xA0 | seq | status(3) | 8ch×3 | 0xC0  = 30 byte
static bool streaming=false;
static uint32_t stFrames=0, stHdrBad=0, stDrdy0=0, stMissed=0, stDown=0, stLastDrdy=0;
static uint8_t seq=0;
static uint8_t chMask=0xFF;   // 사용 채널 (bit0=CH1). 'K'+16진수 2자리로 설정, 예: K01 = CH1만
static void startStream(char mode) {
  hwReset(); cmd(C_SDATAC);
  spiHz = bestHz;
  wreg(R_CONFIG3, CFG3_BIAS); delay(200);
  wreg(R_CONFIG1, CFG1_2K);
  wreg(R_CONFIG2, mode=='G' ? CFG2_TEST : CFG2_NORMAL);
  wreg(R_MISC1, 0x00);                                  // SRB1 off (차동 입력)
  // BIAS 는 '전극이 붙은 채널'만으로 만든다. 떠 있는 채널을 넣으면 그 채널의 전원잡음이 BIAS 로 몸에 되먹임됨
  wreg(R_BIAS_SENSP, mode=='L' ? chMask : 0x00);
  wreg(R_BIAS_SENSN, mode=='L' ? chMask : 0x00);
  wreg(R_LOFF_SENSP, 0); wreg(R_LOFF_SENSN, 0); wreg(R_CONFIG4, 0);
  uint8_t mux = mode=='L' ? MUX_NORMAL : (mode=='S' ? MUX_SHORT : MUX_TEST);
  for (int i=0;i<8;i++) {
    bool on = (mode!='L') || ((chMask>>i)&1);
    wreg(R_CH1SET+i, on ? (GAIN_24 | mux) : (0x80 | MUX_SHORT));   // 미사용 채널: 전원차단 + 입력단락 (출력 0)
  }
  const char* mname = mode=='L' ? "emg" : (mode=='S' ? "short" : "test");
  Serial.printf("#LIVE,fs=2000,gain=24,vref=4.5,mode=%s,spi=%lu,mask=%02X\n", mname, (unsigned long)spiHz, mode=='L' ? chMask : 0xFF);
  Serial.flush();
  delay(50);
  cmd(C_RDATAC);
  drdyFlag=false; seq=0;
  stFrames=stHdrBad=stMissed=stDown=0; stDrdy0=stLastDrdy=drdyCount;
  startConv();
  streaming=true;
}
static void stopStream() {
  stopConv(); cmd(C_SDATAC);
  streaming=false;
  delay(20);
  Serial.printf("\n#STOPPED,frames=%lu,drdy=%lu,missed=%lu,spi_hdr_bad=%lu,spi_khz=%lu,spi_downshift=%lu\n",
                (unsigned long)stFrames, (unsigned long)(drdyCount - stDrdy0), (unsigned long)stMissed,
                (unsigned long)stHdrBad, (unsigned long)(spiHz/1000), (unsigned long)stDown);
}

// ---------------- 전극 입력 DC 검사 (Gain 1) ----------------
// 1) P−N DC 전압: 정상 전극쌍은 두 입력이 모두 몸(BIAS≈2.5 V)에 붙어 수십 mV 이내.
// 2) 끊긴 쪽 찾기: 24 nA 전류를 P에만 / N에만 (양방향) 흘려 본다.
//    붙어 있는 입력은 전류가 몸으로 빠져 전압 변화가 수 mV 이하,
//    떠 있는 입력은 입력 커패시터(≈4.7 nF)가 충전되어 수백 mV 이상 움직인다.
static bool dcMeasure(uint8_t mask, uint8_t sensP, uint8_t sensN, uint8_t flip, float out_mv[8]) {
  wreg(R_LOFF, 0x04);                 // DC, 24 nA
  wreg(R_LOFF_SENSP, sensP); wreg(R_LOFF_SENSN, sensN); wreg(R_LOFF_FLIP, flip);
  Stat st; statInit(st);
  if (!acquire(125, 125, statAdd, &st)) return false;      // 0.5 s 버리고 0.5 s 평균
  for (int c=0;c<8;c++) out_mv[c] = codeToV((int32_t)st.mean[c], 1) * 1000.0f;
  return true;
}
static void dcCheck(uint8_t mask) {
  Serial.println("\n[D] 전극 입력 검사 (Gain 1, BIAS on, 약 6 s) — 전극을 붙인 상태에서 실행");
  hwReset(); cmd(C_SDATAC);
  spiHz = bestHz;
  wreg(R_CONFIG3, CFG3_BIAS); delay(200);
  wreg(R_CONFIG1, CFG1_250); wreg(R_CONFIG2, CFG2_NORMAL); wreg(R_MISC1, 0x00);
  wreg(R_BIAS_SENSP, mask); wreg(R_BIAS_SENSN, mask); wreg(R_CONFIG4, 0x00);
  for (int i=0;i<8;i++) wreg(R_CH1SET+i, ((mask>>i)&1) ? (GAIN_1 | MUX_NORMAL) : (0x80 | MUX_SHORT));
  float v0[8], p1[8], p2[8], n1[8], n2[8];
  bool ok = dcMeasure(mask, 0, 0, 0, v0) &&
            dcMeasure(mask, mask, 0, 0, p1) && dcMeasure(mask, mask, 0, mask, p2) &&
            dcMeasure(mask, 0, mask, 0, n1) && dcMeasure(mask, 0, mask, mask, n2);
  wreg(R_LOFF_SENSP, 0); wreg(R_LOFF_SENSN, 0); wreg(R_LOFF_FLIP, 0); wreg(R_LOFF, 0);
  if (!ok) { Serial.println("  타임아웃 (DRDY 없음)"); hwReset(); cmd(C_SDATAC); return; }
  Serial.println("  CH   P−N(mV)   P전류시 변화   N전류시 변화   판정");
  int bad = 0;
  for (int c=0;c<8;c++) {
    if (!((mask>>c)&1)) { Serial.printf("  %d    (미사용)\n", c+1); continue; }
    float dP = max(fabs(p1[c]-v0[c]), fabs(p2[c]-v0[c]));
    float dN = max(fabs(n1[c]-v0[c]), fabs(n2[c]-v0[c]));
    // 피부-전극 접촉은 직류(DC)에서 수 MΩ 이 정상일 수 있음(근전도 대역에서는 훨씬 낮음).
    // 따라서 '끊김'은 P−N 직류가 크게 벗어나거나 전류 주입에 레일까지(>2 V) 움직일 때만 판정한다.
    String j;
    bool dcBad = fabs(v0[c]) > 1000;
    bool pFloat = dP > 2000, nFloat = dN > 2000;
    if (dcBad || pFloat || nFloat) {
      if (pFloat && nFloat)      j = String("P·N 모두 떠 있음 → ") + (c+1) + "P·" + (c+1) + "N 확인";
      else if (nFloat || (!pFloat && dN >= dP)) j = String(c+1) + "N 끊김 (전극 → 전선 → J1 → 직렬저항 → 칩 순서로 확인)";
      else                       j = String(c+1) + "P 끊김 (전극 → 전선 → J1 → 직렬저항 → 칩 순서로 확인)";
    }
    else if (fabs(v0[c]) > 150) j = "연결됨 · DC 오프셋 큼 → Gain 24 측정범위(±187.5 mV) 포화 위험. 이 채널 전극 2개를 새 것으로 교체";
    else if (max(dP, dN) > 300) j = "연결됨 · 접촉저항 높음 (젤/피부 닦기 권장, 측정은 가능)";
    else j = "정상";
    if (j.indexOf("끊김") >= 0 || j.indexOf("떠 있음") >= 0) bad++;
    Serial.printf("  %d  %9.1f   %10.1f mV   %10.1f mV   %s\n", c+1, v0[c], dP, dN, j.c_str());
  }
  Serial.println("  (끊김: |P−N| > 1 V 또는 전류 주입 시 2 V 이상 이동.  300 mV~2 V 이동 = 연결됐지만 접촉저항 높음)");
  result("D", bad ? "WARN" : "PASS", bad ? String(bad) + "개 채널 끊김 → 표의 판정 위치 확인" : "사용 채널 모두 연결됨");
  hwReset(); cmd(C_SDATAC);
}

void printHelp() {
  Serial.println("\n명령: T=전체검진  W=배선상태  R=선단락검사  V=전압계(GPIO7)  D=전극DC검사  M=측정모드(HIGH)  N=측정모드(LOW)  L=실시간EMG  S=실시간(입력단락)  G=실시간(테스트신호)  X=중지  I=레지스터  H=도움말");
  Serial.println("#READY");
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.begin(921600);
  pinMode(PIN_CS, OUTPUT);    digitalWrite(PIN_CS, HIGH);
  pinMode(PIN_START, OUTPUT); digitalWrite(PIN_START, LOW);
  pinMode(PIN_RESET, OUTPUT); digitalWrite(PIN_RESET, HIGH);
  pinMode(PIN_PWDN, OUTPUT);  digitalWrite(PIN_PWDN, HIGH);
  pinMode(PIN_DRDY, INPUT_PULLUP);
  spi.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, -1);
  gpio_pullup_en((gpio_num_t)PIN_MISO);                   // MISO 단선 시 0xFF로 고정되게
  attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdy, FALLING);
  delay(300);                                             // tPOR (2^18 tCLK ≈ 128 ms) 여유
  hwReset(); cmd(C_SDATAC);
  delay(500);
  Serial.println("\nADS1299 초기 검진 펌웨어 v1.10 (WIRING_MODE " + String(WIRING_MODE) + ")");
  printHelp();
}

void loop() {
  if (Serial.available()) {
    char c = toupper(Serial.read());
    if (streaming) { if (c=='X') { stopStream(); printHelp(); } return; }
    switch (c) {
      case 'T': runSelftest(); printHelp(); break;
      case 'L': case 'S': case 'G': startStream(c); break;
      case 'D': dcCheck(chMask); printHelp(); break;
      case 'K': {                                          // 채널 마스크: K + 16진수 2자리
        char h[3]={0,0,0}; uint32_t t0=millis(); int n=0;
        while (n<2 && millis()-t0<200) if (Serial.available()) h[n++]=Serial.read();
        chMask = (uint8_t)strtol(h, nullptr, 16); if (!chMask) chMask=0xFF;
        Serial.printf("#MASK,%02X\n", chMask); break; }
      case 'I': dumpRegs(); printHelp(); break;
      case 'H': printHelp(); break;
      case 'W': wiringTest(); printHelp(); break;
      case 'R': lineShortTest(); printHelp(); break;
      case 'V': probeMeter(); printHelp(); break;
      case 'M': holdLevels(true);  printHelp(); break;
      case 'N': holdLevels(false); printHelp(); break;
      default: break;
    }
  }
  if (streaming && drdyFlag) {
    drdyFlag = false;
    uint32_t dc = drdyCount;
    if (dc - stLastDrdy > 1) stMissed += dc - stLastDrdy - 1;     // ESP 쪽에서 놓친 샘플
    stLastDrdy = dc;
    uint8_t b[27]; readFrame(b);
    stFrames++;
    if ((b[0] & 0xF0) != 0xC0) {                                 // SPI 읽기 오류 (상태 헤더 깨짐)
      stHdrBad++;
      // 1000 프레임 중 1% 넘게 깨지면 SPI 속도를 절반으로 (최저 500 kHz)
      if (stFrames >= 1000 && stHdrBad * 100 > stFrames && spiHz > 500000) {
        spiHz /= 2; stDown++; stFrames = 0; stHdrBad = 0;
      }
    }
    uint8_t out[30];
    out[0]=0xA0; out[1]=(uint8_t)drdyCount;               // DRDY 기준 순번 → PC에서 누락 검출
    memcpy(out+2, b, 27); out[29]=0xC0;                   // [2..4]=status, [5..28]=8ch×3
    Serial.write(out, 30);
  }
}
