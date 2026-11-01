// 아두이노 자동 프로토타입 생성 때문에 타입은 별도 헤더에 둔다
#pragma once
#include <stdint.h>
typedef void (*SampleFn)(const int32_t ch[8], uint32_t status, void* ctx);
struct Stat { double mean[8], m2[8]; int32_t mn[8], mx[8]; uint32_t n; };
