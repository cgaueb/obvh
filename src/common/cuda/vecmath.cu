#define KR_VECMATH_IMPL
#include "vecmath.h"

#ifdef KR_DEBUG_FACILITIES
#include <stdio.h>

/* https://stackoverflow.com/a/3208376 */
#define KR_BYTE_BIT_PATTERN "%c%c%c%c%c%c%c%c"
#define KR_BYTE_TO_BITS(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

void kr__vprint3(const char* name, kr_vec3 a) {
    printf("%s {%f %f %f}\n", name, a.x, a.y, a.z);
}
void kr__aabb_print3(const char* name, kr_aabb3 a) {
    printf("%s MIN {%f %f %f} MAX {%f %f %f}\n", name, a.min.x, a.min.y, a.min.z, a.max.x, a.max.y, a.max.z);
}
void kr__bit_printu32(const char* name, u32 a) {
    printf("%s " KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" "\n", name, KR_BYTE_TO_BITS(a >> 24), KR_BYTE_TO_BITS(a >> 16), KR_BYTE_TO_BITS(a >> 8), KR_BYTE_TO_BITS(a >> 0));
}
void kr__bit_printu64(const char* name, u64 a) {
    printf("%s " KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" "\n", name, KR_BYTE_TO_BITS(a >> 56), KR_BYTE_TO_BITS(a >> 48), KR_BYTE_TO_BITS(a >> 40), KR_BYTE_TO_BITS(a >> 32), KR_BYTE_TO_BITS(a >> 24), KR_BYTE_TO_BITS(a >> 16), KR_BYTE_TO_BITS(a >> 8), KR_BYTE_TO_BITS(a >> 0));
}
void kr__bit_fprintu32(void* f, const char* name, u32 a) {
    fprintf((FILE*)f, "%s " KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" KR_BYTE_BIT_PATTERN "" "\n", name, KR_BYTE_TO_BITS(a >> 24), KR_BYTE_TO_BITS(a >> 16), KR_BYTE_TO_BITS(a >> 8), KR_BYTE_TO_BITS(a >> 0));
}

#endif /* KR_DEBUG_FACILITIES */