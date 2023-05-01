#include "util.h"

#ifdef KR_WIN32
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//#define KR_RANDOM_STDLIB
#define KR_RANDOM_XOSHIRO128

typedef struct {
  kr_size size;
  char data[1]; 
} kr_block;

void* kr_aligned_allocate(kr_size count, kr_size alignment) {
    void* p1; // original block
    void** p2; // aligned block
	kr_size offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)kr_allocate(count + offset)) == NULL)
    {
       return NULL;
    }
    p2 = (void**)(((kr_size)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void* kr_aligned_reallocate(void* mem, kr_size count, kr_size alignment) {
    void* p1; // original block
    void** p2; // aligned block
    kr_size offset = alignment - 1 + sizeof(void*);
    void** p = (void**)mem;
    if ((p1 = (void*)kr_reallocate(p[-1], count + offset)) == NULL)
    {
        return NULL;
    }
    p2 = (void**)(((kr_size)(p1)+offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void
kr_aligned_free(void** mem)
{
  if(kr_null == mem || kr_null == *mem)
    return;
  void** p = (void**)*mem;
  *mem = kr_null;
  kr_free(&p[-1]);
}

void*
kr_allocate(kr_size count) {
  kr_block* blk = (kr_block*)malloc(count + kr_offsetof(kr_block, data));
  kr_zero_memory(blk, count + kr_offsetof(kr_block, data));
  blk->size = count;

  return (u8*)blk + kr_offsetof(kr_block, data);
}

void 
kr_memcpy(void* dst, const void* src, kr_size count) {
  memcpy(dst, src, count);
}

void 
kr_memset(void* mem, i32 ch, kr_size count) {
  memset(mem, ch, count);
}

void
kr_zero_memory(void* mem, kr_size count) {
  /*for (int i = 0; i < count; i++) {
    ((unsigned char*)mem)[i]= 0;
  }*/
  kr_memset(mem, 0, count);
}

void
kr_free(void** mem) {
  if(kr_null == mem || kr_null == *mem)
    return;
  kr_block* blk = (kr_block*)((u8*)(*mem) - kr_offsetof(kr_block, data));
  kr_zero_memory(blk, blk->size + kr_offsetof(kr_block, data));
  *mem = kr_null;
  free(blk);
}

void*
kr_reallocate(void* mem, kr_size count) {
  if(kr_null == mem)
    return kr_null;

  kr_block* blk = (kr_block*)((u8*)mem - kr_offsetof(kr_block, data));
  kr_size old_size = blk->size;
  blk = (kr_block*)realloc(blk, count + kr_offsetof(kr_block, data));
  blk->size = count;

  kr_zero_memory((u8*)blk + kr_offsetof(kr_block, data) + old_size, blk->size - old_size);
  
  return (u8*)blk + kr_offsetof(kr_block, data);
}

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>

/* This is xoshiro128+ 1.0, our best and fastest 32-bit generator for 32-bit
   floating-point numbers. We suggest to use its upper bits for
   floating-point generation, as it is slightly faster than xoshiro128**.
   It passes all tests we are aware of except for
   linearity tests, as the lowest four bits have low linear complexity, so
   if low linear complexity is not considered an issue (as it is usually
   the case) it can be used to generate 32-bit outputs, too.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. */


static inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}


static uint32_t s[4];

uint32_t next(void) {
    const uint32_t result = s[0] + s[3];

    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 11);

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations. */

void jump(void) {
    static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for (int i = 0; i < sizeof JUMP / sizeof * JUMP; i++)
        for (int b = 0; b < 32; b++) {
            if (JUMP[i] & UINT32_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next();
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^96 calls to next(); it can be used to generate 2^32 starting points,
   from each of which jump() will generate 2^32 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(void) {
    static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for (int i = 0; i < sizeof LONG_JUMP / sizeof * LONG_JUMP; i++)
        for (int b = 0; b < 32; b++) {
            if (LONG_JUMP[i] & UINT32_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next();
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

f32 FloatFromBits(const u32 i) {
    return (i >> 8) * 0x1.0p-24f;
}

void
kr_random_init(kr_random_engine* rng) {
#if defined(KR_RANDOM_XOSHIRO128)
    s[0] = rng->seed;
    s[1] = rng->seed * rng->seed;
    s[2] = rng->seed * rng->seed * rng->seed * rng->seed;
    s[3] = rng->seed * rng->seed * rng->seed * rng->seed * rng->seed * rng->seed * rng->seed * rng->seed;
#elif defined(KR_RANDOM_STDLIB)
    srand(rng->seed);
#endif
}

void
kr_randomf(kr_random_engine* rng, kr_scalar* rands, u32 count) {
  for(u32 i = 0; i < count; i++) {
#if defined(KR_RANDOM_XOSHIRO128)
      rands[i] = FloatFromBits(next());
#elif defined(KR_RANDOM_STDLIB)
      rands[i] = (kr_scalar)rand() / (kr_scalar)RAND_MAX;
#endif
  }
}

kr_vec4
kr_vrandom4(kr_random_engine* rng) {
    kr_vec4 v;
    kr_randomf(rng, v.v, 4);
    return v;
}

kr_vec3 
kr_vrandom3(kr_random_engine* rng) {
    kr_vec3 v;
    kr_randomf(rng, v.v, 3);
    return v;
}

kr_vec2
kr_vrandom2(kr_random_engine* rng) {
    kr_vec2 v;
    kr_randomf(rng, v.v, 2);
    return v;
}

#define KR_FAILED_TO_OPEN_FILE ((kr_error)"Failed to open file")
kr_error
kr_file_size(const char* filename, kr_size* content_size)
{
    FILE* file_handle = 0;
    *content_size = 0;

    file_handle = fopen(filename, "rb");
    if (!file_handle) {
        return KR_FAILED_TO_OPEN_FILE;
    }

    fseek(file_handle, 0L, SEEK_END);
    *content_size = ftell(file_handle);
    fclose(file_handle);

    return kr_success;
}

kr_error
kr_file_contents(const char* filename, char* buffer, kr_size count)
{
    FILE* file_handle = 0;

    file_handle = fopen(filename, "rb");
    if (!file_handle) {
        return KR_FAILED_TO_OPEN_FILE;
    }

    fread(buffer, 1, count, file_handle);

    fclose(file_handle);

    return kr_success;
}

kr_error 
kr_path_filename(const char* path, char* file, kr_size count, kr_b32 keep_ext) {
    const char* at = path;
    const char* last_sep = path;
    while (*at) {
        if (*at == '\\' || *at == '/') {
            last_sep = at;
        }
        at++;
    }

    kr_size processed = 0;
    at = last_sep + 1;
    while (*at && processed < count) {
        if (kr_false == keep_ext && *at == '.') 
          break;
        
        file[processed] = *at++;
        processed++;
    }
    file[processed] = 0;
    return kr_success;
}

kr_error kr_file_base(const char* filename, char* buffer, kr_size count) {
  const char* at = filename;
  const char* last = at;
  while (*at) {
    if (*at == '/' || *at == '\\')  {
      last = at;
    }
    at++;
  }
  at = filename;
  i32 index = 0;
  while (at != last && (count--)) {
    buffer[index++] = *at++;
  }
  buffer[index] = 0;

  return kr_success;
}

kr_error 
kr_path_normalize(const char* path, char* buffer, kr_size count) {
  const char* at = path;
  i32 index = 0;
  while (*at && (count--)) {
    buffer[index++] = (*at == '\\') ? '/' : *at;
    at++;
  }
  buffer[index] = 0;

  return kr_success;
}

kr_error 
kr_file_ext(const char* filename, char* buffer, kr_size count) {
  const char* at = filename;
  const char* last_dot = filename;
  while (*at) {
    if (*at == '.') {
      last_dot = at;
    }
    at++;
  }

  if (last_dot == filename)
    return kr_success;

  at = last_dot + 1;
  while (*at && count--) {
    *buffer = *at;
    buffer++;
    at++;
  }
  *buffer = '\0';

  return kr_success;
}

kr_i32
kr_atomic_addi(kr_i32* mem, kr_i32 val) {
  #if KR_LINUX
    return __atomic_fetch_add(mem, val, __ATOMIC_SEQ_CST); 
  #elif KR_WIN32
	return InterlockedExchangeAdd(mem, val);
  #endif
}

kr_u32 
kr_atomic_addu(kr_u32* mem, kr_u32 val) {
  #if KR_LINUX
    return __atomic_fetch_add(mem, val, __ATOMIC_SEQ_CST); 
  #elif KR_WIN32
	return InterlockedExchangeAdd(mem, val);
  #endif
}

kr_u32
kr_atomic_cmp_exch(kr_u32* mem, kr_u32 val, kr_u32 comp) {
#if KR_LINUX
	return __atomic_fetch_add(mem, val, __ATOMIC_SEQ_CST);
#elif KR_WIN32
	return InterlockedCompareExchange(mem, val, comp);
#endif
}


