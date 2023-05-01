#ifndef _KORANGAR_UTIL_H_
#define _KORANGAR_UTIL_H_

#include "korangar.h"

#define KR_EQUALS_LITERAL(str,literal) (strncmp(str, literal, sizeof(literal)) == 0)

#ifndef _WIN32
#include <dlfcn.h>

#define kr_dlopen(path) dlopen((char*)path, RTLD_NOW | RTLD_GLOBAL)
#define kr_dlsym(handle, symbol) dlsym(handle, symbol)
#define kr_dlclose(handle) dlclose(handle)

#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define kr_dlopen(path) LoadLibraryA((char*)path)
#define kr_dlsym(handle, symbol) (void*)GetProcAddress((HMODULE)handle, symbol)
#define kr_dlclose(handle) (FreeLibrary((HMODULE)handle) ? 0 : -1)
#endif

#define KR_ALLOC_DECLARE(type, name, count) type* name = (type*)kr_allocate((count) * sizeof(*(name)))


#ifdef __cplusplus
extern "C" {
#endif


void* kr_aligned_allocate(kr_size count, kr_size alignment);
void* kr_aligned_reallocate(void* mem, kr_size count, kr_size alignment);
void  kr_aligned_free(void** mem);
void* kr_reallocate(void* ptr, kr_size count);
void* kr_allocate(kr_size count);
void* kr_zero_allocate(kr_size count);
void kr_zero_memory(void* mem, kr_size count);
void kr_memset(void* mem, i32 ch, kr_size count);
void kr_memcpy(void* dst, const void* src, kr_size count);
void kr_free(void** mem);

typedef union {
	u32 seed;
} kr_random_engine;

void kr_random_init(kr_random_engine* rng);
kr_vec4 kr_vrandom4(kr_random_engine* rng);
kr_vec3 kr_vrandom3(kr_random_engine* rng);
kr_vec2 kr_vrandom2(kr_random_engine* rng);
void kr_randomf(kr_random_engine* rng, kr_scalar* rands, u32 count);

kr_error kr_file_size(const char* filename, kr_size* content_size);
kr_error kr_file_contents(const char* filename, char* buffer, kr_size count);
kr_error kr_file_ext(const char* filename, char* buffer, kr_size count);
kr_error kr_file_base(const char* filename, char* buffer, kr_size count);
kr_error kr_path_filename(const char* filename, char* buffer, kr_size count, kr_b32 keep_ext);
kr_error kr_path_normalize(const char* path, char* buffer, kr_size count);

void   kr_atomic_minf(kr_scalar* mem, kr_scalar val);
void   kr_atomic_maxf(kr_scalar* mem, kr_scalar val);
kr_i32 kr_atomic_addi(kr_i32* mem, kr_i32 val);
kr_u32 kr_atomic_addu(kr_u32* mem, kr_u32 val);
kr_u32 kr_atomic_cmp_exch(kr_u32* mem, kr_u32 val, kr_u32 comp);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

struct TimedCall {
    TimedCall() {
    }

    template <typename Callable>
    kr_scalar execute(Callable c) const {
        LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
        LARGE_INTEGER Frequency;

        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
        c();
        QueryPerformanceCounter(&EndingTime);

        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

        f64 ElapsedMiliseconds = (f64)ElapsedMicroseconds.QuadPart / 1000.0;

        kr_scalar ms = (kr_scalar)ElapsedMiliseconds;

        return ms;
    }
};
#endif


#endif /* _KORANGAR_UTIL_H_ */
