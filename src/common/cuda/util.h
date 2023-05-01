#ifndef _KORANGAR_UTIL_CUDA_H_
#define _KORANGAR_UTIL_CUDA_H_

#include "common/korangar.h"

#define KR_CUDA_ALLOC_MANAGED_DECLARE(type, name, count) type* name = (type*)kr_cuda_allocate_managed((count) * sizeof(*(name)))
#define KR_CUDA_ALLOC_DECLARE(type, name, count) type* name = (type*)kr_cuda_allocate((count) * sizeof(*(name)))
#define KR_CUDA_ALLOC_THRUST_DECLARE(type, name, count) thrust::device_ptr<type> name((type*)kr_cuda_allocate((count) * sizeof(type)))

#ifdef __cplusplus
extern "C" {
#endif

void*
kr_cuda_allocate_managed(kr_size count);
void*
kr_cuda_allocate(kr_size count);
void
kr_cuda_free(void** mem);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

struct KernelLaunch {
    KernelLaunch() {
    }

    template <typename Callable, b32 measure_time = true>
    kr_scalar execute(Callable c) const {
        float_t ms = 0.f;
        if constexpr (measure_time) {
            cudaEvent_t start;
            cudaEvent_t end;

            cudaEventCreate(&start);
            cudaEventCreate(&end);

            cudaEventRecord(start);
            c();
            cudaEventRecord(end);

            cudaEventSynchronize(end);
            cudaEventElapsedTime(&ms, start, end);

            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }
        else {
            c();
        }
        return ms;
    }

};

#endif

#endif /* _KORANGAR_UTIL_CUDA_H_ */