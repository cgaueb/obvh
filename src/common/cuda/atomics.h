#ifndef _KORANGAR_ATOMICS_CUDA_H_
#define _KORANGAR_ATOMICS_CUDA_H_

#include "common/korangar.h"

void   kr_atomic_minf(kr_scalar* mem, kr_scalar val);
void   kr_atomic_maxf(kr_scalar* mem, kr_scalar val);

__device__ __inline__ void atomicMax(kr_scalar* ptr, kr_scalar value)
{
    u32 curr = atomicAdd((u32*)ptr, 0);
    while (value > __int_as_float(curr))
    {
        u32 prev = curr;
        curr = atomicCAS((u32*)ptr, curr, __float_as_int(value));
        if (curr == prev)
            break;
    }
}

__device__ __inline__ void atomicMin(kr_scalar* ptr, kr_scalar value)
{
    u32 curr = atomicAdd((u32*)ptr, 0);
    while (value < __int_as_float(curr))
    {
        u32 prev = curr;
        curr = atomicCAS((u32*)ptr, curr, __float_as_int(value));
        if (curr == prev)
            break;
    }
}

#endif /* _KORANGAR_ATOMICS_CUDA_H_ */