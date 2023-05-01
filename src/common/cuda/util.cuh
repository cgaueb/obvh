#ifndef _KORANGAR_UTIL_CUDA_CUH_
#define _KORANGAR_UTIL_CUDA_CUH_

#include "common/korangar.h"

kr_inline_device
i32 kr_cuda_global_index_1D_1D() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

kr_inline_device
i32 kr_cuda_global_index_1D_2D() {
	return blockIdx.x * blockDim.x * blockDim.y
		 + threadIdx.y * blockDim.x + threadIdx.x;
}

#endif /* _KORANGAR_UTIL_CUDA_CUH_ */