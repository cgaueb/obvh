#include "util.h"
#include "../util.h"

#include <cuda_runtime.h>

void*
kr_cuda_allocate_managed(kr_size count) {
    void* ptr = kr_null;
    cudaError_t error = cudaMallocManaged(&ptr, count, cudaMemAttachGlobal);
    if (!ptr || cudaSuccess != error)
      return kr_null;

    kr_zero_memory(ptr, count);

    return ptr;
}

void*
kr_cuda_allocate(kr_size count) {
    void* ptr = kr_null;
    cudaError_t error = cudaMalloc(&ptr, count);
    if (!ptr || cudaSuccess != error)
      return kr_null;

    error = cudaMemset(ptr, 0, count);
    if (cudaSuccess != error)
      return kr_null;

    return ptr;
}

void
kr_cuda_free(void** mem) {
  if(kr_null == mem || kr_null == *mem)
    return;
  cudaFree(*mem);
  *mem = kr_null;
}