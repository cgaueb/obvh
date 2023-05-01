#include "CubWrapper.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define MEASURE_EXECUTION_TIMES

// In and out buffers may be swaped
// Original data is not kept
template <typename T> float DeviceSort(unsigned int numberOfElements, T* keysIn, T* keysOut,
                 unsigned int* valuesIn, unsigned int* valuesOut)
{
    //cub::DoubleBuffer<T> keysBuffer(*keysIn, *keysOut);
    //cub::DoubleBuffer<unsigned int> valuesBuffer(*valuesIn, *valuesOut);

    // Check how much temporary memory will be required
    void* tempStorage = nullptr;
    size_t storageSize = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage, storageSize,
        keysIn, keysOut, valuesIn, valuesOut, numberOfElements);
    //cub::DeviceRadixSort::SortKeys(tempStorage, storageSize, keysBuffer, numberOfElements);

    // Allocate temporary memory
    cudaMalloc(&tempStorage, storageSize);

    float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif

    // Sort
    cub::DeviceRadixSort::SortPairs(tempStorage, storageSize,
        keysIn, keysOut, valuesIn, valuesOut, numberOfElements);

#ifdef MEASURE_EXECUTION_TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

    // Free temporary memory
    cudaFree(tempStorage);
    return elapsedTime;
}

float DeviceSort(unsigned int numberOfElements, unsigned int* keysIn, unsigned int* keysOut,
    unsigned int* valuesIn, unsigned int* valuesOut)
{
    return DeviceSort<unsigned int>(numberOfElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSort(unsigned int numberOfElements, unsigned long long int* keysIn, unsigned long long int* keysOut,
    unsigned int* valuesIn, unsigned int* valuesOut)
{
    return DeviceSort<unsigned long long int>(numberOfElements, keysIn, keysOut, valuesIn, valuesOut);
}
