cmake_minimum_required(VERSION 3.18)

include_directories(${CMAKE_SOURCE_DIR})

enable_language(CUDA)

project (atrbvh_gpu)

add_library(${PROJECT_NAME} SHARED
    atrbvh.cu
    AgglomerativeTreelet.cu
    AgglomerativeScheduler.cpp
    AgglomerativeScheduler.h
    AgglomerativeTreelet.h
    AgglomerativeTreeletOptimizer.cpp
    AgglomerativeTreeletOptimizer.h
    BVHTreeInstanceManager.cpp
    BVHTreeInstanceManager.h
    BVHBuilder.h
    BVHOptimizer.h
    BVHTree.h
    Commons.cuh
    CubWrapper.cu
    CubWrapper.h
    CudaErrorCheck.h
    CudaTimer.cpp
    CudaTimer.h
    LBVH.cu
    LBVH.h
    LBVHBuilder.cpp
    LBVHBuilder.h
    module.c
    Scene.h
    SceneWrapper.cpp
    SceneWrapper.h
    SoABVHTree.cpp
    SoABVHTree.h
    TimeKernelExecution.h
    Treelet.cuh
    TriangularMatrix.cuh
    ${CMAKE_SOURCE_DIR}/ads/common/cuda/intersectors.cu
    ${CMAKE_SOURCE_DIR}/ads/common/cuda/transforms.cu
    ${CMAKE_SOURCE_DIR}/ads/common/transforms.cpp
    ${CMAKE_SOURCE_DIR}/common/algorithm/dito/dito.cu
    ${CMAKE_SOURCE_DIR}/common/algorithm/dito/dito.cpp
    ${CMAKE_SOURCE_DIR}/common/cuda/util.c
    ${CMAKE_SOURCE_DIR}/common/geometry.cpp
    ${CMAKE_SOURCE_DIR}/common/logger.c
    ${CMAKE_SOURCE_DIR}/common/util.c
    ${CMAKE_SOURCE_DIR}/common/ads.c
    ${CMAKE_SOURCE_DIR}/common/sampling.c
    ${CMAKE_SOURCE_DIR}/common/vecmath.c
    ${CMAKE_SOURCE_DIR}/common/scene.c
    ${CMAKE_SOURCE_DIR}/common/queue.c
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DKR_EXPORT_API)

if(WIN32)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../../../bin )
else()
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../../bin )
endif()