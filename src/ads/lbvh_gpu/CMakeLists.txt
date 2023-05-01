cmake_minimum_required(VERSION 3.18)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/../deps)

enable_language(CUDA)

project (lbvh_gpu)

add_library(${PROJECT_NAME} SHARED
    lbvh.cu 
    module.c 
    ${CMAKE_SOURCE_DIR}/common/algorithm/dito/dito.cu
    ${CMAKE_SOURCE_DIR}/common/algorithm/dito/dito.cpp
    ${CMAKE_SOURCE_DIR}/ads/common/cuda/intersectors.cu
    ${CMAKE_SOURCE_DIR}/ads/common/cuda/transforms.cu
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

#target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:O3>) 
set(CMAKE_CUDA_LDFLAGS "${CMAKE_CUDA_LDFLAGS} --ptxas-options=-v -dlto")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DKR_EXPORT_API)

if(WIN32)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../../../bin )
else()
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../../bin )
endif()