cmake_minimum_required(VERSION 3.18)

include_directories(${CMAKE_SOURCE_DIR})

enable_language(CUDA)

project (bench)

add_library(${PROJECT_NAME} SHARED
    bench.cu 
    module.c 
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