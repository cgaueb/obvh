cmake_minimum_required(VERSION 3.18)

include_directories(${CMAKE_SOURCE_DIR})

enable_language(CUDA)

project (pt_gpu)

add_library(${PROJECT_NAME} SHARED
    pt.cu 
    ${CMAKE_SOURCE_DIR}/common/util.c 
    ${CMAKE_SOURCE_DIR}/common/ads.c 
    ${CMAKE_SOURCE_DIR}/common/cuda/util.c
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DKR_EXPORT_API)

if(WIN32)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../../../bin )
else()
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../../bin )
endif()