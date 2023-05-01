cmake_minimum_required(VERSION 2.8.12)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/../deps)

project (korangar)

add_executable(${PROJECT_NAME}
    main.cpp 
    gltf.c
    cmdopt.c
    ${CMAKE_SOURCE_DIR}/common/GLEW/glew.c
    ${CMAKE_SOURCE_DIR}/common/scene.c
    ${CMAKE_SOURCE_DIR}/common/integrator.c
    ${CMAKE_SOURCE_DIR}/common/ads.c
    ${CMAKE_SOURCE_DIR}/common/queue.c
    ${CMAKE_SOURCE_DIR}/common/util.c
    ${CMAKE_SOURCE_DIR}/common/logger.c
    ${CMAKE_SOURCE_DIR}/common/vecmath.c
    ${CMAKE_SOURCE_DIR}/common/texture.c
    ${CMAKE_SOURCE_DIR}/common/geometry.cpp
    ${CMAKE_SOURCE_DIR}/common/algorithm/dito/dito.cpp
)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../deps/cmake/modules/" ${CMAKE_MODULE_PATH})
set(FreeImage_DIR "" CACHE STRING "Custom FreeImage Library path")
find_package(FreeImage)

if(NOT WIN32)
  target_link_libraries(${PROJECT_NAME} m dl X11)
endif()

if(WIN32)
  target_link_libraries (${PROJECT_NAME} opengl32)
  #add_definitions(-DKR_USE_OPENGL=1)
  add_definitions(-DGLEW_NO_GLU)
  add_definitions(-DGLEW_STATIC)
endif()

if (FreeImage_FOUND)
  include_directories(${FREEIMAGE_INCLUDE_DIR})
  target_link_libraries (${PROJECT_NAME} ${FREEIMAGE_LIBRARY})
  add_definitions(-DKR_HAS_FREEIMAGE=1)
endif (FreeImage_FOUND)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../../bin )
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin/$<CONFIG>")
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_COMMAND_ARGUMENTS "-s ../../rsrc/gltf/crown/crown.gltf -integrator bench")

if(WIN32)
if (FreeImage_FOUND)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "\"${FreeImage_DIR}/FreeImage.dll\"" "\"$(SolutionDir)..\\bin\\$(Configuration)\\FreeImage.dll\""
)
endif()
endif()