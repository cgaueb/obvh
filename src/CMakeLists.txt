cmake_minimum_required(VERSION 2.8.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project (Korangar)

add_subdirectory (integrators)
add_subdirectory (ads)
add_subdirectory (app)

add_dependencies(pt_gpu atrbvh_gpu lbvh_gpu)

add_dependencies(korangar pt_gpu bench)