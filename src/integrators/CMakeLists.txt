cmake_minimum_required(VERSION 2.8.12)

add_subdirectory (pt_gpu)
set_target_properties(pt_gpu PROPERTIES FOLDER "integrators")

add_subdirectory (bench)
set_target_properties(bench PROPERTIES FOLDER "integrators")