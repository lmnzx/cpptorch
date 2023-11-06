cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpptorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_INSTALL_RPATH "/Users/lemon/micromamba/envs/cpptorch/lib/python3.11/site-packages/torch/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cpptorch main.cpp)
target_link_libraries(cpptorch ${TORCH_LIBRARIES})
set_property(TARGET cpptorch PROPERTY CXX_STANDARD 17)
