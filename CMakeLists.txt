cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpptorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cpptorch src/main.cpp include/network.h)
target_include_directories(cpptorch PRIVATE include)
target_link_libraries(cpptorch ${TORCH_LIBRARIES})
set_property(TARGET cpptorch PROPERTY CXX_STANDARD 17)
