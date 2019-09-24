project(opencl)
cmake_minimum_required(VERSION 3.0)
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs"
)

add_definitions( -DCL_TARGET_OPENCL_VERSION=200)
link_libraries( mali )

file(GLOB SRCS 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    )

set(CMAKE_CXX_FLAGS "--std=c++11 -Wall -Wno-comment") 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(opencl "${SRCS}")
