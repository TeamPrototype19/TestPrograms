project(pool)
cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -g")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(pool "${SRCS}")

