cmake_minimum_required(VERSION 3.0)
project(flattest)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -g")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(flattest src/main.cpp)

