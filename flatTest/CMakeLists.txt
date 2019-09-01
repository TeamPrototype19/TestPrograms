cmake_minimum_required(VERSION 3.0)
project(flattest)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -g")

add_subdirectory("flatbuffers")
add_executable(flattest src/main.cpp)
