cmake_minimum_required(VERSION 3.10)

file(GLOB SOURCES src/*.cpp)
add_library(digit_recognition STATIC ${SOURCES})

target_include_directories(digit_recognition
                           INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
