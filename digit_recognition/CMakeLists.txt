cmake_minimum_required(VERSION 3.10)

file(GLOB SOURCES src/*.cpp)
add_executable(digit_recognition ${SOURCES})

target_precompile_headers(digit_recognition PRIVATE
                          "${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h")

target_link_libraries(digit_recognition PRIVATE utils armadillo nlohmann_json)
