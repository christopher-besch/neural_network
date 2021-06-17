cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(digit_recognition ${SOURCES})

target_precompile_headers(digit_recognition PRIVATE
                          "${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h")

target_link_libraries(digit_recognition PRIVATE utils armadillo nlohmann_json)
target_include_directories(digit_recognition PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
