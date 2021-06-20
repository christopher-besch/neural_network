cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(neural_net STATIC ${SOURCES})
target_include_directories(neural_net PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_precompile_headers(neural_net PRIVATE
                          "${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h")

target_link_libraries(neural_net PRIVATE utils nlohmann_json)
# armadillo is needed by client
target_link_libraries(neural_net PUBLIC armadillo spdlog)

target_include_directories(neural_net
                           INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
