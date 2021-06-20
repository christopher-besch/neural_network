cmake_minimum_required(VERSION 3.10)

project(neural_network VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/utils")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/vendor/armadillo")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann_json")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/vendor/spdlog")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/digit_recognition")
