cmake_minimum_required(VERSION 3.10)

project(neural_network VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(utils)
list(APPEND EXTRA_LIBS digit_recognition)
list(APPEND EXTRA_LIBS utils)
list(APPEND EXTRA_LIBS utils)
list(APPEND EXTRA_LIBS utils)

add_executable(neural_network neural_network/src/main.cpp)

target_link_libraries(neural_network PUBLIC ${EXTRA_LIBS})
