cmake_minimum_required(VERSION 3.10)

project(neural_network VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# dependencies
add_subdirectory("${CMAKE_SOURCE_DIR}/vendor/armadillo")
add_subdirectory("${CMAKE_SOURCE_DIR}/vendor/nlohmann_json")
add_subdirectory("${CMAKE_SOURCE_DIR}/vendor/spdlog")

# use warnings and multi threaded
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} /MP2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX")
endif()

# neural net static lib
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/neural_net")

# executables
add_subdirectory("${CMAKE_SOURCE_DIR}/digit_recognition")
add_subdirectory("${CMAKE_SOURCE_DIR}/football")
