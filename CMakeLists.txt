cmake_minimum_required(VERSION 3.14)
project(shiva-ai VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

# Include directories for C and C++ bindings
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp
)

# Header-only interface target for C++
add_library(shiva_cpp INTERFACE)
target_include_directories(shiva_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp>
    $<INSTALL_INTERFACE:include>
)

# Header-only interface target for C
add_library(shiva_c INTERFACE)
target_include_directories(shiva_c INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings/c>
    $<INSTALL_INTERFACE:include>
)

# Installation targets
install(FILES bindings/c/shiva.h DESTINATION include)
install(FILES bindings/cpp/shiva.hpp DESTINATION include)
