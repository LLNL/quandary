cmake_minimum_required(VERSION 3.23)

project(
    Quandary
    VERSION 4.3
    LANGUAGES CXX C
)
set(PROJECT_VERSION_SUFFIX "-dev")
set(FULL_VERSION "${PROJECT_VERSION}${PROJECT_VERSION_SUFFIX}")

option(BUILD_QUANDARY "Build Quandary targets" ON)
option(ENABLE_TESTS "Run unit tests" ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type" FORCE)
endif()

set(BLT_CXX_STD "c++17" CACHE STRING "Version of C++ standard")

set(ENABLE_MPI ON CACHE BOOL "")

################################
# BLT
################################
if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR
          "The BLT git submodule is not present. "
          "Either run the following two commands in your git repository: \n"
          "    git submodule init\n"
          "    git submodule update\n"
          "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif ()
  endif ()

  message(STATUS "BLT Source Directory: ${BLT_SOURCE_DIR}")
  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

# Configure version header
configure_file(
    ${CMAKE_SOURCE_DIR}/include/version.hpp.in
    ${CMAKE_BINARY_DIR}/include/version.hpp
    @ONLY
)

if(BUILD_QUANDARY)
  add_subdirectory(src)
  add_subdirectory(tests/unit)
endif()

add_subdirectory(docs)
