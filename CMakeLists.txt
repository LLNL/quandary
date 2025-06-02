cmake_minimum_required(VERSION 3.23)

project(
    Quandary
    VERSION 4.1
    LANGUAGES CXX C
)

set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
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

add_subdirectory(src)
add_subdirectory(docs)
