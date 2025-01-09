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

include(blt/SetupBLT.cmake)

add_subdirectory(src)
