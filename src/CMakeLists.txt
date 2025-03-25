# Local source and header files
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_DIR ${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cpp)
list(REMOVE_ITEM SRC_FILES ${SRC_DIR}/main.cpp)
file(GLOB_RECURSE HEADER_FILES ${INC_DIR}/*.hpp)

# Target definition
blt_add_library(
    NAME quandary_lib
    SOURCES ${SRC_FILES}
    HEADERS ${HEADER_FILES}
    INCLUDES ${INC_DIR}
    DEPENDS_ON blt::mpi
)

blt_add_executable(
    NAME quandary
    SOURCES ${SRC_DIR}/main.cpp
    DEPENDS_ON quandary_lib
)

# PETSc
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_link_libraries(quandary_lib PUBLIC PkgConfig::PETSC)

# ROL optimization package
set(ROL_ROOT "/Users/guenther5/Numerics/rol/build/install/lib/cmake/ROL")
find_package(ROL REQUIRED)
#target_link_libraries(quandary_lib PUBLIC ROL::all_libs)
target_link_libraries(quandary_lib PUBLIC ROL::rol)

# Other libraries
target_link_libraries(quandary_lib PUBLIC m blas lapack)

# Put executable in root directory
set_target_properties(
    quandary
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Optional: Sanity check
option(SANITY_CHECK "Run sanity checks" OFF)
if(SANITY_CHECK)
    target_compile_options(quandary PRIVATE -DSANITY_CHECK)
endif()

# Optional: Link to SLEPC
option(WITH_SLEPC "Link to SLEPC" OFF)
if(WITH_SLEPC)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET SLEPc)
    target_link_libraries(quandary PRIVATE PkgConfig::SLEPC)
    target_compile_options(quandary PRIVATE -DWITH_SLEPC)
endif()

install(
    TARGETS quandary
    RUNTIME DESTINATION bin
)
