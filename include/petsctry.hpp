#pragma once

#include <petscsystypes.h>

/**
 * @brief Call a PETSc function and throw a standard exception if it fails
 * 
 * This macro wraps PETSc function calls and automatically throws a std::runtime_error
 * if the function returns a non-zero error code.
 * 
 * @param func The PETSc function call to execute
 * @throws std::runtime_error if the function fails
 * 
 * Example usage:
 * @code
 * Vec x;
 * PetscTry(VecCreate(PETSC_COMM_WORLD, &x));
 * PetscTry(VecSetSizes(x, PETSC_DECIDE, 100));
 * @endcode
 */
#define PetscTry(func) do { \
    PetscErrorCode ierr = (func); \
    if (PetscUnlikely(ierr)) { \
        throw std::runtime_error("PETSc function " #func " failed with error code " + std::to_string(ierr)); \
    } \
} while(0)
