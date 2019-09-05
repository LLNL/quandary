#include <petscts.h>
#include "hamiltonian.hpp"
#pragma once


/*
 * Evaluate the right-hand side system Matrix (real, vectorized Hamiltonian system matrix)
 * In: ts - time stepper
 *      t - current time
 *      u - solution vector x(t) 
 *      M - right hand side system Matrix
 *      P - ??
 *    ctx - Hamiltonian system 
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx);

/*
 * Create Petsc's time stepper 
 */
PetscErrorCode BuildTimeStepper(TS* ts, Hamiltonian* hamiltonian, int NSteps, double Dt, double Tfinal);