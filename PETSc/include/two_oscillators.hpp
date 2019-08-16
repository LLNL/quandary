#include <petscts.h>
#pragma once


/*
   Petsc's application context containing data needed to perform a time step.
*/
typedef struct {
  PetscInt    nvec;    /* Dimension of vectorized system */
  Mat         IKbMbd, bMbdTKI, aPadTKI, IKaPad, A, B;
  PetscReal   w;       /* Oscillator frequencies */
} TS_App;


/*
 *   Compute the exact solution at a given time.
 *   Input:
 *      t - current time
 *      s - vector in which exact solution will be computed
 *      freq - Oscillator frequency
 *   Output:
 *      s - vector with the newly computed exact solution
 */
PetscErrorCode ExactSolution(PetscReal t,Vec s, PetscReal freq);


/*
 *  Set the initial condition at time t_0
 *  Input:
 *     u - uninitialized solution vector (global)
 *     TS_App - application context
 *  Output Parameter:
 *     u - vector with solution at initial time (global)
 */
PetscErrorCode InitialConditions(Vec x,TS_App *petsc_app);


/*
 * Oscillator 1 (real part)
 */
PetscScalar F(PetscReal t,TS_App *petsc_app);


/*
 * Oscillator 2 (imaginary part)
 */
PetscScalar G(PetscReal t,TS_App *petsc_app);


/*
 * Evaluate the right-hand side system Matrix (real, vectorized Hamiltonian system matrix)
 * In: ts - time stepper
 *      t - current time
 *      u - solution vector x(t) 
 *      M - right hand side system Matrix
 *      P - ??
 *    ctx - Application context 
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx);


/*
 * Initialize fixed matrices for assembling system Hamiltonian
 */
PetscErrorCode SetUpMatrices(TS_App *petsc_app);
