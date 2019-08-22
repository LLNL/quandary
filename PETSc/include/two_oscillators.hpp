#include <petscts.h>
#include "bspline.hpp"
#pragma once


/*
   Petsc's application context containing data needed to perform a time step.
*/
typedef struct {
  PetscInt    nvec;    /* Dimension of vectorized system */
  Mat         IKbMbd, bMbdTKI, aPadTKI, IKaPad, A, B;
  Bspline*    spline;  /* Spline basis functions for oscillator evaluation */
  PetscReal*  spline_coeffs;  /* Spline coefficients (optimization vars) */
} TS_App;


/* 
 * Compute the analytic solution for the 2-oscillator, 2-levels test case.
 */
PetscErrorCode ExactSolution(PetscReal t,Vec s, PetscReal freq);


/*
 *  Set the initial condition at time t_0 to the analytic solution 
 *  of the 2-level, 2-oscillator case.
 */
PetscErrorCode InitialConditions(Vec x,PetscReal freq);


/*
 * Oscillator 1: Evaluate real and imaginary part
 */
PetscScalar F1(PetscReal t,TS_App *petsc_app);  // real 
PetscScalar G1(PetscReal t,TS_App *petsc_app);  // imaginary

/*
 * Oscillator 2: Evaluate real and imaginary part 
 */
PetscScalar F2(PetscReal t,TS_App *petsc_app); // real
PetscScalar G2(PetscReal t,TS_App *petsc_app); // imaginary


/* Real part for Oscillator 1 of analytic solution */
PetscScalar F1(PetscReal t,PetscReal freq);


/* Imaginary part for Oscillator 2 of analytic solution */
PetscScalar G2(PetscReal t,PetscReal freq);

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
