#include <petscts.h>
#include "bspline.hpp"
#include "hamiltonian.hpp"
#pragma once


/*
   Petsc's application context containing data needed to perform a time step.
*/
typedef struct {
  PetscInt    nvec;           /* Dimension of vectorized system */
  PetscInt    nlevels;        /* number of levels */
  Mat         A, B;           /* Real and imaginary part of Hamiltonian */
  Mat         A1, A2, B1, B2; /* Constant matrices for constructing A and B */
  Bspline*    spline;         /* Spline basis functions for oscillator evaluation */
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
 *    ctx - Hamiltonian system 
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx);
