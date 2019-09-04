#include <petscts.h>
#include "bspline.hpp"
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


/*
 *   Compute the matrix C = I_k * (a_1 +- a_1^dg) * I_m
 *   Input:
 *      C - the matrix, which has already been created
 *      n - the number of levels
        s - the sign, +1 => (a_1 + a_1^dg), -1 => (a_1 - a_1^dg)
        k - the number of repetitions of the blocks
        m - the number of repetitions of each entry within the blocks
 *   Output:
 *      C - the assembled matrix with all values inserted
 */
PetscErrorCode BuildingBlock(Mat C, PetscInt n, PetscInt s, PetscInt k, PetscInt m);