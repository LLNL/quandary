#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
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



/* Dervative of RHS wrt control parameters */
PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec y, Mat A, void *ctx);

/*
 * Create Petsc's time stepper 
 */
PetscErrorCode TSInit(TS ts, Hamiltonian* hamiltonian, int NSteps, double Dt, double Tfinal, bool monitor);

/*
 * Monitor the time stepper 
 */
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec x,void *ctx);
PetscErrorCode AdjointMonitor(TS ts,PetscInt step,PetscReal t,Vec x, PetscInt numcost, Vec* lambda, Vec* mu, void *ctx);


/*
 * Routines for splitting Petsc's TSSolve() into individual time steps.
 * TSPreSolve needs to be called BEFORE the time step loop.
 * TSPostSolve needs to be called AFTER the time step loop.
 * A call to TSSetSolution(ts,x) is required before these routines!
 */
PetscErrorCode TSPreSolve(TS ts);
PetscErrorCode TSStepMod(TS ts);
PetscErrorCode TSPostSolve(TS ts);


/*
 * Routines for splitting Petsc's TSAdjointSolve() into individual time steps.
 * TSAdjointPreSolve needs to be called BEFORE the time step loop.
 * TSAdjointPostSolve needs to be called AFTER the time step loop.
 * To run adjoint steps, a call to TSSetSaveTrajectory(ts) is required before the primal run!
 */
PetscErrorCode TSAdjointPreSolve(TS ts);
PetscErrorCode TSAdjointStepMod(TS ts);
PetscErrorCode TSAdjointPostSolve(TS ts);