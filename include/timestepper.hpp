#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscts.h>
#include <petscksp.h>
#include "hamiltonian.hpp"
#include <assert.h> 
#pragma once


/* Base class for time steppers */
class TimeStepper{
  protected:
    int dim;                   /* State vector dimension */
    Hamiltonian* hamiltonian;  

  public: 
    TimeStepper(); 
    TimeStepper(Hamiltonian* hamiltonian_); 
    virtual ~TimeStepper(); 

    /* Evolving x from tstart to tstop (tstart < tstop !) */
    virtual void evolvForward(double tstart, double tstop, Vec x) = 0;
};

/* Implements implicit midpoint rule. 2nd order. Simplectic. 
 * RK tableau:  1/2 |  1/2
 *              ------------
 *                  |   1
 */
class ImplMidpoint : public TimeStepper {

  /* RK variables */
  Vec stage;

  /* Linear system solve */
  Vec rhs;   /* right hand side */
  KSP linearsolver;   /* linear solver context */
  PC  preconditioner; /* Preconditioner for linear solver */

  public:
    ImplMidpoint(Hamiltonian* hamiltonian_);
    ~ImplMidpoint();

    void evolvForward(double tstart, double tstop, Vec x);
};



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
// PetscErrorCode TSInit(TS ts, Hamiltonian* hamiltonian, int NSteps, double Dt, double Tfinal, Vec x, bool monitor);
PetscErrorCode TSInit(TS ts, Hamiltonian* hamiltonian, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal, Vec x, Vec *lambda, Vec *mu, bool monitor);

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
 * Bool tj_store determines, if the trajectory at that step should be saved, or not. 
 */
PetscErrorCode TSPreSolve(TS ts, bool tj_store);
PetscErrorCode TSStepMod(TS ts, bool tj_store);
PetscErrorCode TSPostSolve(TS ts);


/*
 * Routines for splitting Petsc's TSAdjointSolve() into individual time steps.
 * TSAdjointPreSolve needs to be called BEFORE the time step loop.
 * TSAdjointPostSolve needs to be called AFTER the time step loop.
 * To run adjoint steps, a call to TSSetSaveTrajectory(ts) is required before the primal run!
 */
PetscErrorCode TSAdjointPreSolve(TS ts);
PetscErrorCode TSAdjointStepMod(TS ts, bool tj_store);
PetscErrorCode TSAdjointPostSolve(TS ts, bool tj_store);



/* 
 * This sets u to the ts->vec_sensi[0] and ts->vec_sensip[0], which hopefully is PETSC's adjoint variable and reduced gradient
 * This routine closely follows what is done in TSSetSolution. 
 */
PetscErrorCode  TSSetAdjointSolution(TS ts,Vec lambda, Vec mu);