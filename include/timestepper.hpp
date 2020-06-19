#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscts.h>
#include <petscksp.h>
#include "mastereq.hpp"
#include <assert.h> 
#include <iostream> 
#pragma once

enum LinearSolverType{
  GMRES,   // uses Petsc's GMRES solver
  NEUMANN   // uses Neuman power iterations 
};



/* Base class for time steppers */
class TimeStepper{
  protected:
    int dim;                   /* State vector dimension */
    MasterEq* mastereq;  

  public: 
    TimeStepper(); 
    TimeStepper(MasterEq* mastereq_); 
    virtual ~TimeStepper(); 

    /* Evolve state forward from tstart to tstop */
    virtual void evolveFWD(const double tstart, const double tstop, Vec x) = 0;
    /* Evolve adjoint backward from tstop to tstart and update reduced gradient */
    virtual void evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient);
};

class ExplEuler : public TimeStepper {
  Vec stage;
  public:
    ExplEuler(MasterEq* mastereq_);
    ~ExplEuler();

    /* Evolve state forward from tstart to tstop */
    void evolveFWD(const double tstart, const double tstop, Vec x);
    /* Evolve adjoint backward from tstop to tstart and update reduced gradient */
    void evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient);
};



/* Implements implicit midpoint rule. 2nd order. Simplectic. 
 * RK tableau:  1/2 |  1/2
 *              ------------
 *                  |   1
 */
class ImplMidpoint : public TimeStepper {

  Vec stage, stage_adj;  /* Intermediate stage vars */
  Vec rhs, rhs_adj;      /* right hand side */
  KSP ksp;               /* Petsc's linear solver context for running GMRES */
  PC  preconditioner;    /* Preconditioner for linear solver */
  int KSPsolve_iterstaken_avg;  // Computing the average number of iterations taken by KSP solve
  int KSPsolve_counter;            // Counting how often KSPsolve is called
  LinearSolverType linsolve_type;  // Either GMRES or NEUMANN
  int linsolve_maxiter;
  Vec tmp;                /* Auxiliary vector for applying the neuman iterations */

  public:
    ImplMidpoint(MasterEq* mastereq_, LinearSolverType linsolve_type_, int linsolve_maxiter_);
    ~ImplMidpoint();


    /* Evolve state forward from tstart to tstop */
    void evolveFWD(const double tstart, const double tstop, Vec x);
    /* Evolve adjoint backward from tstop to tstart and update reduced gradient */
    void evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient);

    /* Solve (I-alpha*A) * x = b using Neumann iterations */
    // bool transpose=true solves the transposed system (I-alpha A^T)x = b
    void NeumannSolve(Mat A, Vec b, Vec x, double alpha, bool transpose);
};



/*
 * Evaluate the right-hand side system Matrix (real, vectorized system matrix)
 * In: ts - time stepper
 *      t - current time
 *      u - solution vector x(t) 
 *      M - right hand side system Matrix
 *      P - ??
 *    ctx - system 
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx);



/* Dervative of RHS wrt control parameters */
PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec y, Mat A, void *ctx);

/*
 * Create Petsc's time stepper 
 */
PetscErrorCode TSInit(TS ts, MasterEq* mastereq, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal, Vec x, bool monitor);

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