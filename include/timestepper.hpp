#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscts.h>
#include <petscksp.h>
#include "mastereq.hpp"
#include <assert.h> 
#include <iostream> 
#include "defs.hpp"
#include "output.hpp"
#include "optimtarget.hpp"
#pragma once


/* Base class for time steppers */
class TimeStepper{
  protected:
    int dim;             /* State vector dimension */
    Vec x;               // auxiliary vector needed for time stepping
    bool storeFWD;       /* Flag that determines if primal states should be stored during forward evaluation */
    std::vector<Vec> store_states; /* Storage for primal states */
    bool addLeakagePrevent;   /* flag to determine if Leakage preventing term is added to penalty.  */
    int mpirank_world;

  public:
    MasterEq* mastereq;  // Lindblad master equation
    int ntime;           // number of time steps
    double total_time;   // final time
    double dt;           // time step size

    Vec redgrad;                   /* Reduced gradient */

    /* Stuff needed for the penalty integral term */
    // TODO: pass those through the timestepper constructor (currently, they are set manually inside optimproblem constructor), or add up the penalty within the optim_target.
    double penalty_integral;        // output, holds the integral term
    double penalty_param;
    double gamma_penalty;
    OptimTarget* optim_target;

    /* Output */
    Output* output;

  public: 
    TimeStepper(); 
    TimeStepper(MasterEq* mastereq_, int ntime_, double total_time_, Output* output_, bool storeFWD_); 
    virtual ~TimeStepper(); 

    /* Return the state at a certain time index */
    Vec getState(int tindex);

    /* Solve the ODE forward in time with initial condition rho_t0. Return state at final time step */
    Vec solveODE(int initid, Vec rho_t0);

    /* Solve the adjoint ODE backwards in time from terminal condition rho_t0_bar */
    void solveAdjointODE(int initid, Vec rho_t0_bar, Vec finalstate, double Jbar);

    /* evaluate the penalty integral term */
    double penaltyIntegral(double time, const Vec x);
    void penaltyIntegral_diff(double time, const Vec x, Vec xbar, double Jbar);

    /* Evolve state forward from tstart to tstop */
    virtual void evolveFWD(const double tstart, const double tstop, Vec x) = 0;
    /* Evolve adjoint backward from tstop to tstart and update reduced gradient */
    virtual void evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient);
};

class ExplEuler : public TimeStepper {
  Vec stage;
  public:
    ExplEuler(MasterEq* mastereq_, int ntime_, double total_time_, Output* output_, bool storeFWD_);
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
  LinearSolverType linsolve_type;  // Either GMRES or NEUMANN
  int linsolve_maxiter;            // Maximum number of linear solver iterations
  double linsolve_abstol;          // Absolute stopping criteria for linear solver
  double linsolve_reltol;          // Relative stopping criteria for linear solver
  int linsolve_iterstaken_avg;     // Computing the average number of linear solver iterations
  double linsolve_error_avg;       // Computing the average error of linear solver 
  int linsolve_counter;            // Counting how often a linear solve is performed is called
  Vec tmp, err;                    /* Auxiliary vector for applying the neuman iterations */

  public:
    ImplMidpoint(MasterEq* mastereq_, int ntime_, double total_time_, LinearSolverType linsolve_type_, int linsolve_maxiter_, Output* output_, bool storeFWD_);
    ~ImplMidpoint();


    /* Evolve state forward from tstart to tstop */
    virtual void evolveFWD(const double tstart, const double tstop, Vec x);
    /* Evolve adjoint backward from tstop to tstart and update reduced gradient */
    virtual void evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient);

    /* Solve (I-alpha*A) * x = b using Neumann iterations */
    // bool transpose=true solves the transposed system (I-alpha A^T)x = b
    // Return residual norm ||y-yprev||
    int NeumannSolve(Mat A, Vec b, Vec x, double alpha, bool transpose);
};


class CompositionalImplMidpoint : public ImplMidpoint {

  std::vector<double> gamma;    /* Coefficients for the compositional step sizes */
  std::vector<Vec> x_stage;   /* Storage for primal states at stages */
  Vec aux;
  int order;

  public:
    CompositionalImplMidpoint(int order_, MasterEq* mastereq_, int ntime_, double total_time_, LinearSolverType linsolve_type_, int linsolve_maxiter_, Output* output_, bool storeFWD_);
    ~CompositionalImplMidpoint();

    void evolveFWD(const double tstart, const double tstop, Vec x);
    void evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient);
};