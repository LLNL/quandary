#include "math.h"
#include <assert.h>
#include <petsctao.h>
#include "defs.hpp"
#include "timestepper.hpp"
#include <iostream>
#include <algorithm>
#include "optimtarget.hpp"
#pragma once

/* if Petsc version < 3.17: Change interface for Tao Optimizer */
#if PETSC_VERSION_MAJOR<4 && PETSC_VERSION_MINOR<17
#define TaoSetObjective TaoSetObjectiveRoutine
#define TaoSetGradient(tao, NULL, TaoEvalGradient,this)  TaoSetGradientRoutine(tao, TaoEvalGradient,this) 
#define TaoSetObjectiveAndGradient(tao, NULL, TaoEvalObjectiveAndGradient,this) TaoSetObjectiveAndGradientRoutine(tao, TaoEvalObjectiveAndGradient,this)
#define TaoSetSolution(tao, xinit) TaoSetInitialVector(tao, xinit)
#define TaoGetSolution(tao, params) TaoGetSolutionVector(tao, params) 
#endif

class OptimProblem {

  /* ODE stuff */
  int ninit;                            /* Number of initial conditions to be considered (N^2, N, or 1) */
  int ninit_local;                      /* Local number of initial conditions on this processor */
  Vec rho_t0;                            /* Storage for initial condition of the ODE */
  Vec rho_t0_bar;                        /* Adjoint of ODE initial condition */
  InitialConditionType initcond_type;    /* Type of initial conditions */
  std::vector<int> initcond_IDs;         /* Integer list for pure-state initialization */
  std::vector<Vec> store_finalstates;    /* Storage for last time steps for each initial condition */

  OptimTarget* optim_target;      /* Storing the optimization goal */

  /* MPI stuff */
  MPI_Comm comm_init;
  MPI_Comm comm_optim;
  int mpirank_optim, mpisize_optim;
  int mpirank_space, mpisize_space;
  int mpirank_world, mpisize_world;
  int mpirank_init, mpisize_init;

  bool quietmode;

  /* Optimization stuff */
  std::vector<double> obj_weights; /* List of weights for weighting the average objective over initial conditions  */
  int ndesign;                     /* Number of global design parameters */
  double objective;                /* Holds current objective function value */
  double obj_cost;                 /* Final-time term J(T) in objective */
  double obj_regul;                /* Regularization term in objective */
  double obj_penal;                /* Penalty term in objective */
  double obj_penal_dpdm;           /* Penalty term in objective for second order state */
  double obj_penal_energy;         /* Energy Penalty term in objective */
  double fidelity;                 /* Final-time fidelity: 1/ninit \sum_iinit Tr(rhotarget^\dag rho(T)) for Lindblad, or |1/ninit \sum_iinit phitarget^dagger phi |^2 for Schroedinger */
  double gnorm;                    /* Holds current norm of gradient */
  double gamma_tik;                /* Parameter for tikhonov regularization */
  bool gamma_tik_interpolate;      /* Switch to use ||x - x0||^2 for tikhonov regularization instead of ||x||^2 */
  double gamma_penalty;            /* Parameter multiplying integral penalty term */
  double gamma_penalty_dpdm;            /* Parameter multiplying integral penalty term */
  double gamma_penalty_energy;     /* Parameter multiplying energy penalty */
  double penalty_param;            /* Parameter inside integral penalty term w(t) (Gaussian variance) */
  double gatol;                    /* Stopping criterion based on absolute gradient norm */
  double fatol;                    /* Stopping criterion based on objective function value */
  double inftol;                   /* Stopping criterion based on infidelity */
  double grtol;                    /* Stopping criterion based on relative gradient norm */
  int maxiter;                     /* Stopping criterion based on maximum number of iterations */
  Tao tao;                         /* Petsc's Optimization solver */
  std::vector<double> initguess_fromfile;      /* Stores the initial guess, if read from file */
  double* mygrad;  /* Auxiliary */
    
  Vec xtmp;                        /* Temporary storage */
  
  public: 
    Output* output;                 /* Store a reference to the output */
    TimeStepper* timestepper;       /* Store a reference to the time-stepping scheme */
    Vec xlower, xupper;              /* Optimization bounds */
    Vec xprev;                       /* design vector at previous iteration */
    Vec xinit;                       /* Storing initial design vector */

  /* Constructor */
  OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_optim, int ninit_, std::vector<double> gate_rot_freq, Output* output_, bool quietmode=false);
  ~OptimProblem();

  /* Return the number of design variables */
  int getNdesign(){ return ndesign; };

  /* Return the overall objective, final-time costs, regularization and penalty terms */
  double getObjective(){ return objective; };
  double getCostT()    { return obj_cost; };
  double getRegul()    { return obj_regul; };
  double getPenalty()  { return obj_penal; };
  double getPenaltyDpDm()  { return obj_penal_dpdm; };
  double getPenaltyEnergy()  { return obj_penal_energy; };
  double getFidelity() { return fidelity; };
  double getFaTol()    { return fatol; };
  double getInfTol()   { return inftol; };
  int getMPIrank_world() { return mpirank_world;};
  int getMaxIter()     { return maxiter; };

  /* Evaluate the objective function F(x) */
  double evalF(const Vec x);

  /* Evaluate gradient \nabla F(x) */
  void evalGradF(const Vec x, Vec G);

  /* Run optimization solver, starting from initial guess xinit */
  void solve(Vec xinit);

  /* Compute initial guess for optimization variables */
  void getStartingPoint(Vec x);

  /* Call this after TaoSolve() has finished to print out some information */
  void getSolution(Vec* opt);
};

/* Monitor the optimization progress. This routine is called in each iteration of TaoSolve() */
PetscErrorCode TaoMonitor(Tao tao,void*ptr);

/* Petsc's Tao interface routine for evaluating the objective function f = f(x) */
PetscErrorCode TaoEvalObjective(Tao tao, Vec x, PetscReal *f, void*ptr);

/* Petsc's Tao interface routine for evaluating the gradient g = \nabla f(x) */
PetscErrorCode TaoEvalGradient(Tao tao, Vec x, Vec G, void*ptr);

/* Petsc's Tao interface routine for evaluating the gradient g = \nabla f(x) */
PetscErrorCode TaoEvalObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec G, void*ptr);
