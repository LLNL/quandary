#include "braid_wrapper.hpp"
#include "math.h"
#include <assert.h>
#include <petsctao.h>

#pragma once

enum ObjectiveType {GATE,             // Compare final state to linear gate transformation of initial cond.
                    EXPECTEDENERGY,   // Minimizes expected energy levels.
                    GROUNDSTATE};     // Compares final state to groundstate (full matrix)



class OptimProblem {

  public: 

  /* ODE stuff */
  myBraidApp* primalbraidapp;         /* Primal BraidApp to carry out PinT forward sim.*/
  myAdjointBraidApp* adjointbraidapp; /* Adjoint BraidApp to carry out PinT backward sim. */
  int ninit;                            /* Number of initial conditions to be considered (N^2, N, or 1) */
  int ninit_local;                      /* Local number of initial conditions on this processor */
  Vec rho_t0;                            /* Storage for initial condition of the ODE */
  Vec rho_t0_bar;                        /* Adjoint of ODE initial condition */
  InitialConditionType initcond_type;    /* Type of initial conditions */
  std::vector<int> initcond_IDs;         /* IDs of subsystem oscillators considered for initial conditions */


  /* MPI stuff */
  MPI_Comm comm_hiop, comm_init;
  int mpirank_braid, mpisize_braid;
  int mpirank_space, mpisize_space;
  int mpirank_optim, mpisize_optim;
  int mpirank_world, mpisize_world;
  int mpirank_init, mpisize_init;

  /* Optimization stuff */
  ObjectiveType objective_type;    /* Type of objective function (Gate, <N>, Groundstate, ... ) */
  std::vector<int> obj_oscilIDs;   /* List of oscillator IDs that are considered for the optimizer */
  Gate  *targetgate;               /* Target gate */
  int ndesign;                     /* Number of global design parameters */
  double objective;                /* Holds current objective function value */
  double obj_cost;                 /* Regularization term in objective */
  double obj_regul;                /* Cost function term in objective */
  double gnorm;                    /* Holds current norm of gradient */
  double gamma_tik;                /* Parameter for tikhonov regularization */
  double gatol;                    /* Stopping criterion based on absolute gradient norm */
  double grtol;                    /* Stopping criterion based on relative gradient norm */
  int maxiter;                     /* Stopping criterion based on maximum number of iterations */
  Tao tao;                        /* Petsc's Optimization solver */
  Vec xinit;                       /* Initial guess */
  Vec xlower, xupper;              /* Optimization bounds */
  std::string initguess_type;      /* Type of initial guess */
  std::vector<double> initguess_amplitudes; /* Initial amplitudes of controles, or NULL */
  
  /* Output */
  int printlevel;      /* Level of output: 0 - no output, 1 - optimization progress to file */
  FILE* optimfile;     /* Output file to log optimization progress */

  /* Constructor */
  OptimProblem(MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, int ninit_);
  ~OptimProblem();

  /* Evaluate the objective function F(x) */
  double evalF(const Vec x);

  /* Evaluate gradient \nabla F(x) */
  void evalGradF(const Vec x, Vec G);

  /* Run optimization solver */
  void solve();

  /* Compute initial guess for optimization variables */
  void getStartingPoint(Vec x);

  /* Compute final-time part of the objective */
  double objectiveT(Vec finalstate);
  /* Derivative of final-time part of objective times obj_bar */
  void objectiveT_diff(Vec finalstate, const double obj_local, const double obj_bar);

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
