#include "braid_wrapper.hpp"
#include "math.h"
#include <assert.h>
#include <petsctao.h>
#include "optimizer.hpp"

#pragma once


typedef struct OptimCtx {

  /* ODE stuff */
  myBraidApp* primalbraidapp;         /* Primal BraidApp to carry out PinT forward sim.*/
  myAdjointBraidApp* adjointbraidapp; /* Adjoint BraidApp to carry out PinT backward sim. */
  int ninit;                            /* Number of initial conditions to be considered (N^2, N, or 1) */
  int ninit_local;                      /* Local number of initial conditions on this processor */
  Vec rho_t0;                            /* Storage for initial condition of the ODE */
  Vec rho_t0_bar;                        /* Adjoint of ODE initial condition */

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
  double objective;                /* Holds current objective function */
  double gnorm;                    /* Holds current norm of gradient */
  double gamma_tik;                /* Parameter for tikhonov regularization */
  double gatol;                    /* Stopping criterion based on absolute gradient norm */
  double grtol;                    /* Stopping criterion based on relative gradient norm */
  int maxiter;                     /* Stopping criterion based on maximum number of iterations */

  /* Output */
  int printlevel;      /* Level of output: 0 - no output, 1 - optimization progress to file */
  FILE* optimfile;     /* Output file to log optimization progress */

  /* Constructor */
  OptimCtx(MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, std::vector<int> obj_oscilIDs_, InitialConditionType initcondtype_, int ninit_);
  ~OptimCtx();


  /* Compute initial guess for optimization variables */
  void getStartingPoint(Vec x, std::string start_type, std::vector<double> start_amplitudes, std::vector<double> bounds);

  /* Evaluate final time objective J(T) */
  double objectiveT(Vec finalstate);
  /* Derivative of final time objective \nabla J(T) * obj_bar */
  void objectiveT_diff(Vec finalstate, double obj_local, double obj_bar);

} OptimCtx;


/* Initialize the Tao optimizer, set options, starting point, etc */
void OptimTao_Setup(Tao* tao, OptimCtx* ctx, MapParam config, Vec xinit, Vec xlower, Vec xupper);

/* Call this after TaoSolve() has finished to print out some information */
void OptimTao_SolutionCallback(Tao* tao, OptimCtx* ctx);

/* Monitor the optimization progress. This routine is called in each iteration of TaoSolve() */
PetscErrorCode OptimTao_Monitor(Tao tao,void*ptr);

/* Petsc's Tao interface routine for evaluating the objective function f = f(x) */
PetscErrorCode OptimTao_EvalObjective(Tao tao, Vec x, PetscReal *f, void*ptr);

/* Petsc's Tao interface routine for evaluating the gradient g = \nabla f(x) */
PetscErrorCode OptimTao_EvalGradient(Tao tao, Vec x, Vec G, void*ptr);
