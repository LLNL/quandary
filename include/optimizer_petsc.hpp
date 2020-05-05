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
  InitialConditionType initcond_type; /* Type of ODE initial condition (pure, basis, from file, etc.) */
  int ninit;                            /* Number of initial conditions to be considered (N^2, N, or 1) */
  int ninit_local;                      /* Local number of initial conditions on this processor */
  Vec initcond_re, initcond_im;         /* Current initial condition TODO: DESTROY THEM SOME TIME! Or store them in the master eq! */
  Vec initcond_re_bar, initcond_im_bar;   

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
  double gamma_tik;                /* Parameter for tikhonov regularization */

  /* Constructor */
  OptimCtx(MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, std::vector<int> obj_oscilIDs_, InitialConditionType initcondtype_, int ninit_);
  ~OptimCtx();

} OptimCtx;


void OptimTao_Setup(Tao* tao, OptimCtx* ctx, MapParam config, Vec xinit, Vec xlower, Vec xupper);

/* Get initial starting point */
void getStartingPoint(Vec x, OptimCtx* ctx, std::string start_type, std::vector<double> start_amplitudes, std::vector<double> bounds);

/* Evaluate objective function f(x) */
PetscErrorCode optim_evalObjective(Tao tao, Vec x, PetscReal *f, void*ptr);

/* Evaluate gradient g = \nabla f(x) */
PetscErrorCode optim_evalGradient(Tao tao, Vec x, Vec G, void*ptr);

/* Assemble the ODE initial condition */
int optim_assembleInitCond(int iinit, OptimCtx *ctx);

/* Evaluate final time objective function */
double optim_objectiveT(Vec finalstate, OptimCtx *ctx);
void optim_objectiveT_diff(Vec finalstate, double obj_local, double obj_bar, OptimCtx *ctx);