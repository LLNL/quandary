#include "braid_wrapper.hpp"
#include "math.h"
#include <assert.h>
#include <petsctao.h>
#include "optimizer.hpp"

#pragma once

typedef struct {

  myBraidApp* primalbraidapp;         /* Primal BraidApp to carry out PinT forward sim.*/

  /* MPI stuff */
  int mpirank_world;

  /* Optimization parameters */
  std::string optiminit_type;           /* Type of design initialization */

  
} OptimCtx;

/* Get initial starting point */
void optim_getStartingPoint(Vec x, OptimCtx* ctx);

/* Evaluate objective function f(x) */
PetscErrorCode optim_evalObjective(Tao tao, Vec x, PetscReal *f, void*ptr);

/* Evaluate gradient g = \nabla f(x) */
PetscErrorCode optim_evalGradient(Tao tao, Vec x, Vec G, void*ptr);
