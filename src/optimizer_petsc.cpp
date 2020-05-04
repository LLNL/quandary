#include "optimizer_petsc.hpp"

void optim_getStartingPoint(Vec x, OptimCtx* ctx){

  /* Set initial starting point. Do this on one processor only, then broadcast, to make sure that every processor starts with the same initial guess.  */

}

PetscErrorCode optim_evalObjective(Tao tao, Vec x, PetscReal *f, void*ptr){
    OptimCtx* optimctx = (OptimCtx*) ptr;

    /* TODO: Eval objective */

    return 0;
}


PetscErrorCode optim_evalGradient(Tao tao, Vec x, Vec G, void*ptr){
    OptimCtx* optimctx = (OptimCtx*) ptr;

    /* TODO: Eval objective */

    return 0;
}


