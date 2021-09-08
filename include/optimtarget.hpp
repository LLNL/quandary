#include "defs.hpp"
#include "gate.hpp"
#pragma once

/* Collects stuff specifying the optimization target */
struct OptimTarget{
  TargetType target_type;        /* Type of the optimization (pure-state preparation, or gate optimization) */
  ObjectiveType objective_type;  /* Type of the bjective function */
  Gate *targetgate;              /* The target gate (if any) */
  int purestateID;               /* The integer m for pure-state preparation of the m-the state */
  Vec aux;     /* auxiliary vector needed when computing the objective. Typically holding \rho^tar = V\rho(0)V */

  /* Constructor creates the auxiliary vector */
  OptimTarget(int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_){
    target_type = target_type_;
    objective_type = objective_type_;
    targetgate = targetgate_;
    purestateID = purestateID_;

    if (targetgate != NULL) {
      int dim_rho = targetgate->getDimRho();
      VecCreate(PETSC_COMM_WORLD, &aux); 
      VecSetSizes(aux,PETSC_DECIDE, dim_rho*dim_rho);
      VecSetFromOptions(aux);
    }
  };

  /* Destructor */
  ~OptimTarget(){
    if (targetgate != NULL) VecDestroy(&aux);
  };
};

