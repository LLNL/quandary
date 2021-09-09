#include "defs.hpp"
#include "gate.hpp"
#pragma once

/* Collects stuff specifying the optimization target */
struct OptimTarget{
  TargetType target_type;        /* Type of the optimization (pure-state preparation, or gate optimization) */
  ObjectiveType objective_type;  /* Type of the bjective function */
  Gate *targetgate;              /* The target gate (if any) */
  int purestateID;               /* The integer m for pure-state preparation of the m-the state */
  Vec aux;     /* auxiliary vector needed when computing the objective for gate optimization */
  Vec VrhoV;   /* For gate optimization: holds the transformed state VrhoV^\dagger */

  /* Constructor creates the auxiliary vector */
  OptimTarget(int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_){
    target_type = target_type_;
    objective_type = objective_type_;
    targetgate = targetgate_;
    purestateID = purestateID_;

    if (targetgate != NULL) {
      int dim_rho = targetgate->getDimRho();
      VecCreate(PETSC_COMM_WORLD, &aux); 
      VecSetSizes(aux,PETSC_DECIDE, 2*dim_rho*dim_rho);
      VecSetFromOptions(aux);

      /* Allocate transformed target state vector VrhoV^\dagger */
      VecCreate(PETSC_COMM_WORLD, &VrhoV); 
      VecSetSizes(VrhoV,PETSC_DECIDE, 2*dim_rho*dim_rho);
      VecSetFromOptions(VrhoV);
    }
  };

  double FrobeniusDistance(const Vec state){
    /* Frobenius distance F = 1/2 || VrhoV - state ||^2_F */
    double norm;
    VecAYPX(aux, 0.0, VrhoV);    // aux = VrhoV
    VecAXPY(aux, -1.0, state);   // aux = VrhoV - state
    VecNorm(aux, NORM_2, &norm);

    return norm * norm / 2.0;
  };

  void FrobeniusDistance_diff(const Vec state, Vec statebar, const double Jbar){
    // Derivative of frobenius distance : statebar += (VrhoV - state) * (-1) * Jbar 
    VecAXPY(statebar,  1.0*Jbar, state);
    VecAXPY(statebar, -1.0*Jbar, VrhoV);  
  };

  /* Destructor */
  ~OptimTarget(){
    if (targetgate != NULL) {
      VecDestroy(&aux);
      VecDestroy(&VrhoV);
    }
  };
};

