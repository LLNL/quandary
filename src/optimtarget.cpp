#include "optimtarget.hpp"


OptimTarget::OptimTarget(int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_){

  // initialize
  target_type = target_type_;
  objective_type = objective_type_;
  targetgate = targetgate_;
  purestateID = purestateID_;

  /* Allocate gate specific vars */
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
}

OptimTarget::~OptimTarget(){
  if (targetgate != NULL) {
    VecDestroy(&aux);
    VecDestroy(&VrhoV);
  }
}

double OptimTarget::FrobeniusDistance(const Vec state){
  // Frobenius distance F = 1/2 || VrhoV - state ||^2_F  = 1/2 || vec(VrhoV-state)||^2_2
  double norm;
  VecAYPX(aux, 0.0, VrhoV);    // aux = VrhoV
  VecAXPY(aux, -1.0, state);   // aux = VrhoV - state
  VecNorm(aux, NORM_2, &norm);
  double J = norm * norm;

  return J;
}

void OptimTarget::FrobeniusDistance_diff(const Vec state, Vec statebar, const double Jbar){
  
  // Derivative of frobenius distance : statebar += 2 * (VrhoV - state) * (-1) * Jbar 
  VecAXPY(statebar,  2.0*Jbar, state);
  VecAXPY(statebar, -2.0*Jbar, VrhoV);  
}

double OptimTarget::HilbertSchmidtOverlap(const Vec state, bool scalebypurity) {
  // Tr(VrhoV*state) = vec(VrhoV)^dagger vec(state) 
  double J = 0.0;
  VecTDot(VrhoV, state, &J);
  // scale by purity Tr(VrhoV^2) = || vec(VrhoV)||^2_2
  if (scalebypurity){
    double dot;
    VecNorm(VrhoV, NORM_2, &dot);
    J = J / (dot*dot);
  }
  return J;
}

void OptimTarget::HilbertSchmidtOverlap_diff(const Vec state, Vec statebar, const double Jbar, bool scalebypurity){
  // Derivative of Trace: statebar += VrhoV^\dagger Jbar / scale
  double scale = 1.0;
  if (scalebypurity){
    double dot;
    VecNorm(VrhoV, NORM_2, &dot);
    scale = dot*dot;
  }
  VecAXPY(statebar, Jbar/scale, VrhoV);
}



